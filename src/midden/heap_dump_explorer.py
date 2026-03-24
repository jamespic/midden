"""A class that allows exploring heap dumps exported by dump_heap.py.
It uses LMDB to store the data on disk and provides methods for querying objects, their types, and their relationships.
"""

from .long_stack import run_with_long_stack
from dataclasses import dataclass

import struct
from collections import Counter
from lmdb import Environment, Transaction
from threading import local
from functools import wraps
from typing import (
    ParamSpec,
    Callable,
    TypeVar,
    Iterable,
    Generator,
)

from pydantic import BaseModel, ConfigDict


P = ParamSpec("P")
T = TypeVar("T")
F = Callable[P, T]
M = TypeVar("M", bound=BaseModel)


class ObjectSummary(BaseModel):
    id: int
    type: str
    value: str | int | float | None = None
    size: int
    subtree_size: int = 0
    model_config = ConfigDict(extra="ignore")


class _RawObjectRecord(BaseModel):
    id: int
    type: str
    references: list[int]
    value: str | int | float | None = None
    size: int
    subtree_size: int = 0
    model_config = ConfigDict(extra="forbid")


class ObjectRecord(BaseModel):
    id: int
    type: str
    references: list[ObjectSummary]
    referrers: list[ObjectSummary]
    value: str | int | float | None = None
    size: int
    subtree_size: int = 0


class _ObjectRecordNoValue(BaseModel):
    id: int
    type: str
    references: list[int]
    size: int
    model_config = ConfigDict(extra="ignore")


class _SizeIndexEntry(BaseModel):
    size: int
    obj_id: int

    def _pack(self) -> bytes:
        # Pack by size descending, then by obj_id ascending, so we can sort by size in LMDB
        return struct.pack(">qQ", -self.size, self.obj_id)

    @staticmethod
    def _unpack(data: bytes) -> "_SizeIndexEntry":
        size, obj_id = struct.unpack(">qQ", data)
        return _SizeIndexEntry(size=-size, obj_id=obj_id)


PRIMARY_DB = b"primary"
REFERRERS_DB = b"referrers"
TYPES_DB = b"types"
TYPES_SIZE_INDEX_DB = b"types_size_index"
TYPES_SUBTREE_SIZE_INDEX_DB = b"types_subtree_size_index"
PAGE_SIZE = 1000  # Hardcode this for now


def _pack_id(obj_id: int) -> bytes:
    return struct.pack("@n", obj_id)


def _unpack_id(data: bytes) -> int:
    return struct.unpack("@n", data)[0]


_tx_local: local[Transaction] = local()


def tx(f: F) -> F:
    return _wrap_txn(f, write=False)


def tx_write(f: F) -> F:
    return _wrap_txn(f, write=True)


def _wrap_txn(f: F, write=False) -> F:
    @wraps(f)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        self = args[0]
        assert isinstance(self, HeapDumpExplorer)
        if hasattr(_tx_local, "txn"):
            # Support nested transactions by reusing the same one. LMDB supports this.
            return f(*args, **kwargs)
        with self._env.begin(write=write) as txn:
            _tx_local.txn = txn
            try:
                return f(*args, **kwargs)
            finally:
                del _tx_local.txn

    return wrapper


def txn() -> Transaction:
    return _tx_local.txn


class HeapDumpExplorer:
    def __init__(self, db_path="/tmp/heap_dump_db"):
        self._env = Environment(
            db_path, map_size=10 * 1024 * 1024 * 1024, max_dbs=5
        )  # 10 GB
        self._primary_db = self._env.open_db(PRIMARY_DB, integerkey=True, create=True)
        self._referrers_db = self._env.open_db(
            REFERRERS_DB, integerdup=True, create=True
        )
        self._types_db = self._env.open_db(TYPES_DB, dupsort=True, create=True)
        self._types_size_index_db = self._env.open_db(
            TYPES_SIZE_INDEX_DB, dupsort=True, create=True
        )
        self._types_subtree_size_index_db = self._env.open_db(
            TYPES_SUBTREE_SIZE_INDEX_DB, dupsort=True, create=True
        )

    def import_dump(self, dump_path="/tmp/dump.jsonl"):
        with open(dump_path, "rb") as f:
            self.import_lines(f)

    @tx_write
    def import_lines(self, lines: Iterable[bytes]):
        for line in lines:
            record = _RawObjectRecord.model_validate_json(line)
            obj_id = record.id
            encoded_obj_id = _pack_id(obj_id)

            txn().put(encoded_obj_id, line, db=self._primary_db)

            for reference in record.references:
                txn().put(
                    _pack_id(reference),
                    encoded_obj_id,
                    dupdata=True,
                    db=self._referrers_db,
                )

            type_name = record.type
            txn().put(
                type_name.encode(), encoded_obj_id, dupdata=True, db=self._types_db
            )

        self._calculate_subtree_sizes()
        self._build_size_indexes()

    @tx
    def get_object(self, obj_id: int) -> ObjectRecord | None:
        data = self._load_and_validate(obj_id, _RawObjectRecord)
        if data is None:
            return None

        referrers = []
        cursor = txn().cursor(db=self._referrers_db)
        if cursor.set_key(_pack_id(obj_id)):
            for value in cursor.iternext_dup():
                referrers.append(_unpack_id(value))

        return ObjectRecord(
            id=data.id,
            type=data.type,
            value=data.value,
            size=data.size,
            subtree_size=data.subtree_size,
            references=self._get_summaries_for_ids(data.references),
            referrers=self._get_summaries_for_ids(referrers),
        )

    def _get_summaries_for_ids(self, ids: list[int]) -> list[ObjectSummary]:
        summaries = []
        for obj_id in ids:
            summary = self._load_and_validate(obj_id, ObjectSummary)
            if summary is not None:
                summaries.append(summary)
        return summaries

    def _load_and_validate(self, id: int, model: type[M]) -> M | None:
        data = txn().get(_pack_id(id), db=self._primary_db)
        if data is None:
            return None
        return model.model_validate_json(data)

    def _iterate_all_objects(self, model: type[M]) -> Iterable[M]:
        cursor = txn().cursor(db=self._primary_db)
        for value in cursor.iternext(keys=False, values=True):
            assert isinstance(value, bytes)
            yield model.model_validate_json(value)

    @tx
    def get_type_counts(self) -> list[tuple[str, int]]:
        type_counts: Counter[str] = Counter()
        cursor = txn().cursor(db=self._types_db)
        for key in cursor.iternext_nodup(keys=True, values=False):
            assert isinstance(key, bytes)
            type_counts[key.decode()] += cursor.count()
        return type_counts.most_common()

    @tx
    def get_count_for_type(self, type_name: str) -> int:
        cursor = txn().cursor(db=self._types_db)
        if cursor.set_key(type_name.encode()):
            return cursor.count()
        return 0

    @tx
    def get_page_count_for_type(self, type_name: str) -> int:
        count = self.get_count_for_type(type_name)
        return (count + PAGE_SIZE - 1) // PAGE_SIZE

    @tx
    def get_objects_by_type(
        self, type_name: str, page: int | None = None
    ) -> list[ObjectSummary]:
        objects: list[ObjectSummary] = []
        cursor = txn().cursor(db=self._types_db)
        if cursor.set_key(type_name.encode()):
            if page is not None:
                for _ in range(page * PAGE_SIZE):
                    # If this naive pagination proves too slow, we can build a separate index for pagination
                    if not cursor.next_dup():
                        return []
            for i, value in enumerate(cursor.iternext_dup()):
                if page is not None and i >= PAGE_SIZE:
                    break
                obj_id = _unpack_id(value)
                summary = self._load_and_validate(obj_id, ObjectSummary)
                if summary is not None:
                    objects.append(summary)

        return objects

    @tx
    def get_objects_by_type_ordered_by_size(
        self, type_name: str, subtree_size=False, page: int | None = None
    ) -> list[ObjectSummary]:
        objects: list[ObjectSummary] = []
        index_db = (
            self._types_subtree_size_index_db
            if subtree_size
            else self._types_size_index_db
        )
        cursor = txn().cursor(db=index_db)
        if cursor.set_key(type_name.encode()):
            if page is not None:
                for _ in range(page * PAGE_SIZE):
                    if not cursor.next_dup():
                        return []
            for i, value in enumerate(cursor.iternext_dup()):
                if page is not None and i >= PAGE_SIZE:
                    break
                index_entry = _SizeIndexEntry._unpack(value)
                obj_id = index_entry.obj_id
                summary = self._load_and_validate(obj_id, ObjectSummary)
                if summary is not None:
                    objects.append(summary)

        return objects

    def _build_size_indexes(self):
        # Build indexes for types by size and subtree size
        t = txn()
        for obj in self._iterate_all_objects(_RawObjectRecord):
            size_index_entry = _SizeIndexEntry(size=obj.size, obj_id=obj.id)
            t.put(
                obj.type.encode(),
                size_index_entry._pack(),
                dupdata=True,
                db=self._types_size_index_db,
            )
            subtree_size_index_entry = _SizeIndexEntry(
                size=obj.subtree_size, obj_id=obj.id
            )
            t.put(
                obj.type.encode(),
                subtree_size_index_entry._pack(),
                dupdata=True,
                db=self._types_subtree_size_index_db,
            )

    def _calculate_subtree_sizes(self):
        # Calculate subtree sizes using Tarjan
        t = txn()

        @dataclass(slots=True)
        class BookkeepingEntry:
            index: int
            lowlink: int
            on_stack: bool

        @dataclass(slots=True)
        class StackEntry:
            obj_id: int
            size: int

        bookkeeping: dict[int, BookkeepingEntry] = {}
        stack: list[StackEntry] = []
        index = 0
        scc_sizes: dict[int, int] = {}
        obj_id_to_scc: dict[int, int] = {}
        known_module_ids: set[int] = set()

        def strongconnect(
            obj: _ObjectRecordNoValue,
        ) -> Generator[
            set[int], None, set[int]
        ]:  # Returns set of SCCs that are children of this node
            nonlocal index
            entry = BookkeepingEntry(index=index, lowlink=index, on_stack=True)
            bookkeeping[obj.id] = entry
            stack.append(StackEntry(obj_id=obj.id, size=obj.size))
            index += 1
            child_sccs: set[int] = set()

            for ref_id in obj.references:
                # Don't include references from modules in the graph, since they create huge SCCs that aren't interesting
                if ref_id in known_module_ids:
                    continue
                ref = self._load_and_validate(ref_id, _ObjectRecordNoValue)
                if ref is None or ref.type == "builtins.module":
                    known_module_ids.add(ref_id)
                    continue
                bookkeeping_entry = bookkeeping.get(ref_id)
                if bookkeeping_entry is None:
                    child_sccs.update((yield strongconnect(ref)))
                    entry.lowlink = min(entry.lowlink, bookkeeping[ref_id].lowlink)
                elif bookkeeping_entry.on_stack:
                    # Use lowlink variant, so lowlink will point to the root of the SCC, not just the first node we saw
                    entry.lowlink = min(entry.lowlink, bookkeeping_entry.index)
                else:
                    child_sccs.add(obj_id_to_scc[ref_id])

            if entry.index == entry.lowlink:
                # Found an SCC root, pop the stack and calculate size
                scc_size = 0
                scc_members = set()
                while True:
                    member = stack.pop()
                    bookkeeping[member.obj_id].on_stack = False
                    scc_size += member.size
                    scc_members.add(member.obj_id)
                    obj_id_to_scc[member.obj_id] = entry.index
                    if member.obj_id == obj.id:
                        break
                scc_sizes[entry.index] = scc_size

                subtree_size = scc_size + sum(
                    scc_sizes[child_scc] for child_scc in child_sccs
                )
                for member_id in scc_members:
                    record = self._load_and_validate(member_id, _RawObjectRecord)
                    assert record is not None
                    record.subtree_size = subtree_size
                    t.put(
                        _pack_id(member_id),
                        record.model_dump_json().encode(),
                        db=self._primary_db,
                    )

                return child_sccs.union({entry.index})
            return child_sccs

        for obj in self._iterate_all_objects(_ObjectRecordNoValue):
            if obj.id not in bookkeeping:
                run_with_long_stack(strongconnect(obj))
