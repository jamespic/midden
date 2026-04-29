import array
from typing import Iterator

from xxhash import xxh3_128_digest


class PersistentBitSet:
    """A bitset that's immutable, and can be efficiently updated.

    New bitsets share most of the same data"""

    def union(self, other: "PersistentBitSet") -> "PersistentBitSet":
        """Return a new PersistentBitSet that is the union of this and the other."""
        raise NotImplementedError()

    def add(self, element: int) -> "PersistentBitSet":
        """Return a new PersistentBitSet with the element added."""
        raise NotImplementedError()

    def contains(self, element: int) -> bool:
        """Return True if the element is in the set."""
        raise NotImplementedError()

    def __iter__(self) -> Iterator[int]:
        """Iterate over the elements in the set."""
        raise NotImplementedError()

    def unique_hash(self) -> bytes:
        """Return a 128-bit hash of the set that is the same for sets with the same elements."""
        raise NotImplementedError()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PersistentBitSet):
            return NotImplemented
        return self.unique_hash() == other.unique_hash()

    def __hash__(self) -> int:
        return hash(self.unique_hash())

    @staticmethod
    def empty() -> "PersistentBitSet":
        """Return an empty PersistentBitSet."""
        return _EMPTY

    @staticmethod
    def from_elements(elements: Iterator[int]) -> "PersistentBitSet":
        """Create a PersistentBitSet from an iterable of elements."""
        bitset = PersistentBitSet.empty()
        for element in elements:
            bitset = bitset.add(element)
        return bitset

    def __repr__(self) -> str:
        return f"PersistentBitSet.from_elements({list(self)})"

    def __str__(self) -> str:
        return "{" + ", ".join(str(e) for e in self) + "}"


_FANOUT = 8  # Number of child nodes of each branch node
_LEAF_BITS = 128  # Number of bits stored in each leaf node. Use 128 to align with 128-bit hashes and make hashing easier.
_LEAF_DOUBLES = (
    _LEAF_BITS // 64
)  # Number of 64-bit integers needed to store the bits in a leaf node


class _EmptyBitSet(PersistentBitSet):
    """The empty bitset, which is the identity for union."""

    _HASH = bytes(16)  # 128-bit hash of the empty set, all zeros

    def union(self, other: "PersistentBitSet") -> "PersistentBitSet":
        return other

    def contains(self, element: int) -> bool:
        return False

    def add(self, element: int) -> "PersistentBitSet":
        return _with_single_bit_set(element)

    def __iter__(self) -> Iterator[int]:
        return iter(())

    def unique_hash(self) -> bytes:
        return self._HASH


_EMPTY = _EmptyBitSet()


class _LeafBitSet(PersistentBitSet):
    """A leaf node in the bitset tree, which stores a fixed number of bits."""

    __slots__ = ["bits"]
    bits: array.array

    def __init__(self, bits: array.array | None = None):
        if bits is not None:
            self.bits = bits
        else:
            self.bits = array.array("Q", [0] * _LEAF_DOUBLES)

    def union(self, other: "PersistentBitSet") -> "PersistentBitSet":
        if isinstance(other, _EmptyBitSet):
            return self
        elif isinstance(other, _BranchBitSet):
            return other.union(self)
        elif isinstance(other, _LeafBitSet):
            if self.bits == other.bits:
                return self
            new_bits = array.array("Q", (a | b for a, b in zip(self.bits, other.bits)))
            return _LeafBitSet(bits=new_bits)
        else:
            raise TypeError(f"Unexpected PersistentBitSet type: {type(other)}")

    def contains(self, element: int) -> bool:
        if element >= _LEAF_BITS:
            return False
        else:
            bit_index = element // 64
            bit_offset = element % 64
            return (self.bits[bit_index] & (1 << bit_offset)) != 0

    def add(self, element: int) -> "PersistentBitSet":
        if self.contains(element):
            return self
        elif element >= _LEAF_BITS:
            # Need to convert to a branch node to accommodate the new bit
            return _with_single_bit_set(element).union(self)
        else:
            new_bits = array.array("Q", self.bits)
            bit_index = element // 64
            bit_offset = element % 64
            new_bits[bit_index] |= 1 << bit_offset
            return _LeafBitSet(bits=new_bits)

    def unique_hash(self) -> bytes:
        """Just use the bytes underpinning the bits array as the hash, since it's already a fixed-size representation of the set."""
        return self.bits.tobytes()

    def __iter__(self) -> Iterator[int]:
        for bit_index in range(_LEAF_BITS):
            bit_offset = bit_index % 64
            if (self.bits[bit_index // 64] & (1 << bit_offset)) != 0:
                yield bit_index


class _BranchBitSet(PersistentBitSet):
    """A branch node in the bitset tree, which stores up to _FANOUT child nodes."""

    __slots__ = ["level", "children", "hash"]
    children: list[PersistentBitSet]
    level: int
    hash: bytes | None

    def __init__(self, level: int, children: list[PersistentBitSet] | None = None):
        assert level > 0, "_BranchBitSet must have level > 0"
        self.level = level
        if children is not None:
            self.children = children
        else:
            self.children = [_EMPTY for _ in range(_FANOUT)]
        self.hash = None

    def contains(self, element: int) -> bool:
        child_index, child_offset = divmod(
            element, (_FANOUT ** (self.level - 1)) * _LEAF_BITS
        )
        if child_index >= _FANOUT:
            return False
        return self.children[child_index].contains(child_offset)

    def union(self, other: "PersistentBitSet") -> "PersistentBitSet":
        if isinstance(other, _EmptyBitSet):
            return self
        elif self.unique_hash() == other.unique_hash():
            return self
        elif (
            isinstance(other, _LeafBitSet)
            or isinstance(other, _BranchBitSet)
            and self.level > other.level
        ):
            return _BranchBitSet(
                level=self.level,
                children=[self.children[0].union(other), *self.children[1:]],
            )
        elif isinstance(other, _BranchBitSet):
            if self.level == other.level:
                new_children = [
                    c1.union(c2) for c1, c2 in zip(self.children, other.children)
                ]
                return _BranchBitSet(level=self.level, children=new_children)
            elif self.level < other.level:
                return other.union(self)  # Bigger tree handles merging
        assert False, "This code should be unreachable"

    def add(self, element: int) -> "PersistentBitSet":
        if self.contains(element):
            return self
        else:
            return self.union(_with_single_bit_set(element))

    def unique_hash(self) -> bytes:
        """Compute the hash of this branch node based on the hashes of its children."""
        if self.hash is None:
            self.hash = xxh3_128_digest(
                b"".join(child.unique_hash() for child in self.children), self.level
            )
        return self.hash

    def __iter__(self) -> Iterator[int]:
        for i, child in enumerate(self.children):
            child_offset = i * (_FANOUT ** (self.level - 1)) * _LEAF_BITS
            for bit in child:
                yield child_offset + bit


def _with_single_bit_set(bit: int):
    """Return a PersistentBitSet with only the given bit set."""
    if bit < 0:
        raise ValueError("bit must be non-negative")
    leaf_index = bit // _LEAF_BITS
    leaf_bit_index = bit % _LEAF_BITS
    leaf_bits = array.array("Q", [0] * _LEAF_DOUBLES)
    leaf_bits[leaf_bit_index // 64] = 1 << (leaf_bit_index % 64)
    result = _LeafBitSet(bits=leaf_bits)

    # Make parent branch nodes as needed
    current_level = 0
    while leaf_index > 0:
        parent_index = leaf_index % _FANOUT
        children: list[PersistentBitSet] = [_EMPTY for _ in range(_FANOUT)]
        children[parent_index] = result
        new_branch = _BranchBitSet(level=current_level + 1, children=children)
        result = new_branch
        leaf_index //= _FANOUT
        current_level += 1

    return result
