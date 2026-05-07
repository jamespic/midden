from re import S
from enum import Enum
from midden.util.long_stack import run_with_long_stack
from collections.abc import Generator, Callable
from typing import TypeVar, Generic, Iterable
from dataclasses import dataclass

NodeAccT = TypeVar("NodeAccT")
SCCAccT = TypeVar("SCCAccT")
NodeIdT = TypeVar("NodeIdT")
NodeT = TypeVar("NodeT")


@dataclass(slots=True)
class _BookkeepingEntry:
    index: int
    lowlink: int
    on_stack: bool
    scc: int | None = None


@dataclass(slots=True)
class _NodeInfo(Generic[NodeIdT, NodeAccT]):
    id: NodeIdT
    acc: NodeAccT


class GraphSCCVisitor(Generic[NodeT, NodeIdT, NodeAccT, SCCAccT]):
    """An implementation or Tarjan's algorithm for finding strongly connected components (SCCs)
    in a directed graph, with support for accumulating values across nodes and SCCs.
    """

    def iterate_nodes(
        self, already_visited: Callable[[NodeIdT], bool]
    ) -> Iterable[NodeT]:
        """Override this to iterate over all nodes in the graph."""
        raise NotImplementedError

    def get_node_id(self, node: NodeT) -> NodeIdT:
        """Override this to get node information for a given node."""
        raise NotImplementedError

    def get_node_acc(self, node: NodeT) -> NodeAccT:
        """Override this to get the initial accumulator value for a given node."""
        raise NotImplementedError

    def get_successors(self, node: NodeT) -> Iterable[NodeT]:
        """Override this to get the successors of a given node."""
        raise NotImplementedError

    def accumulate_node_values(self, v1: NodeAccT, v2: NodeAccT) -> NodeAccT:
        """Override this to define how to accumulate two node values. This is used to combine values of nodes in the same SCC."""
        raise NotImplementedError

    def accumulate_scc_values(self, scc_acc: SCCAccT, child_scc_acc: SCCAccT) -> SCCAccT:
        """Override this to define how to accumulate values of child SCCs into a parent SCC value."""
        raise NotImplementedError

    def add_node_value_to_scc_value(
        self,
        node_acc: NodeAccT,
        this_scc: int,
        scc_acc: SCCAccT | None,
    ) -> SCCAccT:
        """Override this to define how to accumulate node and child SCC values into an SCC value."""
        raise NotImplementedError

    def emit_result(self, node_id: NodeIdT, scc_acc: SCCAccT):
        """Override this to do something with the result for each node after processing.
        node_id is the ID of the node, and scc_acc is the accumulated value for its SCC."""
        raise NotImplementedError


def visit_sccs(visitor: GraphSCCVisitor[NodeT, NodeIdT, NodeAccT, SCCAccT]) -> None:
    """Run Tarjan's algorithm. For each SCC, visitor.emit_result will be called for each
    member of the SCC with the accumulated value for that SCC."""

    bookkeeping: dict[NodeIdT, _BookkeepingEntry] = {}
    stack: list[NodeIdT] = []
    index = 0
    next_scc_index = 0
    scc_accs: dict[int, SCCAccT] = {}
    

    def strongconnect(
        obj: NodeT,
    ) -> Generator[
        tuple[NodeAccT|None, SCCAccT|None], None, tuple[NodeAccT|None, SCCAccT|None]
    ]:  # Returns set of SCCs that are children of this node
        nonlocal index, next_scc_index
        obj_id = visitor.get_node_id(obj)
        entry = _BookkeepingEntry(index=index, lowlink=index, on_stack=True)
        bookkeeping[obj_id] = entry
        node_acc = visitor.get_node_acc(obj)
        stack.append(obj_id)
        index += 1
        scc_acc: SCCAccT | None = None
        def _acc_scc(other: SCCAccT | None) -> None:
            nonlocal scc_acc
            if scc_acc is None:
                scc_acc = other
                return
            if other is None:
                return
            scc_acc = visitor.accumulate_scc_values(scc_acc, other)

        for successor in visitor.get_successors(obj):
            # Don't include references from modules in the graph, since they create huge SCCs that aren't interesting
            successor_id = visitor.get_node_id(successor)
            bookkeeping_entry = bookkeeping.get(successor_id)
            if bookkeeping_entry is None:
                child_node_acc, child_scc_acc = yield strongconnect(successor)
                if child_node_acc is not None:
                    node_acc = visitor.accumulate_node_values(node_acc, child_node_acc)
                _acc_scc(child_scc_acc)
                entry.lowlink = min(entry.lowlink, bookkeeping[successor_id].lowlink)
            elif bookkeeping_entry.on_stack:
                entry.lowlink = min(entry.lowlink, bookkeeping_entry.index)
            else:
                linked_scc = bookkeeping[successor_id].scc
                assert linked_scc is not None
                _acc_scc(scc_accs[linked_scc])

        if entry.index == entry.lowlink:
            # Found an SCC root, pop the stack and calculate size
            scc = next_scc_index
            next_scc_index += 1

            scc_members: set[NodeIdT] = set()
            while True:
                member = stack.pop()
                scc_members.add(member)
                bookkeeping_item = bookkeeping[member]
                bookkeeping_item.on_stack = False
                bookkeeping_item.scc = scc

                if member == obj_id:
                    break
            scc_acc = visitor.add_node_value_to_scc_value(
                node_acc,
                scc,
                scc_acc
            )
            scc_accs[scc] = scc_acc

            for member_id in scc_members:
                visitor.emit_result(member_id, scc_acc)
            node_acc = None  # Node accumulators only accumulate within SCCs. Propagate none to parent SCCs, since they shouldn't be used there.

        return node_acc, scc_acc

    for obj in visitor.iterate_nodes(bookkeeping.__contains__):
        obj_id = visitor.get_node_id(obj)
        if obj_id not in bookkeeping:
            run_with_long_stack(strongconnect(obj))
