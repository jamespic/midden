from collections.abc import Iterable, Callable
from dataclasses import dataclass
from ..tarjan import GraphSCCVisitor, visit_sccs


@dataclass
class Accumulator:
    ids_in_this_scc: set[int]
    reachable_ids: set[int]


class TestVisitor(GraphSCCVisitor[int, int, set[int], Accumulator]):
    def __init__(self, graph: dict[int, list[int]]):
        self.graph = graph
        self.results: dict[int, Accumulator] = {}

    def iterate_nodes(self, already_visited: Callable[[int], bool]) -> Iterable[int]:
        for node in self.graph:
            if not already_visited(node):
                yield node

    def get_successors(self, node: int) -> list[int]:
        return self.graph[node]

    def get_node_id(self, node: int) -> int:
        return node

    def get_node_acc(self, node: int) -> set[int]:
        return {node}

    def accumulate_node_values(self, v1: set[int], v2: set[int]) -> set[int]:
        return v1.union(v2)

    def accumulate_scc_values(
        self, scc_acc: Accumulator, child_scc_acc: Accumulator
    ) -> Accumulator:
        return Accumulator(
            ids_in_this_scc=scc_acc.ids_in_this_scc,
            reachable_ids=scc_acc.reachable_ids.union(child_scc_acc.reachable_ids),
        )

    def add_node_value_to_scc_value(
        self, node_acc: set[int], this_scc: int, scc_acc: Accumulator | None
    ) -> Accumulator:
        if scc_acc is None:
            return Accumulator(ids_in_this_scc=node_acc, reachable_ids=node_acc)
        else:
            return Accumulator(
                ids_in_this_scc=node_acc,
                reachable_ids=scc_acc.reachable_ids.union(node_acc),
            )

    def emit_result(self, node_id: int, scc_acc: Accumulator):
        self.results[node_id] = scc_acc


test_graph = {
    # Example from Wikipedia page on Tarjan's algorithm
    1: [2],
    2: [3],
    3: [1],
    4: [2, 5],
    5: [4, 6],
    6: [3, 7],
    7: [6],
    8: [5, 7, 8],
}


def test_tarjan():
    visitor = TestVisitor(test_graph)
    visit_sccs(visitor)
    assert visitor.results == {
        1: Accumulator(ids_in_this_scc={1, 2, 3}, reachable_ids={1, 2, 3}),
        2: Accumulator(ids_in_this_scc={1, 2, 3}, reachable_ids={1, 2, 3}),
        3: Accumulator(ids_in_this_scc={1, 2, 3}, reachable_ids={1, 2, 3}),
        4: Accumulator(ids_in_this_scc={4, 5}, reachable_ids={1, 2, 3, 4, 5, 6, 7}),
        5: Accumulator(ids_in_this_scc={4, 5}, reachable_ids={1, 2, 3, 4, 5, 6, 7}),
        6: Accumulator(ids_in_this_scc={6, 7}, reachable_ids={1, 2, 3, 6, 7}),
        7: Accumulator(ids_in_this_scc={6, 7}, reachable_ids={1, 2, 3, 6, 7}),
        8: Accumulator(ids_in_this_scc={8}, reachable_ids={1, 2, 3, 4, 5, 6, 7, 8}),
    }
