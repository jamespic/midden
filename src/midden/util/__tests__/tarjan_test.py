from collections.abc import Iterable, Callable
from dataclasses import dataclass
from ..tarjan import GraphSCCVisitor, visit_sccs


@dataclass
class Count:
    count_in_this_scc: int
    reachable_count: int


class TestVisitor(GraphSCCVisitor[int, int, int, Count]):
    def __init__(self, graph: dict[int, list[int]]):
        self.graph = graph
        self.results: dict[int, Count] = {}

    def iterate_nodes(self, already_visited: Callable[[int], bool]) -> Iterable[int]:
        for node in self.graph:
            if not already_visited(node):
                yield node

    def get_successors(self, node: int) -> list[int]:
        return self.graph[node]

    def get_node_id(self, node: int) -> int:
        return node

    def get_node_acc(self, node: int) -> int:
        return 1

    def accumulate_node_values(self, v1: int, v2: int) -> int:
        return v1 + v2

    def accumulate(
        self, node_acc: int, this_scc: int, scc_values: Iterable[tuple[int, Count]]
    ) -> Count:
        reachable_count = (
            sum(scc_value.count_in_this_scc for _, scc_value in scc_values) + node_acc
        )
        return Count(count_in_this_scc=node_acc, reachable_count=reachable_count)

    def emit_result(self, node_id: int, scc_acc: Count):
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
        1: Count(count_in_this_scc=3, reachable_count=3),
        2: Count(count_in_this_scc=3, reachable_count=3),
        3: Count(count_in_this_scc=3, reachable_count=3),
        4: Count(count_in_this_scc=2, reachable_count=7),
        5: Count(count_in_this_scc=2, reachable_count=7),
        6: Count(count_in_this_scc=2, reachable_count=5),
        7: Count(count_in_this_scc=2, reachable_count=5),
        8: Count(count_in_this_scc=1, reachable_count=8),
    }
