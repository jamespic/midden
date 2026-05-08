from typing import Type
import pytest
from midden_analysis import (
    SummedRadixTree,
    LowPrecisionSizeSketch,
    MediumPrecisionSizeSketch,
    HighPrecisionSizeSketch,
)


def test_happy_path_summed_radix_tree():
    tree = SummedRadixTree()
    tree = tree.add(1, 10)
    tree = tree.add(2, 20)
    tree += (3, 30)

    assert tree.contains(1)
    assert tree.contains(2)
    assert 3 in tree
    assert 4 not in tree

    assert tree[1] == 10
    assert tree[2] == 20
    assert tree[3] == 30

    assert tree == SummedRadixTree({1: 10, 2: 20, 3: 30})
    assert str(tree) == "SummedRadixTree({1: 10, 2: 20, 3: 30})"

    assert tree.total() == 60


@pytest.mark.parametrize(
    "sketch_class",
    [LowPrecisionSizeSketch, MediumPrecisionSizeSketch, HighPrecisionSizeSketch],
)
def test_happy_path_size_sketches(
    sketch_class: Type[
        LowPrecisionSizeSketch | MediumPrecisionSizeSketch | HighPrecisionSizeSketch
    ],
):
    sketch = sketch_class()
    sketch.add(1, 10)
    sketch.add(2, 20)
    sketch.add(3, 30)
    sketch.add(4, 40)

    assert (
        20 <= sketch.total() <= 300
    )  # These are very rough bounds, since the sketch is probabilistic
