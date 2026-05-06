from midden_analysis import SummedRadixTree


def test_happy_path():
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
