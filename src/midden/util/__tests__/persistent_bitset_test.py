import pytest

from ..persistent_bitset import PersistentBitSet


@pytest.mark.parametrize(
    "elements",
    [
        [],
        [
            5,
        ],
        [
            97,
        ],
        [1, 2, 3],
        [0, 1, 2, 3, 4],
        [100, 200, 300],
        [0, 128, 256, 512],
        [0, 127, 128, 129],
        [0, 64, 128, 192],
        [1, 2000000],
        [
            99999999999,
        ],
        [
            0,
            1,
            2,
            4,
            8,
            16,
            32,
            64,
            128,
            256,
            512,
            1024,
            2048,
            4096,
            8192,
            16384,
            32768,
            65536,
        ],
        [128],
        [10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
        [
            65536,
            32768,
            16384,
            8192,
            4096,
            2048,
            1024,
            512,
            256,
            128,
            64,
            32,
            16,
            8,
            4,
            2,
            1,
            0,
        ],
        [
            1000000,
            999999,
            999998,
            999997,
            999996,
            999995,
            999994,
            999993,
            999992,
            999991,
        ],
        [1000000, 999999, 10000, 9999, 100, 99, 5, 2, 1],
        [1, 0],
        [128, 0],
    ],
)
def test_persistent_bitset(elements):
    bitset = PersistentBitSet.empty()
    for element in elements:
        bitset = bitset.add(element)

    for element in elements:
        assert bitset.contains(element)

    assert list(bitset) == sorted(elements)

    # Test that the same elements produce the same hash
    bitset2 = PersistentBitSet.empty()
    for element in elements:
        bitset2 = bitset2.add(element)

    assert bitset == bitset2
    assert bitset.unique_hash() == bitset2.unique_hash()
