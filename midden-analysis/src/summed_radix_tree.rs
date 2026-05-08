use std::{
    array,
    sync::{Arc, LazyLock},
};

use xxhash_rust::xxh3::Xxh3Default;

const FANOUT: usize = 8;

#[derive(Debug)]
pub enum SummedRadixTree {
    Empty,
    Leaf {
        values: [u64; FANOUT],
        hash: u128,
        total: u64,
    },
    Branch {
        level: u8,
        children: [Arc<SummedRadixTree>; FANOUT],
        hash: u128,
        total: u64,
    },
}

pub static EMPTY: LazyLock<Arc<SummedRadixTree>> =
    LazyLock::new(|| Arc::new(SummedRadixTree::Empty));

impl SummedRadixTree {
    pub fn get_value(&self, position: usize) -> u64 {
        match self {
            Self::Empty => 0,
            Self::Leaf { values, .. } => {
                if position >= FANOUT {
                    0
                } else {
                    values[position]
                }
            }
            Self::Branch {
                level, children, ..
            } => {
                let items_per_element = FANOUT.pow(*level as u32);
                let child_index = position / items_per_element;
                if child_index >= FANOUT {
                    0
                } else {
                    let child_offset = position % items_per_element;
                    children[child_index].get_value(child_offset)
                }
            }
        }
    }

    pub fn unique_hash(&self) -> u128 {
        match self {
            Self::Empty => 0,
            Self::Leaf { hash, .. } => *hash,
            Self::Branch { hash, .. } => *hash,
        }
    }

    pub fn total(&self) -> u64 {
        match self {
            Self::Empty => 0,
            Self::Leaf { total, .. } => *total,
            Self::Branch { total, .. } => *total,
        }
    }

    pub fn add(self: &Arc<Self>, position: usize, value: u64) -> Arc<Self> {
        if self.get_value(position) == value {
            self.clone()
        } else {
            let single_position_set = Self::_with_single_position_set(position, value);
            self.union(&Arc::new(single_position_set))
        }
    }

    pub fn union<'a>(self: &Arc<Self>, other: &Arc<Self>) -> Arc<Self> {
        match (self.as_ref(), other.as_ref()) {
            (Self::Empty, _) => other.clone(),
            (_, Self::Empty) => self.clone(),
            (s, o) if s.unique_hash() == o.unique_hash() => {
                if Arc::strong_count(self) >= Arc::strong_count(other) {
                    self.clone()
                } else {
                    other.clone()
                }
            }
            (Self::Leaf { .. }, Self::Branch { .. }) => {
                other.union(self) // Let the bigger tree handle merging
            }
            (
                Self::Branch {
                    level, children, ..
                },
                Self::Leaf { .. },
            ) => {
                let mut new_children = children.clone();
                new_children[0] = new_children[0].clone().union(other);
                let hash = Self::_calculate_branch_hash(&new_children);
                let total = new_children.iter().map(|child| child.total()).sum();
                Arc::new(Self::Branch {
                    level: *level,
                    children: new_children,
                    hash,
                    total,
                })
            }
            (
                Self::Branch {
                    level: l1,
                    children: c1,
                    ..
                },
                Self::Branch {
                    level: l2,
                    children: c2,
                    ..
                },
            ) => {
                match l1.cmp(l2) {
                    std::cmp::Ordering::Greater => {
                        let mut new_children = c1.clone();
                        new_children[0] = new_children[0].clone().union(other);
                        let hash = Self::_calculate_branch_hash(&new_children);
                        let total = new_children.iter().map(|child| child.total()).sum();
                        Arc::new(Self::Branch {
                            level: *l1,
                            children: new_children,
                            hash,
                            total,
                        })
                    }
                    std::cmp::Ordering::Less => {
                        other.union(self) // Let the bigger tree handle merging
                    }
                    std::cmp::Ordering::Equal => {
                        let new_children = array::from_fn(|i| c1[i].clone().union(&c2[i]));
                        let hash = Self::_calculate_branch_hash(&new_children);
                        let total = new_children.iter().map(|child| child.total()).sum();
                        Arc::new(Self::Branch {
                            level: *l1,
                            children: new_children,
                            hash,
                            total,
                        })
                    }
                }
            }
            (Self::Leaf { values: v1, .. }, Self::Leaf { values: v2, .. }) => {
                if v1 == v2 {
                    self.clone()
                } else {
                    let new_values = array::from_fn(|i| v1[i].max(v2[i]));
                    let hash = Self::_calculate_leaf_hash(&new_values);
                    let total = new_values.iter().sum();
                    Arc::new(Self::Leaf {
                        values: new_values,
                        hash,
                        total,
                    })
                }
            }
        }
    }

    fn _with_single_position_set(position: usize, value: u64) -> Self {
        let leaf_index = position / FANOUT;
        let leaf_position = position % FANOUT;

        let mut leaf_values = [0u64; FANOUT];
        leaf_values[leaf_position] = value;
        let mut result = Self::Leaf {
            values: leaf_values,
            hash: Self::_calculate_leaf_hash(&leaf_values),
            total: value,
        };

        // Make parent branch nodes as needed
        let mut current_level = 1;
        let mut child_index = leaf_index;
        while child_index > 0 {
            let parent_index = child_index % FANOUT;
            child_index /= FANOUT;
            let mut children: [Arc<Self>; FANOUT] = array::from_fn(|_| EMPTY.clone());
            children[parent_index] = Arc::new(result);
            let hash = Self::_calculate_branch_hash(&children);
            let new_branch = Self::Branch {
                level: current_level,
                children,
                hash,
                total: value,
            };
            result = new_branch;
            current_level += 1;
        }

        result
    }

    fn _calculate_branch_hash(children: &[Arc<Self>; FANOUT]) -> u128 {
        let mut hasher = Xxh3Default::new();
        for child in children {
            let child_hash = child.unique_hash();
            hasher.update(&child_hash.to_le_bytes());
        }
        hasher.digest128()
    }

    fn _calculate_leaf_hash(values: &[u64; FANOUT]) -> u128 {
        let mut hasher = Xxh3Default::new();
        for bit in values {
            hasher.update(&bit.to_le_bytes());
        }
        hasher.digest128()
    }

    fn _estimate_size(&self) -> usize {
        match self {
            Self::Empty => 0,
            Self::Leaf { .. } => size_of::<Self>(),
            Self::Branch { children, .. } => {
                size_of::<Self>()
                    + children
                        .iter()
                        .map(|child| child._estimate_size_fudging_refcounts())
                        .sum::<usize>()
            }
        }
    }

    pub fn _estimate_size_fudging_refcounts(self: &Arc<Self>) -> usize {
        return (self._estimate_size() / Arc::strong_count(&self)) + size_of::<Arc<()>>();
    }
}

pub struct SummedRadixTreeIterator {
    tree: Arc<SummedRadixTree>,
    offset: usize,
    child_index: usize,
    child_iter: Option<Box<SummedRadixTreeIterator>>,
}

impl SummedRadixTreeIterator {
    pub fn new(tree: Arc<SummedRadixTree>, offset: usize) -> Self {
        Self {
            tree,
            offset,
            child_index: 0,
            child_iter: None,
        }
    }
}

impl Iterator for SummedRadixTreeIterator {
    type Item = (usize, u64);

    fn next(&mut self) -> Option<Self::Item> {
        match self.tree.as_ref() {
            SummedRadixTree::Empty => None,
            SummedRadixTree::Leaf { values, .. } => {
                while self.child_index < FANOUT {
                    let value = values[self.child_index];
                    let current_offset = self.offset + self.child_index;
                    self.child_index += 1;
                    if value > 0 {
                        return Some((current_offset, value));
                    }
                }
                None
            }
            SummedRadixTree::Branch {
                level, children, ..
            } => {
                if let Some(child_iter) = &mut self.child_iter {
                    if let Some(item) = child_iter.next() {
                        return Some(item);
                    } else {
                        self.child_iter = None; // Finished with this child
                    }
                }

                while self.child_index < FANOUT {
                    let child = &children[self.child_index];
                    let current_offset =
                        self.offset + self.child_index * FANOUT.pow((*level) as u32);
                    self.child_index += 1;
                    if child.total() > 0 {
                        self.child_iter = Some(Box::new(SummedRadixTreeIterator::new(
                            child.clone(),
                            current_offset,
                        )));
                        if let Some(item) = self.child_iter.as_mut().unwrap().next() {
                            return Some(item);
                        } else {
                            self.child_iter = None; // Finished with this child
                        }
                    }
                }
                None
            }
        }
    }
}
