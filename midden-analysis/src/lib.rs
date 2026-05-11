use pyo3::prelude::*;

mod set_membership_sketch;
mod size_sketch;
mod summed_radix_tree;
mod tarjan;
mod heap_dump_explorer;
/// A Python module implemented in Rust.
#[pymodule]
mod midden_analysis {
    use std::sync::Arc;

    use crate::set_membership_sketch::DefaultMembershipSketch;
    use crate::size_sketch::{
        HighPrecisionSizeSketch as InnerHighPrecisionSizeSketch,
        LowPrecisionSizeSketch as InnerLowPrecisionSizeSketch,
        MediumPrecisionSizeSketch as InnerMediumPrecisionSizeSketch,
    };
    use crate::summed_radix_tree::{
        EMPTY, SummedRadixTree as InnerSummedRadixTree, SummedRadixTreeIterator,
    };
    use pyo3::{
        exceptions::PyStopIteration,
        prelude::*,
        types::{PyDict, PyTuple},
    };

    #[pyclass(frozen)]
    struct SummedRadixTree(Arc<InnerSummedRadixTree>);

    #[pymethods]
    impl SummedRadixTree {
        #[new]
        #[pyo3(signature = (items=None))]
        fn new(items: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
            let mut bitset = EMPTY.clone();
            if let Some(items) = items {
                for (key, value) in items.iter() {
                    let key: usize = key.extract()?;
                    let value: u64 = value.extract()?;
                    bitset = bitset.add(key, value);
                }
            }
            Ok(SummedRadixTree(bitset))
        }

        fn contains(&self, element: usize) -> bool {
            self.0.get_value(element) != 0
        }

        fn __contains__(&self, element: usize) -> bool {
            self.contains(element)
        }

        fn add(&self, element: usize, value: u64) -> Self {
            SummedRadixTree(self.0.add(element, value))
        }

        fn union(&self, other: &SummedRadixTree) -> Self {
            SummedRadixTree(self.0.union(&other.0))
        }

        fn total(&self) -> u64 {
            self.0.total()
        }

        fn __or__(&self, other: &SummedRadixTree) -> Self {
            self.union(other)
        }

        fn __add__(&self, other: &Bound<'_, PyTuple>) -> PyResult<Self> {
            if other.len() != 2 {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "Expected a tuple of (element, value)",
                ));
            }
            let element: usize = other.get_item(0)?.extract()?;
            let value: u64 = other.get_item(1)?.extract()?;
            Ok(self.add(element, value))
        }

        fn __getitem__(&self, element: usize) -> u64 {
            self.0.get_value(element)
        }

        fn __hash__(&self) -> u64 {
            self.0.unique_hash() as u64
        }

        fn __eq__(&self, other: &SummedRadixTree) -> bool {
            self.0.unique_hash() == other.0.unique_hash()
        }

        fn __iter__(&self) -> _Iterator {
            _Iterator(SummedRadixTreeIterator::new(self.0.clone(), 0))
        }

        fn __str__(&self) -> String {
            let elements: Vec<String> = self
                .__iter__()
                .0
                .map(|e| format!("{}: {}", e.0, e.1))
                .collect();
            format!("SummedRadixTree({{{}}})", elements.join(", "))
        }

        fn __repr__(&self) -> String {
            self.__str__()
        }

        fn __len__(&self) -> usize {
            self.__iter__().0.count()
        }

        fn __sizeof__(&self) -> usize {
            size_of::<Self>() + self.0._estimate_size_fudging_refcounts()
        }
    }

    #[pyclass]
    struct _Iterator(SummedRadixTreeIterator);

    #[pymethods]
    impl _Iterator {
        fn __iter__(slf: PyRef<Self>) -> PyRef<Self> {
            slf
        }

        fn __next__(mut slf: PyRefMut<Self>) -> PyResult<(usize, u64)> {
            slf.0
                .next()
                .ok_or_else(|| PyStopIteration::new_err("No more items"))
        }
    }

    macro_rules! impl_size_sketch {
        ($name:ident, $inner:ty) => {
            #[::pyo3::pyclass]
            struct $name($inner);

            #[::pyo3::pymethods]
            impl $name {
                #[new]
                fn new() -> Self {
                    Self(<$inner>::new())
                }

                fn add<'a>(
                    slf: Bound<'a, Self>,
                    id: Bound<'_, PyAny>,
                    value: f64,
                ) -> PyResult<Bound<'a, Self>> {
                    slf.borrow_mut().0.add(id.hash()?, value);
                    Ok(slf)
                }

                fn union(&self, other: &$name) -> Self {
                    Self(self.0.union(&other.0))
                }

                fn total(&self) -> f64 {
                    self.0.estimate()
                }

                fn __or__(&self, other: &$name) -> Self {
                    self.union(other)
                }

                fn __str__(&self) -> String {
                    format!("{}({:?})", stringify!($name), self.0.estimate())
                }

                fn __repr__(&self) -> String {
                    self.__str__()
                }

                fn __sizeof__(&self) -> usize {
                    size_of::<Self>()
                }
            }
        };
    }
    impl_size_sketch!(LowPrecisionSizeSketch, InnerLowPrecisionSizeSketch);
    impl_size_sketch!(MediumPrecisionSizeSketch, InnerMediumPrecisionSizeSketch);
    impl_size_sketch!(HighPrecisionSizeSketch, InnerHighPrecisionSizeSketch);

    #[pymodule_init]
    fn init(m: &Bound<'_, PyModule>) -> PyResult<()> {
        // Initialize macro-generated classes that #[pymodule can't see]
        m.add_class::<LowPrecisionSizeSketch>()?;
        m.add_class::<MediumPrecisionSizeSketch>()?;
        m.add_class::<HighPrecisionSizeSketch>()?;
        Ok(())
    }

    #[pyclass]
    struct SetMembershipSketch(DefaultMembershipSketch);

    #[pymethods]
    impl SetMembershipSketch {
        #[new]
        fn new() -> Self {
            Self(DefaultMembershipSketch::new())
        }

        fn add<'a>(slf: Bound<'a, Self>, item: &Bound<'_, PyAny>) -> PyResult<Bound<'a, Self>> {
            slf.borrow_mut().0.add(&(item.hash()?));
            Ok(slf)
        }

        fn add_all<'a>(
            slf: Bound<'a, Self>,
            items: &Bound<'_, PyAny>,
        ) -> PyResult<Bound<'a, Self>> {
            let items: Vec<Bound<'_, PyAny>> = items.extract()?;
            for item in items {
                slf.borrow_mut().0.add(&(item.hash()?));
            }
            Ok(slf)
        }

        fn union(&self, other: &SetMembershipSketch) -> Self {
            Self(self.0.union(&other.0))
        }

        fn is_subset_of(&self, other: &SetMembershipSketch) -> bool {
            self.0.is_subset_of(&other.0)
        }

        fn is_empty(&self) -> bool {
            self.0.is_empty()
        }

        fn __or__(&self, other: &SetMembershipSketch) -> Self {
            self.union(other)
        }

        fn __str__(&self) -> String {
            format!(
                "SetMembershipSketch.from_bytes([{}])",
                self.0
                    .to_bytes()
                    .iter()
                    .map(|b| b.to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        }

        fn __repr__(&self) -> String {
            self.__str__()
        }

        fn __bool__(&self) -> bool {
            !self.is_empty()
        }

        fn __sizeof__(&self) -> usize {
            size_of::<Self>()
        }

        fn to_bytes(&self) -> Vec<u8> {
            self.0.to_bytes()
        }

        #[staticmethod]
        fn from_bytes(bytes: &[u8]) -> Self {
            Self(DefaultMembershipSketch::from_bytes(bytes))
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn test_basic_operations() {
            let tree = SummedRadixTree::new(None).unwrap();
            assert!(!tree.contains(5));
            let tree = tree.add(5, 10);
            assert!(tree.contains(5));
            assert_eq!(tree.total(), 10);

            let tree2 = SummedRadixTree::new(None).unwrap().add(5, 15).add(10, 20);
            let union_tree = tree.union(&tree2);
            assert!(union_tree.contains(5));
            assert!(union_tree.contains(10));
            assert_eq!(union_tree.total(), 35);
        }

        fn compare_with_naive_implementation(values: Vec<Vec<(usize, u64)>>) {
            let mut naive_map = std::collections::HashMap::new();
            let mut tree = SummedRadixTree::new(None).unwrap();

            for subset in values {
                let mut inner_tree = SummedRadixTree::new(None).unwrap();
                for (element, value) in subset {
                    inner_tree = inner_tree.add(element, value);
                    naive_map.insert(element, value);
                }
                tree = tree.union(&inner_tree);
            }

            for (element, value) in &naive_map {
                assert!(tree.contains(*element));
                assert_eq!(tree.0.get_value(*element), *value);
            }
            assert!(tree.total() == naive_map.values().sum::<u64>());
        }

        macro_rules! generate_tests {
            ($($name:ident: $values:expr),+) => {
                $(
                    #[test]
                    fn $name() {
                        compare_with_naive_implementation($values);
                    }
                )*
            };
        }

        generate_tests!(
            test_single_element: vec![vec![(5, 10)]],
            test_multiple_elements: vec![vec![(5, 10), (10, 20), (15, 30)]],
            test_overlapping_elements: vec![vec![(5, 10)], vec![(5, 15)], vec![(10, 20)]],
            test_large_numbers: vec![vec![(1000, 1), (2000, 2)], vec![(1000, 3), (3000, 4)]],
            test_empty_tree: vec![],
            test_powers_of_two: vec![vec![(0, 1), (8, 2), (64, 3), (512, 4)]],
            test_descending_powers_of_two: vec![vec![(512, 4), (64, 3), (8, 2), (0, 1)]],
            test_one_to_ten: vec![(0..10).map(|i| (i, i as u64 + 1)).collect()],
            test_descending_one_to_ten: vec![(0..10).rev().map(|i| (i, i as u64 + 1)).collect()]
        );
    }
}
