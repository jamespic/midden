use std::{
    borrow::Cow, convert::Infallible, error::Error, marker::PhantomData, str::FromStr, sync::Arc,
};

use heed::{
    BytesDecode, BytesEncode, Database, DatabaseFlags, Env, EnvOpenOptions, Error as HeedError,
    IntegerComparator,
    byteorder::NativeEndian,
    types::{Lazy, LazyDecode, SerdeJson, U64},
};
// """A class that allows exploring heap dumps exported by dump_heap.py.
// It uses LMDB to store the data on disk and provides methods for querying objects, their types, and their relationships.
use pyo3::{create_exception, prelude::*};
use serde::{Deserialize, Serialize};
// use serde_json::Value;

use crate::{
    set_membership_sketch::{DefaultMembershipSketch, SetMembershipSketch},
    size_sketch::{
        HighPrecisionSizeSketch, LowPrecisionSizeSketch, MediumPrecisionSizeSketch, SizeSketch,
    },
    summed_radix_tree::{EMPTY, SummedRadixTree},
    tarjan::{self, GraphSCCVisitor},
};

#[derive(Serialize, Deserialize, Debug, PartialEq, Clone)]
#[serde(untagged)]
pub enum Value {
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
    None,
}

impl<'py> IntoPyObject<'py> for Value {
    type Target = PyAny;
    type Output = Bound<'py, Self::Target>;
    type Error = Infallible;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        match self {
            Value::Bool(b) => Ok(b.into_pyobject(py)?.as_any().clone()),
            Value::Int(i) => Ok(i.into_pyobject(py)?.as_any().clone()),
            Value::Float(f) => Ok(f.into_pyobject(py)?.as_any().clone()),
            Value::Str(s) => Ok(s.into_pyobject(py)?.as_any().clone()),
            Value::None => Ok(py.None().bind(py).clone()),
        }
    }
}

#[derive(Serialize, Deserialize, Debug, PartialEq, Clone)]
#[pyclass(frozen, skip_from_py_object, get_all)]
pub struct ObjectSummary {
    pub id: Id,
    pub type_name: String,
    pub value: Option<Value>,
    pub size: u64,
    pub subtree_size: Option<u64>,
}

#[derive(Serialize, Deserialize, Debug, PartialEq, Clone)]
struct RawObjectRecord {
    id: Id,
    type_name: String,
    references: Vec<Id>,
    value: Option<Value>,
    size: u64,
    subtree_size: Option<u64>,
}

#[derive(Serialize, Deserialize, Debug, PartialEq, Clone)]
#[pyclass(frozen, skip_from_py_object, get_all)]
pub struct ObjectRecord {
    pub id: Id,
    pub type_name: String,
    pub references: Vec<ObjectSummary>,
    pub referrers: Vec<ObjectSummary>,
    pub value: Option<Value>,
    pub size: u64,
    pub subtree_size: Option<u64>,
}

#[derive(Serialize, Deserialize, Debug, PartialEq, Clone)]
struct ObjectRecordNoValue {
    id: Id,
    type_name: String,
    references: Vec<Id>,
    size: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct SizeIndexEntry {
    size: u64,
    obj_id: u64,
}

impl Ord for SizeIndexEntry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Sort in reverse size order, then by obj_id ascending to break ties, so we can efficiently page through largest objects in LMDB.
        other
            .size
            .cmp(&self.size)
            .then_with(|| self.obj_id.cmp(&other.obj_id))
    }
}

impl PartialOrd for SizeIndexEntry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl BytesEncode<'_> for SizeIndexEntry {
    type EItem = Self;

    fn bytes_encode(item: &Self) -> Result<Cow<'_, [u8]>, Box<dyn Error + Send + Sync>> {
        let mut buf = [0u8; 16];
        // Pack by size descending, then by obj_id ascending, so we can sort by size in LMDB.
        // This enables efficient retrieval of largest objects per type.
        // Use BE bytes so sorting is lexical
        buf[..8].copy_from_slice(&(u64::MAX - item.size).to_be_bytes());
        buf[8..].copy_from_slice(&(item.obj_id).to_be_bytes());
        Ok(Cow::Owned(buf.to_vec()))
    }
}

impl BytesDecode<'_> for SizeIndexEntry {
    type DItem = Self;

    fn bytes_decode(bytes: &[u8]) -> Result<Self::DItem, Box<dyn Error + Send + Sync>> {
        if bytes.len() != 16 {
            return Err(format!(
                "Invalid byte length for SizeIndexEntry: expected 16, got {}",
                bytes.len()
            )
            .into());
        }
        let size = u64::MAX - u64::from_be_bytes(bytes[..8].try_into().unwrap());
        let obj_id = u64::from_be_bytes(bytes[8..16].try_into().unwrap());
        Ok(SizeIndexEntry {
            size: size,
            obj_id: obj_id,
        })
    }
}
#[derive(Debug, PartialEq, Clone)]
#[pyclass(frozen, skip_from_py_object, get_all)]
pub struct TypeSummary {
    pub count: u64,
    pub total_size: u64,
}

impl BytesEncode<'_> for TypeSummary {
    type EItem = Self;

    fn bytes_encode(item: &Self) -> Result<Cow<'_, [u8]>, Box<dyn Error + Send + Sync>> {
        let mut buf = [0u8; 16];
        buf[..8].copy_from_slice(&(item.count).to_ne_bytes());
        buf[8..].copy_from_slice(&(item.total_size).to_ne_bytes());
        Ok(buf.to_vec().into())
    }
}

impl BytesDecode<'_> for TypeSummary {
    type DItem = Self;

    fn bytes_decode(bytes: &[u8]) -> Result<Self::DItem, Box<dyn Error + Send + Sync>> {
        if bytes.len() != 16 {
            return Err(format!(
                "Invalid byte length for TypeSummary: expected 16, got {}",
                bytes.len()
            )
            .into());
        }
        let count = u64::from_ne_bytes(bytes[..8].try_into()?);
        let total_size = u64::from_ne_bytes(bytes[8..16].try_into()?);
        Ok(TypeSummary { count, total_size })
    }
}

const PRIMARY_DB: &str = "primary";
const REFERRERS_DB: &str = "referrers";
const TYPES_DB: &str = "types";
const TYPES_SIZE_INDEX_DB: &str = "types_size_index";
const TYPES_SUBTREE_SIZE_INDEX_DB: &str = "types_subtree_size_index";
const TYPE_SUMMARIES_DB: &str = "type_summaries";
const SCCS_SKETCH_DB: &str = "sccs_sketch";
const ALL_TYPES: &str = "All Types";
const PAGE_SIZE: usize = 1000; // Hardcode this for now

type Id = u64;
type IdDbType = U64<NativeEndian>;

trait SizeEstimator: Clone {
    fn empty() -> Self;
    fn add_in_place(&mut self, element: u64, value: u64);
    fn total(&self) -> u64;
    fn include(&mut self, other: &Self);
}

impl<const N: usize> SizeEstimator for SizeSketch<N> {
    fn empty() -> Self {
        SizeSketch::new()
    }

    fn add_in_place(&mut self, element: u64, value: u64) {
        self.add(element, value as f64);
    }

    fn total(&self) -> u64 {
        self.estimate() as u64
    }

    fn include(&mut self, other: &Self) {
        self.update_in_place(other);
    }
}

impl SizeEstimator for Arc<SummedRadixTree> {
    fn empty() -> Self {
        EMPTY.clone()
    }

    fn add_in_place(&mut self, element: u64, value: u64) {
        *self = self.add(element as usize, value);
    }

    fn total(&self) -> u64 {
        SummedRadixTree::total(self) as u64
    }

    fn include(&mut self, other: &Self) {
        *self = self.union(other);
    }
}

impl<const N: usize> BytesEncode<'_> for SetMembershipSketch<N> {
    type EItem = Self;

    fn bytes_encode(item: &Self) -> Result<Cow<'_, [u8]>, Box<dyn Error + Send + Sync>> {
        Ok(Cow::Owned(item.to_bytes().to_vec()))
    }
}

impl<const N: usize> BytesDecode<'_> for SetMembershipSketch<N> {
    type DItem = Self;

    fn bytes_decode(bytes: &[u8]) -> Result<Self::DItem, Box<dyn Error + Send + Sync>> {
        Ok(SetMembershipSketch::from_bytes(bytes.try_into().map_err(
            |_| {
                format!(
                    "Invalid byte length for SetMembershipSketch: expected {}, got {}",
                    4 * N,
                    bytes.len()
                )
            },
        )?))
    }
}

#[pyclass(frozen, from_py_object)]
#[derive(Debug, Clone)]
pub enum EstimatorPrecision {
    Low,
    Medium,
    High,
    Exact,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[pyclass(frozen, skip_from_py_object, get_all)]
enum Type {
    AllTypes(),
    TypeName(String),
}

impl Type {
    fn from_str(s: &str) -> Self {
        if s == ALL_TYPES {
            Type::AllTypes()
        } else {
            Type::TypeName(s.to_string())
        }
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for Type {
    type Error = PyErr;
    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        Ok(Type::from_str(obj.extract()?))
    }
}

#[pymethods]
impl Type {
    fn __str__(&self) -> String {
        match self {
            Type::AllTypes() => ALL_TYPES.to_string(),
            Type::TypeName(name) => name.clone(),
        }
    }
}

const ALL_TYPES_KEY: &[u8] = b"";

impl<'a> BytesEncode<'a> for Type {
    type EItem = Self;

    fn bytes_encode(item: &Self) -> Result<Cow<'_, [u8]>, Box<dyn Error + Send + Sync>> {
        match item {
            Type::AllTypes() => Ok(Cow::Borrowed(ALL_TYPES_KEY)),
            Type::TypeName(name) => Ok(Cow::Borrowed(name.as_bytes())),
        }
    }
}

impl<'txn> BytesDecode<'txn> for Type {
    type DItem = Self;

    fn bytes_decode(bytes: &[u8]) -> Result<Self::DItem, Box<dyn Error + Send + Sync>> {
        if bytes == ALL_TYPES_KEY {
            Ok(Type::AllTypes())
        } else {
            Ok(Type::TypeName(String::from_str(str::from_utf8(bytes)?)?))
        }
    }
}

#[pyclass]
pub struct HeapDumpExplorer {
    env: Env,
    primary_db: Database<IdDbType, SerdeJson<RawObjectRecord>, IntegerComparator>,
    referrers_db: Database<IdDbType, IdDbType, IntegerComparator>,
    types_db: Database<Type, IdDbType>,
    types_size_index_db: Database<Type, SizeIndexEntry>,
    types_subtree_size_index_db: Database<Type, SizeIndexEntry>,
    type_summaries_db: Database<Type, TypeSummary>,
    sccs_sketch_db: Database<IdDbType, DefaultMembershipSketch, IntegerComparator>,
}

create_exception!(midden_analysis, MdbError, pyo3::exceptions::PyException);
create_exception!(
    midden_analysis,
    EncodingError,
    pyo3::exceptions::PyException
);
create_exception!(
    midden_analysis,
    DecodingError,
    pyo3::exceptions::PyException
);

trait ToPyResult<T> {
    fn to_py_res(self) -> PyResult<T>;
}

impl<T> ToPyResult<T> for heed::Result<T> {
    fn to_py_res(self) -> PyResult<T> {
        self.map_err(|e| match e {
            HeedError::Io(e) => {
                PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("IO Error: {}", e))
            }
            HeedError::Mdb(e) => PyErr::new::<MdbError, _>(format!("MDB Error: {}", e)),
            HeedError::EnvAlreadyOpened => {
                PyErr::new::<MdbError, _>("MDB Error: LMDB environment is already opened")
            }
            HeedError::Decoding(e) => {
                PyErr::new::<DecodingError, _>(format!("Decoding error: {}", e))
            }
            HeedError::Encoding(e) => {
                PyErr::new::<EncodingError, _>(format!("Encoding error: {}", e))
            }
        })
    }
}

trait RollbackOnErr {
    fn rollback_on_err<T, E>(self, f: impl FnOnce(&mut Self) -> Result<T, E>) -> Result<T, E>;
}

impl<'a> RollbackOnErr for heed::RwTxn<'a> {
    fn rollback_on_err<T, E>(mut self, f: impl FnOnce(&mut Self) -> Result<T, E>) -> Result<T, E> {
        match f(&mut self) {
            Ok(result) => Ok(result),
            Err(e) => {
                self.abort();
                Err(e)
            }
        }
    }
}

impl HeapDumpExplorer {
    fn just_import_lines<'a>(&self, lines: &Bound<'_, PyAny>) -> PyResult<()> {
        self.env.write_txn().to_py_res()?.rollback_on_err(|mut tx| {
            for item in lines.try_iter()? {
                let item = item?;
                let line = item.extract::<&[u8]>()?;
                let record: RawObjectRecord = serde_json::from_slice(line).map_err(|e| {
                    PyErr::new::<DecodingError, _>(format!(
                        "Decoding Error: Failed to decode JSON line: {}",
                        e
                    ))
                })?;
                self.primary_db
                    .put(&mut tx, &record.id, &record)
                    .to_py_res()?;

                for reference in &record.references {
                    self.referrers_db
                        .put(&mut tx, reference, &record.id)
                        .to_py_res()?;
                }

                let type_key = Type::TypeName(record.type_name.clone());
                self.types_db
                    .put(&mut tx, &type_key, &record.id)
                    .to_py_res()?;
                self.types_db
                    .put(&mut tx, &Type::AllTypes(), &record.id)
                    .to_py_res()?;
                self.put_size_index_entry(
                    tx,
                    &type_key,
                    &SizeIndexEntry {
                        size: record.size,
                        obj_id: record.id,
                    },
                    self.types_size_index_db,
                )
                .to_py_res()?;
            }
            Ok(())
        })
    }

    fn put_size_index_entry(
        &self,
        tx: &mut heed::RwTxn,
        type_key: &Type,
        entry: &SizeIndexEntry,
        size_index: Database<Type, SizeIndexEntry>,
    ) -> heed::Result<()> {
        size_index.put(tx, type_key, entry)?;
        size_index.put(tx, &Type::AllTypes(), entry)?;
        Ok(())
    }

    fn run_post_import_processing(&self, estimator_precision: EstimatorPrecision) -> PyResult<()> {
        self.env.write_txn().to_py_res()?.rollback_on_err(|mut tx| {
            self.build_type_summaries(&mut tx).to_py_res()?;
            match estimator_precision {
                EstimatorPrecision::Low => self
                    .explore_strongly_connected_components::<'_, '_, LowPrecisionSizeSketch>(
                        &mut tx,
                    ),
                EstimatorPrecision::Medium => self
                    .explore_strongly_connected_components::<'_, '_, MediumPrecisionSizeSketch>(
                        &mut tx,
                    ),
                EstimatorPrecision::High => self
                    .explore_strongly_connected_components::<'_, '_, HighPrecisionSizeSketch>(
                        &mut tx,
                    ),
                EstimatorPrecision::Exact => self
                    .explore_strongly_connected_components::<'_, '_, Arc<SummedRadixTree>>(&mut tx),
            }
            .to_py_res()?;
            Ok(())
        })
    }

    fn build_type_summaries(&self, tx: &mut heed::RwTxn) -> heed::Result<()> {
        let rtx = self.env.read_txn()?;
        let mut last_type_key: Option<Type> = None;
        let mut total_size = 0;
        let mut count = 0;
        for type_size_info in self.types_size_index_db.iter(&rtx)? {
            let (type_key, size_index_entry) = type_size_info?;
            if let Some(ref last_key) = last_type_key {
                if &type_key != last_key {
                    self.type_summaries_db
                        .put(tx, last_key, &TypeSummary { count, total_size })?;
                    total_size = 0;
                    count = 0;
                }
            }
            last_type_key = Some(type_key);
            total_size += size_index_entry.size;
            count += 1;
        }
        if let Some(ref last_key) = last_type_key {
            self.type_summaries_db
                .put(tx, last_key, &TypeSummary { count, total_size })?;
        }
        Ok(())
    }

    fn explore_strongly_connected_components<'vis, 'env, T: SizeEstimator>(
        &'env self,
        tx: &'vis mut heed::RwTxn<'env>,
    ) -> heed::Result<()>
    where
        'env: 'vis,
    {
        let ro_txn = self.env.read_txn()?;
        let ro_iter = self
            .primary_db
            .iter(&ro_txn)?
            .remap_data_type::<SerdeJson<ObjectRecordNoValue>>()
            .lazily_decode_data();
        let known_skips = std::collections::HashSet::new();
        let mut visitor = StronglyConnectedComponentsVisitor {
            explorer: self,
            ro_txn: &ro_txn,
            ro_iter: ro_iter,
            rw_txn: tx,
            known_skips,
            _estimator: PhantomData::<T>,
        };
        tarjan::visit_sccs(&mut visitor)?;
        Ok(())
    }

    fn get_summaries_for_ids(
        &self,
        tx: &heed::RoTxn,
        ids: &Vec<Id>,
    ) -> heed::Result<Vec<ObjectSummary>> {
        let summaries = collect_results(ids.iter().map(|id| self.get_summary(tx, *id)))?;
        Ok(summaries)
    }

    fn retrieve_page_of_object_summaries<T>(
        &self,
        tx: &heed::RoTxn,
        iter: impl Iterator<Item = heed::Result<(T, Id)>>,
        page: Option<usize>,
    ) -> PyResult<Vec<ObjectSummary>> {
        let ids: Vec<heed::Result<(T, Id)>> = if let Some(page) = page {
            iter.skip(page * PAGE_SIZE).take(PAGE_SIZE).collect()
        } else {
            iter.collect()
        };
        let mut result = Vec::with_capacity(ids.len());
        for id_res in ids {
            let (_, id) = id_res.to_py_res()?;
            let summary = self.get_summary(tx, id).to_py_res()?;
            result.push(summary);
        }
        Ok(result)
    }

    fn get_summary(&self, tx: &heed::RoTxn, id: Id) -> heed::Result<ObjectSummary> {
        let remapped_db = self
            .primary_db
            .remap_data_type::<SerdeJson<ObjectSummary>>();
        remapped_db.get(tx, &id).and_then(|opt| {
            opt.ok_or_else(|| {
                HeedError::Decoding(format!("Decoding error: Missing record for ID {}", id).into())
            })
        })
    }
}

#[pymethods]
impl HeapDumpExplorer {
    #[new]
    fn new(db_path: String) -> PyResult<Self> {
        let env = unsafe {
            EnvOpenOptions::new()
                .map_size(10 * 1024 * 1024 * 1024)
                .max_dbs(7)
                .open(db_path)
        }
        .to_py_res()?;
        let mut wtxn = env.write_txn().to_py_res()?;
        let primary_db = env
            .database_options()
            .name(PRIMARY_DB)
            .types()
            .key_comparator()
            .create(&mut wtxn)
            .to_py_res()?;
        let referrers_db = env
            .database_options()
            .name(REFERRERS_DB)
            .flags(DatabaseFlags::DUP_SORT | DatabaseFlags::DUP_FIXED)
            .types()
            .key_comparator()
            .create(&mut wtxn)
            .to_py_res()?;
        let types_db = env
            .database_options()
            .name(TYPES_DB)
            .flags(DatabaseFlags::DUP_SORT | DatabaseFlags::DUP_FIXED)
            .types()
            .create(&mut wtxn)
            .to_py_res()?;
        let types_size_index_db = env
            .database_options()
            .name(TYPES_SIZE_INDEX_DB)
            .flags(DatabaseFlags::DUP_SORT | DatabaseFlags::DUP_FIXED)
            .types()
            .create(&mut wtxn)
            .to_py_res()?;
        let types_subtree_size_index_db = env
            .database_options()
            .name(TYPES_SUBTREE_SIZE_INDEX_DB)
            .flags(DatabaseFlags::DUP_SORT | DatabaseFlags::DUP_FIXED)
            .types()
            .create(&mut wtxn)
            .to_py_res()?;
        let type_summaries_db = env
            .database_options()
            .name(TYPE_SUMMARIES_DB)
            .types()
            .create(&mut wtxn)
            .to_py_res()?;
        let sccs_sketch_db = env
            .database_options()
            .name(SCCS_SKETCH_DB)
            .types()
            .key_comparator()
            .create(&mut wtxn)
            .to_py_res()?;
        drop(wtxn);

        return Ok(Self {
            env,
            primary_db,
            referrers_db,
            types_db,
            types_size_index_db,
            types_subtree_size_index_db,
            type_summaries_db,
            sccs_sketch_db,
        });
    }

    fn import_lines(
        &self,
        lines: &Bound<'_, PyAny>,
        estimator_precision: EstimatorPrecision,
    ) -> PyResult<()> {
        self.just_import_lines(lines)?;
        lines
            .py()
            .detach(|| self.run_post_import_processing(estimator_precision))?;
        Ok(())
    }

    fn get_object(&self, obj_id: Id) -> PyResult<Option<ObjectRecord>> {
        let rtxn = self.env.read_txn().to_py_res()?;
        if let Some(record) = self.primary_db.get(&rtxn, &obj_id).to_py_res()? {
            let referrers_iter_opt = self
                .referrers_db
                .get_duplicates(&rtxn, &obj_id)
                .to_py_res()?;

            let referrers = match referrers_iter_opt {
                Some(iter) => self.retrieve_page_of_object_summaries(&rtxn, iter, None)?,
                None => vec![],
            };

            Ok(Some(ObjectRecord {
                id: record.id,
                type_name: record.type_name,
                value: record.value,
                size: record.size,
                subtree_size: record.subtree_size,
                references: self
                    .get_summaries_for_ids(&rtxn, &record.references)
                    .to_py_res()?,
                referrers,
            }))
        } else {
            Ok(None)
        }
    }

    fn get_type_summaries(&self) -> PyResult<Vec<(Type, TypeSummary)>> {
        let rtxn = self.env.read_txn().to_py_res()?;
        let summaries =
            collect_results(self.type_summaries_db.iter(&rtxn).to_py_res()?).to_py_res()?;
        Ok(summaries)
    }

    fn get_count_for_type(&self, typename: Type) -> PyResult<usize> {
        let rtxn = self.env.read_txn().to_py_res()?;
        Ok(self
            .type_summaries_db
            .get(&rtxn, &typename)
            .to_py_res()?
            .map(|summary| summary.count)
            .unwrap_or(0) as usize)
    }

    fn get_page_count_for_type(&self, typename: Type) -> PyResult<usize> {
        let count = self.get_count_for_type(typename)?;
        Ok((count + PAGE_SIZE - 1) / PAGE_SIZE)
    }

    fn get_objects_by_type(
        &self,
        typename: Type,
        page: Option<usize>,
    ) -> PyResult<Vec<ObjectSummary>> {
        let rtxn = self.env.read_txn().to_py_res()?;
        if let Some(iter) = self.types_db.get_duplicates(&rtxn, &typename).to_py_res()? {
            self.retrieve_page_of_object_summaries(&rtxn, iter, page)
        } else {
            Ok(vec![])
        }
    }

    fn get_objects_by_type_ordered_by_size(
        &self,
        type_name: Type,
        subtree_size: bool,
        page: Option<usize>,
    ) -> PyResult<Vec<ObjectSummary>> {
        let rtxn = self.env.read_txn().to_py_res()?;
        let index_db = if subtree_size {
            &self.types_subtree_size_index_db
        } else {
            &self.types_size_index_db
        };
        if let Some(iter) = index_db.get_duplicates(&rtxn, &type_name).to_py_res()? {
            self.retrieve_page_of_object_summaries(
                &rtxn,
                iter.map(|res| res.map(|(_, index_entry)| (index_entry.size, index_entry.obj_id))),
                page,
            )
        } else {
            Ok(vec![])
        }
    }

    //     @tx
    //     def find_path_between_objects(
    //         self, start_id: int, end_id: int, avoid_ids: set[int] | None = None
    //     ) -> list[ObjectSummary] | None:
    //         """Find a path of references from start_id to end_id, optionally avoiding certain IDs.

    //         Uses SCC sketches to quickly rule out impossible paths.
    //         Returns a list of ObjectSummary representing the path, or None if no path exists.
    //         """
    //         queue = deque([start_id])
    //         predecessors = {start_id: None}  # Doubles as a visited set
    //         dead_ends = set()
    //         start_sketch = self._get_scc_sketch(start_id)
    //         end_sketch = self._get_scc_sketch(end_id)
    //         if not end_sketch.is_subset_of(start_sketch):
    //             return None  # No path can exist if end's reachable SCCs aren't a subset of start's reachable SCCs
    //         while queue:
    //             current_id = queue.popleft()
    //             if current_id == end_id:
    //                 # Reconstruct path
    //                 path = []
    //                 while current_id is not None:
    //                     summary = self._load_and_validate(current_id, ObjectSummary)
    //                     assert summary is not None
    //                     path.append(summary)
    //                     current_id = predecessors[current_id]
    //                 return list(reversed(path))

    //             if current_id in dead_ends:
    //                 continue  # Skip known dead ends
    //             current_sketch = self._get_scc_sketch(current_id)
    //             if not end_sketch.is_subset_of(current_sketch):
    //                 dead_ends.add(current_id)
    //                 continue  # No path can exist from current to end, so skip it

    //             object_record = self._load_and_validate(current_id, _ObjectRecordNoValue)
    //             if self._should_skip_link_in_subtree_exploration(object_record):
    //                 dead_ends.add(current_id)
    //                 continue
    //             assert object_record is not None
    //             for ref_id in object_record.references:
    //                 if avoid_ids is not None and ref_id in avoid_ids:
    //                     continue
    //                 if ref_id not in predecessors:
    //                     predecessors[ref_id] = current_id
    //                     queue.append(ref_id)
}

struct StronglyConnectedComponentsVisitor<'vis, 'env, SizeEstimatorT: SizeEstimator>
where
    'env: 'vis,
{
    explorer: &'vis HeapDumpExplorer,
    ro_txn: &'vis heed::RoTxn<'env>,
    ro_iter: heed::RoIter<'vis, IdDbType, LazyDecode<SerdeJson<ObjectRecordNoValue>>>,
    rw_txn: &'vis mut heed::RwTxn<'env>,
    known_skips: std::collections::HashSet<Id>,
    _estimator: PhantomData<SizeEstimatorT>,
}

impl<T: SizeEstimator> StronglyConnectedComponentsVisitor<'_, '_, T> {
    fn should_skip_link_in_subtree_exploration(&self, record: &ObjectRecordNoValue) -> bool {
        /* Heuristic: skip links to modules, since they often create large SCCs that
        aren't interesting. */
        record.type_name == "builtins.module"
    }
}

#[derive(Debug, Clone)]
struct VisitorState<SizeEstimatorT> {
    scc_sketch: DefaultMembershipSketch,
    size_estimate: SizeEstimatorT,
}

impl<'vis, 'env, T: SizeEstimator> GraphSCCVisitor
    for StronglyConnectedComponentsVisitor<'vis, 'env, T>
{
    type NodeT = ObjectRecordNoValue;
    type NodeIdT = Id;
    type NodeAccT = usize;
    type SCCAccT = VisitorState<T>;
    type ErrorT = HeedError;

    fn next_unvisited_node(
        &mut self,
        mut already_visited: impl FnMut(&Self::NodeIdT) -> bool,
    ) -> Result<Option<Self::NodeT>, Self::ErrorT> {
        while let Some(item) = self.ro_iter.next() {
            let (node_id, record): (Self::NodeIdT, Lazy<SerdeJson<ObjectRecordNoValue>>) = item?;
            if self.known_skips.contains(&node_id) {
                continue;
            }
            if !already_visited(&node_id) {
                let record = record.decode().map_err(|e| HeedError::Decoding(e))?;
                if self.should_skip_link_in_subtree_exploration(&record) {
                    self.known_skips.insert(node_id);
                    continue;
                }
                return Ok(Some(record));
            }
        }
        Ok(None)
    }

    fn get_node_id(&self, node: &Self::NodeT) -> Self::NodeIdT {
        node.id
    }

    fn get_node_acc(&self, node: &Self::NodeT) -> Self::NodeAccT {
        node.size as usize
    }

    fn get_successors(&self, node: &Self::NodeT) -> Result<Vec<Self::NodeT>, Self::ErrorT> {
        let remapped_db = self
            .explorer
            .primary_db
            .remap_data_type::<SerdeJson<ObjectRecordNoValue>>();
        let results = collect_results(node.references.iter().map(|succ_id| {
            match remapped_db.get(&self.ro_txn, succ_id) {
                Ok(Some(record)) => Ok(record),
                Ok(None) => Err(HeedError::Decoding(
                    format!("Missing record for successor ID {}", succ_id).into(),
                )),
                Err(e) => Err(e),
            }
        }))?;
        Ok(results)
    }

    fn accumulate_node_values(&self, v1: &mut Self::NodeAccT, v2: &Self::NodeAccT) {
        *v1 += *v2;
    }

    fn accumulate_scc_values(&self, v1: &mut Self::SCCAccT, v2: &Self::SCCAccT) {
        v1.size_estimate.include(&v2.size_estimate);
        v1.scc_sketch.include(&v2.scc_sketch);
    }

    fn add_node_value_to_scc_value(
        &self,
        node_acc: &Self::NodeAccT,
        this_scc: usize,
        scc_acc: Option<&Self::SCCAccT>,
    ) -> Self::SCCAccT {
        let mut scc_acc = scc_acc.cloned().unwrap_or_else(|| VisitorState {
            scc_sketch: DefaultMembershipSketch::new(),
            size_estimate: T::empty(),
        });
        scc_acc
            .size_estimate
            .add_in_place(this_scc as u64, *node_acc as u64);
        scc_acc.scc_sketch.add(&this_scc);
        scc_acc
    }

    fn emit_result(
        &mut self,
        node_id: Self::NodeIdT,
        _node_acc_this_scc: Self::NodeAccT,
        scc_acc: Self::SCCAccT,
    ) -> Result<(), Self::ErrorT> {
        self.explorer
            .sccs_sketch_db
            .put(&mut self.rw_txn, &node_id, &scc_acc.scc_sketch)?;
        let mut record = self
            .explorer
            .primary_db
            .get(&self.rw_txn, &node_id)?
            .ok_or_else(|| {
                HeedError::Decoding(format!("Missing record for node ID {}", node_id).into())
            })?;
        let subtree_size = scc_acc.size_estimate.total();
        record.subtree_size = Some(subtree_size);
        self.explorer
            .primary_db
            .put(&mut self.rw_txn, &node_id, &record)?;
        self.explorer.put_size_index_entry(
            &mut self.rw_txn,
            &Type::TypeName(record.type_name),
            &SizeIndexEntry {
                size: subtree_size,
                obj_id: node_id,
            },
            self.explorer.types_subtree_size_index_db,
        )?;
        Ok(())
    }
}

fn collect_results<T, E>(iter: impl IntoIterator<Item = Result<T, E>>) -> Result<Vec<T>, E> {
    iter.into_iter().collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_value_deserialization() {
        assert_eq!(
            serde_json::from_str::<Value>("true").unwrap(),
            Value::Bool(true)
        );
        assert_eq!(
            serde_json::from_str::<Value>("false").unwrap(),
            Value::Bool(false)
        );
        assert_eq!(serde_json::from_str::<Value>("42").unwrap(), Value::Int(42));
        assert_eq!(
            serde_json::from_str::<Value>("3.14").unwrap(),
            Value::Float(3.14)
        );
        assert_eq!(
            serde_json::from_str::<Value>(r#""hello""#).unwrap(),
            Value::Str("hello".to_string())
        );
        assert_eq!(serde_json::from_str::<Value>("null").unwrap(), Value::None);
    }

    #[test]
    fn test_object_summary_deserialization() {
        let json = r#"{"id": 1, "type_name": "int", "value": 42, "size": 28, "subtree_size": 100, "references": []}"#;
        let summary: ObjectSummary = serde_json::from_str(json).unwrap();
        assert_eq!(
            summary,
            ObjectSummary {
                id: 1,
                type_name: "int".to_string(),
                value: Some(Value::Int(42)),
                size: 28,
                subtree_size: Some(100),
            }
        );
    }

    #[test]
    fn test_object_summary_deserialization_opt_fields_empty() {
        let json = r#"{"id": 1, "type_name": "int", "size": 28}"#;
        let summary: ObjectSummary = serde_json::from_str(json).unwrap();
        assert_eq!(
            summary,
            ObjectSummary {
                id: 1,
                type_name: "int".to_string(),
                value: None,
                size: 28,
                subtree_size: None,
            }
        );
    }
}
