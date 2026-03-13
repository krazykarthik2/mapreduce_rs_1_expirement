use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use std::sync::{Arc, Mutex};

use mapreduce_engine::{
    execution::{ExecutionConfig, ExecutionEngine},
    job_registry::{JobRegistry, Record},
    logical_plan::{LogicalOp, LogicalPlan},
    physical_plan::PhysicalPlan,
};

const MAP_ADD_PREFIX: &str = "__lambda_map_add__";
const FILTER_EQ_PREFIX: &str = "__lambda_filter_eq__";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FunctionKind {
    Map,
    Filter,
    FlatMap,
    Reduce,
}

impl FunctionKind {
    fn as_str(self) -> &'static str {
        match self {
            FunctionKind::Map => "map",
            FunctionKind::Filter => "filter",
            FunctionKind::FlatMap => "flatmap",
            FunctionKind::Reduce => "reduce",
        }
    }
}

#[pyclass(name = "FunctionRef")]
#[derive(Clone, Debug)]
struct FunctionRef {
    func_name: String,
    kind: String,
}

impl FunctionRef {
    fn new(func_name: String, kind: FunctionKind) -> Self {
        Self {
            func_name,
            kind: kind.as_str().to_string(),
        }
    }
}

#[pymethods]
impl FunctionRef {
    fn name(&self) -> String {
        self.func_name.clone()
    }

    fn kind(&self) -> String {
        self.kind.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "FunctionRef(kind='{}', name='{}')",
            self.kind, self.func_name
        )
    }
}

#[pyclass]
#[derive(Default)]
struct ExprArg;

#[pymethods]
impl ExprArg {
    #[pyo3(signature = (value=None))]
    fn add(&self, value: Option<i64>) -> FunctionRef {
        add_ref(value)
    }

    fn eq(&self, value: String) -> FunctionRef {
        eq_ref(value)
    }

    fn identity(&self) -> FunctionRef {
        identity_ref()
    }

    fn non_empty(&self) -> FunctionRef {
        non_empty_ref()
    }

    fn tokenize(&self) -> FunctionRef {
        tokenize_ref()
    }
}

/// Python Dataset wrapper — lazy evaluation until .collect()
#[pyclass]
struct Dataset {
    plan: Arc<Mutex<LogicalPlan>>,
    last_node_id: Arc<Mutex<Option<usize>>>,
    partitions: usize,
}

impl Dataset {
    /// Clone the current plan and extract the last node inputs — shared by all transform methods.
    fn fork_plan(&self) -> (LogicalPlan, Vec<usize>) {
        let plan = self.plan.lock().unwrap().clone();
        let inputs = self
            .last_node_id
            .lock()
            .unwrap()
            .map(|id| vec![id])
            .unwrap_or_default();
        (plan, inputs)
    }

    /// Build a new Dataset from a modified plan and a new terminal node ID.
    fn with_new_node(plan: LogicalPlan, node_id: usize, partitions: usize) -> Dataset {
        Dataset {
            plan: Arc::new(Mutex::new(plan)),
            last_node_id: Arc::new(Mutex::new(Some(node_id))),
            partitions,
        }
    }

    /// Build a physical plan from the current logical plan.
    fn build_physical(&self) -> PyResult<PhysicalPlan> {
        let plan = self.plan.lock().unwrap().clone();
        PhysicalPlan::from_logical(&plan).map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Create an execution engine configured for this dataset's partition count.
    fn create_engine(&self) -> PyResult<ExecutionEngine> {
        let mut registry = JobRegistry::new();
        let plan = self.plan.lock().unwrap().clone();
        register_dynamic_functions(&plan, &mut registry)?;
        let config = ExecutionConfig {
            num_partitions: self.partitions,
            chunk_size: 10_000,
        };
        Ok(ExecutionEngine::with_config(registry, config))
    }
}

#[pymethods]
impl Dataset {
    /// Create a new Dataset from a text file (lazy)
    #[staticmethod]
    #[pyo3(signature = (path, partitions=None))]
    fn read_text(path: String, partitions: Option<usize>) -> PyResult<Dataset> {
        let mut plan = LogicalPlan::new();
        let id = plan.add_node(LogicalOp::ReadText { path }, vec![]);
        Ok(Dataset::with_new_node(plan, id, partitions.unwrap_or(4)))
    }

    /// Apply a named map function (lazy)
    fn map(&self, py: Python<'_>, func: &Bound<'_, PyAny>) -> PyResult<Dataset> {
        let func_name = resolve_func_name_from_any(py, func, FunctionKind::Map)?;
        let (mut plan, inputs) = self.fork_plan();
        let id = plan.add_node(LogicalOp::Map { func_name }, inputs);
        Ok(Dataset::with_new_node(plan, id, self.partitions))
    }

    /// Apply a named filter function (lazy)
    fn filter(&self, py: Python<'_>, func: &Bound<'_, PyAny>) -> PyResult<Dataset> {
        let func_name = resolve_func_name_from_any(py, func, FunctionKind::Filter)?;
        let (mut plan, inputs) = self.fork_plan();
        let id = plan.add_node(LogicalOp::Filter { func_name }, inputs);
        Ok(Dataset::with_new_node(plan, id, self.partitions))
    }

    /// Apply a named flatmap function (lazy)
    fn flatmap(&self, py: Python<'_>, func: &Bound<'_, PyAny>) -> PyResult<Dataset> {
        let func_name = resolve_func_name_from_any(py, func, FunctionKind::FlatMap)?;
        let (mut plan, inputs) = self.fork_plan();
        let id = plan.add_node(LogicalOp::FlatMap { func_name }, inputs);
        Ok(Dataset::with_new_node(plan, id, self.partitions))
    }

    /// Apply reduce_by_key with a named reduce function (lazy)
    #[pyo3(signature = (func, partitions=None))]
    fn reduce_by_key(
        &self,
        py: Python<'_>,
        func: &Bound<'_, PyAny>,
        partitions: Option<usize>,
    ) -> PyResult<Dataset> {
        let func_name = resolve_func_name_from_any(py, func, FunctionKind::Reduce)?;
        let (mut plan, inputs) = self.fork_plan();
        let parts = partitions.unwrap_or(self.partitions);
        let id = plan.add_node(
            LogicalOp::ReduceByKey {
                func_name,
                partitions: parts,
            },
            inputs,
        );
        Ok(Dataset::with_new_node(plan, id, parts))
    }

    /// Apply group_by_key (lazy)
    #[pyo3(signature = (partitions=None))]
    fn group_by_key(&self, partitions: Option<usize>) -> PyResult<Dataset> {
        let (mut plan, inputs) = self.fork_plan();
        let parts = partitions.unwrap_or(self.partitions);
        let id = plan.add_node(LogicalOp::GroupByKey { partitions: parts }, inputs);
        Ok(Dataset::with_new_node(plan, id, parts))
    }

    /// Trigger execution and return results as a Python list of strings
    fn collect(&self) -> PyResult<Vec<String>> {
        let physical = self.build_physical()?;
        let engine = self.create_engine()?;
        let results = engine.execute(&physical);
        Ok(results.into_iter().map(record_to_string).collect())
    }

    /// Get the number of records without returning them
    fn count(&self) -> PyResult<usize> {
        Ok(self.collect()?.len())
    }

    /// Write results to output directory
    fn write_output(&self, output_dir: String) -> PyResult<()> {
        let physical = self.build_physical()?;
        let engine = self.create_engine()?;
        let results = engine.execute(&physical);
        engine.write_partitions(&results, &output_dir);
        Ok(())
    }

    /// Pretty-print the logical plan
    fn explain(&self) -> PyResult<String> {
        let plan = self.plan.lock().unwrap();
        let mut out = String::new();
        for node in &plan.nodes {
            out.push_str(&format!(
                "Node {}: {:?} <- {:?}\n",
                node.id, node.op, node.inputs
            ));
        }
        Ok(out)
    }
}

fn record_to_string(r: Record) -> String {
    match r {
        Record::Text(s) => s,
        Record::KeyValue(k, v) => format!("{}\t{}", k, v),
        Record::KeyValues(k, vs) => format!("{}\t{}", k, vs.join(",")),
    }
}

fn resolve_func_name_from_any(
    py: Python<'_>,
    func: &Bound<'_, PyAny>,
    expected_kind: FunctionKind,
) -> PyResult<String> {
    if let Ok(func_name) = func.extract::<String>() {
        return Ok(func_name);
    }

    if let Ok(function_ref) = func.extract::<PyRef<'_, FunctionRef>>() {
        return ensure_expected_kind(&function_ref.func_name, &function_ref.kind, expected_kind);
    }

    if func.is_callable() {
        return resolve_from_callable(py, func, expected_kind);
    }

    Err(PyTypeError::new_err(format!(
        "Expected function name string, FunctionRef, or lambda(expr) returning FunctionRef for {}",
        expected_kind.as_str()
    )))
}

fn resolve_from_callable(
    py: Python<'_>,
    func: &Bound<'_, PyAny>,
    expected_kind: FunctionKind,
) -> PyResult<String> {
    let expr_arg = Py::new(py, ExprArg::default())?;
    let value = func.call1((expr_arg.clone_ref(py),))?;
    resolve_func_name_from_any(py, &value, expected_kind)
}

fn ensure_expected_kind(
    func_name: &str,
    actual_kind: &str,
    expected_kind: FunctionKind,
) -> PyResult<String> {
    if actual_kind == expected_kind.as_str() {
        Ok(func_name.to_string())
    } else {
        Err(PyValueError::new_err(format!(
            "FunctionRef kind '{}' cannot be used for '{}'",
            actual_kind,
            expected_kind.as_str()
        )))
    }
}

fn map_add_name(amount: i64) -> String {
    format!("{MAP_ADD_PREFIX}{amount}")
}

fn filter_eq_name(value: &str) -> String {
    format!("{FILTER_EQ_PREFIX}{value}")
}

fn add_ref(value: Option<i64>) -> FunctionRef {
    match value {
        Some(amount) => FunctionRef::new(map_add_name(amount), FunctionKind::Map),
        None => FunctionRef::new("sum".to_string(), FunctionKind::Reduce),
    }
}

fn eq_ref(value: String) -> FunctionRef {
    FunctionRef::new(filter_eq_name(&value), FunctionKind::Filter)
}

fn identity_ref() -> FunctionRef {
    FunctionRef::new("identity".to_string(), FunctionKind::Map)
}

fn non_empty_ref() -> FunctionRef {
    FunctionRef::new("non_empty".to_string(), FunctionKind::Filter)
}

fn tokenize_ref() -> FunctionRef {
    FunctionRef::new("tokenize".to_string(), FunctionKind::FlatMap)
}

fn register_dynamic_functions(plan: &LogicalPlan, registry: &mut JobRegistry) -> PyResult<()> {
    for node in &plan.nodes {
        match &node.op {
            LogicalOp::Map { func_name }
            | LogicalOp::Filter { func_name }
            | LogicalOp::FlatMap { func_name }
            | LogicalOp::ReduceByKey { func_name, .. } => {
                register_dynamic_function(func_name, registry)?
            }
            _ => {}
        }
    }
    Ok(())
}

fn register_dynamic_function(func_name: &str, registry: &mut JobRegistry) -> PyResult<()> {
    if let Some(amount_str) = func_name.strip_prefix(MAP_ADD_PREFIX) {
        let amount = amount_str.parse::<i64>().map_err(|_| {
            PyValueError::new_err(format!(
                "Invalid add() amount in function name '{func_name}'"
            ))
        })?;

        registry.register_map(
            func_name,
            Arc::new(move |record| {
                let updated = match record {
                    Record::Text(text) => match text.parse::<i64>() {
                        Ok(value) => Record::Text((value + amount).to_string()),
                        Err(_) => {
                            log::debug!("add({amount}) skipped non-numeric text record: {text}");
                            Record::Text(text)
                        }
                    },
                    Record::KeyValue(key, value_text) => match value_text.parse::<i64>() {
                        Ok(value) => Record::KeyValue(key, (value + amount).to_string()),
                        Err(_) => {
                            log::debug!(
                                "add({amount}) skipped non-numeric key/value record - key: {key}, value: {value_text}"
                            );
                            Record::KeyValue(key, value_text)
                        }
                    },
                    Record::KeyValues(key, values) => Record::KeyValues(key, values),
                };
                vec![updated]
            }),
        );
    } else if let Some(expected) = func_name.strip_prefix(FILTER_EQ_PREFIX) {
        let expected = expected.to_string();
        registry.register_filter(
            func_name,
            Arc::new(move |record| match record {
                Record::Text(text) => text == &expected,
                Record::KeyValue(key, value) => key == &expected || value == &expected,
                Record::KeyValues(key, values) => {
                    key == &expected || values.iter().any(|v| v == &expected)
                }
            }),
        );
    }

    Ok(())
}

#[pyfunction(signature = (value=None))]
fn add(value: Option<i64>) -> FunctionRef {
    add_ref(value)
}

#[pyfunction]
fn eq(value: String) -> FunctionRef {
    eq_ref(value)
}

#[pyfunction]
fn identity() -> FunctionRef {
    identity_ref()
}

#[pyfunction]
fn non_empty() -> FunctionRef {
    non_empty_ref()
}

#[pyfunction]
fn tokenize() -> FunctionRef {
    tokenize_ref()
}

/// Python module entry point
#[pymodule]
fn mapreduce_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    env_logger::try_init().ok();
    m.add_class::<Dataset>()?;
    m.add_class::<FunctionRef>()?;
    m.add_class::<ExprArg>()?;
    m.add_function(wrap_pyfunction!(add, m)?)?;
    m.add_function(wrap_pyfunction!(eq, m)?)?;
    m.add_function(wrap_pyfunction!(identity, m)?)?;
    m.add_function(wrap_pyfunction!(non_empty, m)?)?;
    m.add_function(wrap_pyfunction!(tokenize, m)?)?;
    Ok(())
}
