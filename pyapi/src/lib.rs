use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::sync::{Arc, Mutex};

use mapreduce_engine::{
    execution::{ExecutionConfig, ExecutionEngine},
    job_registry::JobRegistry,
    logical_plan::{LogicalOp, LogicalPlan},
    physical_plan::PhysicalPlan,
};

/// Python Dataset wrapper — lazy evaluation until .collect()
#[pyclass]
struct Dataset {
    plan: Arc<Mutex<LogicalPlan>>,
    last_node_id: Arc<Mutex<Option<usize>>>,
    partitions: usize,
}

#[pymethods]
impl Dataset {
    /// Create a new Dataset from a text file (lazy)
    #[staticmethod]
    #[pyo3(signature = (path, partitions=None))]
    fn read_text(path: String, partitions: Option<usize>) -> PyResult<Dataset> {
        let mut plan = LogicalPlan::new();
        let id = plan.add_node(LogicalOp::ReadText { path }, vec![]);
        Ok(Dataset {
            plan: Arc::new(Mutex::new(plan)),
            last_node_id: Arc::new(Mutex::new(Some(id))),
            partitions: partitions.unwrap_or(4),
        })
    }

    /// Apply a named map function (lazy)
    fn map(&self, func_name: String) -> PyResult<Dataset> {
        let mut plan = self.plan.lock().unwrap().clone();
        let inputs =
            self.last_node_id.lock().unwrap().map(|id| vec![id]).unwrap_or_default();
        let id = plan.add_node(LogicalOp::Map { func_name }, inputs);
        Ok(Dataset {
            plan: Arc::new(Mutex::new(plan)),
            last_node_id: Arc::new(Mutex::new(Some(id))),
            partitions: self.partitions,
        })
    }

    /// Apply a named filter function (lazy)
    fn filter(&self, func_name: String) -> PyResult<Dataset> {
        let mut plan = self.plan.lock().unwrap().clone();
        let inputs =
            self.last_node_id.lock().unwrap().map(|id| vec![id]).unwrap_or_default();
        let id = plan.add_node(LogicalOp::Filter { func_name }, inputs);
        Ok(Dataset {
            plan: Arc::new(Mutex::new(plan)),
            last_node_id: Arc::new(Mutex::new(Some(id))),
            partitions: self.partitions,
        })
    }

    /// Apply a named flatmap function (lazy)
    fn flatmap(&self, func_name: String) -> PyResult<Dataset> {
        let mut plan = self.plan.lock().unwrap().clone();
        let inputs =
            self.last_node_id.lock().unwrap().map(|id| vec![id]).unwrap_or_default();
        let id = plan.add_node(LogicalOp::FlatMap { func_name }, inputs);
        Ok(Dataset {
            plan: Arc::new(Mutex::new(plan)),
            last_node_id: Arc::new(Mutex::new(Some(id))),
            partitions: self.partitions,
        })
    }

    /// Apply reduce_by_key with a named reduce function (lazy)
    #[pyo3(signature = (func_name, partitions=None))]
    fn reduce_by_key(&self, func_name: String, partitions: Option<usize>) -> PyResult<Dataset> {
        let mut plan = self.plan.lock().unwrap().clone();
        let inputs =
            self.last_node_id.lock().unwrap().map(|id| vec![id]).unwrap_or_default();
        let parts = partitions.unwrap_or(self.partitions);
        let id = plan.add_node(
            LogicalOp::ReduceByKey { func_name, partitions: parts },
            inputs,
        );
        Ok(Dataset {
            plan: Arc::new(Mutex::new(plan)),
            last_node_id: Arc::new(Mutex::new(Some(id))),
            partitions: parts,
        })
    }

    /// Apply group_by_key (lazy)
    #[pyo3(signature = (partitions=None))]
    fn group_by_key(&self, partitions: Option<usize>) -> PyResult<Dataset> {
        let mut plan = self.plan.lock().unwrap().clone();
        let inputs =
            self.last_node_id.lock().unwrap().map(|id| vec![id]).unwrap_or_default();
        let parts = partitions.unwrap_or(self.partitions);
        let id = plan.add_node(LogicalOp::GroupByKey { partitions: parts }, inputs);
        Ok(Dataset {
            plan: Arc::new(Mutex::new(plan)),
            last_node_id: Arc::new(Mutex::new(Some(id))),
            partitions: parts,
        })
    }

    /// Trigger execution and return results as a Python list of strings
    fn collect(&self) -> PyResult<Vec<String>> {
        let plan = self.plan.lock().unwrap().clone();
        let physical = PhysicalPlan::from_logical(&plan)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        let registry = JobRegistry::new();
        let config = ExecutionConfig { num_partitions: self.partitions, chunk_size: 10_000 };
        let engine = ExecutionEngine::with_config(registry, config);
        let results = engine.execute(&physical);

        Ok(results
            .into_iter()
            .map(|r| match r {
                mapreduce_engine::job_registry::Record::Text(s) => s,
                mapreduce_engine::job_registry::Record::KeyValue(k, v) => {
                    format!("{}\t{}", k, v)
                }
                mapreduce_engine::job_registry::Record::KeyValues(k, vs) => {
                    format!("{}\t{}", k, vs.join(","))
                }
            })
            .collect())
    }

    /// Get the number of records without returning them
    fn count(&self) -> PyResult<usize> {
        Ok(self.collect()?.len())
    }

    /// Write results to output directory
    fn write_output(&self, output_dir: String) -> PyResult<()> {
        let plan = self.plan.lock().unwrap().clone();
        let physical = PhysicalPlan::from_logical(&plan)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        let registry = JobRegistry::new();
        let config = ExecutionConfig { num_partitions: self.partitions, chunk_size: 10_000 };
        let engine = ExecutionEngine::with_config(registry, config);
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

/// Python module entry point
#[pymodule]
fn mapreduce_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    env_logger::try_init().ok();
    m.add_class::<Dataset>()?;
    Ok(())
}
