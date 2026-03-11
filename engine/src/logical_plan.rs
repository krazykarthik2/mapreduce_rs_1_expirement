use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LogicalOp {
    ReadText { path: String },
    Map { func_name: String },
    Filter { func_name: String },
    FlatMap { func_name: String },
    ReduceByKey { func_name: String, partitions: usize },
    GroupByKey { partitions: usize },
    Collect,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogicalNode {
    pub id: usize,
    pub op: LogicalOp,
    pub inputs: Vec<usize>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LogicalPlan {
    pub nodes: Vec<LogicalNode>,
}

#[derive(Debug, Error)]
pub enum PlanError {
    #[error("Invalid plan: {0}")]
    InvalidPlan(String),
    #[error("Cycle detected in DAG")]
    CycleDetected,
    #[error("Node {0} not found")]
    NodeNotFound(usize),
}

impl LogicalPlan {
    pub fn new() -> Self {
        Self { nodes: Vec::new() }
    }

    pub fn add_node(&mut self, op: LogicalOp, inputs: Vec<usize>) -> usize {
        let id = self.nodes.len();
        self.nodes.push(LogicalNode { id, op, inputs });
        id
    }

    pub fn validate(&self) -> Result<(), PlanError> {
        let n = self.nodes.len();
        let mut visited = vec![0u8; n];
        for i in 0..n {
            if visited[i] == 0 {
                self.dfs_check(i, &mut visited)?;
            }
        }
        Ok(())
    }

    fn dfs_check(&self, node: usize, visited: &mut Vec<u8>) -> Result<(), PlanError> {
        visited[node] = 1;
        for &inp in &self.nodes[node].inputs {
            if inp >= self.nodes.len() {
                return Err(PlanError::NodeNotFound(inp));
            }
            if visited[inp] == 1 {
                return Err(PlanError::CycleDetected);
            }
            if visited[inp] == 0 {
                self.dfs_check(inp, visited)?;
            }
        }
        visited[node] = 2;
        Ok(())
    }

    pub fn sink_nodes(&self) -> Vec<usize> {
        let mut has_output = vec![false; self.nodes.len()];
        for node in &self.nodes {
            for &inp in &node.inputs {
                has_output[inp] = true;
            }
        }
        (0..self.nodes.len()).filter(|&i| !has_output[i]).collect()
    }

    pub fn debug_print(&self) {
        for node in &self.nodes {
            println!("Node {}: {:?} <- {:?}", node.id, node.op, node.inputs);
        }
    }
}
