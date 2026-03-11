use crate::logical_plan::{LogicalOp, LogicalPlan, PlanError};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PhysicalOp {
    ReadText { path: String },
    Map { func_name: String },
    Filter { func_name: String },
    FlatMap { func_name: String },
    Shuffle { partitions: usize },
    ReduceByKey { func_name: String, partitions: usize },
    GroupByKey { partitions: usize },
    Collect,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Stage {
    pub id: usize,
    pub ops: Vec<PhysicalOp>,
    pub input_stage: Option<usize>,
    pub partitions: usize,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PhysicalPlan {
    pub stages: Vec<Stage>,
}

impl PhysicalPlan {
    pub fn from_logical(logical: &LogicalPlan) -> Result<Self, PlanError> {
        logical.validate()?;

        let sinks = logical.sink_nodes();
        if sinks.is_empty() {
            return Err(PlanError::InvalidPlan("No sink nodes found".to_string()));
        }

        let ordered = topological_sort(logical)?;

        let mut stages: Vec<Stage> = Vec::new();
        let mut current_ops: Vec<PhysicalOp> = Vec::new();
        let mut current_partitions = 1usize;
        let mut stage_input: Option<usize> = None;

        for &node_id in &ordered {
            let node = &logical.nodes[node_id];
            match &node.op {
                LogicalOp::ReadText { path } => {
                    current_ops.push(PhysicalOp::ReadText { path: path.clone() });
                }
                LogicalOp::Map { func_name } => {
                    current_ops.push(PhysicalOp::Map { func_name: func_name.clone() });
                }
                LogicalOp::Filter { func_name } => {
                    current_ops.push(PhysicalOp::Filter { func_name: func_name.clone() });
                }
                LogicalOp::FlatMap { func_name } => {
                    current_ops.push(PhysicalOp::FlatMap { func_name: func_name.clone() });
                }
                LogicalOp::ReduceByKey { func_name, partitions } => {
                    if !current_ops.is_empty() {
                        let stage_id = stages.len();
                        stages.push(Stage {
                            id: stage_id,
                            ops: std::mem::take(&mut current_ops),
                            input_stage: stage_input,
                            partitions: current_partitions,
                        });
                        stage_input = Some(stage_id);
                    }
                    {
                        let shuffle_id = stages.len();
                        stages.push(Stage {
                            id: shuffle_id,
                            ops: vec![PhysicalOp::Shuffle { partitions: *partitions }],
                            input_stage: stage_input,
                            partitions: *partitions,
                        });
                        stage_input = Some(shuffle_id);
                    }
                    let reduce_id = stages.len();
                    stages.push(Stage {
                        id: reduce_id,
                        ops: vec![PhysicalOp::ReduceByKey {
                            func_name: func_name.clone(),
                            partitions: *partitions,
                        }],
                        input_stage: stage_input,
                        partitions: *partitions,
                    });
                    stage_input = Some(reduce_id);
                    current_partitions = *partitions;
                }
                LogicalOp::GroupByKey { partitions } => {
                    if !current_ops.is_empty() {
                        let stage_id = stages.len();
                        stages.push(Stage {
                            id: stage_id,
                            ops: std::mem::take(&mut current_ops),
                            input_stage: stage_input,
                            partitions: current_partitions,
                        });
                        stage_input = Some(stage_id);
                    }
                    let shuffle_id = stages.len();
                    stages.push(Stage {
                        id: shuffle_id,
                        ops: vec![PhysicalOp::Shuffle { partitions: *partitions }],
                        input_stage: stage_input,
                        partitions: *partitions,
                    });
                    stage_input = Some(shuffle_id);
                    let group_id = stages.len();
                    stages.push(Stage {
                        id: group_id,
                        ops: vec![PhysicalOp::GroupByKey { partitions: *partitions }],
                        input_stage: stage_input,
                        partitions: *partitions,
                    });
                    stage_input = Some(group_id);
                    current_partitions = *partitions;
                }
                LogicalOp::Collect => {
                    current_ops.push(PhysicalOp::Collect);
                }
            }
        }

        if !current_ops.is_empty() {
            let stage_id = stages.len();
            stages.push(Stage {
                id: stage_id,
                ops: current_ops,
                input_stage: stage_input,
                partitions: current_partitions,
            });
        }

        Ok(PhysicalPlan { stages })
    }
}

fn topological_sort(plan: &LogicalPlan) -> Result<Vec<usize>, PlanError> {
    let n = plan.nodes.len();
    let mut visited = vec![false; n];
    let mut order = Vec::with_capacity(n);

    fn dfs(node: usize, plan: &LogicalPlan, visited: &mut Vec<bool>, order: &mut Vec<usize>) {
        if visited[node] {
            return;
        }
        visited[node] = true;
        for &inp in &plan.nodes[node].inputs {
            dfs(inp, plan, visited, order);
        }
        order.push(node);
    }

    for i in 0..n {
        dfs(i, plan, &mut visited, &mut order);
    }

    Ok(order)
}
