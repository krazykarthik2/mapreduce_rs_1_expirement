use std::fs::{self, File};
use std::io::{BufRead, BufReader, Write};
use std::sync::Arc;
use std::time::Instant;

use log::{debug, info};
use rayon::prelude::*;

use crate::job_registry::{JobRegistry, Record};
use crate::physical_plan::{PhysicalOp, PhysicalPlan, Stage};
use crate::shuffle::{group_by_key, ShuffleBuffer};

/// Configuration for the execution engine
pub struct ExecutionConfig {
    pub num_partitions: usize,
    pub chunk_size: usize,
}

impl Default for ExecutionConfig {
    fn default() -> Self {
        Self {
            num_partitions: num_cpus(),
            chunk_size: 10_000,
        }
    }
}

fn num_cpus() -> usize {
    std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4)
}

pub struct ExecutionEngine {
    pub registry: Arc<JobRegistry>,
    pub config: ExecutionConfig,
}

impl ExecutionEngine {
    pub fn new(registry: JobRegistry) -> Self {
        Self {
            registry: Arc::new(registry),
            config: ExecutionConfig::default(),
        }
    }

    pub fn with_config(registry: JobRegistry, config: ExecutionConfig) -> Self {
        Self {
            registry: Arc::new(registry),
            config,
        }
    }

    /// Execute a physical plan, return collected results
    pub fn execute(&self, plan: &PhysicalPlan) -> Vec<Record> {
        let mut stage_results: Vec<Vec<Record>> = vec![Vec::new(); plan.stages.len()];

        for stage in &plan.stages {
            let start = Instant::now();
            info!(
                "[Stage {}] Starting execution of stage with {} ops",
                stage.id,
                stage.ops.len()
            );

            let input: Vec<Record> = if let Some(input_stage_id) = stage.input_stage {
                std::mem::take(&mut stage_results[input_stage_id])
            } else {
                Vec::new()
            };

            let result = self.execute_stage(stage, input);
            let count = result.len();
            stage_results[stage.id] = result;

            let elapsed = start.elapsed();
            info!(
                "[Stage {}] Completed in {:.3}s — {} records out",
                stage.id,
                elapsed.as_secs_f64(),
                count
            );
        }

        stage_results.into_iter().last().unwrap_or_default()
    }

    fn execute_stage(&self, stage: &Stage, input: Vec<Record>) -> Vec<Record> {
        let mut records = input;
        for op in &stage.ops {
            records = self.execute_op(op, records);
        }
        records
    }

    fn execute_op(&self, op: &PhysicalOp, records: Vec<Record>) -> Vec<Record> {
        match op {
            PhysicalOp::ReadText { path } => {
                let result = self.read_text_parallel(path);
                info!("[ReadText] Read {} records from {}", result.len(), path);
                result
            }
            PhysicalOp::Map { func_name } => {
                let input_len = records.len();
                let result = self.apply_map(func_name, records);
                debug!("[Map:{}] {} -> {} records", func_name, input_len, result.len());
                result
            }
            PhysicalOp::FlatMap { func_name } => {
                let input_len = records.len();
                let result = self.apply_map(func_name, records);
                debug!("[FlatMap:{}] {} -> {} records", func_name, input_len, result.len());
                result
            }
            PhysicalOp::Filter { func_name } => {
                let f = self.registry.get_filter(func_name).expect("Filter function not found");
                let f = f.clone();
                let input_len = records.len();
                let result: Vec<Record> =
                    records.into_par_iter().filter(|r| f(r)).collect();
                debug!("[Filter:{}] {} -> {} records", func_name, input_len, result.len());
                result
            }
            PhysicalOp::Shuffle { partitions } => {
                let buffer = ShuffleBuffer::new(*partitions);
                records.into_par_iter().for_each(|r| {
                    if let Record::KeyValue(k, v) = r {
                        buffer.insert(k, v);
                    }
                });
                info!(
                    "[Shuffle] {} total records across {} partitions",
                    buffer.total_records(),
                    partitions
                );
                (0..*partitions)
                    .flat_map(|p| {
                        buffer
                            .drain_partition(p)
                            .into_iter()
                            .map(|(k, v)| Record::KeyValue(k, v))
                    })
                    .collect()
            }
            PhysicalOp::ReduceByKey { func_name, partitions } => {
                let f = self
                    .registry
                    .get_reduce(func_name)
                    .expect("Reduce function not found");
                let f = f.clone();
                let kv_pairs: Vec<(String, String)> = records
                    .into_iter()
                    .filter_map(|r| {
                        if let Record::KeyValue(k, v) = r { Some((k, v)) } else { None }
                    })
                    .collect();
                let grouped = group_by_key(kv_pairs);
                let result: Vec<Record> = grouped
                    .into_par_iter()
                    .map(|(k, vals)| {
                        let (k2, v2) = f(k, vals);
                        Record::KeyValue(k2, v2)
                    })
                    .collect();
                info!(
                    "[ReduceByKey:{}] {} key groups, {} partitions",
                    func_name,
                    result.len(),
                    partitions
                );
                result
            }
            PhysicalOp::GroupByKey { partitions } => {
                let kv_pairs: Vec<(String, String)> = records
                    .into_iter()
                    .filter_map(|r| {
                        if let Record::KeyValue(k, v) = r { Some((k, v)) } else { None }
                    })
                    .collect();
                let grouped = group_by_key(kv_pairs);
                info!("[GroupByKey] {} key groups, {} partitions", grouped.len(), partitions);
                grouped.into_iter().map(|(k, vals)| Record::KeyValues(k, vals)).collect()
            }
            PhysicalOp::Collect => records,
        }
    }

    fn read_text_parallel(&self, path: &str) -> Vec<Record> {
        let file =
            File::open(path).unwrap_or_else(|e| panic!("Failed to open input file '{}': {}", path, e));
        let reader = BufReader::new(file);
        reader.lines().filter_map(|l| l.ok()).map(Record::Text).collect()
    }

    fn apply_map(&self, func_name: &str, records: Vec<Record>) -> Vec<Record> {
        let f = self.registry.get_map(func_name).expect("Map function not found");
        let f = f.clone();
        let chunk_size = self.config.chunk_size;
        records
            .par_chunks(chunk_size)
            .flat_map(|chunk| chunk.iter().cloned().flat_map(|r| f(r)).collect::<Vec<_>>())
            .collect()
    }

    /// Write results to output directory, one file per partition
    pub fn write_partitions(&self, results: &[Record], output_dir: &str) {
        fs::create_dir_all(output_dir).expect("Failed to create output directory");
        let num_partitions = self.config.num_partitions;
        let mut partitions: Vec<Vec<String>> = vec![Vec::new(); num_partitions];

        for (i, record) in results.iter().enumerate() {
            let part = i % num_partitions;
            let line = match record {
                Record::Text(s) => s.clone(),
                Record::KeyValue(k, v) => format!("{}\t{}", k, v),
                Record::KeyValues(k, vs) => format!("{}\t{}", k, vs.join(",")),
            };
            partitions[part].push(line);
        }

        partitions.par_iter().enumerate().for_each(|(i, lines)| {
            let path = format!("{}/part-{:05}.txt", output_dir, i);
            let mut f = File::create(&path).expect("Failed to create partition file");
            for line in lines {
                writeln!(f, "{}", line).expect("Failed to write");
            }
            info!("[Output] Wrote {} lines to {}", lines.len(), path);
        });
    }
}
