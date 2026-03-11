use clap::{Parser, Subcommand};
use mapreduce_engine::{
    execution::{ExecutionConfig, ExecutionEngine},
    job_registry::JobRegistry,
    logical_plan::{LogicalOp, LogicalPlan},
    physical_plan::PhysicalPlan,
};
use serde::{Deserialize, Serialize};
use std::fs;

/// Rust MapReduce Engine CLI
#[derive(Parser, Debug)]
#[command(name = "mapreduce", about = "Rust MapReduce/DAG Engine CLI", version = "0.1.0")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Run a job from a JSON spec file
    Run {
        /// Path to the JSON job spec
        #[arg(short, long)]
        spec: String,
        /// Output directory for results
        #[arg(short, long, default_value = "output")]
        output: String,
        /// Number of partitions (default: CPU count)
        #[arg(short, long)]
        partitions: Option<usize>,
    },
    /// Run the built-in wordcount job
    Wordcount {
        /// Input file path
        #[arg(short, long)]
        input: String,
        /// Output directory
        #[arg(short, long, default_value = "output")]
        output: String,
        /// Number of partitions
        #[arg(short, long, default_value_t = 4)]
        partitions: usize,
    },
    /// Print a JSON template job spec
    Template,
}

/// JSON job spec format
#[derive(Debug, Serialize, Deserialize)]
struct JobSpec {
    name: String,
    steps: Vec<StepSpec>,
}

#[derive(Debug, Serialize, Deserialize)]
struct StepSpec {
    op: String,
    #[serde(default)]
    func_name: Option<String>,
    #[serde(default)]
    path: Option<String>,
    #[serde(default)]
    partitions: Option<usize>,
}

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Run { spec, output, partitions } => {
            run_from_spec(&spec, &output, partitions);
        }
        Commands::Wordcount { input, output, partitions } => {
            run_wordcount(&input, &output, partitions);
        }
        Commands::Template => {
            print_template();
        }
    }
}

fn run_from_spec(spec_path: &str, output: &str, partitions: Option<usize>) {
    let content = fs::read_to_string(spec_path).expect("Failed to read spec file");
    let spec: JobSpec = serde_json::from_str(&content).expect("Invalid JSON spec");

    log::info!("Running job: {}", spec.name);

    let mut plan = LogicalPlan::new();
    let mut last_id: Option<usize> = None;
    let default_partitions = partitions.unwrap_or(4);

    for step in &spec.steps {
        let inputs = last_id.map(|id| vec![id]).unwrap_or_default();
        let op = match step.op.as_str() {
            "read_text" => LogicalOp::ReadText {
                path: step.path.clone().expect("read_text requires 'path'"),
            },
            "map" => LogicalOp::Map {
                func_name: step.func_name.clone().expect("map requires 'func_name'"),
            },
            "filter" => LogicalOp::Filter {
                func_name: step.func_name.clone().expect("filter requires 'func_name'"),
            },
            "flatmap" => LogicalOp::FlatMap {
                func_name: step.func_name.clone().expect("flatmap requires 'func_name'"),
            },
            "reduce_by_key" => LogicalOp::ReduceByKey {
                func_name: step
                    .func_name
                    .clone()
                    .expect("reduce_by_key requires 'func_name'"),
                partitions: step.partitions.unwrap_or(default_partitions),
            },
            "group_by_key" => LogicalOp::GroupByKey {
                partitions: step.partitions.unwrap_or(default_partitions),
            },
            "collect" => LogicalOp::Collect,
            other => {
                eprintln!(
                    "Error: Unknown op '{}'. Valid ops: read_text, map, filter, flatmap, reduce_by_key, group_by_key, collect",
                    other
                );
                std::process::exit(1);
            }
        };
        last_id = Some(plan.add_node(op, inputs));
    }

    let physical = PhysicalPlan::from_logical(&plan).expect("Physical planning failed");
    let registry = JobRegistry::new();
    let config = ExecutionConfig {
        num_partitions: default_partitions,
        chunk_size: 10_000,
    };
    let engine = ExecutionEngine::with_config(registry, config);

    let results = engine.execute(&physical);
    log::info!("Total output records: {}", results.len());
    engine.write_partitions(&results, output);
    log::info!("Output written to: {}", output);
}

fn run_wordcount(input: &str, output: &str, partitions: usize) {
    log::info!("Running wordcount on: {}", input);

    let mut plan = LogicalPlan::new();
    let read_id =
        plan.add_node(LogicalOp::ReadText { path: input.to_string() }, vec![]);
    let filter_id = plan.add_node(
        LogicalOp::Filter { func_name: "non_empty".to_string() },
        vec![read_id],
    );
    let map_id = plan.add_node(
        LogicalOp::FlatMap { func_name: "tokenize".to_string() },
        vec![filter_id],
    );
    let reduce_id = plan.add_node(
        LogicalOp::ReduceByKey { func_name: "sum".to_string(), partitions },
        vec![map_id],
    );
    let _ = plan.add_node(LogicalOp::Collect, vec![reduce_id]);

    plan.debug_print();

    let physical = PhysicalPlan::from_logical(&plan).expect("Physical planning failed");
    let registry = JobRegistry::new();
    let config = ExecutionConfig {
        num_partitions: partitions,
        chunk_size: 10_000,
    };
    let engine = ExecutionEngine::with_config(registry, config);

    let results = engine.execute(&physical);
    log::info!("Unique words: {}", results.len());
    engine.write_partitions(&results, output);
    log::info!("Output written to: {}", output);
}

fn print_template() {
    let template = JobSpec {
        name: "my_job".to_string(),
        steps: vec![
            StepSpec {
                op: "read_text".to_string(),
                path: Some("/path/to/input.txt".to_string()),
                func_name: None,
                partitions: None,
            },
            StepSpec {
                op: "filter".to_string(),
                func_name: Some("non_empty".to_string()),
                path: None,
                partitions: None,
            },
            StepSpec {
                op: "flatmap".to_string(),
                func_name: Some("tokenize".to_string()),
                path: None,
                partitions: None,
            },
            StepSpec {
                op: "reduce_by_key".to_string(),
                func_name: Some("sum".to_string()),
                partitions: Some(4),
                path: None,
            },
            StepSpec {
                op: "collect".to_string(),
                func_name: None,
                path: None,
                partitions: None,
            },
        ],
    };
    println!("{}", serde_json::to_string_pretty(&template).unwrap());
}
