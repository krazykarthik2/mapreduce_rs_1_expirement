use mapreduce_engine::{
    execution::ExecutionEngine,
    job_registry::{JobRegistry, Record},
    logical_plan::{LogicalOp, LogicalPlan},
    physical_plan::PhysicalPlan,
};
use tempfile::NamedTempFile;
use std::io::Write;

fn make_wordcount_plan(path: &str, partitions: usize) -> LogicalPlan {
    let mut plan = LogicalPlan::new();
    let read_id =
        plan.add_node(LogicalOp::ReadText { path: path.to_string() }, vec![]);
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
    plan.add_node(LogicalOp::Collect, vec![reduce_id]);
    plan
}

#[test]
fn test_wordcount() {
    let mut tmp = NamedTempFile::new().unwrap();
    writeln!(tmp, "hello world").unwrap();
    writeln!(tmp, "hello rust").unwrap();
    writeln!(tmp, "world rust").unwrap();

    let path = tmp.path().to_str().unwrap().to_string();
    let plan = make_wordcount_plan(&path, 2);

    let physical = PhysicalPlan::from_logical(&plan).unwrap();
    let engine = ExecutionEngine::new(JobRegistry::new());
    let results = engine.execute(&physical);

    let mut counts: std::collections::HashMap<String, i64> =
        std::collections::HashMap::new();
    for r in results {
        if let Record::KeyValue(k, v) = r {
            counts.insert(k, v.parse().unwrap());
        }
    }

    assert_eq!(counts.get("hello"), Some(&2));
    assert_eq!(counts.get("world"), Some(&2));
    assert_eq!(counts.get("rust"), Some(&2));
}

#[test]
fn test_logical_plan_validation() {
    let mut plan = LogicalPlan::new();
    let a = plan.add_node(LogicalOp::ReadText { path: "a.txt".to_string() }, vec![]);
    let b = plan.add_node(
        LogicalOp::Map { func_name: "identity".to_string() },
        vec![a],
    );
    let _ = plan.add_node(LogicalOp::Collect, vec![b]);
    assert!(plan.validate().is_ok());
}

#[test]
fn test_physical_plan_from_logical() {
    let mut plan = LogicalPlan::new();
    let read_id =
        plan.add_node(LogicalOp::ReadText { path: "test.txt".to_string() }, vec![]);
    let map_id = plan.add_node(
        LogicalOp::Map { func_name: "identity".to_string() },
        vec![read_id],
    );
    let _ = plan.add_node(
        LogicalOp::ReduceByKey { func_name: "sum".to_string(), partitions: 2 },
        vec![map_id],
    );

    let physical = PhysicalPlan::from_logical(&plan).unwrap();
    // Should have: [map stage] + [shuffle stage] + [reduce stage]
    assert!(physical.stages.len() >= 3);
}

#[test]
fn test_shuffle_buffer() {
    use mapreduce_engine::shuffle::{group_by_key, ShuffleBuffer};
    let buf = ShuffleBuffer::new(4);
    buf.insert("apple".to_string(), "1".to_string());
    buf.insert("apple".to_string(), "1".to_string());
    buf.insert("banana".to_string(), "1".to_string());
    assert_eq!(buf.total_records(), 3);

    let mut all: Vec<(String, String)> = Vec::new();
    for i in 0..4 {
        all.extend(buf.drain_partition(i));
    }
    let grouped = group_by_key(all);
    let map: std::collections::HashMap<_, _> = grouped.into_iter().collect();
    assert_eq!(map["apple"].len(), 2);
    assert_eq!(map["banana"].len(), 1);
}
