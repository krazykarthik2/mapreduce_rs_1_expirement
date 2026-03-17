# mapreduce_rs_1_expirement

# Rust MapReduce Engine (Experimental)

A lightweight **MapReduce / DAG execution engine written in Rust** featuring:

- Lazy logical planning (DAG based)
- Physical execution planning
- Partitioned shuffle + reduce
- Parallel execution engine
- CLI job runner (JSON spec driven)
- Built-in WordCount example
- Python bindings via PyO3 (Spark-like Dataset API)

This project is designed as a **learning + research implementation** and a future base for **distributed MapReduce systems**.

---

# Project Structure
.
├── engine/ # Core MapReduce execution engine
├── cli/ # Command line job runner (package = mapreduce_cli)
├── pyapi/ # Python bindings
├── todo.md # Roadmap / ideas


---

# Architecture — How It Works

The engine follows a **4-stage execution pipeline**.

---

## 1. Logical Planning (Lazy DAG)

Each transformation builds a node in a DAG.

Supported logical operators:

- ReadText
- Map
- Filter
- FlatMap
- ReduceByKey
- GroupByKey
- Collect

No execution happens here.  
Only a **computation graph is built lazily.**

--- 

## 2. Physical Planning

Logical DAG → Optimized Physical Plan.

Planner decides:

- Partition count
- Shuffle boundaries
- Execution stages
- Pipeline fusion opportunities

---

## 3. Execution Engine

Execution characteristics:

- Partition-parallel processing
- Chunk-based streaming
- Shuffle + merge reduce
- In-memory record pipelines

Execution is controlled via:

```rust
ExecutionConfig {
    num_partitions,
    chunk_size,
}
4. Output Writing

Final results are written as partitioned output:

output/
 ├── part-0.txt
 ├── part-1.txt
 ├── part-2.txt
Prerequisites

Rust (latest stable)

Cargo

Python 3.8+ (for Python bindings)

Install Rust:

curl https://sh.rustup.rs -sSf | sh
Build Project

From project root:

cargo build --release
Running CLI Jobs

The CLI package name inside workspace is:

mapreduce_cli
Run Built-in WordCount
cargo run -p mapreduce_cli -- \
    wordcount \
    --input data.txt \
    --output output \
    --partitions 4
Generate JSON Job Template
cargo run -p mapreduce_cli -- template
Example Job Spec
{
  "name": "wordcount",
  "steps": [
    { "op": "read_text", "path": "data.txt" },
    { "op": "filter", "func_name": "non_empty" },
    { "op": "flatmap", "func_name": "tokenize" },
    { "op": "reduce_by_key", "func_name": "sum", "partitions": 4 },
    { "op": "collect" }
  ]
}
Run Custom Job
cargo run -p mapreduce_cli -- \
    run \
    --spec job.json \
    --output output \
    --partitions 4
Python Bindings

The engine exposes a lazy Dataset API similar to Spark RDD.

Install Python Module

Install maturin:

pip install maturin

Build + install:

cd pyapi
maturin develop --release

Module installed:

mapreduce_py
Python WordCount Example
from mapreduce_py import Dataset

ds = (
    Dataset.read_text("/workspaces/mapreduce_rs_1_expirement/data.txt", partitions=4)
        .filter("non_empty")
        .flatmap("tokenize")
        .reduce_by_key("sum")
)

print(ds.explain())

results = ds.collect()

for r in results:
    print(r.split('\t'))
Write Output
ds.write_output("output")
Count Records
print(ds.count())

Lazy Execution Model

Transformations are not executed immediately.

Execution happens only when an action is called:

collect()

count()

write_output()

Benefits:

DAG optimization

Stage fusion

Reduced memory pressure

Better parallel scheduling potential

Built-in Function Registry

Currently supported:

tokenize

non_empty

sum

Future:

Rust plugin UDFs

Python UDF execution

Distributed function shipping

Testing
cargo test -p engine
Troubleshooting

If CLI package not found:

List workspace packages:

cargo metadata --no-deps --format-version 1 | grep name

Or run CLI directly:

cd cli
cargo run -- wordcount --input ../data.txt --output ../output --partitions 4
Future Goals

Distributed shuffle over TCP

Worker / scheduler model

Fault tolerance

Spill-to-disk execution

Async executor

Streaming MapReduce

SQL layer

Python UDF execution

License

Experimental / Educational.

Author

Karthik Goparaju
Rust Systems • Distributed Computing • Execution Engines