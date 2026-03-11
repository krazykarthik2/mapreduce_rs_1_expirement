# Rust Distributed Data Processing Engine — Task List

## Milestone 1 — Core Engine
- [ ] Implement parallel MapReduce execution
- [ ] Implement basic shuffle system
- [ ] Implement hardcoded wordcount job

## Milestone 2 — Planner Layer
- [ ] Design and implement logical ops enum (ReadText, Map, Filter, FlatMap, ReduceByKey, GroupByKey)
- [ ] Implement stage builder (stage boundary detection)
- [ ] Support native Rust job construction via job registry

## Milestone 3 — CLI Jobs
- [ ] Implement JSON job spec parser
- [ ] Write output partitions to disk
- [ ] Add structured logging and progress display

## Milestone 4 — Python UX
- [ ] Python Dataset wrapper (PyO3)
- [ ] Lazy chaining API (.map(), .filter(), .flatmap(), .reduce_by_key())
- [ ] .collect() trigger

## Core Components
- [ ] Logical Plan Layer (DAG, immutable, validation)
- [ ] Physical Planner (stage boundaries, shuffle insertion, partition decisions)
- [ ] Execution Engine (rayon, worker pool, streaming)
- [ ] Shuffle System (hash partitioning, key grouping, memory buffering)
- [ ] Job Registry (symbolic op → rust function)
- [ ] Python Bindings (PyO3)
- [ ] CLI Interface

## Observability
- [ ] Stage start/end logs
- [ ] Processed record counters
- [ ] Partition size metrics
