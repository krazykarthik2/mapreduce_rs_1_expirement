# Rust Distributed Data Processing Engine — Task List

## Milestone 1 — Core Engine
- [x] Implement parallel MapReduce execution
- [x] Implement basic shuffle system
- [x] Implement hardcoded wordcount job

## Milestone 2 — Planner Layer
- [x] Design and implement logical ops enum (ReadText, Map, Filter, FlatMap, ReduceByKey, GroupByKey)
- [x] Implement stage builder (stage boundary detection)
- [x] Support native Rust job construction via job registry

## Milestone 3 — CLI Jobs
- [x] Implement JSON job spec parser
- [x] Write output partitions to disk
- [x] Add structured logging and progress display

## Milestone 4 — Python UX
- [x] Python Dataset wrapper (PyO3)
- [x] Lazy chaining API (.map(), .filter(), .flatmap(), .reduce_by_key())
- [x] .collect() trigger

## Core Components
- [x] Logical Plan Layer (DAG, immutable, validation)
- [x] Physical Planner (stage boundaries, shuffle insertion, partition decisions)
- [x] Execution Engine (rayon, worker pool, streaming)
- [x] Shuffle System (hash partitioning, key grouping, memory buffering)
- [x] Job Registry (symbolic op → rust function)
- [x] Python Bindings (PyO3)
- [x] CLI Interface

## Observability
- [x] Stage start/end logs
- [x] Processed record counters
- [x] Partition size metrics
