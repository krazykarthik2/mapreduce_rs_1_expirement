use std::collections::HashMap;
use std::sync::Arc;

/// A record in the engine — either a raw text line or a key-value pair
#[derive(Debug, Clone)]
pub enum Record {
    Text(String),
    KeyValue(String, String),
    KeyValues(String, Vec<String>),
}

/// Function type for Map: takes a Record, returns Vec<Record>
pub type MapFn = Arc<dyn Fn(Record) -> Vec<Record> + Send + Sync>;

/// Function type for Filter: takes a Record, returns bool
pub type FilterFn = Arc<dyn Fn(&Record) -> bool + Send + Sync>;

/// Function type for Reduce: takes (key, values), returns (key, value)
pub type ReduceFn = Arc<dyn Fn(String, Vec<String>) -> (String, String) + Send + Sync>;

/// Registry of named functions for job construction
pub struct JobRegistry {
    map_fns: HashMap<String, MapFn>,
    filter_fns: HashMap<String, FilterFn>,
    reduce_fns: HashMap<String, ReduceFn>,
}

impl JobRegistry {
    pub fn new() -> Self {
        let mut registry = Self {
            map_fns: HashMap::new(),
            filter_fns: HashMap::new(),
            reduce_fns: HashMap::new(),
        };
        registry.register_builtins();
        registry
    }

    fn register_builtins(&mut self) {
        // Wordcount tokenizer: splits a line into (word, "1") pairs
        self.map_fns.insert(
            "tokenize".to_string(),
            Arc::new(|record| match record {
                Record::Text(line) => line
                    .split_whitespace()
                    .map(|w| Record::KeyValue(w.to_lowercase(), "1".to_string()))
                    .collect(),
                other => vec![other],
            }),
        );

        // Identity map
        self.map_fns.insert("identity".to_string(), Arc::new(|record| vec![record]));

        // Count reducer: sums string-encoded integers
        self.reduce_fns.insert(
            "sum".to_string(),
            Arc::new(|key, values| {
                let total: i64 =
                    values.iter().filter_map(|v| v.parse::<i64>().ok()).sum();
                (key, total.to_string())
            }),
        );

        // Concat reducer: joins values with comma
        self.reduce_fns.insert(
            "concat".to_string(),
            Arc::new(|key, values| (key, values.join(","))),
        );

        // Filter: non-empty lines
        self.filter_fns.insert(
            "non_empty".to_string(),
            Arc::new(|record| match record {
                Record::Text(s) => !s.trim().is_empty(),
                _ => true,
            }),
        );
    }

    pub fn register_map(&mut self, name: &str, f: MapFn) {
        self.map_fns.insert(name.to_string(), f);
    }

    pub fn register_filter(&mut self, name: &str, f: FilterFn) {
        self.filter_fns.insert(name.to_string(), f);
    }

    pub fn register_reduce(&mut self, name: &str, f: ReduceFn) {
        self.reduce_fns.insert(name.to_string(), f);
    }

    pub fn get_map(&self, name: &str) -> Option<&MapFn> {
        self.map_fns.get(name)
    }

    pub fn get_filter(&self, name: &str) -> Option<&FilterFn> {
        self.filter_fns.get(name)
    }

    pub fn get_reduce(&self, name: &str) -> Option<&ReduceFn> {
        self.reduce_fns.get(name)
    }
}

impl Default for JobRegistry {
    fn default() -> Self {
        Self::new()
    }
}
