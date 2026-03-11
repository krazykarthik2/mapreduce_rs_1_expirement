use dashmap::DashMap;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// Hash a key to determine which partition it belongs to
pub fn partition_for_key(key: &str, num_partitions: usize) -> usize {
    let mut hasher = DefaultHasher::new();
    key.hash(&mut hasher);
    (hasher.finish() as usize) % num_partitions
}

/// A concurrent shuffle buffer that groups records by partition
pub struct ShuffleBuffer {
    /// partition_id -> list of (key, value) pairs
    pub partitions: DashMap<usize, Vec<(String, String)>>,
    pub num_partitions: usize,
}

impl ShuffleBuffer {
    pub fn new(num_partitions: usize) -> Self {
        let partitions = DashMap::new();
        for i in 0..num_partitions {
            partitions.insert(i, Vec::new());
        }
        Self { partitions, num_partitions }
    }

    pub fn insert(&self, key: String, value: String) {
        let partition = partition_for_key(&key, self.num_partitions);
        self.partitions.entry(partition).or_default().push((key, value));
    }

    pub fn drain_partition(&self, partition_id: usize) -> Vec<(String, String)> {
        if let Some(mut entry) = self.partitions.get_mut(&partition_id) {
            std::mem::take(&mut *entry)
        } else {
            Vec::new()
        }
    }

    pub fn total_records(&self) -> usize {
        self.partitions.iter().map(|p| p.value().len()).sum()
    }
}

/// Group sorted/shuffled records by key
pub fn group_by_key(records: Vec<(String, String)>) -> Vec<(String, Vec<String>)> {
    let mut grouped: std::collections::BTreeMap<String, Vec<String>> =
        std::collections::BTreeMap::new();
    for (k, v) in records {
        grouped.entry(k).or_default().push(v);
    }
    grouped.into_iter().collect()
}
