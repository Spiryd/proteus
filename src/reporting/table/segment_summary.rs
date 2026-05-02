#![allow(dead_code)]
use std::fmt::Write as _;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SegmentSummaryRow {
    pub segment_id: usize,
    pub duration_steps: usize,
    pub is_detected: bool,
    pub mean_shift: f64,
}

pub struct SegmentSummaryTableBuilder {
    rows: Vec<SegmentSummaryRow>,
}

impl SegmentSummaryTableBuilder {
    pub fn new() -> Self {
        Self { rows: Vec::new() }
    }

    pub fn add_row(&mut self, row: SegmentSummaryRow) {
        self.rows.push(row);
    }

    pub fn to_csv(&self) -> String {
        let mut out = String::new();
        out.push_str("segment_id,duration_steps,is_detected,mean_shift\n");

        for row in &self.rows {
            let _ = writeln!(
                out,
                "{},{},{},{}",
                row.segment_id, row.duration_steps, row.is_detected, row.mean_shift
            );
        }

        out
    }
}

impl Default for SegmentSummaryTableBuilder {
    fn default() -> Self {
        Self::new()
    }
}
