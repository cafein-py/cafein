//! Optional search statistics, reported under `CAFEIN_MCTBTR_PROF`.

/// Per-search work counters and round-level phase timers, owned by one
/// `passes` call (no shared atomics: parallel origins each fill their
/// own and the caller reduces them afterwards). Increments are plain
/// integer adds so the instrumentation-off cost stays negligible; the
/// report only prints under `CAFEIN_MCTBTR_PROF`.
#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct SearchStats {
    /// Closure edge records read from the CSR (one per edge per sweep).
    pub closure_edge_records_loaded: u64,
    /// Label-against-edge relaxations (per admitted label per edge; in
    /// the label-major loop this equals the records loaded).
    pub closure_label_edge_relaxations: u64,
    /// Direct admissions offered to the closure batch.
    pub closure_points_offered: u64,
    /// Live points compacted for relaxation across all source batches.
    pub closure_points_live: u64,
    /// Points killed by the batch-local gate before relaxation.
    pub closure_points_batch_evicted: u64,
    /// Touched source stops swept (one batch each).
    pub closure_source_batches: u64,
    /// Closure relaxations the target stop bag admitted.
    pub closure_target_admissions: u64,
    /// WalkBoard records the closure sweep queued for boarding.
    pub walk_boards_recorded: u64,
    pub segments_enqueued: u64,
    /// Pending segments cancelled by covering trip-bag admissions.
    pub segments_cancelled_pending: u64,
    /// Queued segments skipped at dequeue because they were cancelled.
    pub segments_skipped_cancelled: u64,
    pub segments_scanned_live: u64,
    pub suffix_context_evaluations: u64,
    /// Closure-path stop-bag calls (diagnostic runs only).
    pub closure_bag_calls: u64,
    /// Calls the stop bag rejected (diagnostic runs only).
    pub closure_bag_rejections: u64,
    /// Entries the rejection scan walked on rejected calls.
    pub closure_bag_reject_entries_examined: u64,
    /// Rejections certified by the front entry (no swap needed).
    pub closure_mtf_front_rejections: u64,
    /// Rejections whose witness was swapped to the front.
    pub closure_mtf_swaps: u64,
    /// Total one-based depth beyond the front across swapped
    /// rejections (equals reject entries examined minus rejections).
    pub closure_mtf_swap_distance: u64,
    /// One-based rejecting depths: 1, 2, 3–4, 5–8, 9–16, 17–32,
    /// 33–64, 65–128, and 129+.
    pub closure_bag_reject_depth_histogram: [u64; 9],
    /// Entries the rejection scan walked on admitted calls.
    pub closure_bag_admit_entries_examined: u64,
    /// Entries the eviction walk examined on admitted calls.
    pub closure_bag_retain_entries_examined: u64,
    /// Pre-call bag lengths: 0, 1, 2, 3–4, 5–8, 9–16, 17–32, 33–128,
    /// and 129+.
    pub closure_bag_length_histogram: [u64; 9],
    pub direct_scan_ns: u64,
    pub closure_ns: u64,
    pub expand_ns: u64,
    pub walk_board_ns: u64,
}

/// Histogram bucket of a pre-call bag length.
pub(super) fn length_bucket(length: u32) -> usize {
    match length {
        0 => 0,
        1 => 1,
        2 => 2,
        3..=4 => 3,
        5..=8 => 4,
        9..=16 => 5,
        17..=32 => 6,
        33..=128 => 7,
        _ => 8,
    }
}

/// Histogram bucket of a one-based rejecting depth.
pub(super) fn depth_bucket(depth: u32) -> usize {
    match depth {
        0..=1 => 0,
        2 => 1,
        3..=4 => 2,
        5..=8 => 3,
        9..=16 => 4,
        17..=32 => 5,
        33..=64 => 6,
        65..=128 => 7,
        _ => 8,
    }
}

impl SearchStats {
    pub(super) fn absorb(&mut self, other: &SearchStats) {
        self.closure_edge_records_loaded += other.closure_edge_records_loaded;
        self.closure_label_edge_relaxations += other.closure_label_edge_relaxations;
        self.closure_points_offered += other.closure_points_offered;
        self.closure_points_live += other.closure_points_live;
        self.closure_points_batch_evicted += other.closure_points_batch_evicted;
        self.closure_source_batches += other.closure_source_batches;
        self.closure_target_admissions += other.closure_target_admissions;
        self.walk_boards_recorded += other.walk_boards_recorded;
        self.segments_enqueued += other.segments_enqueued;
        self.segments_cancelled_pending += other.segments_cancelled_pending;
        self.segments_skipped_cancelled += other.segments_skipped_cancelled;
        self.segments_scanned_live += other.segments_scanned_live;
        self.suffix_context_evaluations += other.suffix_context_evaluations;
        self.closure_bag_calls += other.closure_bag_calls;
        self.closure_bag_rejections += other.closure_bag_rejections;
        self.closure_bag_reject_entries_examined += other.closure_bag_reject_entries_examined;
        self.closure_mtf_front_rejections += other.closure_mtf_front_rejections;
        self.closure_mtf_swaps += other.closure_mtf_swaps;
        self.closure_mtf_swap_distance += other.closure_mtf_swap_distance;
        for (mine, theirs) in self
            .closure_bag_reject_depth_histogram
            .iter_mut()
            .zip(other.closure_bag_reject_depth_histogram)
        {
            *mine += theirs;
        }
        self.closure_bag_admit_entries_examined += other.closure_bag_admit_entries_examined;
        self.closure_bag_retain_entries_examined += other.closure_bag_retain_entries_examined;
        for (mine, theirs) in self
            .closure_bag_length_histogram
            .iter_mut()
            .zip(other.closure_bag_length_histogram)
        {
            *mine += theirs;
        }
        self.direct_scan_ns += other.direct_scan_ns;
        self.closure_ns += other.closure_ns;
        self.expand_ns += other.expand_ns;
        self.walk_board_ns += other.walk_board_ns;
    }

    pub(super) fn report(&self, label: &str) {
        if std::env::var("CAFEIN_MCTBTR_PROF").is_err() {
            return;
        }
        eprintln!(
            "MCTBTR-STATS {label} closure_edge_records_loaded={} \
             closure_label_edge_relaxations={} closure_points_offered={} \
             closure_points_live={} closure_points_batch_evicted={} \
             closure_source_batches={} closure_target_admissions={} \
             walk_boards_recorded={} segments_enqueued={} \
             segments_cancelled_pending={} segments_skipped_cancelled={} \
             segments_scanned_live={} \
             suffix_context_evaluations={} closure_bag_calls={} \
             closure_bag_rejections={} closure_bag_reject_entries_examined={} \
             closure_mtf_front_rejections={} closure_mtf_swaps={} \
             closure_mtf_swap_distance={} closure_bag_reject_depth_histogram={} \
             closure_bag_admit_entries_examined={} \
             closure_bag_retain_entries_examined={} \
             closure_bag_length_histogram={} direct_scan_ms={} closure_ms={} \
             expand_ms={} walk_board_ms={}",
            self.closure_edge_records_loaded,
            self.closure_label_edge_relaxations,
            self.closure_points_offered,
            self.closure_points_live,
            self.closure_points_batch_evicted,
            self.closure_source_batches,
            self.closure_target_admissions,
            self.walk_boards_recorded,
            self.segments_enqueued,
            self.segments_cancelled_pending,
            self.segments_skipped_cancelled,
            self.segments_scanned_live,
            self.suffix_context_evaluations,
            self.closure_bag_calls,
            self.closure_bag_rejections,
            self.closure_bag_reject_entries_examined,
            self.closure_mtf_front_rejections,
            self.closure_mtf_swaps,
            self.closure_mtf_swap_distance,
            self.closure_bag_reject_depth_histogram
                .iter()
                .map(|count| count.to_string())
                .collect::<Vec<_>>()
                .join(","),
            self.closure_bag_admit_entries_examined,
            self.closure_bag_retain_entries_examined,
            self.closure_bag_length_histogram
                .iter()
                .map(|count| count.to_string())
                .collect::<Vec<_>>()
                .join(","),
            self.direct_scan_ns / 1_000_000,
            self.closure_ns / 1_000_000,
            self.expand_ns / 1_000_000,
            self.walk_board_ns / 1_000_000,
        );
    }
}
