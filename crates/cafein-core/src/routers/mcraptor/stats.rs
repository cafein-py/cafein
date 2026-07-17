//! Optional search statistics, reported under `CAFEIN_MCRAPTOR_PROF`.

pub(super) const NONE_U32: u32 = u32::MAX;

/// Per-search attribution counters and round-level phase timers for
/// the McRAPTOR port programme's R0 stage: plain query-local integers
/// (no shared atomics; parallel origins each fill their own and the
/// caller reduces them afterwards). Per-operation fields fill only in
/// single-thread diagnostic runs; the report prints only under
/// `CAFEIN_MCRAPTOR_PROF`.
#[derive(Debug, Default, Clone)]
pub(crate) struct McRaptorStats {
    pub access_ns: u64,
    pub queue_collect_ns: u64,
    pub route_scan_ns: u64,
    pub footpath_ns: u64,
    pub sink_ns: u64,

    pub departure_passes: u64,
    pub rounds_entered: u64,
    pub rounds_ended_empty: u64,
    pub max_round_reached: u32,

    pub access_offers: u64,
    pub access_admissions: u64,
    pub queue_labels: u64,
    pub queue_entries: u64,
    pub lines_touched: u64,

    pub line_positions: u64,
    pub rider_evaluations: u64,
    pub route_bag_calls: u64,
    pub route_bag_admissions: u64,
    pub rode_labels: u64,

    pub footpath_edge_records_loaded: u64,
    pub footpath_label_edge_relaxations: u64,
    pub footpath_cutoff_pruned: u64,
    pub footpath_target_pruned: u64,
    pub footpath_bag_calls: u64,
    pub footpath_bag_rejections: u64,
    pub footpath_target_admissions: u64,

    pub footpath_reject_entries_examined: u64,
    pub footpath_admit_entries_examined: u64,
    pub footpath_retain_entries_examined: u64,
    pub footpath_reject_depth_histogram: [u64; 9],
    pub footpath_bag_length_histogram: [u64; 9],

    pub rode_represented: u64,
    pub rode_stale: u64,
    /// Production batch-gate traffic (strict searches).
    pub batch_points_offered: u64,
    pub batch_points_rejected: u64,
    pub batch_points_evicted: u64,
    pub batch_points_live: u64,
    pub batch_source_stops: u64,

    pub labels_created: u64,
    pub total_bag_entries_at_round_end: u64,
    pub maximum_stop_bag_length: u64,
}

impl McRaptorStats {
    pub(crate) fn absorb(&mut self, other: &McRaptorStats) {
        self.access_ns += other.access_ns;
        self.queue_collect_ns += other.queue_collect_ns;
        self.route_scan_ns += other.route_scan_ns;
        self.footpath_ns += other.footpath_ns;
        self.sink_ns += other.sink_ns;
        self.departure_passes += other.departure_passes;
        self.rounds_entered += other.rounds_entered;
        self.rounds_ended_empty += other.rounds_ended_empty;
        self.max_round_reached = self.max_round_reached.max(other.max_round_reached);
        self.access_offers += other.access_offers;
        self.access_admissions += other.access_admissions;
        self.queue_labels += other.queue_labels;
        self.queue_entries += other.queue_entries;
        self.lines_touched += other.lines_touched;
        self.line_positions += other.line_positions;
        self.rider_evaluations += other.rider_evaluations;
        self.route_bag_calls += other.route_bag_calls;
        self.route_bag_admissions += other.route_bag_admissions;
        self.rode_labels += other.rode_labels;
        self.footpath_edge_records_loaded += other.footpath_edge_records_loaded;
        self.footpath_label_edge_relaxations += other.footpath_label_edge_relaxations;
        self.footpath_cutoff_pruned += other.footpath_cutoff_pruned;
        self.footpath_target_pruned += other.footpath_target_pruned;
        self.footpath_bag_calls += other.footpath_bag_calls;
        self.footpath_bag_rejections += other.footpath_bag_rejections;
        self.footpath_target_admissions += other.footpath_target_admissions;
        self.footpath_reject_entries_examined += other.footpath_reject_entries_examined;
        self.footpath_admit_entries_examined += other.footpath_admit_entries_examined;
        self.footpath_retain_entries_examined += other.footpath_retain_entries_examined;
        for (mine, theirs) in self
            .footpath_reject_depth_histogram
            .iter_mut()
            .zip(other.footpath_reject_depth_histogram)
        {
            *mine += theirs;
        }
        for (mine, theirs) in self
            .footpath_bag_length_histogram
            .iter_mut()
            .zip(other.footpath_bag_length_histogram)
        {
            *mine += theirs;
        }
        self.rode_represented += other.rode_represented;
        self.rode_stale += other.rode_stale;
        self.batch_points_offered += other.batch_points_offered;
        self.batch_points_rejected += other.batch_points_rejected;
        self.batch_points_evicted += other.batch_points_evicted;
        self.batch_points_live += other.batch_points_live;
        self.batch_source_stops += other.batch_source_stops;
        self.labels_created += other.labels_created;
        self.total_bag_entries_at_round_end += other.total_bag_entries_at_round_end;
        self.maximum_stop_bag_length = self
            .maximum_stop_bag_length
            .max(other.maximum_stop_bag_length);
    }

    pub(crate) fn report(&self, label: &str) {
        if std::env::var_os("CAFEIN_MCRAPTOR_PROF").is_none() {
            return;
        }
        let histogram = |counts: &[u64; 9]| {
            counts
                .iter()
                .map(|count| count.to_string())
                .collect::<Vec<_>>()
                .join(",")
        };
        eprintln!(
            "MCRAPTOR-STATS {label} access_ms={} queue_collect_ms={} \
             route_scan_ms={} footpath_ms={} sink_ms={} \
             departure_passes={} rounds_entered={} rounds_ended_empty={} \
             max_round_reached={} access_offers={} access_admissions={} \
             queue_labels={} queue_entries={} lines_touched={} \
             line_positions={} rider_evaluations={} route_bag_calls={} \
             route_bag_admissions={} rode_labels={} \
             footpath_edge_records_loaded={} \
             footpath_label_edge_relaxations={} footpath_cutoff_pruned={} \
             footpath_target_pruned={} footpath_bag_calls={} \
             footpath_bag_rejections={} footpath_target_admissions={} \
             footpath_reject_entries_examined={} \
             footpath_admit_entries_examined={} \
             footpath_retain_entries_examined={} \
             footpath_reject_depth_histogram={} \
             footpath_bag_length_histogram={} rode_represented={} \
             rode_stale={} batch_points_offered={} batch_points_rejected={} \
             batch_points_evicted={} batch_points_live={} \
             batch_source_stops={} labels_created={} \
             total_bag_entries_at_round_end={} maximum_stop_bag_length={}",
            self.access_ns / 1_000_000,
            self.queue_collect_ns / 1_000_000,
            self.route_scan_ns / 1_000_000,
            self.footpath_ns / 1_000_000,
            self.sink_ns / 1_000_000,
            self.departure_passes,
            self.rounds_entered,
            self.rounds_ended_empty,
            self.max_round_reached,
            self.access_offers,
            self.access_admissions,
            self.queue_labels,
            self.queue_entries,
            self.lines_touched,
            self.line_positions,
            self.rider_evaluations,
            self.route_bag_calls,
            self.route_bag_admissions,
            self.rode_labels,
            self.footpath_edge_records_loaded,
            self.footpath_label_edge_relaxations,
            self.footpath_cutoff_pruned,
            self.footpath_target_pruned,
            self.footpath_bag_calls,
            self.footpath_bag_rejections,
            self.footpath_target_admissions,
            self.footpath_reject_entries_examined,
            self.footpath_admit_entries_examined,
            self.footpath_retain_entries_examined,
            histogram(&self.footpath_reject_depth_histogram),
            histogram(&self.footpath_bag_length_histogram),
            self.rode_represented,
            self.rode_stale,
            self.batch_points_offered,
            self.batch_points_rejected,
            self.batch_points_evicted,
            self.batch_points_live,
            self.batch_source_stops,
            self.labels_created,
            self.total_bag_entries_at_round_end,
            self.maximum_stop_bag_length,
        );
    }
}
