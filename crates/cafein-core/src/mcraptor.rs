//! McRAPTOR: multicriteria RAPTOR over (arrival, emissions).
//!
//! Round-based label bags per stop over (arrival, grams CO₂e). Arrivals
//! compare exactly; grams compare at a configurable bucket width, so
//! labels within one bucket count as equal and bag sizes — and the
//! search — stay bounded. Each bag insertion may substitute a
//! same-bucket representative arriving no later, so a reported
//! journey's emissions sit within one bucket of a true frontier value
//! per insertion its labels survived — a worst case of
//! `(2 × rides + 1) × bucket`, in practice well under one bucket. A
//! vanishing bucket (one microgram, matching the label quantization)
//! reproduces the exhaustive oracle's exact frontier.
//!
//! The emissions firewall holds: a label's grams update is one
//! cumulative-distance subtraction per alight, nothing per-leg beyond
//! that enters the search. Trips without a resolved emission factor are
//! skipped — journeys riding them can never sit on an emissions
//! frontier. Boarding considers, besides the earliest boardable trip,
//! the later trips of the line whose factor strictly improves on every
//! earlier boardable one: waiting for a cleaner vehicle can hold a true
//! Pareto point. On lines with uniform factors — the common case — this
//! collapses to the classic earliest-trip rule.

use rayon::prelude::*;

use crate::exhaustive::quantized;
use crate::fares::FareLeg;
use crate::geometry::{wkb_multi_line_string, TripGeometry};
use crate::journey::{Journey, Leg};
use crate::raptor::{departure_candidates, CostInputs, CostRow};
use crate::router::Request;
use crate::tbtr::{earliest_boardable, DayView, ViewTrip};
use crate::timetable::{StopIdx, Timetable, TripIdx};
use crate::transfers::Transfers;

/// How a label reached its stop; parents index the label arena.
#[derive(Debug, Clone, Copy)]
enum Origin {
    Access,
    Ride {
        parent: u32,
        trip: ViewTrip,
        board: u16,
        alight: u16,
    },
    Walk {
        parent: u32,
        duration: u32,
    },
}

#[derive(Debug, Clone, Copy)]
struct Label {
    arrival: u32,
    grams: f64,
    stop: StopIdx,
    /// The profile pass that grew this label's chain: label chains
    /// never cross passes, so travel time is `arrival - departure`.
    departure: u32,
    /// Route-penalty seconds accumulated along the chain (soft-penalty
    /// diverse); the bags dominate on `arrival + penalty` while the
    /// reconstructed journey keeps the true `arrival`. Zero without
    /// penalties.
    penalty: u32,
    origin: Origin,
}

/// One bag entry; `arrival` is the true arrival, `penalty` the
/// accumulated route penalty (0 without one), `key` the grams bucket,
/// `grams` the exact (microgram-quantized) value behind it, `rides` the
/// transit legs the label used to reach the stop.
#[derive(Debug, Clone, Copy)]
struct Entry {
    arrival: u32,
    penalty: u32,
    key: i64,
    grams: f64,
    rides: u8,
}

/// A per-stop label bag under bucketed dominance, cumulative across
/// rounds and profile passes. Shared with the trip-based engine, whose
/// stop bags follow the same contract.
#[derive(Debug, Clone, Default)]
pub(crate) struct Bag {
    entries: Vec<Entry>,
}

/// The strict (`penalty = 0`, `slack = 0`) rejection relation: does
/// `entry` reject the candidate? Equivalent to the rejection scan of
/// `insert_slack(arrival, 0, grams, key, rides, 0)`, including the
/// entry-penalty reads.
fn rejects_strict(entry: &Entry, arrival: u32, grams: f64, key: i64, rides: u8) -> bool {
    if entry.key <= key && entry.rides <= rides && entry.arrival <= arrival {
        if entry.arrival == arrival
            && entry.penalty == 0
            && entry.key == key
            && entry.rides == rides
        {
            grams >= entry.grams
        } else {
            entry.arrival.saturating_add(entry.penalty) <= arrival
        }
    } else {
        false
    }
}

impl Bag {
    /// Inserts unless an entry arriving no later, in the same or a
    /// cleaner bucket, AND on no more rides already dominates it; evicts
    /// what the newcomer covers. The `rides` axis is what makes the
    /// cumulative-across-passes bag sound under the second criterion: a
    /// later-departure label may only suppress an earlier-departure one
    /// when it also used no more transit legs, so it keeps at least the
    /// onward-transfer budget to reproduce every continuation. Dropping
    /// it lets a later-but-more-transferred journey wrongly evict a
    /// cleaner earlier one that still had transfers to spend. An entry
    /// equal on arrival, bucket and rides but strictly dirtier in exact
    /// grams is refined (replaced), keeping the bucket's representative
    /// as clean as the search has seen. The trip-based engine ranks the
    /// transit round in `rides` for its direct and closure arrivals.
    ///
    /// The strict path self-organises the entry vector: a rejection
    /// swaps its witness to slot 0 and an admission swaps the new entry
    /// there, so the workload's recent certificates reject in one
    /// probe. Entry order has no semantic consumer — rejection is an
    /// existential query, eviction is set-based, and a cleaner exact
    /// tie continues the scan rather than admitting early — so only the
    /// private vector permutation differs from a stable-order bag.
    pub(crate) fn insert(&mut self, arrival: u32, grams: f64, key: i64, rides: u8) -> bool {
        for index in 0..self.entries.len() {
            if rejects_strict(&self.entries[index], arrival, grams, key, rides) {
                if index != 0 {
                    self.entries.swap(0, index);
                }
                return false;
            }
        }
        self.entries.retain(|entry| {
            !((key <= entry.key
                && rides <= entry.rides
                && arrival <= entry.arrival
                && arrival <= entry.arrival.saturating_add(entry.penalty))
                || (entry.arrival == arrival
                    && entry.penalty == 0
                    && entry.key == key
                    && entry.rides == rides
                    && grams < entry.grams))
        });
        self.entries.push(Entry {
            arrival,
            penalty: 0,
            key,
            grams,
            rides,
        });
        let last = self.entries.len() - 1;
        if last != 0 {
            self.entries.swap(0, last);
        }
        true
    }

    /// `insert` under a route penalty and a time slack. Dominance runs on
    /// two time axes: an entry may reject the newcomer only when it reaches
    /// the stop no later in **true arrival** — so it catches every onward
    /// connection the newcomer could — and is at least `slack` seconds
    /// earlier on the **effective arrival** (`arrival + penalty`), no
    /// dirtier and on no more rides. A penalized label arriving physically
    /// earlier is therefore never suppressed by an unpenalized one arriving
    /// later, even though its effective arrival is worse. Same-class
    /// (`arrival`, `penalty`, `key`, `rides`) duplicates reduce to the
    /// cleanest representative, and eviction likewise needs the full `slack`
    /// margin. Without penalties effective equals true, so this is exactly
    /// the single-axis `(arrival, key, rides)` dominance; `slack = 0` is
    /// strict `insert`, the only form the trip-based and exhaustive engines
    /// call.
    pub(crate) fn insert_slack(
        &mut self,
        arrival: u32,
        penalty: u32,
        grams: f64,
        key: i64,
        rides: u8,
        slack: u32,
    ) -> bool {
        let effective = arrival.saturating_add(penalty);
        for entry in &self.entries {
            if entry.key <= key && entry.rides <= rides && entry.arrival <= arrival {
                let entry_effective = entry.arrival.saturating_add(entry.penalty);
                if entry.arrival == arrival
                    && entry.penalty == penalty
                    && entry.key == key
                    && entry.rides == rides
                {
                    if grams >= entry.grams {
                        return false;
                    }
                } else if entry_effective.saturating_add(slack) <= effective {
                    return false;
                }
            }
        }
        self.entries.retain(|entry| {
            let entry_effective = entry.arrival.saturating_add(entry.penalty);
            !((key <= entry.key
                && rides <= entry.rides
                && arrival <= entry.arrival
                && effective.saturating_add(slack) <= entry_effective)
                || (entry.arrival == arrival
                    && entry.penalty == penalty
                    && entry.key == key
                    && entry.rides == rides
                    && grams < entry.grams))
        });
        self.entries.push(Entry {
            arrival,
            penalty,
            key,
            grams,
            rides,
        });
        true
    }

    /// Strict `insert` with probe accounting for the trip-based
    /// engine's closure diagnostics: identical decisions and identical
    /// self-organising swaps, recording the bag length before the
    /// call, the one-based rejecting depth (or the complete pre-call
    /// length on admission), and the eviction-walk depth.
    pub(crate) fn insert_probed(
        &mut self,
        arrival: u32,
        grams: f64,
        key: i64,
        rides: u8,
        probes: &mut InsertProbes,
    ) -> bool {
        probes.length = self.entries.len() as u32;
        probes.examined = 0;
        probes.retained = 0;
        for index in 0..self.entries.len() {
            probes.examined += 1;
            if rejects_strict(&self.entries[index], arrival, grams, key, rides) {
                if index != 0 {
                    self.entries.swap(0, index);
                }
                return false;
            }
        }
        let retained = &mut probes.retained;
        self.entries.retain(|entry| {
            *retained += 1;
            !((key <= entry.key
                && rides <= entry.rides
                && arrival <= entry.arrival
                && arrival <= entry.arrival.saturating_add(entry.penalty))
                || (entry.arrival == arrival
                    && entry.penalty == 0
                    && entry.key == key
                    && entry.rides == rides
                    && grams < entry.grams))
        });
        self.entries.push(Entry {
            arrival,
            penalty: 0,
            key,
            grams,
            rides,
        });
        let last = self.entries.len() - 1;
        if last != 0 {
            self.entries.swap(0, last);
        }
        true
    }

    /// A bag with a prescribed entry order, for order-sensitivity
    /// tests (reachable strict bags are antichains; tests may build
    /// unreachable orders deliberately).
    #[cfg(test)]
    pub(crate) fn from_entries(entries: Vec<(u32, u32, i64, f64, u8)>) -> Bag {
        Bag {
            entries: entries
                .into_iter()
                .map(|(arrival, penalty, key, grams, rides)| Entry {
                    arrival,
                    penalty,
                    key,
                    grams,
                    rides,
                })
                .collect(),
        }
    }

    /// Stable-order `insert_slack` with probe accounting for the
    /// R0 attribution runs: identical decisions and identical entry
    /// order to `insert_slack`, recording the bag length before the
    /// call, the one-based depth of the rejection scan (or the
    /// complete pre-call length on admission), and the eviction-walk
    /// depth.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn insert_slack_probed(
        &mut self,
        arrival: u32,
        penalty: u32,
        grams: f64,
        key: i64,
        rides: u8,
        slack: u32,
        probes: &mut InsertProbes,
    ) -> bool {
        probes.length = self.entries.len() as u32;
        probes.examined = 0;
        probes.retained = 0;
        let effective = arrival.saturating_add(penalty);
        for entry in &self.entries {
            probes.examined += 1;
            if entry.key <= key && entry.rides <= rides && entry.arrival <= arrival {
                let entry_effective = entry.arrival.saturating_add(entry.penalty);
                if entry.arrival == arrival
                    && entry.penalty == penalty
                    && entry.key == key
                    && entry.rides == rides
                {
                    if grams >= entry.grams {
                        return false;
                    }
                } else if entry_effective.saturating_add(slack) <= effective {
                    return false;
                }
            }
        }
        let retained = &mut probes.retained;
        self.entries.retain(|entry| {
            *retained += 1;
            let entry_effective = entry.arrival.saturating_add(entry.penalty);
            !((key <= entry.key
                && rides <= entry.rides
                && arrival <= entry.arrival
                && effective.saturating_add(slack) <= entry_effective)
                || (entry.arrival == arrival
                    && entry.penalty == penalty
                    && entry.key == key
                    && entry.rides == rides
                    && grams < entry.grams))
        });
        self.entries.push(Entry {
            arrival,
            penalty,
            key,
            grams,
            rides,
        });
        true
    }

    /// The R1 dispatch: a strict search (zero slack, no route
    /// penalties) inserts through the self-organising strict path —
    /// identical decisions, recent witnesses swapped to the front —
    /// while slack and penalty searches keep the stable-order general
    /// path byte-for-byte. Set evolution is order-independent in both
    /// relations, so which path ran is invisible in results.
    #[allow(clippy::too_many_arguments)]
    fn insert_label(
        &mut self,
        strict: bool,
        arrival: u32,
        penalty: u32,
        grams: f64,
        key: i64,
        rides: u8,
        slack: u32,
    ) -> bool {
        if strict {
            debug_assert!(penalty == 0, "a strict search carries no penalty");
            self.insert(arrival, grams, key, rides)
        } else {
            self.insert_slack(arrival, penalty, grams, key, rides, slack)
        }
    }

    /// The probed twin of `insert_label`: identical decisions and
    /// identical permutations per mode, with probe accounting.
    #[allow(clippy::too_many_arguments)]
    fn insert_label_probed(
        &mut self,
        strict: bool,
        arrival: u32,
        penalty: u32,
        grams: f64,
        key: i64,
        rides: u8,
        slack: u32,
        probes: &mut InsertProbes,
    ) -> bool {
        if strict {
            debug_assert!(penalty == 0, "a strict search carries no penalty");
            self.insert_probed(arrival, grams, key, rides, probes)
        } else {
            self.insert_slack_probed(arrival, penalty, grams, key, rides, slack, probes)
        }
    }

    /// The number of retained entries, for diagnostic bag censuses.
    fn entry_count(&self) -> usize {
        self.entries.len()
    }

    /// Whether the exact label tuple is still retained — the stale-
    /// `rode` audit's membership test.
    fn contains_exact(&self, arrival: u32, penalty: u32, key: i64, grams: f64, rides: u8) -> bool {
        self.entries.iter().any(|entry| {
            entry.arrival == arrival
                && entry.penalty == penalty
                && entry.key == key
                && entry.grams == grams
                && entry.rides == rides
        })
    }

    /// The bag's entries as comparable tuples, for differential tests.
    #[cfg(test)]
    pub(crate) fn snapshot(&self) -> Vec<(u32, u32, i64, u64, u8)> {
        self.entries
            .iter()
            .map(|entry| {
                (
                    entry.arrival,
                    entry.penalty,
                    entry.key,
                    entry.grams.to_bits(),
                    entry.rides,
                )
            })
            .collect()
    }
}

/// Per-call probe depths of one strict `insert_probed`.
#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct InsertProbes {
    /// Entries walked by the rejection scan before returning.
    pub examined: u32,
    /// Entries the eviction walk examined (admissions only).
    pub retained: u32,
    /// Bag length before the call.
    pub length: u32,
}

/// Histogram bucket of a pre-call bag length.
fn length_bucket(length: u32) -> usize {
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
fn depth_bucket(depth: u32) -> usize {
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

const NONE_U32: u32 = u32::MAX;

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
    pub footpath_edge_prepass_ns: u64,

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

    pub access_floor_rejectable_route: u64,
    pub access_floor_rejectable_footpath: u64,
    pub access_floor_nonfront_saved_probes: u64,
}

impl McRaptorStats {
    pub(crate) fn absorb(&mut self, other: &McRaptorStats) {
        self.access_ns += other.access_ns;
        self.queue_collect_ns += other.queue_collect_ns;
        self.route_scan_ns += other.route_scan_ns;
        self.footpath_ns += other.footpath_ns;
        self.sink_ns += other.sink_ns;
        self.footpath_edge_prepass_ns += other.footpath_edge_prepass_ns;
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
        self.access_floor_rejectable_route += other.access_floor_rejectable_route;
        self.access_floor_rejectable_footpath += other.access_floor_rejectable_footpath;
        self.access_floor_nonfront_saved_probes += other.access_floor_nonfront_saved_probes;
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
             route_scan_ms={} footpath_ms={} sink_ms={} edge_prepass_ms={} \
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
             total_bag_entries_at_round_end={} maximum_stop_bag_length={} \
             access_floor_rejectable_route={} \
             access_floor_rejectable_footpath={} \
             access_floor_nonfront_saved_probes={}",
            self.access_ns / 1_000_000,
            self.queue_collect_ns / 1_000_000,
            self.route_scan_ns / 1_000_000,
            self.footpath_ns / 1_000_000,
            self.sink_ns / 1_000_000,
            self.footpath_edge_prepass_ns / 1_000_000,
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
            self.access_floor_rejectable_route,
            self.access_floor_rejectable_footpath,
            self.access_floor_nonfront_saved_probes,
        );
    }
}

/// One pending footpath source point: strict (arrival, key) with the
/// exact-grams refinement, linked per source stop in `rode` order.
#[derive(Clone, Copy)]
struct FootpathPoint {
    label: u32,
    arrival: u32,
    grams: f64,
    key: i64,
    next: u32,
    live: bool,
}

/// A live point compacted for relaxation, without the list link.
#[derive(Clone, Copy)]
struct ActiveFootpathPoint {
    label: u32,
    arrival: u32,
    grams: f64,
    key: i64,
}

/// The round's edge-major footpath batch (strict searches only): the
/// round's `rode` labels queue per source stop under the strict
/// batch-local gate, then each footpath edge is loaded once per
/// touched source and relaxed against the source's compacted live
/// points. A batch never spans two rounds, two departure passes, two
/// origin requests, or walked sources — only ride labels enter.
struct FootpathBatch {
    heads: Box<[u32]>,
    tails: Box<[u32]>,
    touched: Vec<StopIdx>,
    points: Vec<FootpathPoint>,
    active: Vec<ActiveFootpathPoint>,
    rejected: u64,
    evicted: u64,
}

impl FootpathBatch {
    fn new(stop_count: usize) -> FootpathBatch {
        FootpathBatch {
            heads: vec![NONE_U32; stop_count].into_boxed_slice(),
            tails: vec![NONE_U32; stop_count].into_boxed_slice(),
            touched: Vec::new(),
            points: Vec::new(),
            active: Vec::new(),
            rejected: 0,
            evicted: 0,
        }
    }

    /// Queues a ride label's point under the fixed-departure,
    /// fixed-rides strict relation with the exact-grams equal-slot
    /// refinement; a queued candidate kills the pending points it
    /// covers. Accepted points keep `rode` order.
    fn offer(&mut self, source: StopIdx, label: u32, arrival: u32, key: i64, grams: f64) {
        let slot = source.0 as usize;
        let mut cursor = self.heads[slot];
        while cursor != NONE_U32 {
            let point = &mut self.points[cursor as usize];
            let next = point.next;
            if point.live {
                if point.arrival <= arrival
                    && point.key <= key
                    && !(point.arrival == arrival && point.key == key && grams < point.grams)
                {
                    self.rejected += 1;
                    return;
                }
                if arrival <= point.arrival && key <= point.key {
                    point.live = false;
                    self.evicted += 1;
                }
            }
            cursor = next;
        }
        let index = self.points.len() as u32;
        self.points.push(FootpathPoint {
            label,
            arrival,
            grams,
            key,
            next: NONE_U32,
            live: true,
        });
        if self.heads[slot] == NONE_U32 {
            self.heads[slot] = index;
            self.touched.push(source);
        } else {
            self.points[self.tails[slot] as usize].next = index;
        }
        self.tails[slot] = index;
    }

    fn reset(&mut self) {
        for &stop in &self.touched {
            self.heads[stop.0 as usize] = NONE_U32;
            self.tails[stop.0 as usize] = NONE_U32;
        }
        self.touched.clear();
        self.points.clear();
        self.active.clear();
        self.rejected = 0;
        self.evicted = 0;
    }
}

/// A destination frontier entry; `departure` is the profile pass that
/// produced it.
#[derive(Debug, Clone, Copy)]
struct Arrived {
    departure: u32,
    arrival: u32,
    /// Accumulated route-penalty seconds; the frontier dominates on the
    /// effective arrival while the journey reports the true `arrival`.
    penalty: u32,
    key: i64,
    grams: f64,
    label: u32,
}

impl Arrived {
    /// The penalized arrival the frontier dominates on: true `arrival`
    /// plus accumulated route penalty (equal to `arrival` without one).
    fn effective(&self) -> u32 {
        self.arrival.saturating_add(self.penalty)
    }
}

/// The destination bag: Pareto over (departure descending, arrival,
/// grams bucket). Passes run at descending departures, so entries from
/// later departures are never evicted by earlier ones.
#[derive(Debug, Default)]
struct DestinationBag {
    entries: Vec<Arrived>,
}

impl DestinationBag {
    /// Inserts under the same time slack as the stop bags: a frontier entry
    /// rejects the candidate only when it is at least `slack` seconds
    /// earlier and no dirtier, so suboptimal arrivals within the band are
    /// retained. `slack = 0` is the strict (departure↓, arrival, bucket)
    /// Pareto frontier.
    fn insert(&mut self, candidate: Arrived, slack: u32) -> bool {
        for entry in &self.entries {
            if entry.departure >= candidate.departure && entry.key <= candidate.key {
                if entry.departure == candidate.departure
                    && entry.effective() == candidate.effective()
                    && entry.key == candidate.key
                {
                    if candidate.grams >= entry.grams {
                        return false;
                    }
                } else if entry.effective().saturating_add(slack) <= candidate.effective() {
                    return false;
                }
            }
        }
        self.entries.retain(|entry| {
            !((candidate.departure >= entry.departure
                && candidate.key <= entry.key
                && candidate.effective().saturating_add(slack) <= entry.effective())
                || (entry.departure == candidate.departure
                    && entry.effective() == candidate.effective()
                    && entry.key == candidate.key
                    && candidate.grams < entry.grams))
        });
        self.entries.push(candidate);
        true
    }
}

/// A label riding a trip during one line scan. `kappa` folds the grams
/// at boarding and the boarding position's cumulative distance into one
/// value comparable between same-trip riders: at every future alight
/// both share the trip's arrival, so the lower `kappa` alights with
/// fewer grams everywhere.
#[derive(Debug, Clone, Copy)]
struct Rider {
    trip: ViewTrip,
    board: u16,
    kappa: f64,
    grams: f64,
    factor: f64,
    parent: u32,
    departure: u32,
    /// The boarding label's accumulated route penalty; a ride adds this
    /// line's route penalty on top when it alights.
    penalty: u32,
}

/// Per-destination fold state for the emissions matrix: the cleanest
/// (then fastest) label seen per destination, folded at label creation
/// so cross-pass bag evictions cannot lose a budget-qualifying
/// candidate. Mirrors the interim fold's rules: lower grams win, ties
/// resolve toward the shorter travel time, a travel-time budget
/// disqualifies outright.
struct MatrixFold<'a> {
    /// Per stop: destination slot + 1, or 0 when not a destination.
    slots: &'a [u32],
    /// Per stop: the destinations within a final walk of it —
    /// ``(destination slot, walk seconds, walk meters)``. Empty unless a
    /// street egress is folded (the door-to-door emissions matrix), in which
    /// case a label alighting at the stop also credits each of those
    /// destinations with the final walk added.
    egress: &'a [Vec<(u32, u32, f64)>],
    /// Whether a street egress is folded (door-to-door mode). When set, every
    /// located destination carries its own egress self-entry — its connector to
    /// its coordinate — so the zero-walk direct credit below is left to that
    /// entry and skipped, matching the single-pair door-to-door route's arrival
    /// at the coordinate.
    egress_active: bool,
    budget: Option<u32>,
    /// Per slot: (grams, seconds, label, egress meters).
    best: &'a mut [Option<(f64, u32, u32, f64)>],
}

impl MatrixFold<'_> {
    fn fold(&mut self, label: &Label, index: u32) {
        let base = label.arrival - label.departure;
        let stop = label.stop.0 as usize;
        // In closure mode the label alighting at its own stop, when that stop is
        // a destination, is credited with no final walk. In door-to-door mode
        // that credit is left to the egress map's self-entry (the destination's
        // connector), so the arrival lands at the coordinate as the single-pair
        // route reports it.
        if !self.egress_active {
            let slot = self.slots[stop];
            if slot != 0 {
                self.credit(slot as usize - 1, label.grams, base, index, 0.0);
            }
            return;
        }
        // Door-to-door mode. An access seed has not ridden; it never folds a
        // cell here. Its walking-only journey to a destination is the explicit
        // coordinate-to-coordinate direct walk, overlaid in the matrix layer
        // (`merge_direct_walk_cells`) where the diagonal is a true zero — folding
        // the access label would instead credit the access-walk-to-the-stop cost.
        if matches!(label.origin, Origin::Access) {
            return;
        }
        // A label that has ridden (a ride's alight, or a transfer off one — every
        // transfer label is a footpath off a ride, so `Access` above is the only
        // zero-ride origin) takes a final egress walk, bounded by
        // `max_walking_time`, to every reachable destination, matching what the
        // single-pair route folds. Read by index so the immutable borrow of
        // `egress` does not overlap `credit`'s mutable borrow of `best`.
        for i in 0..self.egress[stop].len() {
            let (dest, walk_seconds, walk_meters) = self.egress[stop][i];
            self.credit(
                dest as usize,
                label.grams,
                base.saturating_add(walk_seconds),
                index,
                walk_meters,
            );
        }
    }

    /// Keeps the cleanest (then fastest) crediting of `slot`; a travel-time
    /// budget disqualifies outright. `egress_meters` is the final walk's
    /// distance, zero when the label alights at the destination itself.
    fn credit(&mut self, slot: usize, grams: f64, seconds: u32, index: u32, egress_meters: f64) {
        if self.budget.is_some_and(|budget| seconds > budget) {
            return;
        }
        let best = &mut self.best[slot];
        let better = match best {
            None => true,
            Some((at_grams, at, _, _)) => {
                grams < *at_grams || (grams == *at_grams && seconds < *at)
            }
        };
        if better {
            *best = Some((grams, seconds, index, egress_meters));
        }
    }
}

/// Per-destination-slot frontier fold: every label reached after a
/// round feeds its slot's destination bag exactly as the one-pair
/// search feeds `Search::destination`, so a batched cell's frontier
/// equals the single-pair query's. Stop mode credits a destination
/// stop's own labels with no final walk; door-to-door mode
/// (`egress_active`) walks each label's final egress instead, leaving
/// the walking-only journey to the caller's direct-walk overlay.
/// Access seeds never reach the fold — only ridden or transferred
/// labels do, matching the one-pair egress check.
struct FrontierFold<'a> {
    /// Per stop: destination slot + 1, or 0 when not a destination.
    slots: &'a [u32],
    /// Per stop: `(destination slot, walk seconds, walk meters)` final
    /// egress; consulted only in door-to-door mode.
    egress: &'a [Vec<(u32, u32, f64)>],
    egress_active: bool,
    /// Per slot: the destination bag the cell's journeys assemble from.
    bags: &'a mut [DestinationBag],
}

impl FrontierFold<'_> {
    fn fold(&mut self, label: &Label, index: u32, departure: u32, key: i64, slack: u32) {
        let arrived = |arrival: u32| Arrived {
            departure,
            arrival,
            penalty: label.penalty,
            key,
            grams: label.grams,
            label: index,
        };
        let stop = label.stop.0 as usize;
        if !self.egress_active {
            let slot = self.slots[stop];
            if slot != 0 {
                self.bags[slot as usize - 1].insert(arrived(label.arrival), slack);
            }
            return;
        }
        for &(slot, walk_seconds, _) in &self.egress[stop] {
            self.bags[slot as usize]
                .insert(arrived(label.arrival.saturating_add(walk_seconds)), slack);
        }
    }
}

struct Search<'a> {
    view: &'a DayView,
    timetable: &'a Timetable,
    footpaths: &'a Transfers,
    geometry: &'a TripGeometry,
    factors: &'a [f64],
    bucket: f64,
    /// The time-slack band, in seconds; 0 is the strict frontier.
    slack: u32,
    /// Per route index: seconds added to a ride's effective arrival for
    /// using it (soft-penalty diverse), `u32::MAX` to skip the route's
    /// lines outright (a hard ban), 0 to leave it free. Empty means none.
    route_penalties: &'a [u32],
    arena: Vec<Label>,
    bags: Vec<Bag>,
    destination: DestinationBag,
    /// Whether labels prune against the destination bag (target
    /// pruning) — set per pass, only when a single destination is
    /// routed (a non-empty `request.egress`); the matrix and batched
    /// folds have no single target.
    prune_target: bool,
    /// The `max_slower` per-stop arrival cutoffs of the current pass —
    /// the plain resolved-trip bound plus the band, floored at the
    /// pass's destination bound so the fastest journey always survives.
    /// Empty when the restriction is off; `max_slower` is pareto-only,
    /// so no route penalties exist and true arrivals compare.
    cutoff: Vec<u32>,
    /// Per line: the (position, label) boardings queued this round.
    queue: Vec<Vec<(u16, u32)>>,
    touched: Vec<u32>,
    /// A strict search (zero slack, no route penalties): stop-bag
    /// admissions dispatch to the self-organising strict insert.
    strict_bags: bool,
    /// R0 attribution state: counters always fill (plain adds); the
    /// per-operation probes and audits run only under the
    /// diagnostic environment flags read once at `start`.
    stats: Box<McRaptorStats>,
    ops: bool,
    edges_prepass: bool,
    /// Diagnostic minimum zero-gram access arrival per stop (ops runs).
    access_floor: Vec<u32>,
    /// The edge-major footpath batch (strict searches only); general
    /// searches keep the label-major loop.
    batch: Option<FootpathBatch>,
}

/// The multicriteria journeys for a single departure: the Pareto set
/// over (arrival, emissions bucket), as full journeys. A positive `slack`
/// (seconds) widens the set to the suboptimal journeys arriving within the
/// band; `max_options`, when set, caps the returned count.
#[allow(clippy::too_many_arguments)]
pub fn route(
    view: &DayView,
    timetable: &Timetable,
    footpaths: &Transfers,
    geometry: &TripGeometry,
    factors: &[f64],
    request: &Request,
    bucket: f64,
    slack: u32,
    max_options: Option<usize>,
    route_penalties: &[u32],
    max_slower: Option<u32>,
) -> Vec<Journey> {
    profile(
        view,
        timetable,
        footpaths,
        geometry,
        factors,
        request,
        &[request.departure],
        bucket,
        slack,
        max_options,
        route_penalties,
        max_slower,
    )
}

/// The multicriteria departure-window profile: the Pareto set over
/// (departure, arrival, emissions bucket), each journey's departure
/// being the latest time the origin can be left to catch it.
#[allow(clippy::too_many_arguments)]
pub fn route_range(
    view: &DayView,
    timetable: &Timetable,
    footpaths: &Transfers,
    geometry: &TripGeometry,
    factors: &[f64],
    request: &Request,
    window: u32,
    bucket: f64,
    slack: u32,
    max_options: Option<usize>,
    route_penalties: &[u32],
    max_slower: Option<u32>,
) -> Vec<Journey> {
    let departures = departure_candidates(timetable, request, window);
    profile(
        view,
        timetable,
        footpaths,
        geometry,
        factors,
        request,
        &departures,
        bucket,
        slack,
        max_options,
        route_penalties,
        max_slower,
    )
}

/// The least-emissions cost matrix over McRAPTOR's candidate set: per
/// origin–destination cell, the cleanest journey (ties toward the
/// shorter travel time) among the (departure, arrival, emissions
/// bucket) Pareto candidates of the departure window — the same
/// widened set `journey_frontier`'s pareto candidates draw from, so a
/// cell can be strictly cleaner than the interim objective's, which
/// only sees time-optimal journeys. Candidates fold per pass at label
/// creation, so a `budget` (travel-time cap in seconds) is applied
/// against each label's own departure. Requests fan out over origins
/// with rayon; the access seeds are the zero-ride floor of the
/// origin's own cell.
#[allow(clippy::too_many_arguments)]
pub fn least_emissions_matrix(
    view: &DayView,
    timetable: &Timetable,
    footpaths: &Transfers,
    inputs: &CostInputs<'_>,
    requests: &[Request],
    destinations: &[StopIdx],
    egress: &[Vec<(u32, u32, f64)>],
    access_meters: &[Vec<(StopIdx, f64)>],
    egress_active: bool,
    window: u32,
    budget: Option<u32>,
    bucket: f64,
) -> Vec<Vec<CostRow>> {
    assert!(
        bucket.is_finite() && bucket > 0.0,
        "the emissions bucket must be positive"
    );
    assert_eq!(
        egress.len(),
        timetable.stop_count() as usize,
        "the egress map must be per stop"
    );
    assert_eq!(
        access_meters.len(),
        requests.len(),
        "the access-meter map must be per request"
    );
    // Door-to-door mode is set by the caller, not inferred from the egress map:
    // an all-empty map (every located destination unsnappable or beyond the cap)
    // must still keep the zero-walk direct credit off, leaving those
    // destinations unreachable as the stop-as-coordinate route would.
    let mut slots = vec![0u32; timetable.stop_count() as usize];
    for (index, stop) in destinations.iter().enumerate() {
        slots[stop.0 as usize] = index as u32 + 1;
    }
    let runs: Vec<(Vec<CostRow>, Box<McRaptorStats>)> = requests
        .par_iter()
        .zip(access_meters.par_iter())
        .map(|(request, access_meters)| {
            let mut search = Search::start(
                view,
                timetable,
                footpaths,
                inputs.geometry,
                inputs.factors,
                bucket,
                0,
                &[],
            );
            let departures = departure_candidates(timetable, request, window);
            let mut best: Vec<Option<(f64, u32, u32, f64)>> = vec![None; destinations.len()];
            for &departure in &departures {
                let mut fold = Some(MatrixFold {
                    slots: &slots,
                    egress,
                    egress_active,
                    budget,
                    best: &mut best,
                });
                search.pass(request, departure, &mut fold, &mut None);
            }
            let rows = best
                .into_iter()
                .enumerate()
                .filter_map(|(slot, winner)| {
                    winner.map(|winner| {
                        search.cost_row(inputs, winner, destinations[slot].0, access_meters)
                    })
                })
                .collect();
            (rows, search.stats)
        })
        .collect();
    if std::env::var_os("CAFEIN_MCRAPTOR_PROF").is_some() {
        let mut reduced = McRaptorStats::default();
        for (_, stats) in &runs {
            reduced.absorb(stats);
        }
        reduced.report("least_emissions_matrix");
    }
    runs.into_iter().map(|run| run.0).collect()
}

/// The plain time-only bounds behind `max_slower`: per departure pass
/// (descending, range-RAPTOR shared state and round-capped like the
/// multicriteria search) the earliest per-stop arrival over the trips
/// with a **resolved emission factor** — the same trip set the
/// multicriteria search can board, so its per-pass fastest journey
/// always achieves the destination bound and the cutoff floor provably
/// keeps that journey alive. Returns one per-stop snapshot per
/// departure, in the departures' (descending) order; unreachable stops
/// hold `u32::MAX`.
fn resolved_bounds(
    view: &DayView,
    timetable: &Timetable,
    footpaths: &Transfers,
    factors: &[f64],
    request: &Request,
    departures: &[u32],
) -> Vec<Vec<u32>> {
    let stop_count = timetable.stop_count() as usize;
    let mut best = vec![u32::MAX; stop_count];
    let mut queue: Vec<Vec<(u16, u32)>> = vec![Vec::new(); view.line_count() as usize];
    let mut touched: Vec<u32> = Vec::new();
    let mut snapshots = Vec::with_capacity(departures.len());
    for &departure in departures {
        let mut fresh: Vec<StopIdx> = Vec::new();
        for &(stop, seconds) in &request.access {
            let arrival = departure.saturating_add(seconds);
            if arrival < best[stop.0 as usize] {
                best[stop.0 as usize] = arrival;
                fresh.push(stop);
            }
        }
        for _ in 1..=request.max_transfers as u32 + 1 {
            if fresh.is_empty() {
                break;
            }
            // Boarding times are captured at queueing so a round's own
            // alights cannot fuel same-round boardings (round separation,
            // matching the multicriteria search's label queue).
            for &stop in &fresh {
                let at = best[stop.0 as usize];
                for served in timetable.patterns_at_stop(stop) {
                    let positions = timetable.pattern_stops(served.pattern).len();
                    if served.position as usize + 1 >= positions {
                        continue;
                    }
                    for line in view.lines_of_pattern(served.pattern).into_iter().flatten() {
                        if queue[line as usize].is_empty() {
                            touched.push(line);
                        }
                        queue[line as usize].push((served.position, at));
                    }
                }
            }
            let mut rode: Vec<StopIdx> = Vec::new();
            let lines = std::mem::take(&mut touched);
            for &line in &lines {
                let mut entries = std::mem::take(&mut queue[line as usize]);
                entries.sort_unstable_by_key(|&(position, _)| position);
                let pattern = view.line_pattern(line);
                let stops = timetable.pattern_stops(pattern);
                let offset = view.line_day_offset(line);
                let mut current: Option<ViewTrip> = None;
                let mut queued = 0usize;
                for (position, &stop) in stops.iter().enumerate().skip(entries[0].0 as usize) {
                    if let Some(trip) = current {
                        let arrival = view.stored_times(timetable, trip)[position].arrival - offset;
                        if arrival < best[stop.0 as usize] {
                            best[stop.0 as usize] = arrival;
                            rode.push(stop);
                        }
                    }
                    while queued < entries.len() && entries[queued].0 as usize == position {
                        let (_, at) = entries[queued];
                        queued += 1;
                        // The earliest boardable trip whose factor resolves.
                        let Some(first) =
                            earliest_boardable(view, timetable, line, position as u16, at)
                        else {
                            continue;
                        };
                        for rank in first.0..view.line_trips(line).end {
                            let trip = ViewTrip(rank);
                            if !factors[view.backing(trip).0 as usize].is_finite() {
                                continue;
                            }
                            let departs =
                                view.stored_times(timetable, trip)[position].departure - offset;
                            let held_departs = current.map(|held| {
                                view.stored_times(timetable, held)[position].departure - offset
                            });
                            if held_departs.is_none_or(|held| departs < held) {
                                current = Some(trip);
                            }
                            break;
                        }
                    }
                }
                entries.clear();
                queue[line as usize] = entries;
            }
            let mut next = rode.clone();
            for &stop in &rode {
                let at = best[stop.0 as usize];
                for footpath in footpaths.from_stop(stop) {
                    let arrival = at.saturating_add(footpath.duration);
                    if arrival < best[footpath.to.0 as usize] {
                        best[footpath.to.0 as usize] = arrival;
                        next.push(footpath.to);
                    }
                }
            }
            fresh = next;
        }
        snapshots.push(best.clone());
    }
    snapshots
}

/// The batched Pareto frontiers: per request × destination slot, the
/// (departure, arrival, emissions bucket) Pareto journeys of the
/// departure window — each cell exactly the single-pair `route_range`
/// set (strict frontier: no slack, no cap, no penalties). Stop mode
/// takes the destination stops; door-to-door mode (`egress_active`)
/// takes a per-stop final-egress map over `slot_count` destination
/// points, the walking-only journey being the caller's overlay as in
/// the one-pair coordinate route. One window profile per request
/// serves every slot; requests fan out with rayon.
#[allow(clippy::too_many_arguments)]
pub fn frontier_matrix(
    view: &DayView,
    timetable: &Timetable,
    footpaths: &Transfers,
    geometry: &TripGeometry,
    factors: &[f64],
    requests: &[Request],
    destinations: &[StopIdx],
    egress: &[Vec<(u32, u32, f64)>],
    egress_active: bool,
    slot_count: usize,
    window: u32,
    bucket: f64,
    max_slower: Option<u32>,
) -> Vec<Vec<Vec<Journey>>> {
    assert!(
        bucket.is_finite() && bucket > 0.0,
        "the emissions bucket must be positive"
    );
    if egress_active {
        assert_eq!(
            egress.len(),
            timetable.stop_count() as usize,
            "the egress map must be per stop"
        );
    } else {
        assert_eq!(
            destinations.len(),
            slot_count,
            "stop mode takes one slot per destination stop"
        );
    }
    // A stop holds one slot, so repeated destination stops share the
    // first occurrence's slot and their cells are re-expanded to the
    // requested order after assembly.
    let mut slots = vec![0u32; timetable.stop_count() as usize];
    let mut cell_of: Vec<usize> = Vec::with_capacity(destinations.len());
    let mut unique = 0usize;
    for &stop in destinations {
        let slot = slots[stop.0 as usize];
        if slot == 0 {
            unique += 1;
            slots[stop.0 as usize] = unique as u32;
            cell_of.push(unique - 1);
        } else {
            cell_of.push(slot as usize - 1);
        }
    }
    let bag_count = if egress_active { slot_count } else { unique };
    // Every (stop, slot, final-walk seconds) pair, for the per-slot
    // destination bounds of the `max_slower` restriction.
    let slot_egress: Vec<(StopIdx, usize, u32)> = if max_slower.is_none() {
        Vec::new()
    } else if egress_active {
        egress
            .iter()
            .enumerate()
            .flat_map(|(stop, entries)| {
                entries
                    .iter()
                    .map(move |&(slot, seconds, _)| (StopIdx(stop as u32), slot as usize, seconds))
            })
            .collect()
    } else {
        destinations
            .iter()
            .zip(&cell_of)
            .map(|(&stop, &slot)| (stop, slot, 0))
            .collect()
    };
    let runs: Vec<(Vec<Vec<Journey>>, Box<McRaptorStats>, u64)> = requests
        .par_iter()
        .map(|request| {
            let origin_started = std::time::Instant::now();
            // Per pass, the restriction's per-slot destination bounds; the
            // cutoff floor is the farthest reachable slot's bound, so a
            // label needed by any cell survives, and each cell's output
            // band anchors at its own slot's bound.
            let departures = departure_candidates(timetable, request, window);
            let restricted = max_slower.map(|band| {
                let bounds =
                    resolved_bounds(view, timetable, footpaths, factors, request, &departures);
                let floors: Vec<Vec<u32>> = bounds
                    .iter()
                    .map(|per_stop| {
                        let mut slot_floors = vec![u32::MAX; bag_count];
                        for &(stop, slot, seconds) in &slot_egress {
                            let bound = per_stop[stop.0 as usize].saturating_add(seconds);
                            slot_floors[slot] = slot_floors[slot].min(bound);
                        }
                        slot_floors
                    })
                    .collect();
                (band, bounds, floors)
            });
            let mut search = Search::start(
                view,
                timetable,
                footpaths,
                geometry,
                factors,
                bucket,
                0,
                &[],
            );
            let mut bags: Vec<DestinationBag> = std::iter::repeat_with(DestinationBag::default)
                .take(bag_count)
                .collect();
            for (index, &departure) in departures.iter().enumerate() {
                if let Some((band, bounds, floors)) = &restricted {
                    let floor = floors[index]
                        .iter()
                        .copied()
                        .filter(|&floor| floor != u32::MAX)
                        .max()
                        .unwrap_or(u32::MAX);
                    search.set_cutoff(&bounds[index], floor, *band);
                }
                let mut frontier = Some(FrontierFold {
                    slots: &slots,
                    egress,
                    egress_active,
                    bags: &mut bags,
                });
                search.pass(request, departure, &mut None, &mut frontier);
            }
            let cells: Vec<Vec<Journey>> = bags
                .iter()
                .enumerate()
                .map(|(slot, bag)| {
                    let mut journeys: Vec<Journey> = bag
                        .entries
                        .iter()
                        .filter(|entry| match &restricted {
                            None => true,
                            Some((band, _, floors)) => {
                                let index = departures
                                    .binary_search_by(|probe| probe.cmp(&entry.departure).reverse())
                                    .expect("every entry comes from a pass departure");
                                entry.arrival <= floors[index][slot].saturating_add(*band)
                            }
                        })
                        .map(|arrived| search.assemble(arrived))
                        .collect();
                    journeys.sort_by_key(|journey| {
                        (journey.departure, journey.arrival, journey.rides())
                    });
                    journeys
                })
                .collect();
            let cells: Vec<Vec<Journey>> = if egress_active || unique == destinations.len() {
                cells
            } else {
                cell_of.iter().map(|&cell| cells[cell].clone()).collect()
            };
            (
                cells,
                search.stats,
                origin_started.elapsed().as_nanos() as u64,
            )
        })
        .collect();
    if std::env::var_os("CAFEIN_MCRAPTOR_PROF").is_some() && !runs.is_empty() {
        // The per-origin service-time distribution: each timer starts
        // inside the Rayon task, so queue waiting is excluded.
        let mut walls: Vec<u64> = runs.iter().map(|run| run.2).collect();
        walls.sort_unstable();
        let total_relaxations: u64 = runs
            .iter()
            .map(|run| run.1.footpath_label_edge_relaxations)
            .sum();
        let mut slowest_share = 0.0f64;
        for (index, (cells, stats, wall)) in runs.iter().enumerate() {
            let rows: usize = cells.iter().map(|cell| cell.len()).sum();
            eprintln!(
                "MCRAPTOR-ORIGIN index={index} wall_ms={} departure_passes={} \
                 rounds={} rode_labels={} footpath_relaxations={} \
                 bag_calls={} labels_created={} rows={rows}",
                wall / 1_000_000,
                stats.departure_passes,
                stats.rounds_entered,
                stats.rode_labels,
                stats.footpath_label_edge_relaxations,
                stats.route_bag_calls + stats.footpath_bag_calls,
                stats.labels_created,
            );
            if *wall == walls[walls.len() - 1] && total_relaxations > 0 {
                slowest_share =
                    stats.footpath_label_edge_relaxations as f64 / total_relaxations as f64;
            }
        }
        let percentile = |fraction: f64| walls[((walls.len() - 1) as f64 * fraction) as usize];
        let p50 = percentile(0.5);
        eprintln!(
            "MCRAPTOR-ORIGINS min_ms={} p50_ms={} p90_ms={} max_ms={} \
             max_over_p50={:.2} slowest_relaxation_share={:.3}",
            walls[0] / 1_000_000,
            p50 / 1_000_000,
            percentile(0.9) / 1_000_000,
            walls[walls.len() - 1] / 1_000_000,
            walls[walls.len() - 1] as f64 / p50.max(1) as f64,
            slowest_share,
        );
        let mut reduced = McRaptorStats::default();
        for (_, stats, _) in &runs {
            reduced.absorb(stats);
        }
        reduced.report("frontier_matrix");
    }
    runs.into_iter().map(|run| run.0).collect()
}

#[allow(clippy::too_many_arguments)]
fn profile(
    view: &DayView,
    timetable: &Timetable,
    footpaths: &Transfers,
    geometry: &TripGeometry,
    factors: &[f64],
    request: &Request,
    departures: &[u32],
    bucket: f64,
    slack: u32,
    max_options: Option<usize>,
    route_penalties: &[u32],
    max_slower: Option<u32>,
) -> Vec<Journey> {
    assert!(
        bucket.is_finite() && bucket > 0.0,
        "the emissions bucket must be positive"
    );
    // The `max_slower` restriction: per pass, the resolved-trip bounds
    // anchor the per-stop band and the destination bound floors it (so
    // the pass's fastest journey survives the pruning); the same floor
    // drives the output band below.
    let restricted = max_slower.map(|band| {
        let bounds = resolved_bounds(view, timetable, footpaths, factors, request, departures);
        let floors: Vec<u32> = bounds
            .iter()
            .map(|per_stop| {
                request
                    .egress
                    .iter()
                    .map(|&(stop, seconds)| per_stop[stop.0 as usize].saturating_add(seconds))
                    .min()
                    .unwrap_or(u32::MAX)
            })
            .collect();
        (band, bounds, floors)
    });
    let mut search = Search::start(
        view,
        timetable,
        footpaths,
        geometry,
        factors,
        bucket,
        slack,
        route_penalties,
    );
    for (index, &departure) in departures.iter().enumerate() {
        if let Some((band, bounds, floors)) = &restricted {
            search.set_cutoff(&bounds[index], floors[index], *band);
        }
        search.pass(request, departure, &mut None, &mut None);
    }
    // The output band: a journey stays within `max_slower` of its own
    // pass's plain destination bound (an unreachable bound keeps
    // everything — nothing anchors the band that pass).
    let banded: Vec<Arrived>;
    let entries: &[Arrived] = match &restricted {
        Some((band, _, floors)) => {
            banded = search
                .destination
                .entries
                .iter()
                .filter(|entry| {
                    let index = departures
                        .binary_search_by(|probe| probe.cmp(&entry.departure).reverse())
                        .expect("every entry comes from a pass departure");
                    entry.arrival <= floors[index].saturating_add(*band)
                })
                .copied()
                .collect();
            &banded
        }
        None => &search.destination.entries,
    };
    let kept = cap_entries(entries, max_options);
    let mut journeys: Vec<Journey> = kept
        .into_iter()
        .map(|arrived| search.assemble(arrived))
        .collect();
    journeys.sort_by_key(|journey| (journey.departure, journey.arrival, journey.rides()));
    search.stats.report("profile");
    journeys
}

/// Strict (departure↓, arrival, emissions bucket) domination between two
/// destination entries — the relation that ranks suboptimal arrivals under
/// `max_options`.
fn strictly_dominates(a: &Arrived, b: &Arrived) -> bool {
    a.departure >= b.departure
        && a.effective() <= b.effective()
        && a.key <= b.key
        && (a.departure > b.departure || a.effective() < b.effective() || a.key < b.key)
}

/// The destination entries to assemble. Without a cap (or when the set
/// already fits) every entry is kept; otherwise the strict frontier — the
/// entries no other entry strictly dominates — is kept in full and the
/// suboptimal arrivals of smallest time-gap above it fill the remainder up
/// to `max_options`, ties toward the cleaner emissions. A suboptimal entry's
/// gap is the seconds by which its nearest strict-frontier dominator arrives
/// earlier. The cap never drops a frontier (optimal) journey, so the result
/// can exceed `max_options` when the frontier itself is larger.
fn cap_entries(entries: &[Arrived], max_options: Option<usize>) -> Vec<&Arrived> {
    let cap = match max_options {
        Some(cap) if entries.len() > cap => cap,
        _ => return entries.iter().collect(),
    };
    let on_frontier: Vec<bool> = entries
        .iter()
        .map(|entry| !entries.iter().any(|other| strictly_dominates(other, entry)))
        .collect();
    let mut ranked: Vec<(&Arrived, bool, u32)> = entries
        .iter()
        .zip(&on_frontier)
        .map(|(entry, &frontier)| {
            let gap = if frontier {
                0
            } else {
                entries
                    .iter()
                    .zip(&on_frontier)
                    .filter(|(other, &f)| f && strictly_dominates(other, entry))
                    .map(|(other, _)| entry.effective().saturating_sub(other.effective()))
                    .min()
                    .unwrap_or(u32::MAX)
            };
            (entry, frontier, gap)
        })
        .collect();
    // Frontier entries first (always kept), then suboptimals by time-gap.
    ranked.sort_by(|(a, fa, ga), (b, fb, gb)| {
        fb.cmp(fa)
            .then(ga.cmp(gb))
            .then(a.key.cmp(&b.key))
            .then(a.grams.total_cmp(&b.grams))
    });
    let frontier = on_frontier.iter().filter(|&&f| f).count();
    let keep = cap.max(frontier);
    ranked
        .into_iter()
        .take(keep)
        .map(|(entry, _, _)| entry)
        .collect()
}

impl<'a> Search<'a> {
    #[allow(clippy::too_many_arguments)]
    fn start(
        view: &'a DayView,
        timetable: &'a Timetable,
        footpaths: &'a Transfers,
        geometry: &'a TripGeometry,
        factors: &'a [f64],
        bucket: f64,
        slack: u32,
        route_penalties: &'a [u32],
    ) -> Search<'a> {
        let mut search = Search {
            view,
            timetable,
            footpaths,
            geometry,
            factors,
            bucket,
            slack,
            route_penalties,
            arena: Vec::new(),
            bags: vec![Bag::default(); timetable.stop_count() as usize],
            destination: DestinationBag::default(),
            prune_target: false,
            cutoff: Vec::new(),
            queue: vec![Vec::new(); view.line_count() as usize],
            touched: Vec::new(),
            strict_bags: slack == 0 && route_penalties.is_empty(),
            stats: Box::default(),
            ops: std::env::var_os("CAFEIN_MCRAPTOR_PROF_OPS").is_some(),
            edges_prepass: std::env::var_os("CAFEIN_MCRAPTOR_PROF_EDGES").is_some(),
            access_floor: Vec::new(),
            batch: None,
        };
        if search.strict_bags {
            search.batch = Some(FootpathBatch::new(timetable.stop_count() as usize));
        }
        search.arm_diagnostics();
        search
    }

    /// Sizes the diagnostic structures when the flags ask for them.
    fn arm_diagnostics(&mut self) {
        if !self.ops {
            return;
        }
        self.access_floor = vec![u32::MAX; self.timetable.stop_count() as usize];
    }

    /// Installs the pass's `max_slower` cutoffs: per stop the plain
    /// bound plus the band, floored at the pass's destination bound
    /// (`floor`); unreachable stops never prune.
    fn set_cutoff(&mut self, bounds: &[u32], floor: u32, band: u32) {
        self.cutoff.clear();
        self.cutoff.extend(bounds.iter().map(|&bound| {
            if bound == u32::MAX {
                u32::MAX
            } else {
                bound.saturating_add(band).max(floor)
            }
        }));
    }

    /// Whether the pass's `max_slower` cutoff drops an arrival at a stop.
    fn beyond_cutoff(&self, stop: StopIdx, arrival: u32) -> bool {
        !self.cutoff.is_empty() && arrival > self.cutoff[stop.0 as usize]
    }

    fn key(&self, grams: f64) -> i64 {
        (grams / self.bucket).floor() as i64
    }

    /// Whether a label can be dropped against the destination bag:
    /// along a journey the arrival, the penalty, and the grams only
    /// grow, so a continuation's (departure, effective arrival, bucket)
    /// never improves on the label's and `DestinationBag::insert`'s
    /// rejection applies transitively. The carve-out keeps the label
    /// when a continuation could still *refine* a same-class entry —
    /// equal departure, effective arrival, and bucket with strictly
    /// lower exact grams — which the bag accepts as a replacement.
    fn target_pruned(&self, departure: u32, effective: u32, key: i64, grams: f64) -> bool {
        if !self.prune_target {
            return false;
        }
        self.destination.entries.iter().any(|entry| {
            entry.departure >= departure
                && entry.key <= key
                && entry.effective().saturating_add(self.slack) <= effective
                && !(entry.departure == departure
                    && entry.effective() == effective
                    && entry.key == key
                    && grams < entry.grams)
        })
    }

    /// One profile pass: seed the access labels at `departure`, then
    /// ride/walk/join for each round. Bags persist across passes —
    /// labels of later departures suppress what they dominate on
    /// (arrival, bucket, rides), the range-RAPTOR reuse made sound under
    /// the emissions criterion by ranking the rides used (see
    /// `Bag::insert`). A matrix fold, when given, sees every label the
    /// pass creates (the access seeds are the zero-ride floor of the
    /// origin's own cell); a frontier fold sees the labels reached after
    /// each round, exactly where the one-pair egress check runs.
    fn pass(
        &mut self,
        request: &Request,
        departure: u32,
        fold: &mut Option<MatrixFold<'_>>,
        frontier: &mut Option<FrontierFold<'_>>,
    ) {
        self.prune_target = !request.egress.is_empty();
        self.stats.departure_passes += 1;
        let mut fresh: Vec<u32> = Vec::new();
        let phase = std::time::Instant::now();
        for &(stop, seconds) in &request.access {
            let arrival = departure.saturating_add(seconds);
            let label = self.arena.len() as u32;
            let key = self.key(0.0);
            self.stats.access_offers += 1;
            if self.beyond_cutoff(stop, arrival) || self.target_pruned(departure, arrival, key, 0.0)
            {
                continue;
            }
            if self.bags[stop.0 as usize].insert_label(
                self.strict_bags,
                arrival,
                0,
                0.0,
                key,
                0,
                self.slack,
            ) {
                self.stats.access_admissions += 1;
                if self.ops {
                    let floor = &mut self.access_floor[stop.0 as usize];
                    *floor = (*floor).min(arrival);
                }
                self.arena.push(Label {
                    arrival,
                    grams: 0.0,
                    stop,
                    departure,
                    penalty: 0,
                    origin: Origin::Access,
                });
                if let Some(fold) = fold {
                    fold.fold(&self.arena[label as usize], label);
                }
                fresh.push(label);
            }
        }
        self.stats.access_ns += phase.elapsed().as_nanos() as u64;
        for round in 1..=request.max_transfers as u32 + 1 {
            if fresh.is_empty() {
                self.stats.rounds_ended_empty += 1;
                break;
            }
            self.stats.rounds_entered += 1;
            self.stats.max_round_reached = self.stats.max_round_reached.max(round);
            let rides = round as u8;
            let phase = std::time::Instant::now();
            self.stats.queue_labels += fresh.len() as u64;
            let mut queued_entries = 0u64;
            for &label in &fresh {
                let stop = self.arena[label as usize].stop;
                for served in self.timetable.patterns_at_stop(stop) {
                    let positions = self.timetable.pattern_stops(served.pattern).len();
                    if served.position as usize + 1 >= positions {
                        continue;
                    }
                    for line in self
                        .view
                        .lines_of_pattern(served.pattern)
                        .into_iter()
                        .flatten()
                    {
                        if self.queue[line as usize].is_empty() {
                            self.touched.push(line);
                        }
                        self.queue[line as usize].push((served.position, label));
                        queued_entries += 1;
                    }
                }
            }
            self.stats.queue_entries += queued_entries;
            self.stats.queue_collect_ns += phase.elapsed().as_nanos() as u64;
            let phase = std::time::Instant::now();
            let mut rode: Vec<u32> = Vec::new();
            let touched = std::mem::take(&mut self.touched);
            self.stats.lines_touched += touched.len() as u64;
            for &line in &touched {
                self.scan_line(line, rides, &mut rode);
            }
            self.stats.route_scan_ns += phase.elapsed().as_nanos() as u64;
            self.stats.rode_labels += rode.len() as u64;
            if self.ops {
                self.audit_rode(&rode, rides);
            }
            if self.edges_prepass {
                // Diagnostic: time one edge-only traversal of the same
                // label-major slices. It warms the real loop, so it
                // belongs to separate profiling runs.
                let started = std::time::Instant::now();
                let mut checksum = 0u64;
                for &label in &rode {
                    let stop = self.arena[label as usize].stop;
                    for footpath in self.footpaths.from_stop(stop) {
                        checksum = checksum.wrapping_add(
                            ((footpath.duration as u64) << 32) | footpath.to.0 as u64,
                        );
                    }
                }
                std::hint::black_box(checksum);
                self.stats.footpath_edge_prepass_ns += started.elapsed().as_nanos() as u64;
            }
            // One footpath hop from the improving ride labels; the
            // closed transfer contract makes chains redundant.
            let phase = std::time::Instant::now();
            let mut next = rode.clone();
            if let Some(mut batch) = self.batch.take() {
                if self.ops {
                    self.relax_footpaths_batched_probed(
                        &mut batch, &rode, rides, departure, &mut next,
                    );
                } else {
                    self.relax_footpaths_batched(&mut batch, &rode, rides, departure, &mut next);
                }
                self.batch = Some(batch);
            } else if self.ops {
                self.relax_footpaths_probed(&rode, rides, &mut next);
            } else {
                self.relax_footpaths(&rode, rides, &mut next);
            }
            self.stats.footpath_ns += phase.elapsed().as_nanos() as u64;
            let phase = std::time::Instant::now();
            for &label in &next {
                let reached = self.arena[label as usize];
                if let Some(fold) = fold {
                    fold.fold(&reached, label);
                }
                if let Some(frontier) = frontier {
                    frontier.fold(
                        &reached,
                        label,
                        departure,
                        self.key(reached.grams),
                        self.slack,
                    );
                }
                for &(stop, seconds) in &request.egress {
                    if stop == reached.stop {
                        let arrival = reached.arrival.saturating_add(seconds);
                        self.destination.insert(
                            Arrived {
                                departure,
                                arrival,
                                penalty: reached.penalty,
                                key: self.key(reached.grams),
                                grams: reached.grams,
                                label,
                            },
                            self.slack,
                        );
                    }
                }
            }
            self.stats.sink_ns += phase.elapsed().as_nanos() as u64;
            if self.ops {
                let mut total = 0u64;
                let mut longest = 0u64;
                for bag in &self.bags {
                    let length = bag.entry_count() as u64;
                    total += length;
                    longest = longest.max(length);
                }
                self.stats.total_bag_entries_at_round_end += total;
                self.stats.maximum_stop_bag_length =
                    self.stats.maximum_stop_bag_length.max(longest);
            }
            fresh = next;
        }
        self.stats.labels_created = self.arena.len() as u64;
    }

    /// Whether the diagnostic access-floor entry is an exact rejecting
    /// witness for a candidate under the bag's slack relation: the
    /// floor carries zero grams, penalty, and rides, so it wins every
    /// axis when its key and both arrival comparisons hold (a ridden
    /// candidate can never be its exact-class tie).
    fn access_floor_rejects(&self, stop: StopIdx, arrival: u32, penalty: u32, key: i64) -> bool {
        let floor = self.access_floor[stop.0 as usize];
        floor != u32::MAX
            && key >= 0
            && floor <= arrival
            && floor.saturating_add(self.slack) <= arrival.saturating_add(penalty)
    }

    /// The round's stale-`rode` audit (diagnostic runs): a label is
    /// represented while its exact tuple is still in its bag; a stale
    /// label was evicted by a later admission in the same round.
    fn audit_rode(&mut self, rode: &[u32], rides: u8) {
        for &label in rode {
            let from = self.arena[label as usize];
            let key = self.key(from.grams);
            if self.bags[from.stop.0 as usize].contains_exact(
                from.arrival,
                from.penalty,
                key,
                from.grams,
                rides,
            ) {
                self.stats.rode_represented += 1;
            } else {
                self.stats.rode_stale += 1;
            }
        }
    }

    /// The strict edge-major footpath sweep: the round's `rode`
    /// labels queue per source stop under the batch gate, then every
    /// footpath edge is loaded once per touched source and relaxed
    /// against the source's compacted live points — the same target
    /// admissions as label-major traversal (the walk map is monotone
    /// on every strict bag axis), with transient admissions and
    /// exact-cost ancestries free to differ per the programme
    /// contract. Every `rode` label stays in `next` regardless of the
    /// gate; walked admissions append in sweep order. Resets the
    /// batch when done.
    fn relax_footpaths_batched(
        &mut self,
        batch: &mut FootpathBatch,
        rode: &[u32],
        rides: u8,
        departure: u32,
        next: &mut Vec<u32>,
    ) {
        for &label in rode {
            let from = self.arena[label as usize];
            if self.footpaths.from_stop(from.stop).is_empty() {
                continue;
            }
            let key = self.key(from.grams);
            batch.offer(from.stop, label, from.arrival, key, from.grams);
            self.stats.batch_points_offered += 1;
        }
        for source_index in 0..batch.touched.len() {
            let source = batch.touched[source_index];
            self.stats.batch_source_stops += 1;
            batch.active.clear();
            let mut cursor = batch.heads[source.0 as usize];
            while cursor != NONE_U32 {
                let point = batch.points[cursor as usize];
                if point.live {
                    batch.active.push(ActiveFootpathPoint {
                        label: point.label,
                        arrival: point.arrival,
                        grams: point.grams,
                        key: point.key,
                    });
                }
                cursor = point.next;
            }
            self.stats.batch_points_live += batch.active.len() as u64;
            let slice = self.footpaths.from_stop(source);
            let mut cutoff_pruned = 0u64;
            let mut target_pruned = 0u64;
            let mut admissions = 0u64;
            for footpath in slice {
                for point in &batch.active {
                    let arrival = point.arrival.saturating_add(footpath.duration);
                    let walked = self.arena.len() as u32;
                    if self.beyond_cutoff(footpath.to, arrival) {
                        cutoff_pruned += 1;
                        continue;
                    }
                    if self.target_pruned(departure, arrival, point.key, point.grams) {
                        target_pruned += 1;
                        continue;
                    }
                    if self.bags[footpath.to.0 as usize].insert(
                        arrival,
                        point.grams,
                        point.key,
                        rides,
                    ) {
                        admissions += 1;
                        self.arena.push(Label {
                            arrival,
                            grams: point.grams,
                            stop: footpath.to,
                            departure,
                            penalty: 0,
                            origin: Origin::Walk {
                                parent: point.label,
                                duration: footpath.duration,
                            },
                        });
                        next.push(walked);
                    }
                }
            }
            let relaxations = slice.len() as u64 * batch.active.len() as u64;
            let calls = relaxations - cutoff_pruned - target_pruned;
            self.stats.footpath_edge_records_loaded += slice.len() as u64;
            self.stats.footpath_label_edge_relaxations += relaxations;
            self.stats.footpath_cutoff_pruned += cutoff_pruned;
            self.stats.footpath_target_pruned += target_pruned;
            self.stats.footpath_bag_calls += calls;
            self.stats.footpath_target_admissions += admissions;
            self.stats.footpath_bag_rejections += calls - admissions;
        }
        self.stats.batch_points_rejected += batch.rejected;
        self.stats.batch_points_evicted += batch.evicted;
        batch.reset();
    }

    /// The diagnostic twin of `relax_footpaths_batched`: identical
    /// decisions and permutations through the probed strict insert,
    /// plus depth histograms and the access-floor attribution.
    fn relax_footpaths_batched_probed(
        &mut self,
        batch: &mut FootpathBatch,
        rode: &[u32],
        rides: u8,
        departure: u32,
        next: &mut Vec<u32>,
    ) {
        for &label in rode {
            let from = self.arena[label as usize];
            if self.footpaths.from_stop(from.stop).is_empty() {
                continue;
            }
            let key = self.key(from.grams);
            batch.offer(from.stop, label, from.arrival, key, from.grams);
            self.stats.batch_points_offered += 1;
        }
        for source_index in 0..batch.touched.len() {
            let source = batch.touched[source_index];
            self.stats.batch_source_stops += 1;
            batch.active.clear();
            let mut cursor = batch.heads[source.0 as usize];
            while cursor != NONE_U32 {
                let point = batch.points[cursor as usize];
                if point.live {
                    batch.active.push(ActiveFootpathPoint {
                        label: point.label,
                        arrival: point.arrival,
                        grams: point.grams,
                        key: point.key,
                    });
                }
                cursor = point.next;
            }
            self.stats.batch_points_live += batch.active.len() as u64;
            let slice = self.footpaths.from_stop(source);
            let mut cutoff_pruned = 0u64;
            let mut target_pruned = 0u64;
            let mut admissions = 0u64;
            for footpath in slice {
                for point in &batch.active {
                    let arrival = point.arrival.saturating_add(footpath.duration);
                    let walked = self.arena.len() as u32;
                    if self.beyond_cutoff(footpath.to, arrival) {
                        cutoff_pruned += 1;
                        continue;
                    }
                    if self.target_pruned(departure, arrival, point.key, point.grams) {
                        target_pruned += 1;
                        continue;
                    }
                    let mut probes = InsertProbes::default();
                    let admitted = self.bags[footpath.to.0 as usize].insert_probed(
                        arrival,
                        point.grams,
                        point.key,
                        rides,
                        &mut probes,
                    );
                    self.stats.footpath_bag_length_histogram[length_bucket(probes.length)] += 1;
                    if admitted {
                        self.stats.footpath_admit_entries_examined += probes.examined as u64;
                        self.stats.footpath_retain_entries_examined += probes.retained as u64;
                        admissions += 1;
                        self.arena.push(Label {
                            arrival,
                            grams: point.grams,
                            stop: footpath.to,
                            departure,
                            penalty: 0,
                            origin: Origin::Walk {
                                parent: point.label,
                                duration: footpath.duration,
                            },
                        });
                        next.push(walked);
                    } else {
                        self.stats.footpath_reject_entries_examined += probes.examined as u64;
                        self.stats.footpath_reject_depth_histogram
                            [depth_bucket(probes.examined)] += 1;
                        if self.access_floor_rejects(footpath.to, arrival, 0, point.key) {
                            self.stats.access_floor_rejectable_footpath += 1;
                            if probes.examined > 1 {
                                self.stats.access_floor_nonfront_saved_probes +=
                                    (probes.examined - 1) as u64;
                            }
                        }
                    }
                }
            }
            let relaxations = slice.len() as u64 * batch.active.len() as u64;
            let calls = relaxations - cutoff_pruned - target_pruned;
            self.stats.footpath_edge_records_loaded += slice.len() as u64;
            self.stats.footpath_label_edge_relaxations += relaxations;
            self.stats.footpath_cutoff_pruned += cutoff_pruned;
            self.stats.footpath_target_pruned += target_pruned;
            self.stats.footpath_bag_calls += calls;
            self.stats.footpath_target_admissions += admissions;
            self.stats.footpath_bag_rejections += calls - admissions;
        }
        self.stats.batch_points_rejected += batch.rejected;
        self.stats.batch_points_evicted += batch.evicted;
        batch.reset();
    }

    /// The production footpath hop: one relaxation per (rode label,
    /// closure edge), kept free of per-edge instrumentation — the
    /// counters accumulate in locals and flush per label so the inner
    /// loop compiles exactly as before the R0 stage.
    fn relax_footpaths(&mut self, rode: &[u32], rides: u8, next: &mut Vec<u32>) {
        for &label in rode {
            let from = self.arena[label as usize];
            let key = self.key(from.grams);
            let slice = self.footpaths.from_stop(from.stop);
            let mut cutoff_pruned = 0u64;
            let mut target_pruned = 0u64;
            let mut admissions = 0u64;
            for footpath in slice {
                let arrival = from.arrival.saturating_add(footpath.duration);
                let walked = self.arena.len() as u32;
                if self.beyond_cutoff(footpath.to, arrival) {
                    cutoff_pruned += 1;
                    continue;
                }
                if self.target_pruned(
                    from.departure,
                    arrival.saturating_add(from.penalty),
                    key,
                    from.grams,
                ) {
                    target_pruned += 1;
                    continue;
                }
                // A footpath adds no route penalty; it inherits the chain's.
                if self.bags[footpath.to.0 as usize].insert_label(
                    self.strict_bags,
                    arrival,
                    from.penalty,
                    from.grams,
                    key,
                    rides,
                    self.slack,
                ) {
                    admissions += 1;
                    self.arena.push(Label {
                        arrival,
                        grams: from.grams,
                        stop: footpath.to,
                        departure: from.departure,
                        penalty: from.penalty,
                        origin: Origin::Walk {
                            parent: label,
                            duration: footpath.duration,
                        },
                    });
                    next.push(walked);
                }
            }
            let calls = slice.len() as u64 - cutoff_pruned - target_pruned;
            self.stats.footpath_edge_records_loaded += slice.len() as u64;
            self.stats.footpath_label_edge_relaxations += slice.len() as u64;
            self.stats.footpath_cutoff_pruned += cutoff_pruned;
            self.stats.footpath_target_pruned += target_pruned;
            self.stats.footpath_bag_calls += calls;
            self.stats.footpath_target_admissions += admissions;
            self.stats.footpath_bag_rejections += calls - admissions;
        }
    }

    /// The diagnostic twin of `relax_footpaths`: identical decisions
    /// through the probed insert, plus depth histograms and the
    /// access-floor attribution. Runs only under the ops flag.
    fn relax_footpaths_probed(&mut self, rode: &[u32], rides: u8, next: &mut Vec<u32>) {
        for &label in rode {
            let from = self.arena[label as usize];
            let key = self.key(from.grams);
            let slice = self.footpaths.from_stop(from.stop);
            let mut cutoff_pruned = 0u64;
            let mut target_pruned = 0u64;
            let mut admissions = 0u64;
            for footpath in slice {
                let arrival = from.arrival.saturating_add(footpath.duration);
                let walked = self.arena.len() as u32;
                if self.beyond_cutoff(footpath.to, arrival) {
                    cutoff_pruned += 1;
                    continue;
                }
                if self.target_pruned(
                    from.departure,
                    arrival.saturating_add(from.penalty),
                    key,
                    from.grams,
                ) {
                    target_pruned += 1;
                    continue;
                }
                let mut probes = InsertProbes::default();
                let admitted = self.bags[footpath.to.0 as usize].insert_label_probed(
                    self.strict_bags,
                    arrival,
                    from.penalty,
                    from.grams,
                    key,
                    rides,
                    self.slack,
                    &mut probes,
                );
                self.stats.footpath_bag_length_histogram[length_bucket(probes.length)] += 1;
                if admitted {
                    self.stats.footpath_admit_entries_examined += probes.examined as u64;
                    self.stats.footpath_retain_entries_examined += probes.retained as u64;
                    admissions += 1;
                    self.arena.push(Label {
                        arrival,
                        grams: from.grams,
                        stop: footpath.to,
                        departure: from.departure,
                        penalty: from.penalty,
                        origin: Origin::Walk {
                            parent: label,
                            duration: footpath.duration,
                        },
                    });
                    next.push(walked);
                } else {
                    self.stats.footpath_reject_entries_examined += probes.examined as u64;
                    self.stats.footpath_reject_depth_histogram[depth_bucket(probes.examined)] += 1;
                    if self.access_floor_rejects(footpath.to, arrival, from.penalty, key) {
                        self.stats.access_floor_rejectable_footpath += 1;
                        if probes.examined > 1 {
                            self.stats.access_floor_nonfront_saved_probes +=
                                (probes.examined - 1) as u64;
                        }
                    }
                }
            }
            let calls = slice.len() as u64 - cutoff_pruned - target_pruned;
            self.stats.footpath_edge_records_loaded += slice.len() as u64;
            self.stats.footpath_label_edge_relaxations += slice.len() as u64;
            self.stats.footpath_cutoff_pruned += cutoff_pruned;
            self.stats.footpath_target_pruned += target_pruned;
            self.stats.footpath_bag_calls += calls;
            self.stats.footpath_target_admissions += admissions;
            self.stats.footpath_bag_rejections += calls - admissions;
        }
    }

    /// Scans one line: boarding labels enter the rider bag at their
    /// queued positions, riders alight at every later position. Besides
    /// the earliest boardable trip, later trips join while their factor
    /// strictly improves; same-trip riders reduce to the lowest
    /// `kappa`.
    fn scan_line(&mut self, line: u32, rides: u8, rode: &mut Vec<u32>) {
        let mut entries = std::mem::take(&mut self.queue[line as usize]);
        let pattern = self.view.line_pattern(line);
        let line_penalty = self
            .route_penalties
            .get(self.timetable.pattern_route(pattern) as usize)
            .copied()
            .unwrap_or(0);
        // A banned route (the `u32::MAX` sentinel) skips its lines entirely, so
        // the re-search omits committed corridors; a finite penalty instead adds
        // to each ride's effective arrival, making the route costly but usable.
        if line_penalty == u32::MAX {
            entries.clear();
            self.queue[line as usize] = entries;
            return;
        }
        entries.sort_unstable_by_key(|&(position, _)| position);
        let stops = self.timetable.pattern_stops(pattern);
        let offset = self.view.line_day_offset(line);
        let mut riders: Vec<Rider> = Vec::new();
        let mut queued = 0;
        let mut positions = 0u64;
        let mut evaluations = 0u64;
        let mut bag_calls = 0u64;
        let mut admissions = 0u64;
        for position in entries[0].0 as usize..stops.len() {
            positions += 1;
            for rider in &riders {
                evaluations += 1;
                if (rider.board as usize) >= position {
                    continue;
                }
                let times = self.view.stored_times(self.timetable, rider.trip);
                let arrival = times[position].arrival - offset;
                let meters = self.geometry.leg_distance(
                    self.view.backing(rider.trip),
                    rider.board,
                    position as u16,
                ) as f64;
                let grams = quantized(rider.grams + meters / 1000.0 * rider.factor);
                let label = self.arena.len() as u32;
                let key = self.key(grams);
                // This ride pays the line's route penalty once, on top of the
                // penalty its boarding chain already carried.
                let penalty = rider.penalty.saturating_add(line_penalty);
                if self.beyond_cutoff(stops[position], arrival)
                    || self.target_pruned(
                        rider.departure,
                        arrival.saturating_add(penalty),
                        key,
                        grams,
                    )
                {
                    continue;
                }
                let admitted = self.bags[stops[position].0 as usize].insert_label(
                    self.strict_bags,
                    arrival,
                    penalty,
                    grams,
                    key,
                    rides,
                    self.slack,
                );
                bag_calls += 1;
                if !admitted {
                    if self.ops && self.access_floor_rejects(stops[position], arrival, penalty, key)
                    {
                        self.stats.access_floor_rejectable_route += 1;
                    }
                    continue;
                }
                admissions += 1;
                {
                    self.arena.push(Label {
                        arrival,
                        grams,
                        stop: stops[position],
                        departure: rider.departure,
                        penalty,
                        origin: Origin::Ride {
                            parent: rider.parent,
                            trip: rider.trip,
                            board: rider.board,
                            alight: position as u16,
                        },
                    });
                    rode.push(label);
                }
            }
            while queued < entries.len() && entries[queued].0 as usize == position {
                let (_, label) = entries[queued];
                queued += 1;
                let boarding = self.arena[label as usize];
                let Some(first) = earliest_boardable(
                    self.view,
                    self.timetable,
                    line,
                    position as u16,
                    boarding.arrival,
                ) else {
                    continue;
                };
                let mut cleanest = f64::INFINITY;
                // The latest boarded trip whose factor set `cleanest` — the
                // nearest no-dirtier earlier departure a within-slack trip is
                // measured against (only tracked when relaxing).
                let mut last_clean_departure = if self.slack > 0 {
                    self.view.stored_times(self.timetable, first)[position].departure - offset
                } else {
                    0
                };
                for rank in first.0..self.view.line_trips(line).end {
                    let trip = ViewTrip(rank);
                    let factor = self.factors[self.view.backing(trip).0 as usize];
                    if !factor.is_finite() {
                        continue;
                    }
                    if factor < cleanest {
                        cleanest = factor;
                        if self.slack > 0 {
                            last_clean_departure =
                                self.view.stored_times(self.timetable, trip)[position].departure
                                    - offset;
                        }
                    } else if self.slack == 0 {
                        continue;
                    } else {
                        // Under a slack, board a later same-line trip that is
                        // not strictly cleaner when it departs within the band of
                        // the nearest no-dirtier boarded trip — the "next
                        // departure" alternative strict Pareto drops. The stop
                        // bags prune the ones that fall more than `slack` behind
                        // at their alights.
                        let departure = self.view.stored_times(self.timetable, trip)[position]
                            .departure
                            - offset;
                        if departure.saturating_sub(last_clean_departure) > self.slack {
                            continue;
                        }
                    }
                    let travelled =
                        self.geometry
                            .leg_distance(self.view.backing(trip), 0, position as u16)
                            as f64;
                    let kappa = boarding.grams - travelled / 1000.0 * factor;
                    // Same-trip riders reduce over both grams (`kappa`) and the
                    // accumulated penalty: a cleaner rider that carries a larger
                    // penalty must not suppress a dirtier one that alights on a
                    // lower effective arrival.
                    let penalty = boarding.penalty;
                    if riders.iter().any(|rider| {
                        rider.trip == trip && rider.kappa <= kappa && rider.penalty <= penalty
                    }) {
                        continue;
                    }
                    riders.retain(|rider| {
                        !(rider.trip == trip && kappa <= rider.kappa && penalty <= rider.penalty)
                    });
                    riders.push(Rider {
                        trip,
                        board: position as u16,
                        kappa,
                        grams: boarding.grams,
                        factor,
                        parent: label,
                        departure: boarding.departure,
                        penalty,
                    });
                }
            }
        }
        entries.clear();
        self.queue[line as usize] = entries;
        self.stats.line_positions += positions;
        self.stats.rider_evaluations += evaluations;
        self.stats.route_bag_calls += bag_calls;
        self.stats.route_bag_admissions += admissions;
    }

    /// Walks a winning label's chain into a cost row, mirroring the
    /// interim reconstruction: transit and transfer meters summed leg
    /// by leg, geometry legs reversed into ride order, fare legs
    /// priced in order. Emissions come from the label itself — the
    /// same cumulative-distance sums, already microgram-quantized.
    fn cost_row(
        &self,
        inputs: &CostInputs<'_>,
        winner: (f64, u32, u32, f64),
        to: u32,
        access_meters: &[(StopIdx, f64)],
    ) -> CostRow {
        let (grams, seconds, mut at, egress_meters) = winner;
        let mut rides = 0u32;
        let mut transit_meters = 0.0;
        // The final walk to the destination (zero when the label alights there).
        let mut walk_meters = egress_meters;
        let mut legs: Vec<(TripIdx, u16, u16)> = Vec::new();
        let mut fare_legs: Vec<FareLeg> = Vec::new();
        loop {
            let label = &self.arena[at as usize];
            match label.origin {
                Origin::Access => break,
                Origin::Ride {
                    parent,
                    trip,
                    board,
                    alight,
                } => {
                    rides += 1;
                    let backing = self.view.backing(trip);
                    transit_meters += inputs.geometry.leg_distance(backing, board, alight) as f64;
                    if inputs.with_geometry {
                        legs.push((backing, board, alight));
                    }
                    if inputs.fares.is_some() {
                        let pattern = self.timetable.trip_pattern(backing);
                        let stops = self.timetable.pattern_stops(pattern);
                        fare_legs.push(FareLeg {
                            route: self.timetable.pattern_route(pattern),
                            board_stop: stops[board as usize].0,
                            alight_stop: stops[alight as usize].0,
                            board_time: self.timetable.trip_stop_times(backing)[board as usize]
                                .departure
                                .saturating_sub(self.view.day_offset(trip)),
                        });
                    }
                    at = parent;
                }
                Origin::Walk { parent, .. } => {
                    let from = &self.arena[parent as usize];
                    walk_meters += self
                        .footpaths
                        .from_stop(from.stop)
                        .iter()
                        .find(|transfer| transfer.to == label.stop)
                        .map(|transfer| transfer.meters)
                        .unwrap_or(0.0);
                    at = parent;
                }
            }
        }
        // The initial walk from the origin coordinate to the boarded stop, in
        // door-to-door mode (empty access-meter map otherwise).
        let access_stop = self.arena[at as usize].stop;
        walk_meters += access_meters
            .iter()
            .find(|(stop, _)| *stop == access_stop)
            .map(|(_, meters)| *meters)
            .unwrap_or(0.0);
        let geometry = match (inputs.with_geometry, inputs.leg_geometry) {
            (true, Some(shapes)) => {
                let parts: Vec<Vec<(f64, f64)>> = legs
                    .iter()
                    .rev()
                    .map(|&(trip, board, alight)| shapes.leg_coordinates(trip, board, alight))
                    .collect();
                Some(wkb_multi_line_string(&parts))
            }
            _ => None,
        };
        let fare = match inputs.fares {
            Some(tables) => {
                fare_legs.reverse();
                tables.price(&fare_legs)
            }
            None => f64::NAN,
        };
        CostRow {
            to,
            seconds,
            rides,
            transit_meters,
            walk_meters,
            emission_grams: grams,
            fare,
            geometry,
        }
    }

    /// Walks a destination entry's label chain back into the journey
    /// contract.
    fn assemble(&self, arrived: &Arrived) -> Journey {
        let last = &self.arena[arrived.label as usize];
        let mut legs = vec![Leg::Egress {
            from_stop: last.stop,
            departure: last.arrival,
            arrival: arrived.arrival,
        }];
        let mut at = arrived.label;
        loop {
            let label = &self.arena[at as usize];
            match label.origin {
                Origin::Access => {
                    legs.push(Leg::Access {
                        to_stop: label.stop,
                        departure: arrived.departure,
                        arrival: label.arrival,
                    });
                    break;
                }
                Origin::Ride {
                    parent,
                    trip,
                    board,
                    alight,
                } => {
                    let line = self.view.line_of(trip);
                    let offset = self.view.line_day_offset(line);
                    let stops = self.timetable.pattern_stops(self.view.line_pattern(line));
                    let times = self.view.stored_times(self.timetable, trip);
                    legs.push(Leg::Transit {
                        trip: self.view.backing(trip),
                        board_stop: stops[board as usize],
                        alight_stop: stops[alight as usize],
                        board_position: board,
                        alight_position: alight,
                        board_time: times[board as usize].departure - offset,
                        alight_time: times[alight as usize].arrival - offset,
                    });
                    at = parent;
                }
                Origin::Walk { parent, duration } => {
                    let from = &self.arena[parent as usize];
                    legs.push(Leg::Transfer {
                        from_stop: from.stop,
                        to_stop: label.stop,
                        departure: from.arrival,
                        arrival: from.arrival.saturating_add(duration),
                    });
                    at = parent;
                }
            }
        }
        legs.reverse();
        Journey {
            departure: arrived.departure,
            arrival: arrived.arrival,
            legs,
        }
    }
}

#[cfg(test)]
#[path = "mcraptor_tests.rs"]
mod tests;
