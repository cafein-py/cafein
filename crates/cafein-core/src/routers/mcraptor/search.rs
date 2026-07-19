//! The multicriteria scan: per-round line scans, footpath
//! relaxation, and journey assembly.

use super::*;

/// One pending footpath source point: strict (arrival, key) with the
/// exact-grams refinement, linked per source stop in `rode` order.
#[derive(Clone, Copy)]
pub(super) struct FootpathPoint {
    pub(super) label: u32,
    pub(super) arrival: u32,
    pub(super) grams: f64,
    pub(super) key: i64,
    pub(super) next: u32,
    pub(super) live: bool,
}

/// A live point compacted for relaxation, without the list link.
#[derive(Clone, Copy)]
pub(super) struct ActiveFootpathPoint {
    pub(super) label: u32,
    pub(super) arrival: u32,
    pub(super) grams: f64,
    pub(super) key: i64,
}

/// The round's edge-major footpath batch (strict searches only): the
/// round's `rode` labels queue per source stop under the strict
/// batch-local gate, then each footpath edge is loaded once per
/// touched source and relaxed against the source's compacted live
/// points. A batch never spans two rounds, two departure passes, two
/// origin requests, or walked sources — only ride labels enter.
pub(super) struct FootpathBatch {
    pub(super) heads: Box<[u32]>,
    pub(super) tails: Box<[u32]>,
    pub(super) touched: Vec<StopIdx>,
    pub(super) points: Vec<FootpathPoint>,
    pub(super) active: Vec<ActiveFootpathPoint>,
    pub(super) rejected: u64,
    pub(super) evicted: u64,
}

impl FootpathBatch {
    pub(super) fn new(stop_count: usize) -> FootpathBatch {
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
    pub(super) fn offer(
        &mut self,
        source: StopIdx,
        label: u32,
        arrival: u32,
        key: i64,
        grams: f64,
    ) {
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

    pub(super) fn reset(&mut self) {
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
pub(super) struct Arrived {
    pub(super) departure: u32,
    pub(super) arrival: u32,
    /// Accumulated route-penalty seconds; the frontier dominates on the
    /// effective arrival while the journey reports the true `arrival`.
    pub(super) penalty: u32,
    pub(super) key: i64,
    pub(super) grams: f64,
    pub(super) label: u32,
}

impl Arrived {
    /// The penalized arrival the frontier dominates on: true `arrival`
    /// plus accumulated route penalty (equal to `arrival` without one).
    pub(super) fn effective(&self) -> u32 {
        self.arrival.saturating_add(self.penalty)
    }
}

/// The destination bag: Pareto over (departure descending, arrival,
/// grams bucket). Passes run at descending departures, so entries from
/// later departures are never evicted by earlier ones.
#[derive(Debug, Default)]
pub(super) struct DestinationBag {
    pub(super) entries: Vec<Arrived>,
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
pub(super) struct Rider {
    pub(super) trip: ViewTrip,
    pub(super) board: u16,
    pub(super) kappa: f64,
    pub(super) grams: f64,
    pub(super) factor: f64,
    pub(super) parent: u32,
    pub(super) departure: u32,
    /// The boarding label's accumulated route penalty; a ride adds this
    /// line's route penalty on top when it alights.
    pub(super) penalty: u32,
}

/// Per-destination fold state for the emissions matrix: the cleanest
/// (then fastest) label seen per destination, folded at label creation
/// so cross-pass bag evictions cannot lose a budget-qualifying
/// candidate. Mirrors the interim fold's rules: lower grams win, ties
/// resolve toward the shorter travel time, a travel-time budget
/// disqualifies outright.
pub(super) struct MatrixFold<'a> {
    /// Per stop: destination slot + 1, or 0 when not a destination.
    pub(super) slots: &'a [u32],
    /// Per stop: the destinations within a final walk of it —
    /// ``(destination slot, walk seconds, walk meters)``. Empty unless a
    /// street egress is folded (the door-to-door emissions matrix), in which
    /// case a label alighting at the stop also credits each of those
    /// destinations with the final walk added.
    pub(super) egress: &'a [Vec<(u32, u32, f64)>],
    /// Whether a street egress is folded (door-to-door mode). When set, every
    /// located destination carries its own egress self-entry — its connector to
    /// its coordinate — so the zero-walk direct credit below is left to that
    /// entry and skipped, matching the single-pair door-to-door route's arrival
    /// at the coordinate.
    pub(super) egress_active: bool,
    pub(super) budget: Option<u32>,
    /// Per slot: (grams, seconds, label, egress meters).
    pub(super) best: &'a mut [Option<(f64, u32, u32, f64)>],
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
pub(super) struct FrontierFold<'a> {
    /// Per stop: destination slot + 1, or 0 when not a destination.
    pub(super) slots: &'a [u32],
    /// Per stop: `(destination slot, walk seconds, walk meters)` final
    /// egress; consulted only in door-to-door mode.
    pub(super) egress: &'a [Vec<(u32, u32, f64)>],
    pub(super) egress_active: bool,
    /// Per slot: the destination bag the cell's journeys assemble from.
    pub(super) bags: &'a mut [DestinationBag],
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

pub(super) struct Search<'a> {
    pub(super) view: &'a DayView,
    pub(super) timetable: &'a Timetable,
    pub(super) footpaths: &'a Transfers,
    pub(super) geometry: &'a TripGeometry,
    pub(super) factors: &'a [f64],
    pub(super) bucket: f64,
    /// The time-slack band, in seconds; 0 is the strict frontier.
    pub(super) slack: u32,
    /// Per route index: seconds added to a ride's effective arrival for
    /// using it (soft-penalty diverse), `u32::MAX` to skip the route's
    /// lines outright (a hard ban), 0 to leave it free. Empty means none.
    pub(super) route_penalties: &'a [u32],
    pub(super) exclusions: Option<&'a Exclusions>,
    pub(super) arena: Vec<Label>,
    pub(super) bags: Vec<Bag>,
    pub(super) destination: DestinationBag,
    /// Whether labels prune against the destination bag (target
    /// pruning) — set per pass, only when a single destination is
    /// routed (a non-empty `request.egress`); the matrix and batched
    /// folds have no single target.
    pub(super) prune_target: bool,
    /// The `max_slower` per-stop arrival cutoffs of the current pass —
    /// the plain resolved-trip bound plus the band, floored at the
    /// pass's destination bound so the fastest journey always survives.
    /// Empty when the restriction is off; `max_slower` is pareto-only,
    /// so no route penalties exist and true arrivals compare.
    pub(super) cutoff: Vec<u32>,
    /// Per line: the (position, label) boardings queued this round.
    pub(super) queue: Vec<Vec<(u16, u32)>>,
    pub(super) touched: Vec<u32>,
    /// A strict search (zero slack, no route penalties): stop-bag
    /// admissions dispatch to the self-organising strict insert.
    pub(super) strict_bags: bool,
    /// R0 attribution state: counters always fill (plain adds); the
    /// per-operation probes and audits run only under the
    /// diagnostic environment flags read once at `start`.
    pub(super) stats: Box<McRaptorStats>,
    pub(super) ops: bool,
    /// The edge-major footpath batch (strict searches only); general
    /// searches keep the label-major loop.
    pub(super) batch: Option<FootpathBatch>,
}

impl<'a> Search<'a> {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn start(
        view: &'a DayView,
        timetable: &'a Timetable,
        footpaths: &'a Transfers,
        geometry: &'a TripGeometry,
        factors: &'a [f64],
        bucket: f64,
        slack: u32,
        route_penalties: &'a [u32],
        exclusions: Option<&'a Exclusions>,
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
            exclusions,
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
            batch: None,
        };
        if search.strict_bags {
            search.batch = Some(FootpathBatch::new(timetable.stop_count() as usize));
        }
        search
    }

    /// Installs the pass's `max_slower` cutoffs: per stop the plain
    /// bound plus the band, floored at the pass's destination bound
    /// (`floor`); unreachable stops never prune.
    pub(super) fn set_cutoff(&mut self, bounds: &[u32], floor: u32, band: u32) {
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
    pub(super) fn pass(
        &mut self,
        request: &Request,
        departure: u32,
        fold: &mut Option<MatrixFold<'_>>,
        frontier: &mut Option<FrontierFold<'_>>,
    ) {
        // Single-target pruning is sound only for the one-pair query: a
        // fold serves many destination cells, and a pruned label may be
        // another cell's winner.
        self.prune_target = !request.egress.is_empty() && fold.is_none() && frontier.is_none();
        self.stats.departure_passes += 1;
        let mut fresh: Vec<u32> = Vec::new();
        let phase = std::time::Instant::now();
        for &(stop, seconds) in &request.access {
            let arrival = departure.saturating_add(seconds);
            let label = self.arena.len() as u32;
            let key = self.key(0.0);
            self.stats.access_offers += 1;
            if self.stop_excluded(stop)
                || self.beyond_cutoff(stop, arrival)
                || self.target_pruned(departure, arrival, key, 0.0)
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
        // The bags store ride counts as `u8`, so 255 rides (254
        // transfers) is the representable cap; beyond it the count would
        // wrap and evict labels as zero-ride candidates.
        for round in 1..=request.max_transfers.min(254) as u32 + 1 {
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
                    if self.stop_excluded(stop) {
                        continue;
                    }
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
                    if self.stop_excluded(footpath.to) || self.beyond_cutoff(footpath.to, arrival) {
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
    /// plus the probe-depth histograms.
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
                    if self.stop_excluded(footpath.to) || self.beyond_cutoff(footpath.to, arrival) {
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
                if self.stop_excluded(footpath.to) || self.beyond_cutoff(footpath.to, arrival) {
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
    /// through the probed insert, plus the probe-depth histograms.
    /// Runs only under the ops flag.
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
                if self.stop_excluded(footpath.to) || self.beyond_cutoff(footpath.to, arrival) {
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
    /// Whether the exclusions refuse this stop for boarding, alighting,
    /// transfers, and access/egress (riding through stays allowed).
    fn stop_excluded(&self, stop: StopIdx) -> bool {
        self.exclusions
            .is_some_and(|excluded| excluded.excludes_stop(stop))
    }

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
        let route_excluded = self
            .exclusions
            .is_some_and(|excluded| excluded.excludes_route(self.timetable.pattern_route(pattern)));
        if line_penalty == u32::MAX || route_excluded {
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
                if self.stop_excluded(stops[position])
                    || self.beyond_cutoff(stops[position], arrival)
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
                    if !factor.is_finite()
                        || self
                            .exclusions
                            .is_some_and(|excluded| excluded.excludes_trip(self.view.backing(trip)))
                    {
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
    pub(super) fn cost_row(
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
    pub(super) fn assemble(&self, arrived: &Arrived) -> Journey {
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
