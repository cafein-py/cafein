//! The query engine: construction, one-pair and range profiles,
//! and the shared scan passes.

use super::*;

/// The multicriteria trip-based engine for one query date: the day
/// view, the dominance-aware transfer set, and per-trip factors.
/// Queries return the same journeys McRAPTOR's would — verified
/// against it and the exhaustive oracle — via segment scanning.
pub struct McTbtrEngine<'a> {
    pub(super) timetable: &'a Timetable,
    pub(super) footpaths: &'a Transfers,
    pub(super) geometry: &'a TripGeometry,
    pub(super) factors: &'a [f64],
    pub(super) view: DayView,
    pub(super) set: std::borrow::Cow<'a, TransferSet>,
    pub(super) chains: CleanerChains,
}

impl<'a> McTbtrEngine<'a> {
    /// The date's multicriteria transfer set alone — what a caller
    /// caches to skip the expensive precompute on later engines
    /// (`from_set`). Keyed by the date's view and the factors; the
    /// query-time footpaths never enter the precompute, so one set
    /// serves every footpath choice.
    pub fn transfers_for_date(
        timetable: &Timetable,
        geometry: &TripGeometry,
        factors: &[f64],
        active_services: &[bool],
        active_services_previous: &[bool],
    ) -> TransferSet {
        let view = DayView::for_date(timetable, active_services, active_services_previous);
        transfer_set(&view, timetable, geometry, factors).transfers
    }

    pub fn for_date(
        timetable: &'a Timetable,
        footpaths: &'a Transfers,
        geometry: &'a TripGeometry,
        factors: &'a [f64],
        active_services: &[bool],
        active_services_previous: &[bool],
    ) -> McTbtrEngine<'a> {
        let set = Self::transfers_for_date(
            timetable,
            geometry,
            factors,
            active_services,
            active_services_previous,
        );
        let view = DayView::for_date(timetable, active_services, active_services_previous);
        let chains = CleanerChains::build(&view, factors);
        McTbtrEngine {
            timetable,
            footpaths,
            geometry,
            factors,
            view,
            set: std::borrow::Cow::Owned(set),
            chains,
        }
    }

    /// The engine over a **prebuilt** multicriteria transfer set — the
    /// reused path when the caller cached the date's set
    /// (`transfers_for_date`), skipping the dominance-aware precompute.
    /// The set must have been built for these `active_services` and
    /// these `factors`; only the cheap per-engine state (the day view)
    /// is rebuilt.
    pub fn from_set(
        timetable: &'a Timetable,
        footpaths: &'a Transfers,
        geometry: &'a TripGeometry,
        factors: &'a [f64],
        active_services: &[bool],
        active_services_previous: &[bool],
        set: &'a TransferSet,
    ) -> McTbtrEngine<'a> {
        let view = DayView::for_date(timetable, active_services, active_services_previous);
        let chains = CleanerChains::build(&view, factors);
        McTbtrEngine {
            timetable,
            footpaths,
            geometry,
            factors,
            view,
            set: std::borrow::Cow::Borrowed(set),
            chains,
        }
    }

    /// The Pareto set over (arrival, emissions bucket) for a single
    /// departure, as full journeys.
    pub fn route(&self, request: &Request, bucket: f64) -> Vec<Journey> {
        self.profile(request, &[request.departure], bucket, &mut None)
    }

    /// The departure-window profile over (departure, arrival,
    /// emissions bucket).
    pub fn route_range(&self, request: &Request, window: u32, bucket: f64) -> Vec<Journey> {
        let departures = departure_candidates(self.timetable, request, window);
        self.profile(request, &departures, bucket, &mut None)
    }

    fn profile(
        &self,
        request: &Request,
        departures: &[u32],
        bucket: f64,
        fold: &mut Option<MatrixSink<'_>>,
    ) -> Vec<Journey> {
        let mut stats = SearchStats::default();
        let (arena, destination) =
            self.passes(request, departures, bucket, fold, &mut None, &mut stats);
        let mut journeys: Vec<Journey> = destination
            .iter()
            .map(|arrived| self.assemble(arrived, &arena))
            .collect();
        journeys.sort_by_key(|journey| (journey.departure, journey.arrival, journey.rides()));
        journeys
    }

    /// The pass loop shared by the journey profile and the matrix:
    /// returns the segment arena and the destination frontier.
    pub(super) fn passes(
        &self,
        request: &Request,
        departures: &[u32],
        bucket: f64,
        fold: &mut Option<MatrixSink<'_>>,
        frontier: &mut Option<FrontierSink<'_>>,
        stats: &mut SearchStats,
    ) -> (Vec<Segment>, Vec<Arrived>) {
        assert!(
            bucket.is_finite() && bucket > 0.0,
            "the emissions bucket must be positive"
        );
        // The trip bags store ride counts as `u8`, so 255 rides (254
        // transfers) is the representable cap.
        let rounds = request.max_transfers.min(254) as usize + 1;
        let key = |grams: f64| (grams / bucket).floor() as i64;
        let mut arena: Vec<Segment> = Vec::new();
        let mut segment_states: Vec<SegmentState> = Vec::new();
        let mut queue: Vec<Vec<u32>> = vec![Vec::new(); rounds + 1];
        let mut trip_bags: Vec<TripBag> =
            vec![TripBag::default(); self.view.trip_count() as usize * rounds];
        let mut stop_bags: Vec<Bag> = vec![Bag::default(); self.timetable.stop_count() as usize];
        let mut closure = ClosureBatch::new(self.timetable.stop_count() as usize);
        // Per-operation bag accounting (single-thread profiling runs).
        let profile_bag_ops = std::env::var_os("CAFEIN_MCTBTR_PROF_OPS").is_some();
        let mut destination: Vec<Arrived> = Vec::new();
        for &departure in departures {
            // Seed: board from every access stop.
            for &(stop, seconds) in &request.access {
                let ready = departure.saturating_add(seconds);
                // rides = 0: the access seed precedes every ride, so
                // it ranks below all round-ranked arrivals — see
                // mcraptor::Bag::insert.
                let admitted = stop_bags[stop.0 as usize].insert(ready, 0.0, key(0.0), 0);
                if !admitted {
                    continue;
                }
                if let Some(sink) = fold {
                    // The zero-ride floor of the origin's own cell.
                    sink.fold(
                        stop,
                        ready.saturating_sub(departure),
                        0.0,
                        ACCESS_LEAF,
                        0,
                        false,
                    );
                }
                self.board(
                    stop,
                    ready,
                    0.0,
                    departure,
                    |_, _| SegOrigin::Access { stop, seconds },
                    0,
                    key,
                    &mut arena,
                    &mut segment_states,
                    &mut trip_bags,
                    &mut queue[1],
                    stats,
                );
            }
            for round in 1..=rounds {
                let queued = std::mem::take(&mut queue[round]);
                // Consume in original queue order, dropping segments a
                // covering admission cancelled before this round.
                let mut segments = Vec::with_capacity(queued.len());
                for index in queued {
                    match segment_states[index as usize] {
                        SegmentState::Pending => {
                            segment_states[index as usize] = SegmentState::Scanned;
                            segments.push(index);
                        }
                        SegmentState::Cancelled => stats.segments_skipped_cancelled += 1,
                        SegmentState::Scanned => {
                            debug_assert!(false, "a segment was queued twice")
                        }
                    }
                }
                // The direct scans first: alights feed the stop bags,
                // every destination, and the closure batch. The
                // edge-major closure sweep then relaxes each touched
                // source's live points (recording admitted footpath
                // boardings), and only then the expansion sweep runs
                // under the round's fully tightened pruning envelope —
                // Baum et al.'s improved trip-based query, on both
                // expansion channels. Only the one-pair profile
                // prunes; see [`PruneEnvelope`].
                let mut walk_boards: Vec<WalkBoard> = Vec::new();
                let mut admitted: Vec<(u32, u16)> = Vec::new();
                stats.segments_scanned_live += segments.len() as u64;
                let scan_started = std::time::Instant::now();
                for &index in &segments {
                    self.scan_direct_alights(
                        index,
                        round,
                        request,
                        key,
                        &arena,
                        &mut stop_bags,
                        &mut destination,
                        fold,
                        frontier,
                        &mut closure,
                        &mut admitted,
                        stats,
                    );
                }
                stats.direct_scan_ns += scan_started.elapsed().as_nanos() as u64;
                let closure_started = std::time::Instant::now();
                self.relax_closure_batch(
                    &mut closure,
                    round,
                    request,
                    key,
                    &arena,
                    &mut stop_bags,
                    &mut destination,
                    fold,
                    frontier,
                    (round < rounds).then_some(&mut walk_boards),
                    profile_bag_ops,
                    stats,
                );
                stats.closure_ns += closure_started.elapsed().as_nanos() as u64;
                if round < rounds {
                    // Only the one-pair profile prunes; the batched
                    // frontier and fold modes never build an envelope.
                    let envelope = if frontier.is_none() && fold.is_none() {
                        PruneEnvelope::build(&destination)
                    } else {
                        PruneEnvelope::none()
                    };
                    let expand_started = std::time::Instant::now();
                    self.expand_admitted(
                        &admitted,
                        round,
                        rounds,
                        key,
                        &envelope,
                        &mut arena,
                        &mut segment_states,
                        &mut trip_bags,
                        &mut queue,
                        stats,
                    );
                    stats.expand_ns += expand_started.elapsed().as_nanos() as u64;
                    let walk_started = std::time::Instant::now();
                    for board in walk_boards {
                        if envelope.prunes(board.reached, key(board.grams)) {
                            continue;
                        }
                        let departure = arena[board.segment as usize].departure;
                        let (parent, alight, duration) =
                            (board.segment, board.alight, board.duration);
                        self.board(
                            board.to,
                            board.reached,
                            board.grams,
                            departure,
                            |_, _| SegOrigin::Walked {
                                parent,
                                alight,
                                duration,
                            },
                            round,
                            key,
                            &mut arena,
                            &mut segment_states,
                            &mut trip_bags,
                            &mut queue[round + 1],
                            stats,
                        );
                    }
                    stats.walk_board_ns += walk_started.elapsed().as_nanos() as u64;
                }
            }
        }
        stats.segments_enqueued += arena.len() as u64;
        #[cfg(debug_assertions)]
        for arrived in &destination {
            // A cancelled segment was never scanned, so it can be
            // neither a destination leaf nor any chain parent.
            let mut cursor = arrived.leaf;
            loop {
                debug_assert!(
                    segment_states[cursor as usize] != SegmentState::Cancelled,
                    "a cancelled segment reached a destination chain"
                );
                match arena[cursor as usize].origin {
                    SegOrigin::Access { .. } => break,
                    SegOrigin::Transfer { parent, .. } | SegOrigin::Walked { parent, .. } => {
                        cursor = parent
                    }
                }
            }
        }
        (arena, destination)
    }

    /// Boards the earliest catchable trip of every line serving `stop`,
    /// plus the strictly-decreasing-factor suffix, admitting through
    /// the (board, κ) bags of `round + 1`... the caller passes the
    /// target round's queue.
    #[allow(clippy::too_many_arguments)]
    fn board(
        &self,
        stop: StopIdx,
        ready: u32,
        grams: f64,
        departure: u32,
        origin: impl Fn(ViewTrip, u16) -> SegOrigin,
        round: usize,
        key: impl Fn(f64) -> i64,
        arena: &mut Vec<Segment>,
        segment_states: &mut Vec<SegmentState>,
        trip_bags: &mut [TripBag],
        queue: &mut Vec<u32>,
        stats: &mut SearchStats,
    ) {
        let rounds_stride = trip_bags.len() / self.view.trip_count().max(1) as usize;
        for served in self.timetable.patterns_at_stop(stop) {
            for line in self
                .view
                .lines_of_pattern(served.pattern)
                .into_iter()
                .flatten()
            {
                let Some(first) =
                    earliest_boardable(&self.view, self.timetable, line, served.position, ready)
                else {
                    continue;
                };
                for rank in self.chains.candidates(first.0) {
                    let trip = ViewTrip(rank);
                    let ridden = absolute_grams(
                        self.geometry,
                        &self.view,
                        self.factors,
                        trip,
                        served.position,
                    );
                    let kappa = quantized(grams - ridden);
                    let slot = rank as usize * rounds_stride + round;
                    let index = arena.len() as u32;
                    let admitted = trip_bags[slot].admits(
                        served.position,
                        kappa,
                        key(kappa),
                        index,
                        |cancelled| {
                            if segment_states[cancelled as usize] == SegmentState::Pending {
                                segment_states[cancelled as usize] = SegmentState::Cancelled;
                                stats.segments_cancelled_pending += 1;
                            }
                        },
                    );
                    if admitted {
                        arena.push(Segment {
                            trip,
                            board: served.position,
                            grams,
                            departure,
                            origin: origin(trip, served.position),
                        });
                        segment_states.push(SegmentState::Pending);
                        queue.push(index);
                    }
                }
            }
        }
    }

    /// The direct scan over one segment: alight everywhere ahead,
    /// feed the stop bags and every direct destination, record the
    /// admitted pairs for the expansion sweep, and queue each
    /// admission with the round's closure batch.
    #[allow(clippy::too_many_arguments)]
    fn scan_direct_alights(
        &self,
        index: u32,
        round: usize,
        request: &Request,
        key: impl Fn(f64) -> i64 + Copy,
        arena: &[Segment],
        stop_bags: &mut [Bag],
        destination: &mut Vec<Arrived>,
        fold: &mut Option<MatrixSink<'_>>,
        frontier: &mut Option<FrontierSink<'_>>,
        closure: &mut ClosureBatch,
        admitted: &mut Vec<(u32, u16)>,
        stats: &mut SearchStats,
    ) {
        let segment = arena[index as usize];
        let trip = segment.trip;
        let line = self.view.line_of(trip);
        let offset = self.view.line_day_offset(line);
        let stops = self.timetable.pattern_stops(self.view.line_pattern(line));
        let times = self.view.stored_times(self.timetable, trip);
        let boarded_from =
            absolute_grams(self.geometry, &self.view, self.factors, trip, segment.board);
        stats.suffix_context_evaluations +=
            (stops.len() - segment.board as usize).saturating_sub(1) as u64;
        for alight in segment.board as usize + 1..stops.len() {
            let arrival = times[alight].arrival - offset;
            let ridden =
                absolute_grams(self.geometry, &self.view, self.factors, trip, alight as u16)
                    - boarded_from;
            let grams = quantized(segment.grams + ridden);
            let bucket_key = key(grams);
            let stop = stops[alight];
            // Alights, direct egress joins, and query-time footpaths are
            // all gated on stop-bag improvement (the T4b semantics) —
            // the same admission McRAPTOR's egress check applies, so a
            // dominated arrival (the origin's own access seed included)
            // never joins a destination. Rank the rides used (this
            // segment's round) so a later-departure pass that reached
            // the stop on more rides cannot suppress a cleaner
            // fewer-rides arrival across the profile — the same
            // cross-pass soundness the McRAPTOR bag needs.
            if !stop_bags[stop.0 as usize].insert(arrival, grams, bucket_key, round as u8) {
                continue;
            }
            admitted.push((index, alight as u16));
            self.join_direct_alight(
                request,
                key,
                segment.departure,
                stop,
                arrival,
                grams,
                index,
                alight as u16,
                destination,
                fold,
                frontier,
            );
            if !self.footpaths.from_stop(stop).is_empty() {
                stats.closure_points_offered += 1;
                closure.offer(stop, index, alight as u16, arrival, grams, bucket_key);
            }
        }
    }

    /// The direct-alight sink joins for a bag-admitted on-trip
    /// arrival: egress destinations, the frontier, and the fold.
    #[allow(clippy::too_many_arguments)]
    fn join_direct_alight(
        &self,
        request: &Request,
        key: impl Fn(f64) -> i64 + Copy,
        departure: u32,
        stop: StopIdx,
        arrival: u32,
        grams: f64,
        index: u32,
        alight: u16,
        destination: &mut Vec<Arrived>,
        fold: &mut Option<MatrixSink<'_>>,
        frontier: &mut Option<FrontierSink<'_>>,
    ) {
        for &(egress, seconds) in &request.egress {
            if egress == stop {
                self.join(
                    destination,
                    key,
                    departure,
                    arrival.saturating_add(seconds),
                    grams,
                    index,
                    alight,
                    None,
                );
            }
        }
        if let Some(sink) = frontier {
            self.frontier_join(
                sink, stop, key, departure, arrival, grams, index, alight, None,
            );
        }
        if let Some(sink) = fold {
            sink.fold(stop, arrival - departure, grams, index, alight, false);
        }
    }

    /// The round's edge-major closure sweep: each touched source's
    /// live points are compacted once, then every footpath edge is
    /// loaded once and relaxed against all of them — the same target
    /// admissions as label-major traversal (the relaxation map is
    /// monotone on every dominance axis, so batch-gate evictions and
    /// target rejections are always covered). On admission the fold,
    /// egress, frontier, and WalkBoard joins run in the label-major
    /// order; walking adds no emissions, so the stored exact grams and
    /// key relax without new floating-point work. Resets the batch.
    #[allow(clippy::too_many_arguments)]
    fn relax_closure_batch(
        &self,
        closure: &mut ClosureBatch,
        round: usize,
        request: &Request,
        key: impl Fn(f64) -> i64 + Copy,
        arena: &[Segment],
        stop_bags: &mut [Bag],
        destination: &mut Vec<Arrived>,
        fold: &mut Option<MatrixSink<'_>>,
        frontier: &mut Option<FrontierSink<'_>>,
        mut walk_boards: Option<&mut Vec<WalkBoard>>,
        profile_bag_ops: bool,
        stats: &mut SearchStats,
    ) {
        for source_index in 0..closure.touched.len() {
            let source = closure.touched[source_index];
            stats.closure_source_batches += 1;
            closure.active.clear();
            let mut cursor = closure.heads[source.0 as usize];
            while cursor != NONE_U32 {
                let point = closure.points[cursor as usize];
                if point.live {
                    closure.active.push(ActiveClosurePoint {
                        grams: point.grams,
                        key: point.key,
                        segment: point.segment,
                        arrival: point.arrival,
                        alight: point.alight,
                    });
                }
                cursor = point.next;
            }
            stats.closure_points_live += closure.active.len() as u64;
            for edge in self.footpaths.from_stop(source) {
                stats.closure_edge_records_loaded += 1;
                stats.closure_label_edge_relaxations += closure.active.len() as u64;
                let (to, duration) = (edge.to, edge.duration);
                for point in &closure.active {
                    let reached = point.arrival.saturating_add(duration);
                    let admitted = if profile_bag_ops {
                        let mut probes = InsertProbes::default();
                        let admitted = stop_bags[to.0 as usize].insert_probed(
                            reached,
                            point.grams,
                            point.key,
                            round as u8,
                            &mut probes,
                        );
                        stats.closure_bag_calls += 1;
                        stats.closure_bag_length_histogram[length_bucket(probes.length)] += 1;
                        if admitted {
                            stats.closure_bag_admit_entries_examined += probes.examined as u64;
                            stats.closure_bag_retain_entries_examined += probes.retained as u64;
                        } else {
                            stats.closure_bag_rejections += 1;
                            stats.closure_bag_reject_entries_examined += probes.examined as u64;
                            if probes.examined == 1 {
                                stats.closure_mtf_front_rejections += 1;
                            } else {
                                stats.closure_mtf_swaps += 1;
                                stats.closure_mtf_swap_distance += (probes.examined - 1) as u64;
                            }
                            stats.closure_bag_reject_depth_histogram
                                [depth_bucket(probes.examined)] += 1;
                        }
                        admitted
                    } else {
                        stop_bags[to.0 as usize].insert(
                            reached,
                            point.grams,
                            point.key,
                            round as u8,
                        )
                    };
                    if !admitted {
                        continue;
                    }
                    stats.closure_target_admissions += 1;
                    let departure = arena[point.segment as usize].departure;
                    if let Some(sink) = fold {
                        sink.fold(
                            to,
                            reached - departure,
                            point.grams,
                            point.segment,
                            point.alight,
                            true,
                        );
                    }
                    for &(egress, seconds) in &request.egress {
                        if egress == to {
                            self.join(
                                destination,
                                key,
                                departure,
                                reached.saturating_add(seconds),
                                point.grams,
                                point.segment,
                                point.alight,
                                Some((to, duration)),
                            );
                        }
                    }
                    if let Some(sink) = frontier {
                        self.frontier_join(
                            sink,
                            to,
                            key,
                            departure,
                            reached,
                            point.grams,
                            point.segment,
                            point.alight,
                            Some((to, duration)),
                        );
                    }
                    if let Some(boards) = walk_boards.as_deref_mut() {
                        stats.walk_boards_recorded += 1;
                        boards.push(WalkBoard {
                            segment: point.segment,
                            alight: point.alight,
                            to,
                            reached,
                            grams: point.grams,
                            duration,
                        });
                    }
                }
            }
        }
        stats.closure_points_batch_evicted += closure.evicted;
        closure.reset();
    }

    /// The expansion sweep, under the round's pruning envelope: the
    /// join sweep's bag-admitted (segment, alight) pairs relax the
    /// precomputed transfers — a dominated alight's transfers are
    /// covered by its dominator's, whose reduction-complete transfer
    /// set reaches every outcome no later, no dirtier, and on no more
    /// rides, with a departure no earlier (passes descend). A pruned
    /// pair ends its segment's expansion outright: arrivals and grams
    /// only grow along the trip while the envelope's bound never rises.
    #[allow(clippy::too_many_arguments)]
    fn expand_admitted(
        &self,
        admitted: &[(u32, u16)],
        round: usize,
        rounds: usize,
        key: impl Fn(f64) -> i64 + Copy,
        envelope: &PruneEnvelope,
        arena: &mut Vec<Segment>,
        segment_states: &mut Vec<SegmentState>,
        trip_bags: &mut [TripBag],
        queue: &mut [Vec<u32>],
        stats: &mut SearchStats,
    ) {
        let mut at = 0;
        while at < admitted.len() {
            let (index, _) = admitted[at];
            let segment = arena[index as usize];
            let trip = segment.trip;
            let line = self.view.line_of(trip);
            let offset = self.view.line_day_offset(line);
            let times = self.view.stored_times(self.timetable, trip);
            let boarded_from =
                absolute_grams(self.geometry, &self.view, self.factors, trip, segment.board);
            while at < admitted.len() && admitted[at].0 == index {
                let alight = admitted[at].1;
                at += 1;
                let arrival = times[alight as usize].arrival - offset;
                let ridden = absolute_grams(self.geometry, &self.view, self.factors, trip, alight)
                    - boarded_from;
                let grams = quantized(segment.grams + ridden);
                if envelope.prunes(arrival, key(grams)) {
                    while at < admitted.len() && admitted[at].0 == index {
                        at += 1;
                    }
                    break;
                }
                for transfer in self.set.from_trip_position(trip, alight) {
                    let boarded = transfer.trip;
                    let ridden = absolute_grams(
                        self.geometry,
                        &self.view,
                        self.factors,
                        boarded,
                        transfer.position,
                    );
                    let kappa = quantized(grams - ridden);
                    let stride = rounds;
                    let slot = boarded.0 as usize * stride + round;
                    let next = arena.len() as u32;
                    let admitted = trip_bags[slot].admits(
                        transfer.position,
                        kappa,
                        key(kappa),
                        next,
                        |cancelled| {
                            if segment_states[cancelled as usize] == SegmentState::Pending {
                                segment_states[cancelled as usize] = SegmentState::Cancelled;
                                stats.segments_cancelled_pending += 1;
                            }
                        },
                    );
                    if admitted {
                        arena.push(Segment {
                            trip: boarded,
                            board: transfer.position,
                            grams,
                            departure: segment.departure,
                            origin: SegOrigin::Transfer {
                                parent: index,
                                alight,
                            },
                        });
                        segment_states.push(SegmentState::Pending);
                        queue[round + 1].push(next);
                    }
                }
            }
        }
    }

    /// Inserts a destination candidate under (departure desc, arrival,
    /// bucket) dominance with equal-slot refinement — McRAPTOR's
    /// destination-bag rules.
    #[allow(clippy::too_many_arguments)]
    fn join(
        &self,
        destination: &mut Vec<Arrived>,
        key: impl Fn(f64) -> i64,
        departure: u32,
        arrival: u32,
        grams: f64,
        leaf: u32,
        alight: u16,
        walk: Option<(StopIdx, u32)>,
    ) {
        let key = key(grams);
        for entry in destination.iter() {
            if entry.departure >= departure
                && entry.arrival <= arrival
                && entry.key <= key
                && !(entry.departure == departure
                    && entry.arrival == arrival
                    && entry.key == key
                    && grams < entry.grams)
            {
                return;
            }
        }
        destination.retain(|entry| {
            !(departure >= entry.departure && arrival <= entry.arrival && key <= entry.key)
        });
        destination.push(Arrived {
            departure,
            arrival,
            key,
            grams,
            leaf,
            alight,
            walk,
        });
    }

    /// Routes an egress join into every destination slot the stop
    /// serves, under the one-pair `join` rules: the slot's own stop in
    /// stop mode (no final walk), the per-stop final-egress map in
    /// door-to-door mode (`arrival` is before the final walk).
    #[allow(clippy::too_many_arguments)]
    fn frontier_join(
        &self,
        sink: &mut FrontierSink<'_>,
        stop: StopIdx,
        key: impl Fn(f64) -> i64 + Copy,
        departure: u32,
        arrival: u32,
        grams: f64,
        leaf: u32,
        alight: u16,
        walk: Option<(StopIdx, u32)>,
    ) {
        if !sink.egress_active {
            let slot = sink.slots[stop.0 as usize];
            if slot != 0 {
                self.join(
                    &mut sink.bags[slot as usize - 1],
                    key,
                    departure,
                    arrival,
                    grams,
                    leaf,
                    alight,
                    walk,
                );
            }
            return;
        }
        for index in 0..sink.egress[stop.0 as usize].len() {
            let (slot, seconds, _) = sink.egress[stop.0 as usize][index];
            self.join(
                &mut sink.bags[slot as usize],
                key,
                departure,
                arrival.saturating_add(seconds),
                grams,
                leaf,
                alight,
                walk,
            );
        }
    }
}
