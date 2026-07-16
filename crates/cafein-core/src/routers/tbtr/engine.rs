//! The query engine: construction over a cached or ad-hoc set, and
//! the one-pair, profile, one-to-all, and percentile scans.

use super::*;

/// The TBTR router: builds a day engine per request and queries it.
///
/// Building the engine (the day view and its reduced transfer set) is
/// the expensive part — seconds on a metropolitan feed — so batch
/// workloads should hold a [`TbtrEngine`] per date instead of routing
/// through the trait.
pub struct Tbtr;

impl TransitRouter for Tbtr {
    fn route(
        &self,
        timetable: &Timetable,
        transfers: &Transfers,
        request: &Request,
    ) -> Vec<Journey> {
        TbtrEngine::for_date(
            timetable,
            transfers,
            &request.active_services,
            &request.active_services_previous,
        )
        .query(
            request.departure,
            &request.access,
            &request.egress,
            request.max_transfers,
        )
    }
}

/// A TBTR query engine for one service date: the day view, its reduced
/// transfer set, and the reversed footpaths egress joins need.
pub struct TbtrEngine<'a> {
    pub(super) timetable: &'a Timetable,
    pub(super) footpaths: &'a Transfers,
    pub(super) view: DayView,
    /// Owned when built ad hoc (`for_date`), borrowed when the caller
    /// cached the date's set (`from_set`) — the engine only reads it.
    pub(super) set: std::borrow::Cow<'a, TransferSet>,
    /// Reversed footpath adjacency: for each stop, the `(from, seconds)`
    /// edges walking *into* it — the one-hop closure behind egress
    /// stops, mirroring RAPTOR's post-round transfer relaxation.
    pub(super) incoming_offsets: Vec<u32>,
    pub(super) incoming: Vec<(StopIdx, u32)>,
}

/// Dense per-stop scratch for the stops a round's footpaths improved,
/// drained in first-improvement insertion order. A `HashMap` here would
/// board walked stops in random per-process order: arrivals would not
/// change (strict-improvement labels are order-independent), but at an
/// exact arrival tie the first-enqueued segment wins the label, so
/// journey reconstruction would elect a different equal journey from
/// process to process.
pub(super) struct WalkedScratch {
    /// `(ready, parent segment, alight position)` per stop;
    /// `ready == UNREACHED` marks an empty slot.
    pub(super) slots: Vec<(u32, u32, u16)>,
    pub(super) touched: Vec<u32>,
}

impl WalkedScratch {
    pub(super) fn new(stop_count: usize) -> WalkedScratch {
        WalkedScratch {
            slots: vec![(UNREACHED, 0, 0); stop_count],
            touched: Vec::new(),
        }
    }

    pub(super) fn clear(&mut self) {
        for &stop in &self.touched {
            self.slots[stop as usize].0 = UNREACHED;
        }
        self.touched.clear();
    }

    /// Records (or overwrites, keeping the stop's original insertion
    /// rank) the walk that improved `stop`, exactly as the map insert
    /// this replaces did.
    pub(super) fn insert(&mut self, stop: u32, value: (u32, u32, u16)) {
        let slot = &mut self.slots[stop as usize];
        if slot.0 == UNREACHED {
            self.touched.push(stop);
        }
        *slot = value;
    }

    pub(super) fn iter(&self) -> impl Iterator<Item = (u32, (u32, u32, u16))> + '_ {
        self.touched
            .iter()
            .map(move |&stop| (stop, self.slots[stop as usize]))
    }
}

/// A queued trip segment: `trip` boarded at `board`, reached through
/// `origin`.
pub(super) struct Segment {
    pub(super) trip: ViewTrip,
    pub(super) board: u16,
    pub(super) origin: SegmentOrigin,
}

pub(super) enum SegmentOrigin {
    /// Seeded from the origin: the access stop, its walk, and the
    /// pass departure that seeded it (the canonical chain root).
    Access {
        stop: StopIdx,
        seconds: u32,
        departure: u32,
    },
    /// Reached by a transfer leaving `parent` at `alight`.
    Transfer { parent: u32, alight: u16 },
}

/// A way to reach the destination from a line: alight at `position`,
/// then (optionally) walk the `via` footpath, then the egress walk.
pub(super) struct Target {
    pub(super) position: u16,
    /// Alight-to-destination seconds: footpath (if any) plus egress.
    pub(super) total: u32,
    /// The footpath hop to the egress stop, when the alight stop is
    /// not itself the egress stop: `(egress stop, footpath seconds)`.
    pub(super) via: Option<(StopIdx, u32)>,
    /// Seconds of the final egress walk.
    pub(super) egress_seconds: u32,
}

/// The profile scan's at-most-round label improvement, shared by the
/// matrix pass: a suffix write over the non-increasing round axis.
pub(super) fn improve_labels(
    labels: &mut [u32],
    rounds: usize,
    stop: StopIdx,
    time: u32,
    round: usize,
) -> bool {
    let base = stop.0 as usize * rounds;
    if time >= labels[base + round] {
        return false;
    }
    for slot in &mut labels[base + round..base + rounds] {
        if time < *slot {
            *slot = time;
        } else {
            break;
        }
    }
    true
}

impl<'a> TbtrEngine<'a> {
    /// Builds the engine for one service date.
    ///
    /// The precomputed transfer set covers same-stop transfers only:
    /// the transitively closed footpath set is quadratic in dense
    /// areas, far beyond what generation and reduction can enumerate.
    /// Footpaths join at query time instead — per-stop arrival labels
    /// relax them RAPTOR-style, and boardings are searched from the
    /// stops a walk improved — so the two engines see exactly the same
    /// journeys while the timetable side keeps TBTR's precomputed
    /// pruning.
    pub fn for_date(
        timetable: &'a Timetable,
        footpaths: &'a Transfers,
        active_services: &[bool],
        active_services_previous: &[bool],
    ) -> TbtrEngine<'a> {
        let view = DayView::for_date(timetable, active_services, active_services_previous);
        let same_stop = Transfers::empty(timetable.stop_count());
        let set = TransferSet::for_view(&view, timetable, &same_stop).transfers;
        Self::build_engine(timetable, footpaths, view, std::borrow::Cow::Owned(set))
    }

    /// The engine over a **prebuilt** transfer set — the reused path when the
    /// caller cached the date's set (via [`TransferSet::for_view`]), skipping
    /// the expensive precompute. The set must have been built for these
    /// `active_services`; only the cheap per-engine state (the day view and the
    /// reversed footpath adjacency) is rebuilt.
    pub fn from_set(
        timetable: &'a Timetable,
        footpaths: &'a Transfers,
        active_services: &[bool],
        active_services_previous: &[bool],
        set: &'a TransferSet,
    ) -> TbtrEngine<'a> {
        let view = DayView::for_date(timetable, active_services, active_services_previous);
        Self::build_engine(timetable, footpaths, view, std::borrow::Cow::Borrowed(set))
    }

    fn build_engine(
        timetable: &'a Timetable,
        footpaths: &'a Transfers,
        view: DayView,
        set: std::borrow::Cow<'a, TransferSet>,
    ) -> TbtrEngine<'a> {
        // Reverse the footpath adjacency once per engine.
        let stop_count = timetable.stop_count() as usize;
        let mut counts = vec![0u32; stop_count + 1];
        for stop in (0..timetable.stop_count()).map(StopIdx) {
            for footpath in footpaths.from_stop(stop) {
                counts[footpath.to.0 as usize + 1] += 1;
            }
        }
        for index in 1..counts.len() {
            counts[index] += counts[index - 1];
        }
        let mut incoming = vec![(StopIdx(0), 0u32); *counts.last().unwrap() as usize];
        let mut cursors = counts.clone();
        for stop in (0..timetable.stop_count()).map(StopIdx) {
            for footpath in footpaths.from_stop(stop) {
                let slot = &mut cursors[footpath.to.0 as usize];
                incoming[*slot as usize] = (stop, footpath.duration);
                *slot += 1;
            }
        }
        TbtrEngine {
            timetable,
            footpaths,
            view,
            set,
            incoming_offsets: counts,
            incoming,
        }
    }

    /// Builds only the transfer set for a date — the cacheable precompute that
    /// [`from_set`](Self::from_set) later reuses.
    pub fn transfers_for_date(
        timetable: &Timetable,
        active_services: &[bool],
        active_services_previous: &[bool],
    ) -> TransferSet {
        let view = DayView::for_date(timetable, active_services, active_services_previous);
        let same_stop = Transfers::empty(timetable.stop_count());
        TransferSet::for_view(&view, timetable, &same_stop).transfers
    }

    fn incoming(&self, stop: StopIdx) -> &[(StopIdx, u32)] {
        let start = self.incoming_offsets[stop.0 as usize] as usize;
        let end = self.incoming_offsets[stop.0 as usize + 1] as usize;
        &self.incoming[start..end]
    }

    /// The Pareto set over (arrival, rides) for one departure — the
    /// same contract [`TransitRouter`] documents.
    pub fn query(
        &self,
        departure: u32,
        access: &[(StopIdx, u32)],
        egress: &[(StopIdx, u32)],
        max_transfers: u8,
    ) -> Vec<Journey> {
        self.profile(&[departure], access, egress, max_transfers)
    }

    /// The range counterpart of [`Raptor::route_range`]: the journeys
    /// of the request's departure window no later departure dominates,
    /// enumerated over the same candidate departures.
    pub fn route_range(&self, request: &Request, window: u32) -> Vec<Journey> {
        let departures = crate::raptor::departure_candidates(self.timetable, request, window);
        self.profile(
            &departures,
            &request.access,
            &request.egress,
            request.max_transfers,
        )
    }

    /// One rTBTR pass per departure — which must be strictly
    /// decreasing — on shared state: the reached horizons persist, so
    /// each pass explores only what its departure improves, and the
    /// per-round thresholds suppress journeys dominated by a later
    /// departure. Journeys come back sorted by (departure, rides).
    pub fn profile(
        &self,
        departures: &[u32],
        access: &[(StopIdx, u32)],
        egress: &[(StopIdx, u32)],
        max_transfers: u8,
    ) -> Vec<Journey> {
        // Per-line ways to reach the destination: alight positions with
        // their walks, direct or over one incoming footpath — the same
        // one-hop closure RAPTOR reaches with its post-round transfer
        // relaxation before the egress join.
        let mut targets: HashMap<u32, Vec<Target>> = HashMap::new();
        let mut add_targets = |at: StopIdx, total: u32, via, egress_seconds| {
            for served in self.timetable.patterns_at_stop(at) {
                for line in self
                    .view
                    .lines_of_pattern(served.pattern)
                    .into_iter()
                    .flatten()
                {
                    if served.position == 0 {
                        // An alight at the first position boards nothing.
                        continue;
                    }
                    targets.entry(line).or_default().push(Target {
                        position: served.position,
                        total,
                        via,
                        egress_seconds,
                    });
                }
            }
        };
        for &(stop, seconds) in egress {
            add_targets(stop, seconds, None, seconds);
            for &(from, walk) in self.incoming(stop) {
                add_targets(
                    from,
                    walk.saturating_add(seconds),
                    Some((stop, walk)),
                    seconds,
                );
            }
        }

        let rounds = max_transfers as usize + 1;
        let mut reached = self.horizons(rounds);
        let mut arena: Vec<Segment> = Vec::new();
        let mut queues: Vec<Vec<(u32, u16)>> = vec![Vec::new(); rounds];
        let mut journeys = Vec::new();
        // The best arrival emitted with at most `round` rides so far,
        // across the (descending) departures: a pass may only emit
        // strictly earlier arrivals, so journeys dominated by a later
        // departure never surface.
        let mut thresholds = vec![UNREACHED; rounds];
        // Per-(round, stop) arrival labels back the query-time footpath
        // relaxation; like the reached horizons they persist across the
        // descending departures, and like the horizons they need the
        // round dimension — an earlier departure may reach a stop later
        // in time yet with fewer rides, and boarding from it then is
        // not dominated.
        let stop_count = self.timetable.stop_count() as usize;
        let mut labels = vec![UNREACHED; stop_count * rounds];
        // The label suffix is non-increasing across rounds (each slot is the
        // min over rounds up to it), so a time that does not beat the first
        // slot beats none, and the write loop can stop at the first
        // non-improving slot — the hot footpath scans then cost one read per
        // non-improving target.
        let improve = move |labels: &mut Vec<u32>, stop: StopIdx, time: u32, round: usize| {
            let base = stop.0 as usize * rounds;
            if time >= labels[base + round] {
                return false;
            }
            for slot in &mut labels[base + round..base + rounds] {
                if time < *slot {
                    *slot = time;
                } else {
                    break;
                }
            }
            true
        };
        let mut walked = WalkedScratch::new(self.timetable.stop_count() as usize);

        for &departure in departures {
            // Access stops carry their labels from the start: a ride
            // looping back to one never improves it, so — matching
            // RAPTOR — nothing relaxes onward from such an arrival.
            for &(stop, seconds) in access {
                improve(&mut labels, stop, departure.saturating_add(seconds), 0);
            }
            self.seed(
                departure,
                access,
                &mut reached,
                rounds,
                &mut arena,
                &mut queues[0],
            );
            for round in 0..rounds {
                if queues[round].is_empty() {
                    break;
                }
                let mut round_best: Option<(u32, u32, u16, &Target)> = None;
                let segments = std::mem::take(&mut queues[round]);
                walked.clear();
                for &(segment, end) in &segments {
                    let trip = arena[segment as usize].trip;
                    let board = arena[segment as usize].board;
                    let line = self.view.line_of(trip);
                    let offset = self.view.line_day_offset(line);
                    let stops = self.timetable.pattern_stops(self.view.line_pattern(line));
                    let times = self.view.stored_times(self.timetable, trip);
                    // Alights run past the old horizon inclusively: the
                    // segment that set it boarded there and never
                    // alighted at its own boarding position.
                    let last = (end as usize + 1).min(times.len());
                    // Direct destination joins: alighting at an
                    // egress stop itself. Never better than an earlier
                    // round's join at the same stop, so no label gate
                    // is needed — the thresholds filter duplicates.
                    if let Some(line_targets) = targets.get(&line) {
                        for target in line_targets {
                            if target.via.is_some()
                                || target.position <= board
                                || target.position as usize >= last
                            {
                                continue;
                            }
                            let arrival = (times[target.position as usize].arrival - offset)
                                .saturating_add(target.total);
                            let current = round_best.map_or(thresholds[round], |(at, _, _, _)| {
                                at.min(thresholds[round])
                            });
                            if arrival < current {
                                round_best = Some((arrival, segment, target.position, target));
                            }
                        }
                    }
                    let expand = round + 1 < rounds;
                    let expansion_bar = if expand {
                        round_best.map_or(thresholds[round + 1], |(at, _, _, _)| {
                            at.min(thresholds[round + 1])
                        })
                    } else {
                        0
                    };
                    for alight in board + 1..last as u16 {
                        let arrival = times[alight as usize].arrival - offset;
                        let stop = stops[alight as usize];
                        // Everything walking onward from this arrival —
                        // via-joins to the destination, footpath
                        // boardings — is gated on the arrival improving
                        // the stop's label, exactly like RAPTOR's
                        // marked-stop transfer relaxation. An arrival a
                        // ride looping back cannot beat never improves.
                        let improved = improve(&mut labels, stop, arrival, round);
                        if improved {
                            if let Some(line_targets) = targets.get(&line) {
                                for target in line_targets {
                                    if target.via.is_none() || target.position != alight {
                                        continue;
                                    }
                                    let joined = arrival.saturating_add(target.total);
                                    let current = round_best
                                        .map_or(thresholds[round], |(at, _, _, _)| {
                                            at.min(thresholds[round])
                                        });
                                    if joined < current {
                                        round_best = Some((joined, segment, alight, target));
                                    }
                                }
                            }
                            if expand {
                                for footpath in self.footpaths.from_stop(stop) {
                                    let walked_at = arrival.saturating_add(footpath.duration);
                                    if improve(&mut labels, footpath.to, walked_at, round) {
                                        walked.insert(footpath.to.0, (walked_at, segment, alight));
                                    }
                                }
                            }
                        }
                        // Transfers into the next round. A continuation
                        // can still be emitted at any later round; the
                        // weakest bar it must clear is the next round's
                        // threshold (they are non-increasing), tightened
                        // by what this round is about to emit.
                        if expand && arrival < expansion_bar {
                            for transfer in self.set.from_trip_position(trip, alight) {
                                enqueue(
                                    &self.view,
                                    &mut reached,
                                    rounds,
                                    round + 1,
                                    &mut arena,
                                    &mut queues[round + 1],
                                    Segment {
                                        trip: transfer.trip,
                                        board: transfer.position,
                                        origin: SegmentOrigin::Transfer {
                                            parent: segment,
                                            alight,
                                        },
                                    },
                                );
                            }
                        }
                    }
                }
                if round + 1 < rounds {
                    for (stop, (ready, parent, alight)) in walked.iter() {
                        self.board_walked(
                            StopIdx(stop),
                            ready,
                            parent,
                            alight,
                            &mut reached,
                            rounds,
                            round + 1,
                            &mut arena,
                            &mut queues[round + 1],
                        );
                    }
                }
                if let Some((arrival, segment, alight, target)) = round_best {
                    if arrival < thresholds[round] {
                        for threshold in &mut thresholds[round..] {
                            *threshold = (*threshold).min(arrival);
                        }
                        journeys.push(self.assemble(departure, &arena, segment, alight, target));
                    }
                }
            }
        }
        journeys.sort_by_key(|journey| (journey.departure, journey.rides()));
        journeys
    }

    /// Boards every line catchable at `stop` from `ready` — the
    /// query-time counterpart of the precomputed same-stop transfers,
    /// used for the stops a footpath improved.
    #[allow(clippy::too_many_arguments)]
    fn board_walked(
        &self,
        stop: StopIdx,
        ready: u32,
        parent: u32,
        alight: u16,
        reached: &mut [u16],
        rounds: usize,
        round: usize,
        arena: &mut Vec<Segment>,
        queue: &mut Vec<(u32, u16)>,
    ) {
        for served in self.timetable.patterns_at_stop(stop) {
            for line in self
                .view
                .lines_of_pattern(served.pattern)
                .into_iter()
                .flatten()
            {
                let Some(boarded) =
                    earliest_boardable(&self.view, self.timetable, line, served.position, ready)
                else {
                    continue;
                };
                enqueue(
                    &self.view,
                    reached,
                    rounds,
                    round,
                    arena,
                    queue,
                    Segment {
                        trip: boarded,
                        board: served.position,
                        origin: SegmentOrigin::Transfer { parent, alight },
                    },
                );
            }
        }
    }

    /// The earliest arrival at every stop for one departure, with any
    /// number of rides up to the transfer cap — the TBTR counterpart of
    /// [`Raptor::one_to_all`]: access-seeded stops at their walk time,
    /// everything else over rides and one-hop footpaths; unreachable
    /// stops are `None`.
    pub fn one_to_all(
        &self,
        departure: u32,
        access: &[(StopIdx, u32)],
        max_transfers: u8,
    ) -> Vec<Option<u32>> {
        let rounds = max_transfers as usize + 1;
        let mut best = vec![UNREACHED; self.timetable.stop_count() as usize];
        for &(stop, seconds) in access {
            let at = departure.saturating_add(seconds);
            best[stop.0 as usize] = best[stop.0 as usize].min(at);
        }
        let mut reached = self.horizons(rounds);
        let mut arena: Vec<Segment> = Vec::new();
        let mut queues: Vec<Vec<(u32, u16)>> = vec![Vec::new(); rounds];
        let mut walked = WalkedScratch::new(self.timetable.stop_count() as usize);
        self.seed(
            departure,
            access,
            &mut reached,
            rounds,
            &mut arena,
            &mut queues[0],
        );
        for round in 0..rounds {
            if queues[round].is_empty() {
                break;
            }
            let segments = std::mem::take(&mut queues[round]);
            walked.clear();
            for &(segment, end) in &segments {
                let trip = arena[segment as usize].trip;
                let board = arena[segment as usize].board;
                let line = self.view.line_of(trip);
                let offset = self.view.line_day_offset(line);
                let stops = self.timetable.pattern_stops(self.view.line_pattern(line));
                let times = self.view.stored_times(self.timetable, trip);
                let last = (end as usize + 1).min(times.len()) as u16;
                for alight in board + 1..last {
                    let arrival = times[alight as usize].arrival - offset;
                    let stop = stops[alight as usize];
                    // Walks relax only from arrivals that improve the
                    // stop — RAPTOR's marked-stop semantics; a ride
                    // looping back to a better-known stop goes nowhere.
                    let improved = arrival < best[stop.0 as usize];
                    if improved {
                        best[stop.0 as usize] = arrival;
                        for footpath in self.footpaths.from_stop(stop) {
                            let walked_at = arrival.saturating_add(footpath.duration);
                            let slot = &mut best[footpath.to.0 as usize];
                            if walked_at < *slot {
                                *slot = walked_at;
                                walked.insert(footpath.to.0, (walked_at, segment, alight));
                            }
                        }
                    }
                    if round + 1 < rounds {
                        for transfer in self.set.from_trip_position(trip, alight) {
                            enqueue(
                                &self.view,
                                &mut reached,
                                rounds,
                                round + 1,
                                &mut arena,
                                &mut queues[round + 1],
                                Segment {
                                    trip: transfer.trip,
                                    board: transfer.position,
                                    origin: SegmentOrigin::Transfer {
                                        parent: segment,
                                        alight,
                                    },
                                },
                            );
                        }
                    }
                }
            }
            if round + 1 < rounds {
                for (stop, (ready, parent, alight)) in walked.iter() {
                    self.board_walked(
                        StopIdx(stop),
                        ready,
                        parent,
                        alight,
                        &mut reached,
                        rounds,
                        round + 1,
                        &mut arena,
                        &mut queues[round + 1],
                    );
                }
            }
        }
        best.into_iter()
            .map(|arrival| (arrival != UNREACHED).then_some(arrival))
            .collect()
    }

    /// [`TbtrEngine::one_to_all`] fanned out over origins with rayon —
    /// the matrix primitive on the TBTR engine. The engine is shared
    /// read-only and each origin runs independently, so the output is
    /// deterministic regardless of scheduling.
    pub fn one_to_all_many(
        &self,
        departure: u32,
        accesses: &[Vec<(StopIdx, u32)>],
        max_transfers: u8,
    ) -> Vec<Vec<Option<u32>>> {
        accesses
            .par_iter()
            .map(|access| self.one_to_all(departure, access, max_transfers))
            .collect()
    }

    /// Travel-time percentiles over a departure window, per request — the
    /// TBTR counterpart of [`Raptor::percentile_matrix`], fanned out over
    /// requests with rayon. Semantics and output layout match RAPTOR's
    /// exactly: `stop_count × percentiles.len()` nearest-rank travel times
    /// flat by stop, `u32::MAX` for an unreachable percentile.
    pub fn percentile_matrix(
        &self,
        requests: &[Request],
        window: u32,
        percentiles: &[f64],
    ) -> Vec<Vec<u32>> {
        let stop_count = self.timetable.stop_count() as usize;
        requests
            .par_iter()
            .map(|request| {
                let arrivals = self.window_samples(request, window);
                let access_floor = crate::raptor::access_floor(stop_count, request);
                let mut out = Vec::with_capacity(stop_count * percentiles.len());
                let mut samples = vec![0u32; arrivals.len()];
                for stop in 0..stop_count {
                    for (sample, (mark, marked)) in samples.iter_mut().zip(&arrivals) {
                        *sample =
                            crate::raptor::travel_time(marked[stop], *mark, access_floor[stop]);
                    }
                    samples.sort_unstable();
                    for &percentile in percentiles {
                        out.push(crate::raptor::nearest_rank(&samples, percentile));
                    }
                }
                out
            })
            .collect()
    }

    /// Travel-time percentiles from each request to each destination
    /// *point*, joined through the points' egress link tables — the TBTR
    /// counterpart of [`Raptor::percentile_matrix_to_points`], sharing its
    /// propagation (`propagate_point_percentiles`), so the two engines'
    /// door-to-door windowed matrices agree cell for cell.
    pub fn percentile_matrix_to_points(
        &self,
        requests: &[Request],
        egress: &[Vec<(StopIdx, u32, f64)>],
        window: u32,
        percentiles: &[f64],
    ) -> Vec<Vec<u32>> {
        let stop_count = self.timetable.stop_count() as usize;
        requests
            .par_iter()
            .map(|request| {
                let arrivals = self.window_samples(request, window);
                let access_floor = crate::raptor::access_floor(stop_count, request);
                crate::raptor::propagate_point_percentiles(
                    &arrivals,
                    &access_floor,
                    stop_count,
                    egress,
                    percentiles,
                )
            })
            .collect()
    }

    /// For every minute mark within `[departure, departure + window)`,
    /// the per-stop earliest arrival when leaving at or after it — the
    /// TBTR counterpart of RAPTOR's `window_samples`. One descending pass
    /// per minute mark on shared state (persistent reached horizons and
    /// per-(round, stop) `labels`, as in [`profile`](Self::profile), but
    /// with no egress targets so every stop is explored), snapshotting
    /// the labels at each mark. A mark's travel times match
    /// [`one_to_all`](Self::one_to_all) run for that departure once the
    /// access floor is applied — an access stop's raw label is the next
    /// boardable departure, not the mark itself. Marks come back
    /// ascending.
    pub(super) fn window_samples(&self, request: &Request, window: u32) -> Vec<(u32, Vec<u32>)> {
        let rounds = request.max_transfers as usize + 1;
        let stop_count = self.timetable.stop_count() as usize;
        let mut reached = self.horizons(rounds);
        let mut arena: Vec<Segment> = Vec::new();
        let mut queues: Vec<Vec<(u32, u16)>> = vec![Vec::new(); rounds];
        // Per-(round, stop) arrival labels persist across the descending
        // departures; the last-round slot is the earliest arrival over all
        // rounds, so it is what each mark snapshots.
        let mut labels = vec![UNREACHED; stop_count * rounds];
        // The label suffix is non-increasing across rounds (each slot is the
        // min over rounds up to it), so a time that does not beat the first
        // slot beats none, and the write loop can stop at the first
        // non-improving slot — the hot footpath scans then cost one read per
        // non-improving target.
        let improve = move |labels: &mut Vec<u32>, stop: StopIdx, time: u32, round: usize| {
            let base = stop.0 as usize * rounds;
            if time >= labels[base + round] {
                return false;
            }
            for slot in &mut labels[base + round..base + rounds] {
                if time < *slot {
                    *slot = time;
                } else {
                    break;
                }
            }
            true
        };
        let mut walked = WalkedScratch::new(self.timetable.stop_count() as usize);
        let sample_count = (window as u64).div_ceil(60).max(1) as u32;
        let mut samples = Vec::with_capacity(sample_count as usize);
        for step in (0..sample_count).rev() {
            let Some(mark) = request.departure.checked_add(step * 60) else {
                continue;
            };
            // One pass per minute mark, descending, on the shared labels and
            // horizons (range-TBTR). Seeding at `mark` boards the earliest
            // catchable trip per line — exactly `one_to_all(mark)` — so after
            // the pass the labels hold the earliest arrivals for leaving at or
            // after `mark`; per-trip-departure passes in between add nothing
            // to the minute-mark samples.
            {
                let departure = mark;
                for &(stop, seconds) in &request.access {
                    improve(&mut labels, stop, departure.saturating_add(seconds), 0);
                }
                self.seed(
                    departure,
                    &request.access,
                    &mut reached,
                    rounds,
                    &mut arena,
                    &mut queues[0],
                );
                for round in 0..rounds {
                    if queues[round].is_empty() {
                        break;
                    }
                    let segments = std::mem::take(&mut queues[round]);
                    walked.clear();
                    for &(segment, end) in &segments {
                        let trip = arena[segment as usize].trip;
                        let board = arena[segment as usize].board;
                        let line = self.view.line_of(trip);
                        let offset = self.view.line_day_offset(line);
                        let stops = self.timetable.pattern_stops(self.view.line_pattern(line));
                        let times = self.view.stored_times(self.timetable, trip);
                        let last = (end as usize + 1).min(times.len());
                        for alight in board + 1..last as u16 {
                            let arrival = times[alight as usize].arrival - offset;
                            let stop = stops[alight as usize];
                            // Walks relax only from arrivals that improve the
                            // stop's label — RAPTOR's marked-stop semantics.
                            // Unlike `profile` this relaxes at the last round
                            // too, so a stop reachable only by a final walk is
                            // captured, matching `one_to_all`.
                            if improve(&mut labels, stop, arrival, round) {
                                for footpath in self.footpaths.from_stop(stop) {
                                    let walked_at = arrival.saturating_add(footpath.duration);
                                    if improve(&mut labels, footpath.to, walked_at, round) {
                                        walked.insert(footpath.to.0, (walked_at, segment, alight));
                                    }
                                }
                            }
                            if round + 1 < rounds {
                                for transfer in self.set.from_trip_position(trip, alight) {
                                    enqueue(
                                        &self.view,
                                        &mut reached,
                                        rounds,
                                        round + 1,
                                        &mut arena,
                                        &mut queues[round + 1],
                                        Segment {
                                            trip: transfer.trip,
                                            board: transfer.position,
                                            origin: SegmentOrigin::Transfer {
                                                parent: segment,
                                                alight,
                                            },
                                        },
                                    );
                                }
                            }
                        }
                    }
                    if round + 1 < rounds {
                        for (stop, (ready, parent, alight)) in walked.iter() {
                            self.board_walked(
                                StopIdx(stop),
                                ready,
                                parent,
                                alight,
                                &mut reached,
                                rounds,
                                round + 1,
                                &mut arena,
                                &mut queues[round + 1],
                            );
                        }
                    }
                }
            }
            let snapshot = (0..stop_count)
                .map(|stop| labels[stop * rounds + (rounds - 1)])
                .collect();
            samples.push((mark, snapshot));
        }
        samples.reverse();
        samples
    }

    /// Fresh per-(trip, round) reached horizons: a trip's is its
    /// pattern length.
    pub(super) fn horizons(&self, rounds: usize) -> Vec<u16> {
        let mut horizons = Vec::with_capacity(self.view.trip_count() as usize * rounds);
        for trip in 0..self.view.trip_count() {
            let line = self.view.line_of(ViewTrip(trip));
            let length = self
                .timetable
                .pattern_stops(self.view.line_pattern(line))
                .len() as u16;
            horizons.extend(std::iter::repeat_n(length, rounds));
        }
        horizons
    }

    /// Seeds round 0 from the access stops for one departure.
    #[allow(clippy::too_many_arguments)]
    fn seed(
        &self,
        departure: u32,
        access: &[(StopIdx, u32)],
        reached: &mut [u16],
        rounds: usize,
        arena: &mut Vec<Segment>,
        queue: &mut Vec<(u32, u16)>,
    ) {
        for &(stop, seconds) in access {
            let ready = departure.saturating_add(seconds);
            for served in self.timetable.patterns_at_stop(stop) {
                for line in self
                    .view
                    .lines_of_pattern(served.pattern)
                    .into_iter()
                    .flatten()
                {
                    let Some(boarded) = earliest_boardable(
                        &self.view,
                        self.timetable,
                        line,
                        served.position,
                        ready,
                    ) else {
                        continue;
                    };
                    enqueue(
                        &self.view,
                        reached,
                        rounds,
                        0,
                        arena,
                        queue,
                        Segment {
                            trip: boarded,
                            board: served.position,
                            origin: SegmentOrigin::Access {
                                stop,
                                seconds,
                                departure,
                            },
                        },
                    );
                }
            }
        }
    }

    /// Walks a winning segment chain back into the journey contract.
    fn assemble(
        &self,
        departure: u32,
        arena: &[Segment],
        leaf: u32,
        alight: u16,
        target: &Target,
    ) -> Journey {
        let mut legs = Vec::new();
        let mut segment = &arena[leaf as usize];
        let mut alight_position = alight;
        // The egress (and its footpath hop, if any) come first — legs
        // are assembled back to front.
        let times = self.view.stored_times(self.timetable, segment.trip);
        let offset = self.view.day_offset(segment.trip);
        let alight_arrival = times[alight_position as usize].arrival - offset;
        let alight_stop = self
            .timetable
            .pattern_stops(self.view.line_pattern(self.view.line_of(segment.trip)))
            [alight_position as usize];
        match target.via {
            Some((stop, walk)) => {
                let reached = alight_arrival.saturating_add(walk);
                legs.push(Leg::Egress {
                    from_stop: stop,
                    departure: reached,
                    arrival: reached.saturating_add(target.egress_seconds),
                });
                legs.push(Leg::Transfer {
                    from_stop: alight_stop,
                    to_stop: stop,
                    departure: alight_arrival,
                    arrival: reached,
                });
            }
            None => {
                legs.push(Leg::Egress {
                    from_stop: alight_stop,
                    departure: alight_arrival,
                    arrival: alight_arrival.saturating_add(target.egress_seconds),
                });
            }
        }
        loop {
            let trip = segment.trip;
            let line = self.view.line_of(trip);
            let offset = self.view.line_day_offset(line);
            let stops = self.timetable.pattern_stops(self.view.line_pattern(line));
            let times = self.view.stored_times(self.timetable, trip);
            let board_stop = stops[segment.board as usize];
            legs.push(Leg::Transit {
                trip: self.view.backing(trip),
                board_stop,
                alight_stop: stops[alight_position as usize],
                board_position: segment.board,
                alight_position,
                board_time: times[segment.board as usize].departure - offset,
                alight_time: times[alight_position as usize].arrival - offset,
            });
            match segment.origin {
                SegmentOrigin::Access { stop, seconds, .. } => {
                    if stop != board_stop {
                        unreachable!("access seeds board at their own stop");
                    }
                    legs.push(Leg::Access {
                        to_stop: stop,
                        departure,
                        arrival: departure.saturating_add(seconds),
                    });
                    break;
                }
                SegmentOrigin::Transfer { parent, alight } => {
                    let parent_segment = &arena[parent as usize];
                    let parent_line = self.view.line_of(parent_segment.trip);
                    let parent_stops = self
                        .timetable
                        .pattern_stops(self.view.line_pattern(parent_line));
                    let parent_stop = parent_stops[alight as usize];
                    if parent_stop != board_stop {
                        let parent_times =
                            self.view.stored_times(self.timetable, parent_segment.trip);
                        let left = parent_times[alight as usize].arrival
                            - self.view.line_day_offset(parent_line);
                        let duration = self
                            .footpaths
                            .from_stop(parent_stop)
                            .iter()
                            .find(|footpath| footpath.to == board_stop)
                            .map(|footpath| footpath.duration)
                            .unwrap_or(0);
                        legs.push(Leg::Transfer {
                            from_stop: parent_stop,
                            to_stop: board_stop,
                            departure: left,
                            arrival: left.saturating_add(duration),
                        });
                    }
                    alight_position = alight;
                    segment = parent_segment;
                }
            }
        }
        legs.reverse();
        let arrival = match legs.last() {
            Some(Leg::Egress { arrival, .. }) => *arrival,
            _ => unreachable!("journeys end with an egress leg"),
        };
        Journey {
            departure,
            arrival,
            legs,
        }
    }
}
