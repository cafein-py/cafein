//! Trip-Based Transit Routing: the query-day trip universe and the
//! precomputed trip-to-trip transfer set.
//!
//! TBTR (Witt, ESA 2015) replaces RAPTOR's per-stop labels with trip
//! segments linked by precomputed transfers: alight a trip at a
//! position, walk (or stay at the stop), board another trip at a
//! position. Generation keeps, per reachable (line, position), only
//! the earliest boardable trip; a reduction pass then drops transfers
//! that improve no stop's arrival over staying on the trip or over the
//! transfers already kept — typically the large majority — leaving the
//! set the query engine scans.
//!
//! Both passes run over a [`DayView`]: the virtual trips one query
//! date sees. Restricting to a date before the reduction is what keeps
//! the reduction exact — dropped transfers are judged against exactly
//! the trips that run — and it folds the previous service day's
//! over-midnight tails in as *lines of their own*, shifted back a day,
//! so no service check or day arithmetic is left inside the query
//! loop. The all-trips [`DayView::universal`] view serves calendar-free
//! uses (and the whole-feed diagnostics the tests pin).

use std::collections::HashMap;

use rayon::prelude::*;

use crate::journey::{Journey, Leg};
use crate::router::{Request, TransitRouter};
use crate::timetable::{PatternIdx, StopIdx, StopTime, Timetable, TripIdx};
use crate::transfers::Transfers;

const UNREACHED: u32 = u32::MAX;

/// Seconds in a service day: the shift of previous-day lines.
const DAY_SECONDS: u32 = 86_400;

/// A virtual trip of a [`DayView`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ViewTrip(pub u32);

/// The trip universe one query date sees, grouped into FIFO lines.
///
/// A line is a pattern's active trips on one day class: today's trips
/// with their stored times, or the previous day's over-midnight tails
/// with times shifted back a day. Within a line, departures at every
/// position are non-decreasing with rank (a subset of a FIFO chain
/// stays FIFO), so boarding searches stay binary.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct DayView {
    /// Per virtual trip: the backing timetable trip. Virtual trips are
    /// contiguous per line, in line order.
    trips: Vec<TripIdx>,
    /// Per virtual trip: its line.
    trip_lines: Vec<u32>,
    /// Per virtual trip: the first position boardable on the query
    /// day's clock (nonzero only on previous-day tails).
    first_boardable: Vec<u16>,
    /// Per line: the pattern its stops come from.
    line_patterns: Vec<PatternIdx>,
    /// Per line: subtracted from stored times to land on the query
    /// day's clock (0 today, 86 400 for the previous day).
    line_offsets: Vec<u32>,
    /// CSR offsets into `trips`, one per line plus a tail.
    line_trips_offsets: Vec<u32>,
    /// Per pattern: its today and previous-day lines, where active.
    pattern_lines: Vec<[Option<u32>; 2]>,
}

impl DayView {
    /// Every trip, one line per pattern, on the stored clock — the
    /// calendar-free view. Virtual trip indexes equal timetable trip
    /// indexes.
    pub fn universal(timetable: &Timetable) -> DayView {
        DayView::assemble(timetable, |_| Some(0), |_| false)
    }

    /// The trip universe of one query date: trips of the services
    /// active on the date, plus the previous day's active trips that
    /// still have boardable track past midnight, as shifted lines.
    pub fn for_date(
        timetable: &Timetable,
        active_services: &[bool],
        active_services_previous: &[bool],
    ) -> DayView {
        let runs = |mask: &[bool], trip: TripIdx| {
            mask.get(timetable.trip_service(trip) as usize)
                .copied()
                .unwrap_or(false)
        };
        DayView::assemble(
            timetable,
            |trip| {
                if runs(active_services, trip) {
                    Some(0)
                } else {
                    None
                }
            },
            |trip| runs(active_services_previous, trip),
        )
    }

    /// Builds the line structure: per pattern, a today line of the
    /// trips `today` admits, then a previous-day line of the trips
    /// `previous` admits that are still boardable after the shift.
    fn assemble(
        timetable: &Timetable,
        today: impl Fn(TripIdx) -> Option<u32>,
        previous: impl Fn(TripIdx) -> bool,
    ) -> DayView {
        let mut view = DayView {
            trips: Vec::new(),
            trip_lines: Vec::new(),
            first_boardable: Vec::new(),
            line_patterns: Vec::new(),
            line_offsets: Vec::new(),
            line_trips_offsets: vec![0],
            pattern_lines: vec![[None; 2]; timetable.pattern_count() as usize],
        };
        for pattern in (0..timetable.pattern_count()).map(PatternIdx) {
            let stops = timetable.pattern_stops(pattern).len();
            let mut today_line = None;
            let mut previous_line = None;
            let members: Vec<TripIdx> = timetable
                .pattern_trips(pattern)
                .filter(|&trip| today(trip).is_some())
                .collect();
            if !members.is_empty() {
                today_line = Some(view.push_line(pattern, 0, &members, |_| 0));
            }
            let members: Vec<(TripIdx, u16)> = timetable
                .pattern_trips(pattern)
                .filter(|&trip| previous(trip))
                .filter_map(|trip| {
                    let times = timetable.trip_stop_times(trip);
                    let boardable = times.partition_point(|time| time.departure < DAY_SECONDS);
                    // Still boardable with track ahead after the shift.
                    (boardable + 1 < stops).then_some((trip, boardable as u16))
                })
                .collect();
            if !members.is_empty() {
                let boardable: Vec<u16> = members.iter().map(|&(_, at)| at).collect();
                let trips: Vec<TripIdx> = members.into_iter().map(|(trip, _)| trip).collect();
                previous_line =
                    Some(view.push_line(pattern, DAY_SECONDS, &trips, |rank| boardable[rank]));
            }
            view.pattern_lines[pattern.0 as usize] = [today_line, previous_line];
        }
        view
    }

    fn push_line(
        &mut self,
        pattern: PatternIdx,
        offset: u32,
        members: &[TripIdx],
        first_boardable: impl Fn(usize) -> u16,
    ) -> u32 {
        let line = self.line_patterns.len() as u32;
        self.line_patterns.push(pattern);
        self.line_offsets.push(offset);
        for (rank, &trip) in members.iter().enumerate() {
            self.trips.push(trip);
            self.trip_lines.push(line);
            self.first_boardable.push(first_boardable(rank));
        }
        self.line_trips_offsets.push(self.trips.len() as u32);
        line
    }

    /// The number of virtual trips in the view.
    pub fn trip_count(&self) -> u32 {
        self.trips.len() as u32
    }

    pub fn line_count(&self) -> u32 {
        self.line_patterns.len() as u32
    }

    /// The backing timetable trip of a virtual trip.
    pub fn backing(&self, trip: ViewTrip) -> TripIdx {
        self.trips[trip.0 as usize]
    }

    /// Subtracted from the backing trip's stored times to land on the
    /// query day's clock.
    pub fn day_offset(&self, trip: ViewTrip) -> u32 {
        self.line_offsets[self.line_of(trip) as usize]
    }

    pub fn line_of(&self, trip: ViewTrip) -> u32 {
        self.trip_lines[trip.0 as usize]
    }

    pub fn line_pattern(&self, line: u32) -> PatternIdx {
        self.line_patterns[line as usize]
    }

    pub fn line_day_offset(&self, line: u32) -> u32 {
        self.line_offsets[line as usize]
    }

    /// The virtual trips of a line, in FIFO order.
    pub fn line_trips(&self, line: u32) -> std::ops::Range<u32> {
        self.line_trips_offsets[line as usize]..self.line_trips_offsets[line as usize + 1]
    }

    /// The today and previous-day lines of a pattern, where active.
    pub fn lines_of_pattern(&self, pattern: PatternIdx) -> [Option<u32>; 2] {
        self.pattern_lines[pattern.0 as usize]
    }

    /// The first position of a virtual trip boardable on the query
    /// day's clock.
    pub fn first_boardable(&self, trip: ViewTrip) -> u16 {
        self.first_boardable[trip.0 as usize]
    }

    /// The backing trip's stored stop times (shift by the day offset to
    /// reach the query day's clock).
    pub fn stored_times<'t>(&self, timetable: &'t Timetable, trip: ViewTrip) -> &'t [StopTime] {
        timetable.trip_stop_times(self.backing(trip))
    }
}

/// One precomputed transfer: board `trip` at `position` of its pattern.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct TripTransfer {
    pub trip: ViewTrip,
    pub position: u16,
}

/// The reduced trip-to-trip transfer set, in CSR layout keyed by
/// (virtual trip, alight position).
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct TransferSet {
    /// Base slot of each virtual trip's positions plus a tail: alight
    /// position `i` of trip `t` is slot `trip_base[t] + i`.
    trip_base: Vec<u32>,
    /// CSR offsets into `transfers`, one per slot plus a tail.
    offsets: Vec<u32>,
    transfers: Vec<TripTransfer>,
}

/// The outcome of building a [`TransferSet`]: the reduced set and how
/// many feasible transfers generation produced before the reduction.
#[derive(Debug)]
pub struct TransferSetBuild {
    pub transfers: TransferSet,
    pub generated: usize,
}

impl TransferSet {
    /// The calendar-free set over every trip: [`TransferSet::for_view`]
    /// of the universal view.
    pub fn build(timetable: &Timetable, footpaths: &Transfers) -> TransferSetBuild {
        TransferSet::for_view(&DayView::universal(timetable), timetable, footpaths)
    }

    /// Generates and reduces the transfer set of a day view, fanned out
    /// over virtual trips with rayon. Deterministic: each trip's
    /// transfers depend only on the shared inputs.
    pub fn for_view(
        view: &DayView,
        timetable: &Timetable,
        footpaths: &Transfers,
    ) -> TransferSetBuild {
        let per_trip: Vec<(Vec<Vec<TripTransfer>>, usize)> = (0..view.trip_count())
            .into_par_iter()
            .map_init(
                || Labels::new(timetable.stop_count()),
                |labels, trip| {
                    let trip = ViewTrip(trip);
                    let mut generated = generate(view, timetable, footpaths, trip);
                    let count = generated.iter().map(Vec::len).sum();
                    reduce(view, timetable, footpaths, trip, labels, &mut generated);
                    (generated, count)
                },
            )
            .collect();
        TransferSet::assemble(per_trip)
    }

    /// Lays per-trip kept transfers out as the CSR set; shared with the
    /// multicriteria builder.
    pub(crate) fn assemble(per_trip: Vec<(Vec<Vec<TripTransfer>>, usize)>) -> TransferSetBuild {
        let generated = per_trip.iter().map(|(_, count)| count).sum();
        let mut trip_base = Vec::with_capacity(per_trip.len() + 1);
        let mut offsets = Vec::new();
        let mut transfers = Vec::new();
        let mut base = 0u32;
        for (positions, _) in &per_trip {
            trip_base.push(base);
            base += positions.len() as u32;
            for kept in positions {
                offsets.push(transfers.len() as u32);
                transfers.extend_from_slice(kept);
            }
        }
        trip_base.push(base);
        offsets.push(transfers.len() as u32);
        TransferSetBuild {
            transfers: TransferSet {
                trip_base,
                offsets,
                transfers,
            },
            generated,
        }
    }

    /// The transfers available when alighting `trip` at `position`.
    pub fn from_trip_position(&self, trip: ViewTrip, position: u16) -> &[TripTransfer] {
        let slot = (self.trip_base[trip.0 as usize] + position as u32) as usize;
        let start = self.offsets[slot] as usize;
        let end = self.offsets[slot + 1] as usize;
        &self.transfers[start..end]
    }

    /// The number of transfers kept after the reduction.
    pub fn len(&self) -> usize {
        self.transfers.len()
    }

    pub fn is_empty(&self) -> bool {
        self.transfers.is_empty()
    }
}

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
    timetable: &'a Timetable,
    footpaths: &'a Transfers,
    view: DayView,
    set: TransferSet,
    /// Reversed footpath adjacency: for each stop, the `(from, seconds)`
    /// edges walking *into* it — the one-hop closure behind egress
    /// stops, mirroring RAPTOR's post-round transfer relaxation.
    incoming_offsets: Vec<u32>,
    incoming: Vec<(StopIdx, u32)>,
}

/// A queued trip segment: `trip` boarded at `board`, reached through
/// `origin`.
struct Segment {
    trip: ViewTrip,
    board: u16,
    origin: SegmentOrigin,
}

enum SegmentOrigin {
    /// Seeded from the origin: the access stop and its walk.
    Access { stop: StopIdx, seconds: u32 },
    /// Reached by a transfer leaving `parent` at `alight`.
    Transfer { parent: u32, alight: u16 },
}

/// A way to reach the destination from a line: alight at `position`,
/// then (optionally) walk the `via` footpath, then the egress walk.
struct Target {
    position: u16,
    /// Alight-to-destination seconds: footpath (if any) plus egress.
    total: u32,
    /// The footpath hop to the egress stop, when the alight stop is
    /// not itself the egress stop: `(egress stop, footpath seconds)`.
    via: Option<(StopIdx, u32)>,
    /// Seconds of the final egress walk.
    egress_seconds: u32,
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
        let improve = move |labels: &mut Vec<u32>, stop: StopIdx, time: u32, round: usize| {
            let base = stop.0 as usize * rounds;
            let gate = time < labels[base + round];
            for slot in &mut labels[base + round..base + rounds] {
                if time < *slot {
                    *slot = time;
                }
            }
            gate
        };
        let mut walked: HashMap<u32, (u32, u32, u16)> = HashMap::new();

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
                    for (&stop, &(ready, parent, alight)) in &walked {
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
        let mut walked: HashMap<u32, (u32, u32, u16)> = HashMap::new();
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
                for (&stop, &(ready, parent, alight)) in &walked {
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

    /// Fresh per-(trip, round) reached horizons: a trip's is its
    /// pattern length.
    fn horizons(&self, rounds: usize) -> Vec<u16> {
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
                            origin: SegmentOrigin::Access { stop, seconds },
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
                SegmentOrigin::Access { stop, seconds } => {
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

/// Queues a segment when it boards earlier than anything seen on its
/// trip with this many rides or fewer, and marks the trip and its
/// later line siblings reached from this round on: under FIFO,
/// boarding a later sibling at the same or a later position with the
/// same or more rides can never beat this one. The horizons are per
/// (trip, round) — profile passes at earlier departures may re-board a
/// trip already used by a later departure when they do so with fewer
/// rides, and a single per-trip horizon would wrongly suppress them.
fn enqueue(
    view: &DayView,
    reached: &mut [u16],
    rounds: usize,
    round: usize,
    arena: &mut Vec<Segment>,
    queue: &mut Vec<(u32, u16)>,
    segment: Segment,
) {
    let trip = segment.trip;
    let board = segment.board;
    let slot = trip.0 as usize * rounds + round;
    if board >= reached[slot] {
        return;
    }
    queue.push((arena.len() as u32, reached[slot]));
    arena.push(segment);
    let line_end = view.line_trips(view.line_of(trip)).end;
    for later in trip.0..line_end {
        let base = later as usize * rounds;
        for horizon in &mut reached[base + round..base + rounds] {
            *horizon = (*horizon).min(board);
        }
    }
}

/// The feasible transfers of one virtual trip, per alight position: for
/// every stop reachable from the alight stop (itself, or over a
/// footpath), the earliest boardable trip of each (line, position)
/// serving it — skipping same-line transfers that stay on the trip or
/// board a later sibling no earlier along the pattern (they cannot help
/// under FIFO), and U-turns (reboarding the segment just ridden when
/// the boarded trip was already catchable at the previous stop).
fn generate(
    view: &DayView,
    timetable: &Timetable,
    footpaths: &Transfers,
    trip: ViewTrip,
) -> Vec<Vec<TripTransfer>> {
    let line = view.line_of(trip);
    let pattern = view.line_pattern(line);
    let offset = view.line_day_offset(line);
    let stops = timetable.pattern_stops(pattern);
    let times = view.stored_times(timetable, trip);
    let mut per_position: Vec<Vec<TripTransfer>> = vec![Vec::new(); stops.len()];
    let alight_from = view.first_boardable(trip) as usize + 1;
    for (alight, kept) in per_position.iter_mut().enumerate().skip(alight_from) {
        // On the query day's clock; non-negative past the first
        // boardable position.
        let arrival = times[alight].arrival - offset;
        let stop = stops[alight];
        let mut board_from = |at: StopIdx, ready: u32| {
            for served in timetable.patterns_at_stop(at) {
                for candidate_line in view.lines_of_pattern(served.pattern).into_iter().flatten() {
                    let candidate =
                        earliest_boardable(view, timetable, candidate_line, served.position, ready);
                    let Some(boarded) = candidate else { continue };
                    if candidate_line == line
                        && boarded.0 >= trip.0
                        && served.position as usize >= alight
                    {
                        continue;
                    }
                    if u_turn(
                        view,
                        timetable,
                        stops,
                        times,
                        offset,
                        alight,
                        boarded,
                        served.position,
                    ) {
                        continue;
                    }
                    kept.push(TripTransfer {
                        trip: boarded,
                        position: served.position,
                    });
                }
            }
        };
        board_from(stop, arrival);
        for footpath in footpaths.from_stop(stop) {
            board_from(footpath.to, arrival.saturating_add(footpath.duration));
        }
    }
    per_position
}

/// The earliest trip of `line` boardable at `position` no earlier than
/// `ready` on the query day's clock; `None` when none departs in time
/// or the position is the pattern's last (nothing left to ride).
pub(crate) fn earliest_boardable(
    view: &DayView,
    timetable: &Timetable,
    line: u32,
    position: u16,
    ready: u32,
) -> Option<ViewTrip> {
    let pattern = view.line_pattern(line);
    if position as usize + 1 >= timetable.pattern_stops(pattern).len() {
        return None;
    }
    // Compare on the line's stored clock; on previous-day lines this
    // also rules out pre-midnight positions (their stored departures
    // sit below the offset).
    let ready = ready as u64 + view.line_day_offset(line) as u64;
    let departs_before = |trip: u32| {
        (view.stored_times(timetable, ViewTrip(trip))[position as usize].departure as u64) < ready
    };
    let range = view.line_trips(line);
    let (mut low, mut high) = (range.start, range.end);
    while low < high {
        let middle = low + (high - low) / 2;
        if departs_before(middle) {
            low = middle + 1;
        } else {
            high = middle;
        }
    }
    (low < range.end).then_some(ViewTrip(low))
}

/// Whether boarding `boarded` at `position` just rides back over the
/// segment `trip` arrived on, when it was already catchable one stop
/// earlier — the classic redundant U-turn.
#[allow(clippy::too_many_arguments)]
pub(crate) fn u_turn(
    view: &DayView,
    timetable: &Timetable,
    stops: &[StopIdx],
    times: &[StopTime],
    offset: u32,
    alight: usize,
    boarded: ViewTrip,
    position: u16,
) -> bool {
    let boarded_stops = timetable.pattern_stops(view.line_pattern(view.line_of(boarded)));
    let j = position as usize;
    j + 1 < boarded_stops.len()
        && boarded_stops[j] == stops[alight - 1]
        && boarded_stops[j + 1] == stops[alight]
        && times[alight - 1].arrival as i64 - offset as i64
            <= view.stored_times(timetable, boarded)[j].departure as i64
                - view.day_offset(boarded) as i64
}

/// Witt's transfer reduction for one virtual trip: walking the alight
/// positions back to front, a transfer survives only if riding the
/// boarded trip onward improves the arrival at some stop (directly or
/// over a footpath) over staying on the trip or over the transfers
/// already kept. Labels are per-trip scratch state, pooled per worker.
fn reduce(
    view: &DayView,
    timetable: &Timetable,
    footpaths: &Transfers,
    trip: ViewTrip,
    labels: &mut Labels,
    per_position: &mut [Vec<TripTransfer>],
) {
    labels.clear();
    let offset = view.line_day_offset(view.line_of(trip));
    let stops = timetable.pattern_stops(view.line_pattern(view.line_of(trip)));
    let times = view.stored_times(timetable, trip);
    let alight_from = view.first_boardable(trip) as usize + 1;
    for alight in (alight_from..stops.len()).rev() {
        let arrival = times[alight].arrival - offset;
        labels.improve(stops[alight], arrival);
        for footpath in footpaths.from_stop(stops[alight]) {
            labels.improve(footpath.to, arrival.saturating_add(footpath.duration));
        }
        per_position[alight].retain(|transfer| {
            let boarded_offset = view.day_offset(transfer.trip);
            let boarded_stops =
                timetable.pattern_stops(view.line_pattern(view.line_of(transfer.trip)));
            let boarded_times = view.stored_times(timetable, transfer.trip);
            let mut keeps = false;
            for k in transfer.position as usize + 1..boarded_stops.len() {
                let reached = boarded_times[k].arrival - boarded_offset;
                if labels.improve(boarded_stops[k], reached) {
                    keeps = true;
                }
                for footpath in footpaths.from_stop(boarded_stops[k]) {
                    if labels.improve(footpath.to, reached.saturating_add(footpath.duration)) {
                        keeps = true;
                    }
                }
            }
            keeps
        });
    }
}

/// Per-stop earliest-arrival scratch labels with cheap reuse: only the
/// touched stops reset between trips.
struct Labels {
    arrival: Vec<u32>,
    touched: Vec<u32>,
}

impl Labels {
    fn new(stop_count: u32) -> Labels {
        Labels {
            arrival: vec![UNREACHED; stop_count as usize],
            touched: Vec::new(),
        }
    }

    fn clear(&mut self) {
        for &stop in &self.touched {
            self.arrival[stop as usize] = UNREACHED;
        }
        self.touched.clear();
    }

    fn improve(&mut self, stop: StopIdx, time: u32) -> bool {
        let slot = &mut self.arrival[stop.0 as usize];
        if time < *slot {
            if *slot == UNREACHED {
                self.touched.push(stop.0);
            }
            *slot = time;
            true
        } else {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::timetable::TimetableBuilder;

    fn time(at: u32) -> StopTime {
        StopTime {
            arrival: at,
            departure: at,
        }
    }

    /// Line A rides 0→1→2; line B rides 1→3 at 90, 120, and 200. B's
    /// trips carry services 0, 1, and 2.
    fn crossing() -> Timetable {
        let mut builder = TimetableBuilder::new(4);
        let a = builder
            .add_pattern(&[StopIdx(0), StopIdx(1), StopIdx(2)], 0)
            .unwrap();
        let b = builder.add_pattern(&[StopIdx(1), StopIdx(3)], 1).unwrap();
        builder
            .add_trip(a, vec![time(0), time(100), time(300)], 0, 0)
            .unwrap();
        builder
            .add_trip(b, vec![time(90), time(400)], 1, 0)
            .unwrap();
        builder
            .add_trip(b, vec![time(120), time(500)], 2, 1)
            .unwrap();
        builder
            .add_trip(b, vec![time(200), time(600)], 3, 2)
            .unwrap();
        builder.finish()
    }

    #[test]
    fn boards_the_earliest_catchable_trip_only() {
        let timetable = crossing();
        let build = TransferSet::build(&timetable, &Transfers::empty(4));
        let set = &build.transfers;
        // Alighting A at stop 1 (arrival 100) catches B's 120 trip —
        // not the missed 90 nor the later 200.
        assert_eq!(
            set.from_trip_position(ViewTrip(0), 1),
            &[TripTransfer {
                trip: ViewTrip(2),
                position: 0,
            }]
        );
        // Nothing rides on from A's or B's last stop.
        assert!(set.from_trip_position(ViewTrip(0), 2).is_empty());
        assert!(set.from_trip_position(ViewTrip(2), 1).is_empty());
        assert_eq!(build.generated, set.len());
    }

    #[test]
    fn date_views_board_the_earliest_running_trip() {
        let timetable = crossing();
        // B's 120 trip (service 1) does not run: the 200 trip is the
        // day's earliest catchable — judged against the date, not the
        // whole timetable.
        let active = vec![true, false, true];
        let view = DayView::for_date(&timetable, &active, &[false; 3]);
        assert_eq!(view.trip_count(), 3);
        let build = TransferSet::for_view(&view, &timetable, &Transfers::empty(4));
        let set = &build.transfers;
        // Virtual trips: 0 = A's trip, 1 = B's 90, 2 = B's 200.
        assert_eq!(view.backing(ViewTrip(2)), TripIdx(3));
        assert_eq!(
            set.from_trip_position(ViewTrip(0), 1),
            &[TripTransfer {
                trip: ViewTrip(2),
                position: 0,
            }]
        );
    }

    #[test]
    fn previous_day_tails_join_as_shifted_lines() {
        // A today trip arrives stop 1 at 01:40; the connecting line's
        // only run today left at 00:01:40 stored (missed yesterday's
        // clock is irrelevant), but yesterday's 25:00 tail — 01:00
        // shifted — no: 25:33:20 stored = 01:33:20 shifted misses too;
        // 26:06:40 stored = 02:06:40 shifted connects.
        let mut builder = TimetableBuilder::new(4);
        let a = builder.add_pattern(&[StopIdx(0), StopIdx(1)], 0).unwrap();
        let b = builder.add_pattern(&[StopIdx(1), StopIdx(3)], 1).unwrap();
        builder
            .add_trip(a, vec![time(5_000), time(6_000)], 0, 0)
            .unwrap();
        // Stored times: one trip wholly before midnight (never joins),
        // one over-midnight tail alighting too early, one connecting.
        builder
            .add_trip(b, vec![time(80_000), time(81_000)], 1, 1)
            .unwrap();
        builder
            .add_trip(b, vec![time(86_400 + 5_600), time(86_400 + 9_000)], 2, 1)
            .unwrap();
        builder
            .add_trip(b, vec![time(86_400 + 7_600), time(86_400 + 11_000)], 3, 1)
            .unwrap();
        let timetable = builder.finish();
        let view = DayView::for_date(&timetable, &[true, false], &[false, true]);
        // The wholly-pre-midnight trip is unboardable and stays out.
        assert_eq!(view.trip_count(), 3);
        assert_eq!(view.line_count(), 2);
        assert_eq!(view.day_offset(ViewTrip(1)), DAY_SECONDS);
        let build = TransferSet::for_view(&view, &timetable, &Transfers::empty(4));
        // A arrives stop 1 at 6 000: yesterday's 5 600-shifted tail has
        // left; the 7 600-shifted one boards.
        assert_eq!(
            build.transfers.from_trip_position(ViewTrip(0), 1),
            &[TripTransfer {
                trip: ViewTrip(2),
                position: 0,
            }]
        );
    }

    #[test]
    fn reduction_drops_transfers_that_improve_nothing() {
        // Line C parallels A from stop 1 to stop 2, arriving later than
        // staying on A: feasible, but improves no arrival.
        let mut builder = TimetableBuilder::new(3);
        let a = builder
            .add_pattern(&[StopIdx(0), StopIdx(1), StopIdx(2)], 0)
            .unwrap();
        let c = builder.add_pattern(&[StopIdx(1), StopIdx(2)], 1).unwrap();
        builder
            .add_trip(a, vec![time(0), time(100), time(300)], 0, 0)
            .unwrap();
        builder
            .add_trip(c, vec![time(150), time(400)], 1, 0)
            .unwrap();
        let timetable = builder.finish();
        let build = TransferSet::build(&timetable, &Transfers::empty(3));
        assert_eq!(build.generated, 1);
        assert!(build.transfers.is_empty());
    }

    #[test]
    fn same_pattern_transfers_cannot_help_under_fifo() {
        // Two trips of one line: the earlier trip never "transfers" to
        // the later one at the same or a later position.
        let mut builder = TimetableBuilder::new(3);
        let a = builder
            .add_pattern(&[StopIdx(0), StopIdx(1), StopIdx(2)], 0)
            .unwrap();
        builder
            .add_trip(a, vec![time(0), time(100), time(300)], 0, 0)
            .unwrap();
        builder
            .add_trip(a, vec![time(50), time(150), time(350)], 1, 0)
            .unwrap();
        let timetable = builder.finish();
        let build = TransferSet::build(&timetable, &Transfers::empty(3));
        assert_eq!(build.generated, 0);
        assert!(build.transfers.is_empty());
    }

    fn request(services: usize, origin: StopIdx, destination: StopIdx, departure: u32) -> Request {
        Request {
            departure,
            access: vec![(origin, 0)],
            egress: vec![(destination, 0)],
            active_services: vec![true; services],
            active_services_previous: vec![false; services],
            max_transfers: 7,
        }
    }

    fn pareto(journeys: &[Journey]) -> Vec<(u32, usize)> {
        journeys
            .iter()
            .map(|journey| (journey.arrival, journey.rides()))
            .collect()
    }

    #[test]
    fn query_matches_raptor_on_the_crossing() {
        use crate::raptor::Raptor;

        let timetable = crossing();
        let footpaths = Transfers::empty(4);
        let request = request(3, StopIdx(0), StopIdx(3), 0);
        let raptor = Raptor.route(&timetable, &footpaths, &request);
        let tbtr = Tbtr.route(&timetable, &footpaths, &request);
        assert!(!raptor.is_empty());
        assert_eq!(pareto(&tbtr), pareto(&raptor));
        // The winning chain: A boarded at the origin, B caught at 120.
        assert_eq!(tbtr[0].rides(), 2);
        assert_eq!(tbtr[0].arrival, 500);
    }

    #[test]
    fn footpath_egress_joins_like_raptors_transfer_relaxation() {
        use crate::raptor::Raptor;

        // The u-turn fixture: a destination 30 s from stop 2, best
        // reached by alighting at stop 1 and walking the footpath —
        // the one-hop closure behind the egress stop.
        let mut builder = TimetableBuilder::new(4);
        let a = builder
            .add_pattern(&[StopIdx(0), StopIdx(1), StopIdx(2)], 0)
            .unwrap();
        builder
            .add_trip(a, vec![time(0), time(100), time(200)], 0, 0)
            .unwrap();
        let timetable = builder.finish();
        let footpaths = Transfers::from_edges(
            4,
            &[
                (StopIdx(1), StopIdx(2), 60, 50.0),
                (StopIdx(2), StopIdx(1), 60, 50.0),
            ],
        )
        .unwrap();
        let mut request = request(1, StopIdx(0), StopIdx(2), 0);
        request.egress = vec![(StopIdx(2), 30)];
        let raptor = Raptor.route(&timetable, &footpaths, &request);
        let tbtr = Tbtr.route(&timetable, &footpaths, &request);
        assert_eq!(pareto(&tbtr), pareto(&raptor));
        assert_eq!(tbtr[0].arrival, 190);
        // Egress leaves from the footpath's far end.
        assert!(matches!(
            tbtr[0].legs.last(),
            Some(Leg::Egress { from_stop, .. }) if *from_stop == StopIdx(2)
        ));
    }

    #[test]
    fn query_matches_raptor_over_midnight() {
        use crate::raptor::Raptor;

        // The shifted-tails fixture: the connection rides yesterday's
        // 26:06:40 tail, 02:06:40 on the query day's clock.
        let mut builder = TimetableBuilder::new(4);
        let a = builder.add_pattern(&[StopIdx(0), StopIdx(1)], 0).unwrap();
        let b = builder.add_pattern(&[StopIdx(1), StopIdx(3)], 1).unwrap();
        builder
            .add_trip(a, vec![time(5_000), time(6_000)], 0, 0)
            .unwrap();
        builder
            .add_trip(b, vec![time(80_000), time(81_000)], 1, 1)
            .unwrap();
        builder
            .add_trip(b, vec![time(86_400 + 5_600), time(86_400 + 9_000)], 2, 1)
            .unwrap();
        builder
            .add_trip(b, vec![time(86_400 + 7_600), time(86_400 + 11_000)], 3, 1)
            .unwrap();
        let timetable = builder.finish();
        let footpaths = Transfers::empty(4);
        let mut request = request(2, StopIdx(0), StopIdx(3), 5_000);
        request.active_services = vec![true, false];
        request.active_services_previous = vec![false, true];
        let raptor = Raptor.route(&timetable, &footpaths, &request);
        let tbtr = Tbtr.route(&timetable, &footpaths, &request);
        assert_eq!(pareto(&tbtr), pareto(&raptor));
        assert_eq!(tbtr[0].arrival, 11_000);
    }

    #[test]
    fn range_profiles_match_raptor() {
        use crate::raptor::Raptor;

        // The crossing plus a later A trip and a later B trip: the
        // window offers two distinct departures with different
        // connections, and the descending passes must suppress
        // journeys dominated by the later departure.
        let mut builder = TimetableBuilder::new(4);
        let a = builder
            .add_pattern(&[StopIdx(0), StopIdx(1), StopIdx(2)], 0)
            .unwrap();
        let b = builder.add_pattern(&[StopIdx(1), StopIdx(3)], 1).unwrap();
        builder
            .add_trip(a, vec![time(0), time(100), time(300)], 0, 0)
            .unwrap();
        builder
            .add_trip(a, vec![time(150), time(250), time(450)], 1, 0)
            .unwrap();
        builder
            .add_trip(b, vec![time(120), time(500)], 2, 0)
            .unwrap();
        builder
            .add_trip(b, vec![time(300), time(650)], 3, 0)
            .unwrap();
        let timetable = builder.finish();
        let footpaths = Transfers::empty(4);
        let request = request(1, StopIdx(0), StopIdx(3), 0);
        let raptor = Raptor.route_range(&timetable, &footpaths, &request, 400);
        let engine = TbtrEngine::for_date(
            &timetable,
            &footpaths,
            &request.active_services,
            &request.active_services_previous,
        );
        let tbtr = engine.route_range(&request, 400);
        let triples = |journeys: &[Journey]| -> Vec<(u32, u32, usize)> {
            journeys
                .iter()
                .map(|journey| (journey.departure, journey.arrival, journey.rides()))
                .collect()
        };
        assert_eq!(triples(&tbtr), triples(&raptor));
        assert!(tbtr.len() >= 2);
    }

    #[test]
    fn one_to_all_matches_raptor() {
        use crate::raptor::Raptor;

        let timetable = crossing();
        let footpaths = Transfers::from_edges(4, &[(StopIdx(2), StopIdx(3), 45, 40.0)]).unwrap();
        let request = request(3, StopIdx(0), StopIdx(3), 0);
        let raptor = Raptor.one_to_all(&timetable, &footpaths, &request);
        let engine = TbtrEngine::for_date(
            &timetable,
            &footpaths,
            &request.active_services,
            &request.active_services_previous,
        );
        let tbtr = engine.one_to_all(request.departure, &request.access, request.max_transfers);
        assert_eq!(tbtr, raptor);
        // The footpath tail is reachable: stop 3 over stop 2's walk.
        assert!(tbtr[3].is_some());
    }

    #[test]
    fn u_turns_are_dropped_at_generation() {
        // Line A rides 0→1→2; line B rides 1→2→(3); footpaths join
        // stops 1 and 2 both ways. Alighting A at stop 2 and walking
        // back to reboard B over the same 1→2 segment is a U-turn: B
        // was already catchable at stop 1.
        let mut builder = TimetableBuilder::new(4);
        let a = builder
            .add_pattern(&[StopIdx(0), StopIdx(1), StopIdx(2)], 0)
            .unwrap();
        let b = builder
            .add_pattern(&[StopIdx(1), StopIdx(2), StopIdx(3)], 1)
            .unwrap();
        builder
            .add_trip(a, vec![time(0), time(100), time(200)], 0, 0)
            .unwrap();
        builder
            .add_trip(b, vec![time(400), time(500), time(600)], 1, 0)
            .unwrap();
        let timetable = builder.finish();
        let footpaths = Transfers::from_edges(
            4,
            &[
                (StopIdx(1), StopIdx(2), 60, 50.0),
                (StopIdx(2), StopIdx(1), 60, 50.0),
            ],
        )
        .unwrap();
        let build = TransferSet::build(&timetable, &footpaths);
        let set = &build.transfers;
        // Walking from stop 2 back to stop 1 to re-ride the 1→2 segment
        // is dropped at generation: only the three genuine boardings of
        // B are generated (from stop 1 at either end of its footpath,
        // and at stop 2 directly).
        assert_eq!(build.generated, 3);
        // The reduction then collapses them to one representative — all
        // three ride the same B trip to the same arrivals, and the
        // latest alight position is processed first.
        assert!(set.from_trip_position(ViewTrip(0), 1).is_empty());
        assert_eq!(
            set.from_trip_position(ViewTrip(0), 2),
            &[TripTransfer {
                trip: ViewTrip(1),
                position: 1,
            }]
        );
    }
}
