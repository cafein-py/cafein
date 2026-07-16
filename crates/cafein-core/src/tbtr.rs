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
//! set the query engine scans. The reduction is tie-complete: a
//! transfer that exactly ties a kept competitor from a *different*
//! trip is retained too (as is each trip's earliest tied boarding), so
//! cost reconstruction can elect the same journey RAPTOR's
//! tie-breaking does; ties against staying on the trip still prune.
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
use crate::raptor::{CostInputs, CostRow};
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
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
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
    /// Owned when built ad hoc (`for_date`), borrowed when the caller
    /// cached the date's set (`from_set`) — the engine only reads it.
    set: std::borrow::Cow<'a, TransferSet>,
    /// Reversed footpath adjacency: for each stop, the `(from, seconds)`
    /// edges walking *into* it — the one-hop closure behind egress
    /// stops, mirroring RAPTOR's post-round transfer relaxation.
    incoming_offsets: Vec<u32>,
    incoming: Vec<(StopIdx, u32)>,
}

/// Dense per-stop scratch for the stops a round's footpaths improved,
/// drained in first-improvement insertion order. A `HashMap` here would
/// board walked stops in random per-process order: arrivals would not
/// change (strict-improvement labels are order-independent), but at an
/// exact arrival tie the first-enqueued segment wins the label, so
/// journey reconstruction would elect a different equal journey from
/// process to process.
struct WalkedScratch {
    /// `(ready, parent segment, alight position)` per stop;
    /// `ready == UNREACHED` marks an empty slot.
    slots: Vec<(u32, u32, u16)>,
    touched: Vec<u32>,
}

impl WalkedScratch {
    fn new(stop_count: usize) -> WalkedScratch {
        WalkedScratch {
            slots: vec![(UNREACHED, 0, 0); stop_count],
            touched: Vec::new(),
        }
    }

    fn clear(&mut self) {
        for &stop in &self.touched {
            self.slots[stop as usize].0 = UNREACHED;
        }
        self.touched.clear();
    }

    /// Records (or overwrites, keeping the stop's original insertion
    /// rank) the walk that improved `stop`, exactly as the map insert
    /// this replaces did.
    fn insert(&mut self, stop: u32, value: (u32, u32, u16)) {
        let slot = &mut self.slots[stop as usize];
        if slot.0 == UNREACHED {
            self.touched.push(stop);
        }
        *slot = value;
    }

    fn iter(&self) -> impl Iterator<Item = (u32, (u32, u32, u16))> + '_ {
        self.touched
            .iter()
            .map(move |&stop| (stop, self.slots[stop as usize]))
    }
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

/// How a stop's exact-round arrival in a matrix pass was achieved.
#[derive(Clone, Copy)]
enum StopWinner {
    Unreached,
    /// Reached by the access walk alone (round 0).
    Access,
    /// Alighted `arena[segment]` at `alight`.
    Alight {
        segment: u32,
        alight: u16,
    },
    /// Walked a footpath from `from` after alighting `arena[segment]`
    /// at `alight`.
    Walked {
        segment: u32,
        alight: u16,
        from: StopIdx,
    },
}

/// Pooled per-worker state for the cost-matrix passes: the profile
/// scan's scratch plus exact-round arrivals and the winner records the
/// cost reconstruction walks. One state serves an origin's descending
/// departures (horizons, labels, arena, and winners persist across
/// passes, rTBTR-style); `reset` recycles it for the next origin.
pub struct MatrixState {
    rounds: usize,
    stop_count: usize,
    reached: Vec<u16>,
    arena: Vec<Segment>,
    queues: Vec<Vec<(u32, u16)>>,
    /// The at-most-round arrival gate, `stop × rounds` with the suffix
    /// write of the profile scan.
    labels: Vec<u32>,
    walked: WalkedScratch,
    /// Exact-round arrivals, `stop × (rounds + 1)`; slot 0 is the
    /// access-only round, slot `q + 1` holds queue round `q`'s alights
    /// and their same-round walks — RAPTOR's `tau` shape.
    tau: Vec<u32>,
    winners: Vec<StopWinner>,
}

impl MatrixState {
    /// Fresh state sized for an engine and a round count.
    pub fn new(engine: &TbtrEngine<'_>, max_transfers: u8) -> MatrixState {
        let rounds = max_transfers as usize + 1;
        let stop_count = engine.timetable.stop_count() as usize;
        MatrixState {
            rounds,
            stop_count,
            reached: engine.horizons(rounds),
            arena: Vec::new(),
            queues: vec![Vec::new(); rounds],
            labels: vec![UNREACHED; stop_count * rounds],
            walked: WalkedScratch::new(stop_count),
            tau: vec![UNREACHED; stop_count * (rounds + 1)],
            winners: vec![StopWinner::Unreached; stop_count * (rounds + 1)],
        }
    }

    /// Recycles the buffers for another origin on the same engine.
    pub fn reset(&mut self, engine: &TbtrEngine<'_>) {
        self.reached.clear();
        self.reached
            .extend_from_slice(&engine.horizons(self.rounds));
        self.arena.clear();
        for queue in &mut self.queues {
            queue.clear();
        }
        self.labels.fill(UNREACHED);
        self.walked.clear();
        self.tau.fill(UNREACHED);
        self.winners.fill(StopWinner::Unreached);
    }

    fn tau_at(&self, stop: StopIdx, round: usize) -> u32 {
        self.tau[stop.0 as usize * (self.rounds + 1) + round]
    }

    fn record(&mut self, stop: StopIdx, round: usize, arrival: u32, winner: StopWinner) {
        let slot = stop.0 as usize * (self.rounds + 1) + round;
        self.tau[slot] = arrival;
        self.winners[slot] = winner;
    }
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
    fn window_samples(&self, request: &Request, window: u32) -> Vec<(u32, Vec<u32>)> {
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
    /// One scan of a departure onto a `MatrixState`: the profile scan
    /// stripped of destination targets and journey assembly, writing
    /// exact-round arrivals and winner records instead. State persists
    /// across strictly decreasing departures (rTBTR), so an origin's
    /// window runs passes on one state; `reset` recycles it between
    /// origins.
    pub fn matrix_pass(&self, departure: u32, access: &[(StopIdx, u32)], state: &mut MatrixState) {
        let rounds = state.rounds;
        let improve = |labels: &mut Vec<u32>, stop: StopIdx, time: u32, round: usize| {
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
        for &(stop, seconds) in access {
            let at = departure.saturating_add(seconds);
            improve(&mut state.labels, stop, at, 0);
            if at < state.tau_at(stop, 0) {
                state.record(stop, 0, at, StopWinner::Access);
            }
        }
        self.seed(
            departure,
            access,
            &mut state.reached,
            rounds,
            &mut state.arena,
            &mut state.queues[0],
        );
        for round in 0..rounds {
            if state.queues[round].is_empty() {
                break;
            }
            let segments = std::mem::take(&mut state.queues[round]);
            state.walked.clear();
            for &(segment, end) in &segments {
                let trip = state.arena[segment as usize].trip;
                let board = state.arena[segment as usize].board;
                let line = self.view.line_of(trip);
                let offset = self.view.line_day_offset(line);
                let stops = self.timetable.pattern_stops(self.view.line_pattern(line));
                let times = self.view.stored_times(self.timetable, trip);
                let last = (end as usize + 1).min(times.len()) as u16;
                for alight in board + 1..last {
                    let arrival = times[alight as usize].arrival - offset;
                    let stop = stops[alight as usize];
                    // Walks relax only from arrivals that improve the
                    // stop's at-most-round label — RAPTOR's marked-stop
                    // semantics; the same improvements write the
                    // exact-round winner the reconstruction walks.
                    if improve(&mut state.labels, stop, arrival, round) {
                        state.record(
                            stop,
                            round + 1,
                            arrival,
                            StopWinner::Alight { segment, alight },
                        );
                        for footpath in self.footpaths.from_stop(stop) {
                            let walked_at = arrival.saturating_add(footpath.duration);
                            if improve(&mut state.labels, footpath.to, walked_at, round) {
                                state.record(
                                    footpath.to,
                                    round + 1,
                                    walked_at,
                                    StopWinner::Walked {
                                        segment,
                                        alight,
                                        from: stop,
                                    },
                                );
                                state
                                    .walked
                                    .insert(footpath.to.0, (walked_at, segment, alight));
                            }
                        }
                    }
                    if round + 1 < rounds {
                        for transfer in self.set.from_trip_position(trip, alight) {
                            enqueue(
                                &self.view,
                                &mut state.reached,
                                rounds,
                                round + 1,
                                &mut state.arena,
                                &mut state.queues[round + 1],
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
                let boardings: Vec<(u32, (u32, u32, u16))> = state.walked.iter().collect();
                for (stop, (ready, parent, alight)) in boardings {
                    self.board_walked(
                        StopIdx(stop),
                        ready,
                        parent,
                        alight,
                        &mut state.reached,
                        rounds,
                        round + 1,
                        &mut state.arena,
                        &mut state.queues[round + 1],
                    );
                }
            }
        }
    }

    /// The fastest journey's aggregated costs to `stop`, mirroring
    /// `Search::costs_to`: rounds scan ascending, only a strictly
    /// earlier arrival replaces the winner — fewest rides on ties.
    fn matrix_costs_to(
        &self,
        state: &MatrixState,
        stop: StopIdx,
        departure: u32,
        inputs: &CostInputs<'_>,
        access_meters: Option<&HashMap<StopIdx, f64>>,
    ) -> Option<CostRow> {
        let mut best_round = 0;
        let mut best_arrival = UNREACHED;
        for round in 0..=state.rounds {
            let arrival = state.tau_at(stop, round);
            if arrival < best_arrival {
                best_arrival = arrival;
                best_round = round;
            }
        }
        if best_arrival == UNREACHED {
            return None;
        }
        let mut row = self.matrix_cost_row(state, stop, best_round, inputs, access_meters);
        row.seconds = best_arrival - departure;
        Some(row)
    }

    /// The aggregated costs of the journey behind the winner at
    /// `(stop, round)`, walking its chain destination → origin in the
    /// same order (and with the same floating-point accumulation
    /// sequence) as `Raptor::walk_costs`, so identical journeys yield
    /// bit-identical rows.
    fn matrix_cost_row(
        &self,
        state: &MatrixState,
        stop: StopIdx,
        round: usize,
        inputs: &CostInputs<'_>,
        access_meters: Option<&HashMap<StopIdx, f64>>,
    ) -> CostRow {
        let mut rides = 0u32;
        let mut transit_meters = 0.0;
        let mut walk_meters = 0.0;
        let mut grams = 0.0;
        let mut resolved = true;
        let mut legs: Vec<(TripIdx, u16, u16)> = Vec::new();
        let mut fare_legs: Vec<crate::fares::FareLeg> = Vec::new();
        let slot = stop.0 as usize * (state.rounds + 1) + round;
        // Resolve the destination-side walk, if any, then follow the
        // segment chain of the alighted arena entry.
        let (mut segment, mut alight_position, mut at) = match state.winners[slot] {
            StopWinner::Unreached => {
                unreachable!("cost reconstruction hit an unreached winner")
            }
            StopWinner::Access => {
                if let Some(access) = access_meters {
                    walk_meters += access.get(&stop).copied().unwrap_or(0.0);
                }
                return self.finish_cost_row(
                    stop.0,
                    rides,
                    transit_meters,
                    walk_meters,
                    grams,
                    resolved,
                    legs,
                    fare_legs,
                    inputs,
                );
            }
            StopWinner::Alight { segment, alight } => (segment, alight, stop),
            StopWinner::Walked {
                segment,
                alight,
                from,
            } => {
                walk_meters += self
                    .footpaths
                    .from_stop(from)
                    .iter()
                    .find(|transfer| transfer.to == stop)
                    .map(|transfer| transfer.meters)
                    .unwrap_or(0.0);
                (segment, alight, from)
            }
        };
        loop {
            let entry = &state.arena[segment as usize];
            let backing = self.view.backing(entry.trip);
            let offset = self.view.day_offset(entry.trip);
            rides += 1;
            let meters = inputs
                .geometry
                .leg_distance(backing, entry.board, alight_position)
                as f64;
            transit_meters += meters;
            let factor = inputs.factors[backing.0 as usize];
            if factor.is_finite() {
                grams += meters / 1000.0 * factor;
            } else {
                resolved = false;
            }
            if inputs.with_geometry {
                legs.push((backing, entry.board, alight_position));
            }
            let pattern = self.timetable.trip_pattern(backing);
            let board_stop = self.timetable.pattern_stops(pattern)[entry.board as usize];
            if inputs.fares.is_some() {
                fare_legs.push(crate::fares::FareLeg {
                    route: self.timetable.pattern_route(pattern),
                    board_stop: board_stop.0,
                    alight_stop: at.0,
                    board_time: self.timetable.trip_stop_times(backing)[entry.board as usize]
                        .departure
                        .saturating_sub(offset),
                });
            }
            match entry.origin {
                SegmentOrigin::Access { stop: origin, .. } => {
                    if board_stop != origin {
                        walk_meters += self
                            .footpaths
                            .from_stop(origin)
                            .iter()
                            .find(|transfer| transfer.to == board_stop)
                            .map(|transfer| transfer.meters)
                            .unwrap_or(0.0);
                    }
                    if let Some(access) = access_meters {
                        walk_meters += access.get(&origin).copied().unwrap_or(0.0);
                    }
                    break;
                }
                SegmentOrigin::Transfer { parent, alight } => {
                    let parent_entry = &state.arena[parent as usize];
                    let parent_line = self.view.line_of(parent_entry.trip);
                    let parent_stops = self
                        .timetable
                        .pattern_stops(self.view.line_pattern(parent_line));
                    let parent_stop = parent_stops[alight as usize];
                    if parent_stop != board_stop {
                        walk_meters += self
                            .footpaths
                            .from_stop(parent_stop)
                            .iter()
                            .find(|transfer| transfer.to == board_stop)
                            .map(|transfer| transfer.meters)
                            .unwrap_or(0.0);
                    }
                    at = parent_stop;
                    alight_position = alight;
                    segment = parent;
                }
            }
        }
        self.finish_cost_row(
            stop.0,
            rides,
            transit_meters,
            walk_meters,
            grams,
            resolved,
            legs,
            fare_legs,
            inputs,
        )
    }

    /// Assembles the row tail exactly as `Raptor::walk_costs` does:
    /// geometry legs reversed into ride order, fares priced in ride
    /// order, NaN-poisoned emissions.
    #[allow(clippy::too_many_arguments)]
    fn finish_cost_row(
        &self,
        to: u32,
        rides: u32,
        transit_meters: f64,
        walk_meters: f64,
        grams: f64,
        resolved: bool,
        legs: Vec<(TripIdx, u16, u16)>,
        mut fare_legs: Vec<crate::fares::FareLeg>,
        inputs: &CostInputs<'_>,
    ) -> CostRow {
        let geometry = match (inputs.with_geometry, inputs.leg_geometry) {
            (true, Some(shapes)) => {
                let parts: Vec<Vec<(f64, f64)>> = legs
                    .iter()
                    .rev()
                    .map(|&(trip, board, alight)| shapes.leg_coordinates(trip, board, alight))
                    .collect();
                Some(crate::geometry::wkb_multi_line_string(&parts))
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
            seconds: 0,
            rides,
            transit_meters,
            walk_meters,
            emission_grams: if resolved { grams } else { f64::NAN },
            fare,
            geometry,
        }
    }

    /// The fastest journey's aggregated costs from each request to each
    /// destination — the TBTR counterpart of `Raptor::cost_matrix`,
    /// fanned out over the origins with pooled per-worker state.
    pub fn cost_matrix(
        &self,
        inputs: &CostInputs<'_>,
        requests: &[Request],
        destinations: &[StopIdx],
    ) -> Vec<Vec<CostRow>> {
        requests
            .par_iter()
            .map_init(
                || None,
                |pooled: &mut Option<MatrixState>, request| {
                    let state = match pooled {
                        Some(state) if state.rounds == request.max_transfers as usize + 1 => {
                            state.reset(self);
                            state
                        }
                        _ => pooled.insert(MatrixState::new(self, request.max_transfers)),
                    };
                    self.matrix_pass(request.departure, &request.access, state);
                    destinations
                        .iter()
                        .filter_map(|&stop| {
                            self.matrix_costs_to(state, stop, request.departure, inputs, None)
                        })
                        .collect()
                },
            )
            .collect()
    }

    /// The objective-best journey's costs within a travel-time budget,
    /// per destination, over a departure window — the TBTR counterpart
    /// of `Raptor::least_cost_matrix`, with `Search::fold_best`'s exact
    /// admission order: per-round suffix thresholds first (strict), the
    /// budget second, reconstruction third, the objective fold last.
    #[allow(clippy::too_many_arguments)]
    pub fn least_cost_matrix(
        &self,
        inputs: &CostInputs<'_>,
        requests: &[Request],
        destinations: &[StopIdx],
        window: u32,
        budget: Option<u32>,
        objective: crate::raptor::Objective,
    ) -> Vec<Vec<CostRow>> {
        requests
            .par_iter()
            .map_init(
                || None,
                |pooled: &mut Option<MatrixState>, request| {
                    let state = match pooled {
                        Some(state) if state.rounds == request.max_transfers as usize + 1 => {
                            state.reset(self);
                            state
                        }
                        _ => pooled.insert(MatrixState::new(self, request.max_transfers)),
                    };
                    let departures =
                        crate::raptor::departure_candidates(self.timetable, request, window);
                    let mut thresholds = vec![UNREACHED; destinations.len() * (state.rounds + 1)];
                    let mut best: Vec<Option<CostRow>> = vec![None; destinations.len()];
                    for &departure in &departures {
                        self.matrix_pass(departure, &request.access, state);
                        self.fold_matrix_best(
                            state,
                            departure,
                            destinations,
                            budget,
                            inputs,
                            None,
                            objective,
                            &mut thresholds,
                            &mut best,
                        );
                    }
                    best.into_iter().flatten().collect()
                },
            )
            .collect()
    }

    /// One pass's fold onto the per-destination bests — the mirror of
    /// `Search::fold_best`. Stale slots from later departures fail the
    /// strict threshold and are never reconstructed, so each fold reads
    /// a consistent snapshot of what its pass improved.
    #[allow(clippy::too_many_arguments)]
    fn fold_matrix_best(
        &self,
        state: &MatrixState,
        departure: u32,
        destinations: &[StopIdx],
        budget: Option<u32>,
        inputs: &CostInputs<'_>,
        access_meters: Option<&HashMap<StopIdx, f64>>,
        objective: crate::raptor::Objective,
        thresholds: &mut [u32],
        best: &mut [Option<CostRow>],
    ) {
        for (slot, &stop) in destinations.iter().enumerate() {
            let thresholds = &mut thresholds[slot * (state.rounds + 1)..][..state.rounds + 1];
            for round in 0..=state.rounds {
                let arrival = state.tau_at(stop, round);
                if arrival >= thresholds[round] {
                    continue;
                }
                for threshold in &mut thresholds[round..] {
                    *threshold = (*threshold).min(arrival);
                }
                let seconds = arrival - departure;
                if budget.is_some_and(|budget| seconds > budget) {
                    continue;
                }
                let mut row = self.matrix_cost_row(state, stop, round, inputs, access_meters);
                row.seconds = seconds;
                crate::raptor::fold_better(&mut best[slot], row, objective);
            }
        }
    }

    /// The fastest journey's costs to each destination *point*, joined
    /// through egress link tables — the TBTR counterpart of
    /// `Raptor::cost_matrix_to_points`, including `costs_to_point`'s
    /// link election: the first strictly smaller joined arrival wins,
    /// with no ride-count comparison across equal links.
    pub fn cost_matrix_to_points(
        &self,
        inputs: &CostInputs<'_>,
        requests: &[Request],
        access_meters: &[HashMap<StopIdx, f64>],
        egress: &[Vec<(StopIdx, u32, f64)>],
    ) -> Vec<Vec<CostRow>> {
        assert_eq!(requests.len(), access_meters.len());
        requests
            .par_iter()
            .zip(access_meters.par_iter())
            .map_init(
                || None,
                |pooled: &mut Option<MatrixState>, (request, access)| {
                    let state = match pooled {
                        Some(state) if state.rounds == request.max_transfers as usize + 1 => {
                            state.reset(self);
                            state
                        }
                        _ => pooled.insert(MatrixState::new(self, request.max_transfers)),
                    };
                    self.matrix_pass(request.departure, &request.access, state);
                    egress
                        .iter()
                        .enumerate()
                        .filter_map(|(point, links)| {
                            self.matrix_costs_to_point(
                                state,
                                point as u32,
                                links,
                                request.departure,
                                inputs,
                                access,
                            )
                        })
                        .collect()
                },
            )
            .collect()
    }

    /// `Raptor::costs_to_point`'s mirror over the matrix state.
    fn matrix_costs_to_point(
        &self,
        state: &MatrixState,
        point: u32,
        egress: &[(StopIdx, u32, f64)],
        departure: u32,
        inputs: &CostInputs<'_>,
        access_meters: &HashMap<StopIdx, f64>,
    ) -> Option<CostRow> {
        let mut chosen: Option<(u32, StopIdx, f64)> = None;
        for &(stop, seconds, meters) in egress {
            let at_stop = (0..=state.rounds)
                .map(|round| state.tau_at(stop, round))
                .min()
                .expect("a matrix state always has a round");
            if at_stop == UNREACHED {
                continue;
            }
            let Some(arrival) = at_stop.checked_add(seconds).filter(|&at| at != UNREACHED) else {
                continue;
            };
            if chosen.is_none_or(|(current, _, _)| arrival < current) {
                chosen = Some((arrival, stop, meters));
            }
        }
        let (arrival, stop, egress_meters) = chosen?;
        let mut row = self.matrix_costs_to(state, stop, departure, inputs, Some(access_meters))?;
        row.to = point;
        row.seconds = arrival - departure;
        row.walk_meters += egress_meters;
        Some(row)
    }

    /// `Raptor::least_cost_matrix_to_points`'s TBTR counterpart.
    #[allow(clippy::too_many_arguments)]
    pub fn least_cost_matrix_to_points(
        &self,
        inputs: &CostInputs<'_>,
        requests: &[Request],
        access_meters: &[HashMap<StopIdx, f64>],
        egress: &[Vec<(StopIdx, u32, f64)>],
        window: u32,
        budget: Option<u32>,
        objective: crate::raptor::Objective,
    ) -> Vec<Vec<CostRow>> {
        assert_eq!(requests.len(), access_meters.len());
        requests
            .par_iter()
            .zip(access_meters.par_iter())
            .map_init(
                || None,
                |pooled: &mut Option<MatrixState>, (request, access)| {
                    let state = match pooled {
                        Some(state) if state.rounds == request.max_transfers as usize + 1 => {
                            state.reset(self);
                            state
                        }
                        _ => pooled.insert(MatrixState::new(self, request.max_transfers)),
                    };
                    let departures =
                        crate::raptor::departure_candidates(self.timetable, request, window);
                    let mut thresholds = vec![UNREACHED; egress.len() * (state.rounds + 1)];
                    let mut best: Vec<Option<CostRow>> = vec![None; egress.len()];
                    for &departure in &departures {
                        self.matrix_pass(departure, &request.access, state);
                        self.fold_matrix_best_points(
                            state,
                            departure,
                            egress,
                            budget,
                            inputs,
                            access,
                            objective,
                            &mut thresholds,
                            &mut best,
                        );
                    }
                    best.into_iter().flatten().collect()
                },
            )
            .collect()
    }

    /// `Search::fold_best_points`' mirror: rounds from 1 (the access
    /// floor is the caller's direct-walk overlay), the first strictly
    /// smaller joined link wins each round, thresholds and budget in
    /// the same order.
    #[allow(clippy::too_many_arguments)]
    fn fold_matrix_best_points(
        &self,
        state: &MatrixState,
        departure: u32,
        egress: &[Vec<(StopIdx, u32, f64)>],
        budget: Option<u32>,
        inputs: &CostInputs<'_>,
        access_meters: &HashMap<StopIdx, f64>,
        objective: crate::raptor::Objective,
        thresholds: &mut [u32],
        best: &mut [Option<CostRow>],
    ) {
        for (point, links) in egress.iter().enumerate() {
            let thresholds = &mut thresholds[point * (state.rounds + 1)..][..state.rounds + 1];
            for round in 1..=state.rounds {
                let mut chosen: Option<(u32, StopIdx, f64)> = None;
                for &(stop, seconds, meters) in links {
                    let at_stop = state.tau_at(stop, round);
                    if at_stop == UNREACHED {
                        continue;
                    }
                    let Some(arrival) = at_stop.checked_add(seconds).filter(|&at| at != UNREACHED)
                    else {
                        continue;
                    };
                    if chosen.is_none_or(|(current, _, _)| arrival < current) {
                        chosen = Some((arrival, stop, meters));
                    }
                }
                let Some((arrival, stop, egress_meters)) = chosen else {
                    continue;
                };
                if arrival >= thresholds[round] {
                    continue;
                }
                for threshold in &mut thresholds[round..] {
                    *threshold = (*threshold).min(arrival);
                }
                let seconds = arrival - departure;
                if budget.is_some_and(|budget| seconds > budget) {
                    continue;
                }
                let mut row = self.matrix_cost_row(state, stop, round, inputs, Some(access_meters));
                row.to = point as u32;
                row.seconds = seconds;
                row.walk_meters += egress_meters;
                crate::raptor::fold_better(&mut best[point], row, objective);
            }
        }
    }

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

/// Witt's transfer reduction for one virtual trip, tie-complete: walking
/// the alight positions back to front, each alight runs two phases.
/// First every candidate of the alight contributes to the labels
/// (alongside the stays), so same-alight competitors converge on each
/// trip's earliest tied boarding; then the alight's candidates are
/// retained exactly when they witness a label — a strict best, or their
/// trip's minimal tied boarding. A tie against staying on the trip
/// prunes (fewer rides wins, as in RAPTOR's round-ascending tie-break);
/// a tie between different boarded trips keeps both, since which one
/// RAPTOR elects depends on the query. Only same-or-later alight state
/// ever competes — an earlier alight's labels are unavailable to a
/// query that boards between the two positions — which the backward
/// walk preserves. Labels are per-trip scratch state, pooled per worker.
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
        labels.improve_stay(stops[alight], arrival);
        for footpath in footpaths.from_stop(stops[alight]) {
            labels.improve_stay(footpath.to, arrival.saturating_add(footpath.duration));
        }
        for transfer in per_position[alight].iter() {
            let boarded_offset = view.day_offset(transfer.trip);
            let boarded_stops =
                timetable.pattern_stops(view.line_pattern(view.line_of(transfer.trip)));
            let boarded_times = view.stored_times(timetable, transfer.trip);
            for k in transfer.position as usize + 1..boarded_stops.len() {
                let reached = boarded_times[k].arrival - boarded_offset;
                labels.improve_transfer(
                    boarded_stops[k],
                    reached,
                    transfer.trip,
                    transfer.position,
                );
                for footpath in footpaths.from_stop(boarded_stops[k]) {
                    labels.improve_transfer(
                        footpath.to,
                        reached.saturating_add(footpath.duration),
                        transfer.trip,
                        transfer.position,
                    );
                }
            }
        }
        per_position[alight].retain(|transfer| {
            let boarded_offset = view.day_offset(transfer.trip);
            let boarded_stops =
                timetable.pattern_stops(view.line_pattern(view.line_of(transfer.trip)));
            let boarded_times = view.stored_times(timetable, transfer.trip);
            for k in transfer.position as usize + 1..boarded_stops.len() {
                let reached = boarded_times[k].arrival - boarded_offset;
                if labels.witnesses(boarded_stops[k], reached, transfer.trip, transfer.position) {
                    return true;
                }
                for footpath in footpaths.from_stop(boarded_stops[k]) {
                    if labels.witnesses(
                        footpath.to,
                        reached.saturating_add(footpath.duration),
                        transfer.trip,
                        transfer.position,
                    ) {
                        return true;
                    }
                }
            }
            false
        });
    }
}

/// Per-stop earliest-arrival scratch labels with cheap reuse: only the
/// touched stops reset between trips. Each label carries how it was
/// reached, so an exact arrival tie can distinguish a fewer-rides stay
/// (the candidate loses outright, as it would against RAPTOR's
/// round-ascending tie-break) from same-ride competitors. A tied label
/// tracks every retained trip with its minimum boarding position —
/// RAPTOR boards a trip at its earliest catchable position, so among
/// same-trip ties only the earliest boarding is electable, however many
/// other trips tie in between.
struct Labels {
    arrival: Vec<u32>,
    /// Whether the label's arrival level is stay-witnessed (equal
    /// candidates die against it). Meaningful only while `arrival` is
    /// set; guarded by the `UNREACHED` checks below.
    stay: Vec<bool>,
    /// The transfer-witnessed trips at the label's arrival level, each
    /// with the minimum boarding position retained so far. Tiny in
    /// practice (a tie rarely involves more than a couple of trips), so
    /// a linear scan beats any keyed structure.
    ties: Vec<Vec<(ViewTrip, u16)>>,
    touched: Vec<u32>,
}

impl Labels {
    fn new(stop_count: u32) -> Labels {
        Labels {
            arrival: vec![UNREACHED; stop_count as usize],
            stay: vec![false; stop_count as usize],
            ties: vec![Vec::new(); stop_count as usize],
            touched: Vec::new(),
        }
    }

    fn clear(&mut self) {
        for &stop in &self.touched {
            self.arrival[stop as usize] = UNREACHED;
            self.ties[stop as usize].clear();
        }
        self.touched.clear();
    }

    /// A stay-side improvement: strictly earlier claims the label, and an
    /// exact tie demotes the label to Stay — the stayed path rides less,
    /// so equal candidates must stop surviving off it. (A tie at
    /// `UNREACHED` is a saturated walk, not a label; state behind an
    /// `UNREACHED` slot is stale and must stay unread.)
    fn improve_stay(&mut self, stop: StopIdx, time: u32) {
        let slot = &mut self.arrival[stop.0 as usize];
        if time < *slot {
            if *slot == UNREACHED {
                self.touched.push(stop.0);
            }
            *slot = time;
            self.stay[stop.0 as usize] = true;
            self.ties[stop.0 as usize].clear();
        } else if time == *slot && time != UNREACHED {
            self.stay[stop.0 as usize] = true;
            self.ties[stop.0 as usize].clear();
        }
    }

    /// A candidate transfer's contribution to the labels: strictly
    /// earlier claims the label outright. An exact tie never survives a
    /// fewer-rides stay (nor a stale label behind an `UNREACHED` slot);
    /// against other transfers the tied trips accumulate, each at its
    /// minimum boarding position — a *different* trip is a genuinely
    /// distinct journey whose election depends on the query, while a
    /// same-trip later boarding can never be elected (RAPTOR boards at
    /// the earliest catchable position), whichever competitor happens to
    /// have contributed first.
    fn improve_transfer(&mut self, stop: StopIdx, time: u32, trip: ViewTrip, position: u16) {
        let slot = &mut self.arrival[stop.0 as usize];
        if time < *slot {
            if *slot == UNREACHED {
                self.touched.push(stop.0);
            }
            *slot = time;
            self.stay[stop.0 as usize] = false;
            let ties = &mut self.ties[stop.0 as usize];
            ties.clear();
            ties.push((trip, position));
        } else if time == *slot && time != UNREACHED && !self.stay[stop.0 as usize] {
            let ties = &mut self.ties[stop.0 as usize];
            for (kept, kept_position) in ties.iter_mut() {
                if *kept == trip {
                    if position < *kept_position {
                        *kept_position = position;
                    }
                    return;
                }
            }
            ties.push((trip, position));
        }
    }

    /// Whether a candidate's reach of `stop` at `time` witnesses the
    /// final label: the arrival matches, no fewer-rides stay claimed it,
    /// and the candidate is its trip's minimal tied boarding there.
    fn witnesses(&self, stop: StopIdx, time: u32, trip: ViewTrip, position: u16) -> bool {
        time != UNREACHED
            && self.arrival[stop.0 as usize] == time
            && !self.stay[stop.0 as usize]
            && self.ties[stop.0 as usize]
                .iter()
                .any(|&(kept, kept_position)| kept == trip && kept_position == position)
    }
}

#[cfg(test)]
#[path = "tbtr_tests.rs"]
mod tests;
