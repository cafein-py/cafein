//! Transfer-set generation and its tie-complete reduction.

use super::*;

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

/// Queues a segment when it boards earlier than anything seen on its
/// trip with this many rides or fewer, and marks the trip and its
/// later line siblings reached from this round on: under FIFO,
/// boarding a later sibling at the same or a later position with the
/// same or more rides can never beat this one. The horizons are per
/// (trip, round) — profile passes at earlier departures may re-board a
/// trip already used by a later departure when they do so with fewer
/// rides, and a single per-trip horizon would wrongly suppress them.
pub(super) fn enqueue(
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
