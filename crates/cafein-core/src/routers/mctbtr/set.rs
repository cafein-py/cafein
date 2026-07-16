//! The factor-bound multicriteria transfer-set precompute.

use super::*;

/// Builds the multicriteria transfer set of a day view by **global
/// candidate/witness enumeration** (Baum et al. 2023's integrated
/// preprocessing, adapted to (arrival, grams) and cafein's trip-based
/// set): from every source stop, a descending run per unique departure
/// time boards the first-trip frontier, transfers same-stop onto
/// second-trip candidates, and keeps a trip-transfer only when a two-trip journey
/// using it survives every witness — one-trip journeys and competing
/// alternatives from the same origin context. Fanned out over source
/// stops with rayon; `factors` are grams CO₂e per passenger-kilometre
/// per backing trip, NaN where unresolved. `TransferSetBuild::generated`
/// reports the kept-transfer count (the enumeration never materializes
/// an unreduced set).
pub fn transfer_set(
    view: &DayView,
    timetable: &Timetable,
    geometry: &TripGeometry,
    factors: &[f64],
) -> TransferSetBuild {
    let chains = CleanerChains::build(view, factors);
    let per_source: Vec<Vec<(u32, u16, TripTransfer)>> = (0..timetable.stop_count())
        .into_par_iter()
        .map_init(
            || SetSearch::new(view, timetable, geometry, factors, &chains),
            |search, stop| search.run(StopIdx(stop)),
        )
        .collect();
    let mut all: Vec<(u32, u16, TripTransfer)> = per_source.into_iter().flatten().collect();
    all.sort_unstable_by_key(|&(trip, alight, transfer)| {
        (trip, alight, transfer.trip.0, transfer.position)
    });
    all.dedup();
    let mut per_trip: Vec<(Vec<Vec<TripTransfer>>, usize)> = (0..view.trip_count())
        .map(|trip| {
            let line = view.line_of(ViewTrip(trip));
            let positions = timetable.pattern_stops(view.line_pattern(line)).len();
            (vec![Vec::new(); positions], 0)
        })
        .collect();
    for (trip, alight, transfer) in all {
        let (positions, kept) = &mut per_trip[trip as usize];
        positions[alight as usize].push(transfer);
        *kept += 1;
    }
    TransferSet::assemble(per_trip)
}

/// The boarding candidates of every line, precomputed: from any first
/// boardable trip, the trips worth boarding are the first one with a
/// resolved factor followed by its strictly-decreasing-factor chain —
/// exactly the set the naive suffix walk keeps, without visiting the
/// skipped trips. `first_candidate[rank]` is the first rank at or after
/// `rank` in the same line whose factor resolves; `next_cleaner[rank]`
/// is the next rank after `rank` in the same line whose factor is
/// strictly cleaner. `u32::MAX` ends a chain.
pub(crate) struct CleanerChains {
    pub(super) first_candidate: Vec<u32>,
    pub(super) next_cleaner: Vec<u32>,
}

impl CleanerChains {
    pub(crate) fn build(view: &DayView, factors: &[f64]) -> CleanerChains {
        let trips = view.trip_count() as usize;
        let mut first_candidate = vec![u32::MAX; trips];
        let mut next_cleaner = vec![u32::MAX; trips];
        let mut stack: Vec<u32> = Vec::new();
        for line in 0..view.line_count() {
            // Backward over the line: the classic next-smaller-element
            // stack yields each rank's next strictly cleaner trip, and
            // the running first resolved rank fills `first_candidate`.
            stack.clear();
            let mut first = u32::MAX;
            for rank in view.line_trips(line).rev() {
                let factor = factors[view.backing(ViewTrip(rank)).0 as usize];
                if factor.is_finite() {
                    while let Some(&top) = stack.last() {
                        let top_factor = factors[view.backing(ViewTrip(top)).0 as usize];
                        if top_factor >= factor {
                            stack.pop();
                        } else {
                            break;
                        }
                    }
                    next_cleaner[rank as usize] = stack.last().copied().unwrap_or(u32::MAX);
                    stack.push(rank);
                    first = rank;
                }
                first_candidate[rank as usize] = first;
            }
        }
        CleanerChains {
            first_candidate,
            next_cleaner,
        }
    }

    /// The boarding candidates from the first boardable rank, in order:
    /// the first resolved trip, then each strictly cleaner successor.
    pub(crate) fn candidates(&self, first: u32) -> CleanerChain<'_> {
        CleanerChain {
            chains: self,
            next: self.first_candidate[first as usize],
            started: false,
        }
    }
}

pub(crate) struct CleanerChain<'a> {
    pub(super) chains: &'a CleanerChains,
    pub(super) next: u32,
    pub(super) started: bool,
}

impl Iterator for CleanerChain<'_> {
    type Item = u32;

    fn next(&mut self) -> Option<u32> {
        if self.started && self.next != u32::MAX {
            self.next = self.chains.next_cleaner[self.next as usize];
        }
        self.started = true;
        (self.next != u32::MAX).then_some(self.next)
    }
}

/// Grams CO₂e accumulated along `trip` from its first position to
/// `position`; callers subtract the boarding position's value to
/// anchor a leg's grams at the real boarding.
pub(super) fn absolute_grams(
    geometry: &TripGeometry,
    view: &DayView,
    factors: &[f64],
    trip: ViewTrip,
    position: u16,
) -> f64 {
    let backing = view.backing(trip);
    geometry.leg_distance(backing, 0, position) as f64 / 1000.0 * factors[backing.0 as usize]
}

/// A one-trip label during the set enumeration: the journey's arrival
/// and grams at a stop, and — when it rode a trip — the alight event a
/// kept transfer would depart from. `None` marks the zero-trip witness
/// (staying at the source).
#[derive(Debug, Clone, Copy)]
pub(super) struct OneCtx {
    pub(super) arrival: u32,
    pub(super) grams: f64,
    pub(super) event: Option<(ViewTrip, u16)>,
}

/// A two-trip label: the journey's arrival and grams, and the candidate
/// transfer it would emit — `None` for witnesses (one- or zero-trip
/// journeys mirrored into the two-trip bags).
#[derive(Debug, Clone, Copy)]
pub(super) struct TwoCtx {
    pub(super) arrival: u32,
    pub(super) grams: f64,
    pub(super) pair: Option<(ViewTrip, u16, TripTransfer)>,
}

impl OneCtx {
    /// The label's context-independent identity, witnesses first.
    fn rank(&self) -> (u8, u32, u16) {
        match self.event {
            None => (0, 0, 0),
            Some((trip, position)) => (1, trip.0, position),
        }
    }
}

/// Whether `a` dominates `b` in a one-trip bag: strictly better on
/// (arrival, grams), or an exact tie resolved by the global identity
/// order — witnesses beat candidates, then the lower event wins. The
/// tie order is context-independent so every enumeration context
/// elects the same canonical label, keeping witness chains
/// substitutable across contexts (an inconsistent tie-break can evict
/// a transfer in every context that needs it, each citing the other).
pub(super) fn dominates_one(a: &OneCtx, b: &OneCtx) -> bool {
    if a.arrival > b.arrival || a.grams > b.grams {
        return false;
    }
    if a.arrival < b.arrival || a.grams < b.grams {
        return true;
    }
    a.rank() <= b.rank()
}

/// The two-trip counterpart of [`dominates_one`], with the candidate
/// transfer pair as the identity.
pub(super) fn dominates_two(a: &TwoCtx, b: &TwoCtx) -> bool {
    if a.arrival > b.arrival || a.grams > b.grams {
        return false;
    }
    if a.arrival < b.arrival || a.grams < b.grams {
        return true;
    }
    a.rank() <= b.rank()
}

impl TwoCtx {
    /// The label's context-independent identity, witnesses first.
    fn rank(&self) -> (u8, u32, u16, u32, u16) {
        match self.pair {
            None => (0, 0, 0, 0, 0),
            Some((first, alight, transfer)) => {
                (1, first.0, alight, transfer.trip.0, transfer.position)
            }
        }
    }

    /// Whether `other` is the same frontier point — used at extraction
    /// to detect a candidate its bag has since evicted.
    fn same(&self, other: &TwoCtx) -> bool {
        self.arrival == other.arrival
            && self.grams.to_bits() == other.grams.to_bits()
            && match (self.pair, other.pair) {
                (None, None) => true,
                (Some(a), Some(b)) => a == b,
                _ => false,
            }
    }
}

/// A per-worker enumeration from one source stop: per-stop one- and
/// two-trip Pareto bags under [`dominates_one`]/[`dominates_two`] —
/// strict dominance always evicts, an exact tie resolves by the
/// global identity order — shared across the source's descending
/// departure runs
/// (a witness from a later departure is catchable from every earlier
/// context, so its evictions are sound), with each run's candidates
/// extracted **at the end of that run**, before any earlier-departure
/// label exists.
pub(super) struct SetSearch<'a> {
    pub(super) view: &'a DayView,
    pub(super) timetable: &'a Timetable,
    pub(super) geometry: &'a TripGeometry,
    pub(super) factors: &'a [f64],
    pub(super) chains: &'a CleanerChains,
    pub(super) one: Vec<Vec<OneCtx>>,
    pub(super) two: Vec<Vec<TwoCtx>>,
    pub(super) touched: Vec<u32>,
    pub(super) marked: Vec<bool>,
    pub(super) out: Vec<(u32, u16, TripTransfer)>,
}

impl<'a> SetSearch<'a> {
    fn new(
        view: &'a DayView,
        timetable: &'a Timetable,
        geometry: &'a TripGeometry,
        factors: &'a [f64],
        chains: &'a CleanerChains,
    ) -> SetSearch<'a> {
        let stops = timetable.stop_count() as usize;
        SetSearch {
            view,
            timetable,
            geometry,
            factors,
            chains,
            one: vec![Vec::new(); stops],
            two: vec![Vec::new(); stops],
            touched: Vec::new(),
            marked: vec![false; stops],
            out: Vec::new(),
        }
    }

    /// Every unique origin departure at `source`, descending.
    fn departures_at(&self, source: StopIdx) -> Vec<u32> {
        let mut departures = Vec::new();
        for served in self.timetable.patterns_at_stop(source) {
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
                let offset = self.view.line_day_offset(line);
                for rank in self.view.line_trips(line) {
                    let times = self.view.stored_times(self.timetable, ViewTrip(rank));
                    // A previous-day trip's pre-midnight events are
                    // yesterday's departures, not boardable today.
                    if let Some(departure) = times[served.position as usize]
                        .departure
                        .checked_sub(offset)
                    {
                        departures.push(departure);
                    }
                }
            }
        }
        departures.sort_unstable_by(|a, b| b.cmp(a));
        departures.dedup();
        departures
    }

    fn touch(&mut self, stop: StopIdx) {
        if !self.marked[stop.0 as usize] {
            self.marked[stop.0 as usize] = true;
            self.touched.push(stop.0);
        }
    }

    fn insert_one(&mut self, stop: StopIdx, label: OneCtx) -> bool {
        let bag = &self.one[stop.0 as usize];
        if bag.iter().any(|entry| dominates_one(entry, &label)) {
            return false;
        }
        self.touch(stop);
        let bag = &mut self.one[stop.0 as usize];
        bag.retain(|entry| !dominates_one(&label, entry));
        bag.push(label);
        true
    }

    fn insert_two(&mut self, stop: StopIdx, label: TwoCtx) -> bool {
        let bag = &self.two[stop.0 as usize];
        if bag.iter().any(|entry| dominates_two(entry, &label)) {
            return false;
        }
        self.touch(stop);
        let bag = &mut self.two[stop.0 as usize];
        bag.retain(|entry| !dominates_two(&label, entry));
        bag.push(label);
        true
    }

    fn run(&mut self, source: StopIdx) -> Vec<(u32, u16, TripTransfer)> {
        for &stop in &self.touched {
            self.one[stop as usize].clear();
            self.two[stop as usize].clear();
            self.marked[stop as usize] = false;
        }
        self.touched.clear();
        self.out.clear();
        for departure in self.departures_at(source) {
            self.run_departure(source, departure);
        }
        std::mem::take(&mut self.out)
    }

    fn run_departure(&mut self, source: StopIdx, departure: u32) {
        // The zero-trip witness: staying at the source.
        self.insert_one(
            source,
            OneCtx {
                arrival: departure,
                grams: 0.0,
                event: None,
            },
        );
        self.insert_two(
            source,
            TwoCtx {
                arrival: departure,
                grams: 0.0,
                pair: None,
            },
        );
        // Round 1: the first-trip frontier from the source.
        let mut fresh: Vec<(StopIdx, OneCtx)> = Vec::new();
        for served in self.timetable.patterns_at_stop(source) {
            let stops = self.timetable.pattern_stops(served.pattern);
            if served.position as usize + 1 >= stops.len() {
                continue;
            }
            for line in self
                .view
                .lines_of_pattern(served.pattern)
                .into_iter()
                .flatten()
            {
                let Some(first) =
                    earliest_boardable(self.view, self.timetable, line, served.position, departure)
                else {
                    continue;
                };
                let offset = self.view.line_day_offset(line);
                for rank in self.chains.candidates(first.0) {
                    let trip = ViewTrip(rank);
                    let times = self.view.stored_times(self.timetable, trip);
                    let boarded_from = absolute_grams(
                        self.geometry,
                        self.view,
                        self.factors,
                        trip,
                        served.position,
                    );
                    for alight in served.position as usize + 1..stops.len() {
                        let arrival = times[alight].arrival - offset;
                        let grams = quantized(
                            absolute_grams(
                                self.geometry,
                                self.view,
                                self.factors,
                                trip,
                                alight as u16,
                            ) - boarded_from,
                        );
                        let label = OneCtx {
                            arrival,
                            grams,
                            event: Some((trip, alight as u16)),
                        };
                        if self.insert_one(stops[alight], label) {
                            fresh.push((stops[alight], label));
                            self.insert_two(
                                stops[alight],
                                TwoCtx {
                                    arrival,
                                    grams,
                                    pair: None,
                                },
                            );
                        }
                    }
                }
            }
        }
        // Round 2: transfer same-stop onto second-trip candidates —
        // walked transfers never enter the set; the query boards over
        // installed footpaths from every scanned alight itself.
        let mut inserted: Vec<(StopIdx, TwoCtx)> = Vec::new();
        for (stop, context) in fresh {
            let Some((first_trip, alight)) = context.event else {
                continue;
            };
            self.board_second(
                stop,
                context.arrival,
                first_trip,
                alight,
                context.grams,
                &mut inserted,
            );
        }
        // Extraction seals this departure's candidates before any
        // earlier-departure label can exist.
        for (stop, label) in inserted {
            if self.two[stop.0 as usize]
                .iter()
                .any(|entry| entry.same(&label))
            {
                let (trip, alight, transfer) = label.pair.expect("recorded labels are candidates");
                let record = (trip.0, alight, transfer);
                if !self.out.contains(&record) {
                    self.out.push(record);
                }
            }
        }
    }

    /// Boards every second-trip candidate at `board` ready at `ready`,
    /// riding it everywhere ahead, under the first trip's same-line
    /// skip. The `u_turn` predicate is structurally unreachable here:
    /// it requires boarding at the first trip's *previous* stop, while
    /// a same-stop transfer boards at the alight stop itself — equal
    /// only where a pattern repeats a stop consecutively, and that
    /// repeat's later context is bag-dominated before round 2 runs
    /// (it arrives no earlier and rides no less than the first
    /// visit), so no candidate the filter could drop is generated.
    fn board_second(
        &mut self,
        board: StopIdx,
        ready: u32,
        first_trip: ViewTrip,
        alight: u16,
        grams: f64,
        inserted: &mut Vec<(StopIdx, TwoCtx)>,
    ) {
        let first_line = self.view.line_of(first_trip);
        let first_factor = self.factors[self.view.backing(first_trip).0 as usize];
        for served in self.timetable.patterns_at_stop(board) {
            let stops = self.timetable.pattern_stops(served.pattern);
            if served.position as usize + 1 >= stops.len() {
                continue;
            }
            for line in self
                .view
                .lines_of_pattern(served.pattern)
                .into_iter()
                .flatten()
            {
                let Some(first) =
                    earliest_boardable(self.view, self.timetable, line, served.position, ready)
                else {
                    continue;
                };
                let offset = self.view.line_day_offset(line);
                for rank in self.chains.candidates(first.0) {
                    let boarded = ViewTrip(rank);
                    let boarded_factor = self.factors[self.view.backing(boarded).0 as usize];
                    if line == first_line
                        && boarded.0 >= first_trip.0
                        && served.position >= alight
                        && boarded_factor >= first_factor
                    {
                        continue;
                    }
                    let pair = Some((
                        first_trip,
                        alight,
                        TripTransfer {
                            trip: boarded,
                            position: served.position,
                        },
                    ));
                    let times = self.view.stored_times(self.timetable, boarded);
                    let boarded_from = absolute_grams(
                        self.geometry,
                        self.view,
                        self.factors,
                        boarded,
                        served.position,
                    );
                    for pos in served.position as usize + 1..stops.len() {
                        let arrival = times[pos].arrival - offset;
                        let ridden = absolute_grams(
                            self.geometry,
                            self.view,
                            self.factors,
                            boarded,
                            pos as u16,
                        ) - boarded_from;
                        let label = TwoCtx {
                            arrival,
                            grams: quantized(grams + ridden),
                            pair,
                        };
                        if self.insert_two(stops[pos], label) {
                            inserted.push((stops[pos], label));
                        }
                    }
                }
            }
        }
    }
}
