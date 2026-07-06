//! Trip-Based Transit Routing: the precomputed trip-to-trip transfer set.
//!
//! TBTR (Witt, ESA 2015) replaces RAPTOR's per-stop labels with trip
//! segments linked by precomputed transfers: alight a trip at a
//! position, walk (or stay at the stop), board another trip at a
//! position. Generation keeps, per reachable (pattern, position), only
//! the earliest boardable trip; a reduction pass then drops transfers
//! that improve no stop's arrival over staying on the trip or over the
//! transfers already kept — typically the large majority — leaving the
//! set the query engine scans.
//!
//! Transfers are computed on the timetable's stored clock times, so
//! they hold within one service day (over-midnight trips included,
//! their times simply exceed 24 h). How the previous service day joins
//! a query is the range engine's concern, not the set's.

use rayon::prelude::*;

use crate::timetable::{PatternIdx, StopIdx, Timetable, TripIdx};
use crate::transfers::Transfers;

const UNREACHED: u32 = u32::MAX;

/// One precomputed transfer: board `trip` at `position` of its pattern.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct TripTransfer {
    pub trip: TripIdx,
    pub position: u16,
}

/// The reduced trip-to-trip transfer set, in CSR layout keyed by
/// (trip, alight position).
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct TransferSet {
    /// Base slot of each trip's positions plus a tail: alight position
    /// `i` of trip `t` is slot `trip_base[t] + i`.
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
    /// Generates and reduces the transfer set of a timetable and its
    /// footpaths, fanned out over trips with rayon. Deterministic:
    /// each trip's transfers depend only on the shared inputs.
    pub fn build(timetable: &Timetable, footpaths: &Transfers) -> TransferSetBuild {
        let trip_count = timetable.trip_count();
        let per_trip: Vec<(Vec<Vec<TripTransfer>>, usize)> = (0..trip_count)
            .into_par_iter()
            .map_init(
                || Labels::new(timetable.stop_count()),
                |labels, trip| {
                    let trip = TripIdx(trip);
                    let mut generated = generate(timetable, footpaths, trip);
                    let count = generated.iter().map(Vec::len).sum();
                    reduce(timetable, footpaths, trip, labels, &mut generated);
                    (generated, count)
                },
            )
            .collect();
        let generated = per_trip.iter().map(|(_, count)| count).sum();
        let mut trip_base = Vec::with_capacity(trip_count as usize + 1);
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
    pub fn from_trip_position(&self, trip: TripIdx, position: u16) -> &[TripTransfer] {
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

/// The feasible transfers of one trip, per alight position: for every
/// stop reachable from the alight stop (itself, or over a footpath),
/// the earliest boardable trip of each (pattern, position) serving it —
/// skipping same-pattern transfers that stay on the trip or board a
/// later sibling no earlier along the pattern (they cannot help under
/// FIFO), and U-turns (reboarding the segment just ridden when the
/// boarded trip was already catchable at the previous stop).
fn generate(timetable: &Timetable, footpaths: &Transfers, trip: TripIdx) -> Vec<Vec<TripTransfer>> {
    let pattern = timetable.trip_pattern(trip);
    let stops = timetable.pattern_stops(pattern);
    let times = timetable.trip_stop_times(trip);
    let mut per_position: Vec<Vec<TripTransfer>> = vec![Vec::new(); stops.len()];
    for (alight, kept) in per_position.iter_mut().enumerate().skip(1) {
        let arrival = times[alight].arrival;
        let stop = stops[alight];
        let mut board_from = |at: StopIdx, ready: u32| {
            for served in timetable.patterns_at_stop(at) {
                let candidate =
                    earliest_boardable(timetable, served.pattern, served.position, ready);
                let Some(boarded) = candidate else { continue };
                if served.pattern == pattern
                    && boarded.0 >= trip.0
                    && served.position as usize >= alight
                {
                    continue;
                }
                if u_turn(timetable, stops, times, alight, boarded, served.position) {
                    continue;
                }
                kept.push(TripTransfer {
                    trip: boarded,
                    position: served.position,
                });
            }
        };
        board_from(stop, arrival);
        for footpath in footpaths.from_stop(stop) {
            board_from(footpath.to, arrival.saturating_add(footpath.duration));
        }
    }
    per_position
}

/// The earliest trip of `pattern` boardable at `position` no earlier
/// than `ready`; `None` when none departs in time or the position is
/// the pattern's last (nothing left to ride).
fn earliest_boardable(
    timetable: &Timetable,
    pattern: PatternIdx,
    position: u16,
    ready: u32,
) -> Option<TripIdx> {
    if position as usize + 1 >= timetable.pattern_stops(pattern).len() {
        return None;
    }
    let range = timetable.pattern_trip_range(pattern);
    // Patterns are FIFO chains: departures at every position are
    // non-decreasing with trip rank, so binary search is valid.
    let departs_before =
        |trip: u32| timetable.trip_stop_times(TripIdx(trip))[position as usize].departure < ready;
    let (mut low, mut high) = (range.start, range.end);
    while low < high {
        let middle = low + (high - low) / 2;
        if departs_before(middle) {
            low = middle + 1;
        } else {
            high = middle;
        }
    }
    (low < range.end).then_some(TripIdx(low))
}

/// Whether boarding `boarded` at `position` just rides back over the
/// segment `trip` arrived on, when it was already catchable one stop
/// earlier — the classic redundant U-turn.
fn u_turn(
    timetable: &Timetable,
    stops: &[StopIdx],
    times: &[crate::timetable::StopTime],
    alight: usize,
    boarded: TripIdx,
    position: u16,
) -> bool {
    let boarded_stops = timetable.pattern_stops(timetable.trip_pattern(boarded));
    let j = position as usize;
    j + 1 < boarded_stops.len()
        && boarded_stops[j] == stops[alight - 1]
        && boarded_stops[j + 1] == stops[alight]
        && times[alight - 1].arrival <= timetable.trip_stop_times(boarded)[j].departure
}

/// Witt's transfer reduction for one trip: walking the alight positions
/// back to front, a transfer survives only if riding the boarded trip
/// onward improves the arrival at some stop (directly or over a
/// footpath) over staying on the trip or over the transfers already
/// kept. Labels are per-trip scratch state, pooled per worker.
fn reduce(
    timetable: &Timetable,
    footpaths: &Transfers,
    trip: TripIdx,
    labels: &mut Labels,
    per_position: &mut [Vec<TripTransfer>],
) {
    labels.clear();
    let stops = timetable.pattern_stops(timetable.trip_pattern(trip));
    let times = timetable.trip_stop_times(trip);
    for alight in (1..stops.len()).rev() {
        let arrival = times[alight].arrival;
        labels.improve(stops[alight], arrival);
        for footpath in footpaths.from_stop(stops[alight]) {
            labels.improve(footpath.to, arrival.saturating_add(footpath.duration));
        }
        per_position[alight].retain(|transfer| {
            let boarded_stops = timetable.pattern_stops(timetable.trip_pattern(transfer.trip));
            let boarded_times = timetable.trip_stop_times(transfer.trip);
            let mut keeps = false;
            for k in transfer.position as usize + 1..boarded_stops.len() {
                let reached = boarded_times[k].arrival;
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
    use crate::timetable::{StopTime, TimetableBuilder};

    fn time(at: u32) -> StopTime {
        StopTime {
            arrival: at,
            departure: at,
        }
    }

    /// Line A rides 0→1→2; line B rides 1→3 at 90, 120, and 200.
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
            .add_trip(b, vec![time(120), time(500)], 2, 0)
            .unwrap();
        builder
            .add_trip(b, vec![time(200), time(600)], 3, 0)
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
            set.from_trip_position(TripIdx(0), 1),
            &[TripTransfer {
                trip: TripIdx(2),
                position: 0,
            }]
        );
        // Nothing rides on from A's or B's last stop.
        assert!(set.from_trip_position(TripIdx(0), 2).is_empty());
        assert!(set.from_trip_position(TripIdx(2), 1).is_empty());
        assert_eq!(build.generated, set.len());
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
        assert!(set.from_trip_position(TripIdx(0), 1).is_empty());
        assert_eq!(
            set.from_trip_position(TripIdx(0), 2),
            &[TripTransfer {
                trip: TripIdx(1),
                position: 1,
            }]
        );
    }
}
