//! The TBTR trip-to-trip transfer set over the Helsinki region GTFS
//! feed shared with r5py (r5py.sampledata.helsinki v1.1.1).

mod common;

use std::sync::OnceLock;

use cafein_core::tbtr::{TransferSet, TransferSetBuild};
use cafein_core::timetable::{Timetable, TripIdx};
use cafein_core::transfers::Transfers;
use cafein_gtfs::{build_timetable, Feed};

fn helsinki() -> Option<&'static (Timetable, TransferSetBuild)> {
    static DATA: OnceLock<Option<(Timetable, TransferSetBuild)>> = OnceLock::new();
    DATA.get_or_init(|| {
        let path = common::helsinki_gtfs_path()?;
        let feed = Feed::from_path(path).unwrap();
        let timetable = build_timetable(&feed).unwrap().timetable;
        // Same-stop transfers only: footpaths come from OSM streets,
        // which the Rust integration layer does not build.
        let build = TransferSet::build(&timetable, &Transfers::empty(timetable.stop_count()));
        Some((timetable, build))
    })
    .as_ref()
}

#[test]
fn reduction_keeps_a_fraction_of_the_feasible_transfers() {
    let Some((_, build)) = helsinki() else {
        return;
    };
    assert!(!build.transfers.is_empty());
    assert!(build.transfers.len() < build.generated);
    // Pinned counts: deterministic for the pinned fixture. The
    // reduction keeps ~12 % of the feasible transfers, in line with
    // the reductions Witt reports.
    assert_eq!(build.generated, 42_937_748);
    assert_eq!(build.transfers.len(), 5_064_961);
}

#[test]
fn kept_transfers_are_feasible_and_earliest() {
    let Some((timetable, build)) = helsinki() else {
        return;
    };
    let set = &build.transfers;
    // Every kept transfer boards a catchable trip that still has track
    // ahead, and no earlier trip of the same pattern position was
    // catchable — sampled across the trip space.
    for trip in (0..timetable.trip_count()).step_by(997).map(TripIdx) {
        let stops = timetable.pattern_stops(timetable.trip_pattern(trip));
        let times = timetable.trip_stop_times(trip);
        for alight in 1..stops.len() {
            for transfer in set.from_trip_position(trip, alight as u16) {
                let boarded_pattern = timetable.trip_pattern(transfer.trip);
                let boarded_stops = timetable.pattern_stops(boarded_pattern);
                let position = transfer.position as usize;
                assert!(position + 1 < boarded_stops.len());
                assert_eq!(boarded_stops[position], stops[alight]);
                let departure = timetable.trip_stop_times(transfer.trip)[position].departure;
                assert!(departure >= times[alight].arrival);
                // Earliest boardable of its pattern position: the
                // previous trip of the pattern departs too early.
                let range = timetable.pattern_trip_range(boarded_pattern);
                if transfer.trip.0 > range.start {
                    let previous = TripIdx(transfer.trip.0 - 1);
                    assert!(
                        timetable.trip_stop_times(previous)[position].departure
                            < times[alight].arrival
                    );
                }
            }
        }
    }
}
