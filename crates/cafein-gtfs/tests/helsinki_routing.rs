//! RAPTOR routing on the Helsinki region GTFS feed shared with r5py
//! (r5py.sampledata.helsinki v1.1.1), at r5py's canonical test departure
//! of 2022-02-22 08:30.

mod common;

use std::sync::OnceLock;

use cafein_core::journey::Leg;
use cafein_core::raptor::Raptor;
use cafein_core::router::{Request, TransitRouter};
use cafein_core::timetable::StopIdx;
use cafein_core::transfers::Transfers;
use cafein_gtfs::{build_timetable, Feed, TimetableBuild};
use chrono::NaiveDate;

fn helsinki() -> Option<&'static (Feed, TimetableBuild)> {
    static DATA: OnceLock<Option<(Feed, TimetableBuild)>> = OnceLock::new();
    DATA.get_or_init(|| {
        let path = common::helsinki_gtfs_path()?;
        let feed = Feed::from_path(path).unwrap();
        let build = build_timetable(&feed).unwrap();
        Some((feed, build))
    })
    .as_ref()
}

fn stop_index(feed: &Feed, stop_id: &str) -> StopIdx {
    StopIdx(
        feed.stops
            .iter()
            .position(|stop| stop.id == stop_id)
            .unwrap() as u32,
    )
}

#[test]
fn finds_the_earliest_direct_k_train() {
    let Some((feed, build)) = helsinki() else {
        return;
    };
    // Korso → Käpylä, both served directly by the K commuter train. The
    // expected values are computed independently from the raw GTFS tables:
    // the earliest direct ride departing 08:30:00 or later on 2022-02-22
    // leaves Korso at 08:36:00 and arrives at Käpylä at 08:58:00 on trip
    // 3001K_20220222_S1_2_0831.
    let korso = stop_index(feed, "4810551");
    let kapyla = stop_index(feed, "1250551");
    let date = NaiveDate::from_ymd_opt(2022, 2, 22).unwrap();
    let request = Request {
        departure: 8 * 3600 + 30 * 60,
        access: vec![(korso, 0)],
        egress: vec![(kapyla, 0)],
        active_services: build.services.active_on(date),
        max_transfers: 4,
    };
    let transfers = Transfers::empty(build.timetable.stop_count());
    let journeys = Raptor.route(&build.timetable, &transfers, &request);

    assert!(!journeys.is_empty());
    let direct = &journeys[0];
    assert_eq!(direct.rides(), 1);
    assert_eq!(direct.arrival, 8 * 3600 + 58 * 60);
    let Leg::Transit {
        trip,
        board_stop,
        alight_stop,
        board_time,
        ..
    } = direct.legs[1]
    else {
        panic!("expected a transit leg");
    };
    assert_eq!(board_stop, korso);
    assert_eq!(alight_stop, kapyla);
    assert_eq!(board_time, 8 * 3600 + 36 * 60);
    let source = build.timetable.trip_source(trip) as usize;
    assert_eq!(feed.trips[source].id, "3001K_20220222_S1_2_0831");

    // Journeys form a Pareto set: more rides only if strictly earlier.
    for pair in journeys.windows(2) {
        assert!(pair[1].rides() > pair[0].rides());
        assert!(pair[1].arrival < pair[0].arrival);
    }
}

#[test]
fn journeys_are_time_consistent() {
    let Some((feed, build)) = helsinki() else {
        return;
    };
    let korso = stop_index(feed, "4810551");
    let kapyla = stop_index(feed, "1250551");
    let date = NaiveDate::from_ymd_opt(2022, 2, 22).unwrap();
    let request = Request {
        departure: 8 * 3600 + 30 * 60,
        access: vec![(korso, 120)],
        egress: vec![(kapyla, 180)],
        active_services: build.services.active_on(date),
        max_transfers: 4,
    };
    let transfers = Transfers::empty(build.timetable.stop_count());
    for journey in Raptor.route(&build.timetable, &transfers, &request) {
        let mut clock = journey.departure;
        for leg in &journey.legs {
            let (start, end) = match *leg {
                Leg::Access {
                    departure, arrival, ..
                }
                | Leg::Transfer {
                    departure, arrival, ..
                }
                | Leg::Egress {
                    departure, arrival, ..
                } => (departure, arrival),
                Leg::Transit {
                    board_time,
                    alight_time,
                    ..
                } => (board_time, alight_time),
            };
            assert!(start >= clock);
            assert!(end >= start);
            clock = end;
        }
        assert_eq!(clock, journey.arrival);
    }
}
