//! Service-calendar resolution and feed QA on the Helsinki region GTFS
//! feed shared with r5py (r5py.sampledata.helsinki v1.1.1; the feed covers
//! 2022-02-22 to 2022-04-07, and 2022-02-22 08:30 is r5py's canonical test
//! departure).

mod common;

use std::sync::OnceLock;

use cafein_core::timetable::TripIdx;
use cafein_gtfs::{build_timetable, validate_feed, Feed, TimetableBuild};
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

fn date(year: i32, month: u32, day: u32) -> NaiveDate {
    NaiveDate::from_ymd_opt(year, month, day).unwrap()
}

#[test]
fn resolves_active_services_and_trips_per_date() {
    let Some((_, build)) = helsinki() else {
        return;
    };
    assert_eq!(build.services.service_count(), 4068);

    // Tuesday 2022-02-22 (two calendar_dates exceptions apply on this day)
    // and Saturday 2022-02-26; expected values computed independently from
    // the raw GTFS tables.
    for (day, expected_services, expected_trips) in [
        (date(2022, 2, 22), 469, 24_280),
        (date(2022, 2, 26), 307, 17_427),
    ] {
        let active = build.services.active_on(day);
        assert_eq!(
            active.iter().filter(|running| **running).count(),
            expected_services
        );
        let running_trips = (0..build.timetable.trip_count())
            .filter(|trip| active[build.timetable.trip_service(TripIdx(*trip)) as usize])
            .count();
        assert_eq!(running_trips, expected_trips);
    }
}

#[test]
fn night_bus_runs_on_saturdays_only() {
    let Some((feed, build)) = helsinki() else {
        return;
    };
    let trip = feed
        .trips
        .iter()
        .find(|trip| trip.id == "2235N_20220222_La_1_2835")
        .unwrap();
    let service = build.services.index(trip.feed, &trip.service_id).unwrap();
    assert!(build.services.runs_on(service, date(2022, 2, 26)));
    assert!(!build.services.runs_on(service, date(2022, 2, 22)));
    // Outside the feed's validity window nothing runs.
    assert!(!build.services.runs_on(service, date(2022, 4, 9)));
}

#[test]
fn the_feed_is_qa_clean() {
    let Some((feed, build)) = helsinki() else {
        return;
    };
    assert_eq!(validate_feed(feed, &build.services), vec![]);
}
