//! Tests against the Helsinki region GTFS feed shared with r5py
//! (r5py.sampledata.helsinki v1.1.1, HSL feed for 2022-02-22 to 2022-04-07).

mod common;

use std::sync::OnceLock;

use cafein_gtfs::{Feed, RouteType};

const SECONDS_PER_DAY: u32 = 24 * 60 * 60;

fn helsinki_feed() -> Option<&'static Feed> {
    static FEED: OnceLock<Option<Feed>> = OnceLock::new();
    FEED.get_or_init(|| Some(Feed::from_path(common::helsinki_gtfs_path()?).unwrap()))
        .as_ref()
}

#[test]
fn reads_all_tables() {
    let Some(feed) = helsinki_feed() else {
        return;
    };
    assert_eq!(feed.feed_count, 1);
    assert_eq!(feed.agencies.len(), 1);
    assert_eq!(feed.stops.len(), 8305);
    assert_eq!(feed.routes.len(), 479);
    assert_eq!(feed.trips.len(), 195_351);
    assert_eq!(feed.calendars.len(), 4068);
    assert_eq!(feed.calendar_dates.len(), 32);

    let agency = &feed.agencies[0];
    assert_eq!(agency.name, "Helsingin seudun liikenne");
    assert_eq!(agency.timezone, "Europe/Helsinki");

    let info = &feed.feed_infos[0];
    assert_eq!(info.publisher_name, "Helsingin seudun liikenne");
    assert_eq!(info.version.as_deref(), Some("2022-02-22 22:38:36"));
}

#[test]
fn stops_are_indexed_in_id_order_with_coordinates() {
    let Some(feed) = helsinki_feed() else {
        return;
    };
    let first = &feed.stops[0];
    assert_eq!(first.id, "1000001");
    assert_eq!(
        first.name.as_deref(),
        Some("Kamppi (lähiliikenneterminaali)")
    );
    assert!((first.latitude.unwrap() - 60.169008).abs() < 1e-9);
    assert!((first.longitude.unwrap() - 24.931662).abs() < 1e-9);
    assert!(feed.stops.windows(2).all(|pair| pair[0].id < pair[1].id));
}

#[test]
fn stop_times_are_complete_and_ordered() {
    let Some(feed) = helsinki_feed() else {
        return;
    };
    let total: usize = feed.trips.iter().map(|trip| trip.stop_times.len()).sum();
    assert_eq!(total, 5_353_583);
    for trip in &feed.trips {
        assert!(trip
            .stop_times
            .windows(2)
            .all(|pair| pair[0].stop_sequence < pair[1].stop_sequence));
    }
}

#[test]
fn over_midnight_times_stay_on_their_service_day() {
    let Some(feed) = helsinki_feed() else {
        return;
    };
    let over_midnight: usize = feed
        .trips
        .iter()
        .flat_map(|trip| &trip.stop_times)
        .filter(|stop_time| stop_time.departure.is_some_and(|t| t >= SECONDS_PER_DAY))
        .count();
    assert_eq!(over_midnight, 196_514);

    let night_bus = feed
        .trips
        .iter()
        .find(|trip| trip.id == "2235N_20220222_La_1_2835")
        .unwrap();
    assert_eq!(night_bus.stop_times.len(), 72);
    // Departs 28:35:00 and arrives 29:36:00 on the Saturday service day.
    assert_eq!(night_bus.stop_times[0].departure, Some(28 * 3600 + 35 * 60));
    assert_eq!(
        night_bus.stop_times.last().unwrap().arrival,
        Some(29 * 3600 + 36 * 60)
    );

    let route = &feed.routes[night_bus.route as usize];
    assert_eq!(route.id, "2235N");
    assert_eq!(route.short_name.as_deref(), Some("235N"));
    // Extended route type 701 maps to the core bus type.
    assert_eq!(route.route_type, RouteType::Bus);
}

#[test]
fn merging_two_feeds_namespaces_entities_by_feed_index() {
    let Some(path) = common::helsinki_gtfs_path() else {
        return;
    };
    let merged = Feed::from_paths(&[&path, &path]).unwrap();

    assert_eq!(merged.feed_count, 2);
    assert_eq!(merged.stops.len(), 2 * 8305);
    assert_eq!(merged.routes.len(), 2 * 479);
    assert_eq!(merged.trips.len(), 2 * 195_351);

    assert!(merged.stops[..8305].iter().all(|stop| stop.feed == 0));
    assert!(merged.stops[8305..].iter().all(|stop| stop.feed == 1));
    assert_eq!(merged.stops[0].id, merged.stops[8305].id);

    // Trips of the second feed resolve to that feed's stop and route entries.
    let second_feed_trip = merged.trips.iter().find(|trip| trip.feed == 1).unwrap();
    assert!(second_feed_trip.route >= 479);
    assert!(second_feed_trip
        .stop_times
        .iter()
        .all(|stop_time| stop_time.stop >= 8305));
}
