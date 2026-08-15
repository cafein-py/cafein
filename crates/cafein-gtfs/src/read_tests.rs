use super::*;

/// A one-trip feed whose routes.txt carries the given extra header
/// columns and row values, as zip bytes.
fn minimal_feed_zip(extra_columns: &str, extra_values: &str) -> Vec<u8> {
    let mut writer = zip::ZipWriter::new(Cursor::new(Vec::new()));
    let options = zip::write::SimpleFileOptions::default();
    let files = [
        (
            "agency.txt",
            "agency_id,agency_name,agency_url,agency_timezone\n\
                 A,Agency,http://example.com,Europe/Helsinki\n"
                .to_string(),
        ),
        (
            "stops.txt",
            "stop_id,stop_name,stop_lat,stop_lon\nS1,One,60.0,24.0\nS2,Two,60.01,24.01\n"
                .to_string(),
        ),
        (
            "routes.txt",
            format!("route_id,route_short_name,route_type{extra_columns}\nR1,1,3{extra_values}\n"),
        ),
        (
            "trips.txt",
            "route_id,service_id,trip_id\nR1,SV,T1\n".to_string(),
        ),
        (
            "stop_times.txt",
            "trip_id,arrival_time,departure_time,stop_id,stop_sequence\n\
                 T1,08:00:00,08:00:00,S1,1\nT1,08:10:00,08:10:00,S2,2\n"
                .to_string(),
        ),
        (
            "calendar.txt",
            "service_id,monday,tuesday,wednesday,thursday,friday,saturday,sunday,\
                 start_date,end_date\nSV,1,1,1,1,1,1,1,20220101,20221231\n"
                .to_string(),
        ),
    ];
    for (name, content) in files {
        writer.start_file(name, options).unwrap();
        writer.write_all(content.as_bytes()).unwrap();
    }
    writer.finish().unwrap().into_inner()
}

fn read_zip_bytes(tag: &str, bytes: &[u8]) -> Result<Feed, Error> {
    let path =
        std::env::temp_dir().join(format!("cafein-read-test-{}-{tag}.zip", std::process::id()));
    std::fs::write(&path, bytes).unwrap();
    let feed = Feed::from_path(&path);
    std::fs::remove_file(&path).ok();
    feed
}

#[test]
fn tolerates_invalid_route_colours() {
    // route_text_color "0" is not RRGGBB; the strict read fails and
    // the colour-less retry recovers the feed intact.
    let feed = read_zip_bytes(
        "colours",
        &minimal_feed_zip(",route_color,route_text_color", ",FFFFFF,0"),
    )
    .unwrap();
    assert_eq!(feed.routes.len(), 1);
    assert_eq!(feed.routes[0].id, "R1");
    assert_eq!(feed.trips.len(), 1);
    assert_eq!(feed.trips[0].stop_times.len(), 2);
}

#[test]
fn keeps_routes_errors_that_are_not_colours() {
    // A malformed route_type fails the colour-less retry too: the
    // fallback never masks real routes.txt problems.
    let mut writer = zip::ZipWriter::new(Cursor::new(Vec::new()));
    let options = zip::write::SimpleFileOptions::default();
    writer.start_file("routes.txt", options).unwrap();
    writer
        .write_all(b"route_id,route_short_name,route_type\nR1,1,not-a-number\n")
        .unwrap();
    let bytes = writer.finish().unwrap().into_inner();
    assert!(read_zip_bytes("route-type", &bytes).is_err());
    // A ragged row (extra field) fails the sanitizer, so a shape
    // error is never repaired into a loadable feed either.
    let ragged = minimal_feed_zip(",route_text_color", ",0,i-am-an-extra-field");
    assert!(read_zip_bytes("ragged", &ragged).is_err());
}

/// A feed whose stops and trips carry every wheelchair tri-state, a
/// station for children to inherit from, and an out-of-spec code.
fn wheelchair_feed_zip() -> Vec<u8> {
    let mut writer = zip::ZipWriter::new(Cursor::new(Vec::new()));
    let options = zip::write::SimpleFileOptions::default();
    let files = [
        (
            "agency.txt",
            "agency_id,agency_name,agency_url,agency_timezone\n\
                 A,Agency,http://example.com,Europe/Helsinki\n",
        ),
        (
            "stops.txt",
            "stop_id,stop_name,stop_lat,stop_lon,location_type,parent_station,wheelchair_boarding\n\
                 STATION,Hub,60.0,24.0,1,,1\n\
                 CHILD_BLANK,Inherits,60.0,24.0,0,STATION,\n\
                 CHILD_ZERO,Inherits too,60.0,24.0,0,STATION,0\n\
                 CHILD_OWN,Keeps its own,60.0,24.0,0,STATION,2\n\
                 LONER_YES,Accessible,60.01,24.01,0,,1\n\
                 LONER_NO,Not accessible,60.02,24.02,0,,2\n\
                 LONER_BLANK,Unknown,60.03,24.03,0,,\n\
                 LONER_ODD,Out of spec,60.04,24.04,0,,3\n",
        ),
        (
            "routes.txt",
            "route_id,route_short_name,route_type\nR1,1,3\n",
        ),
        (
            "trips.txt",
            "route_id,service_id,trip_id,wheelchair_accessible\n\
                 R1,SV,T_YES,1\nR1,SV,T_NO,2\nR1,SV,T_BLANK,\nR1,SV,T_ODD,3\n",
        ),
        (
            "stop_times.txt",
            "trip_id,arrival_time,departure_time,stop_id,stop_sequence\n\
                 T_YES,08:00:00,08:00:00,LONER_YES,1\n\
                 T_YES,08:10:00,08:10:00,LONER_NO,2\n\
                 T_NO,09:00:00,09:00:00,LONER_YES,1\n\
                 T_NO,09:10:00,09:10:00,LONER_NO,2\n\
                 T_BLANK,10:00:00,10:00:00,LONER_YES,1\n\
                 T_BLANK,10:10:00,10:10:00,LONER_NO,2\n\
                 T_ODD,11:00:00,11:00:00,LONER_YES,1\n\
                 T_ODD,11:10:00,11:10:00,LONER_NO,2\n",
        ),
        (
            "calendar.txt",
            "service_id,monday,tuesday,wednesday,thursday,friday,saturday,sunday,\
                 start_date,end_date\nSV,1,1,1,1,1,1,1,20220101,20221231\n",
        ),
    ];
    for (name, content) in files {
        writer.start_file(name, options).unwrap();
        writer.write_all(content.as_bytes()).unwrap();
    }
    writer.finish().unwrap().into_inner()
}

#[test]
fn maps_the_wheelchair_tri_states_and_inherits_the_parent_station() {
    let feed = read_zip_bytes("wheelchair", &wheelchair_feed_zip()).unwrap();
    let stop = |id: &str| {
        feed.stops
            .iter()
            .find(|stop| stop.id == id)
            .unwrap()
            .wheelchair_boarding
    };
    assert_eq!(stop("STATION"), Some(true));
    assert_eq!(stop("LONER_YES"), Some(true));
    assert_eq!(stop("LONER_NO"), Some(false));
    assert_eq!(stop("LONER_BLANK"), None);
    // An out-of-spec integer code stays unknown, never a guess.
    assert_eq!(stop("LONER_ODD"), None);
    // Blank and 0 both inherit the parent station's known value; an
    // explicit value of the stop's own wins over the parent's.
    assert_eq!(stop("CHILD_BLANK"), Some(true));
    assert_eq!(stop("CHILD_ZERO"), Some(true));
    assert_eq!(stop("CHILD_OWN"), Some(false));
    let trip = |id: &str| {
        feed.trips
            .iter()
            .find(|trip| trip.id == id)
            .unwrap()
            .wheelchair_accessible
    };
    assert_eq!(trip("T_YES"), Some(true));
    assert_eq!(trip("T_NO"), Some(false));
    assert_eq!(trip("T_BLANK"), None);
    assert_eq!(trip("T_ODD"), None);
}
