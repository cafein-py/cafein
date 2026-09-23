use super::*;

/// A one-trip feed whose routes.txt carries the given extra header
/// columns and row values, plus `extra_files` (replacing a table of
/// the same name), as zip bytes.
fn minimal_feed_zip(
    extra_columns: &str,
    extra_values: &str,
    extra_files: &[(&str, &str)],
) -> Vec<u8> {
    let mut writer = zip::ZipWriter::new(Cursor::new(Vec::new()));
    let options = zip::write::SimpleFileOptions::default();
    let mut files = vec![
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
    for (name, content) in extra_files {
        match files.iter_mut().find(|(existing, _)| existing == name) {
            Some(entry) => entry.1 = content.to_string(),
            None => files.push((name, content.to_string())),
        }
    }
    for (name, content) in files {
        writer.start_file(name, options).unwrap();
        writer.write_all(content.as_bytes()).unwrap();
    }
    writer.finish().unwrap().into_inner()
}

fn read_zip_bytes(tag: &str, bytes: &[u8]) -> Result<Feed, Error> {
    // Tests run concurrently: every call gets its own file.
    static CALLS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let call = CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let path = std::env::temp_dir().join(format!(
        "cafein-read-test-{}-{tag}-{call}.zip",
        std::process::id()
    ));
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
        &minimal_feed_zip(",route_color,route_text_color", ",FFFFFF,0", &[]),
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
    let ragged = minimal_feed_zip(",route_text_color", ",0,i-am-an-extra-field", &[]);
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

#[test]
fn skips_unused_tables_that_fail_to_parse() {
    // rider_categories.txt in its pre-2024 draft layout lacks the
    // column the parser requires; routing never reads the table, so
    // the feed loads and the skip is reported with its cause. A
    // transfer to a stop the feed lacks is dropped silently, and a
    // clean feed_info.txt is kept.
    let feed = read_zip_bytes(
        "skip",
        &minimal_feed_zip(
            "",
            "",
            &[
                (
                    "rider_categories.txt",
                    "rider_category_id,rider_category_name,min_age,max_age\nadult,Adult,,\n",
                ),
                (
                    "transfers.txt",
                    "from_stop_id,to_stop_id,transfer_type\nS1,NOWHERE,0\n",
                ),
                (
                    "feed_info.txt",
                    "feed_publisher_name,feed_publisher_url,feed_lang\nPub,http://example.com,fi\n",
                ),
            ],
        ),
    )
    .unwrap();
    assert_eq!(feed.trips.len(), 1);
    assert_eq!(feed.feed_infos.len(), 1);
    assert_eq!(feed.skipped_files.len(), 1);
    let skipped = &feed.skipped_files[0];
    assert_eq!(skipped.feed, 0);
    assert_eq!(skipped.file_name, "rider_categories.txt");
    assert!(
        skipped.reason.contains("is_default_fare_category") && skipped.reason.contains("line: 2"),
        "{}",
        skipped.reason
    );
}

#[test]
fn keeps_errors_in_tables_routing_needs() {
    // A malformed feed_info.txt is skipped like any unused table, but
    // a malformed calendar.txt would silently change which trips run:
    // it stays fatal, and the message states the cause.
    let feed = read_zip_bytes(
        "feed-info",
        &minimal_feed_zip("", "", &[("feed_info.txt", "feed_lang\nfi\n")]),
    )
    .unwrap();
    assert!(feed.feed_infos.is_empty());
    assert_eq!(feed.skipped_files[0].file_name, "feed_info.txt");
    let error = read_zip_bytes(
        "calendar",
        &minimal_feed_zip(
            "",
            "",
            &[(
                "calendar.txt",
                "service_id,monday,tuesday,wednesday,thursday,friday,saturday,sunday,\
                 start_date,end_date\nSV,1,1,1,1,1,1,1,not-a-date,20221231\n",
            )],
        ),
    )
    .unwrap_err();
    let message = crate::error_chain(&error);
    assert!(
        message.contains("calendar.txt") && message.contains("line: 2"),
        "{message}"
    );
}

#[test]
fn expands_frequency_templates_into_runs() {
    // T1 has two windows (one per exact_times value, adjoining), a row
    // with a zero headway, and a row whose runs would pass the end of
    // the representable clock; T2's only row ends before it starts; T9
    // is not a trip. Runs replace T1 in place, T2 is omitted, and each
    // problem is reported once.
    let feed = read_zip_bytes(
        "frequencies",
        &minimal_feed_zip(
            "",
            "",
            &[
                (
                    "trips.txt",
                    "route_id,service_id,trip_id\nR1,SV,T1\nR1,SV,T2\n",
                ),
                (
                    "stop_times.txt",
                    "trip_id,arrival_time,departure_time,stop_id,stop_sequence\n\
                     T1,08:00:00,08:00:00,S1,1\nT1,08:10:00,08:10:00,S2,2\n\
                     T2,09:00:00,09:00:00,S1,1\nT2,09:10:00,09:10:00,S2,2\n",
                ),
                (
                    "frequencies.txt",
                    "trip_id,start_time,end_time,headway_secs,exact_times\n\
                     T1,06:00:00,07:00:00,1200,1\nT1,07:00:00,07:30:00,900,0\n\
                     T1,08:00:00,09:00:00,0,1\nT1,1193046:20:00,1193046:28:00,240,1\n\
                     T2,10:00:00,09:00:00,600,1\nT9,06:00:00,07:00:00,600,1\n",
                ),
            ],
        ),
    )
    .unwrap();
    let departures: Vec<u32> = feed
        .trips
        .iter()
        .map(|trip| trip.stop_times[0].departure.unwrap())
        .collect();
    assert_eq!(departures, vec![21600, 22800, 24000, 25200, 26100]);
    assert!(feed.trips.iter().all(|trip| trip.id == "T1"));
    let last = feed.trips.last().unwrap();
    assert_eq!(last.stop_times[1].arrival, Some(26100 + 600));
    assert_eq!(last.stop_times.len(), 2);
    let reported: Vec<(&str, &str)> = feed
        .skipped_frequencies
        .iter()
        .map(|skipped| (skipped.trip_id.as_str(), skipped.reason.as_str()))
        .collect();
    assert_eq!(reported.len(), 5, "{reported:?}");
    assert_eq!(reported[0].0, "T1");
    assert!(reported[0].1.contains("headway_secs is 0"), "{reported:?}");
    assert!(reported[1].1.contains("outside the clock"), "{reported:?}");
    assert_eq!(reported[2].0, "T2");
    assert!(
        reported[2].1.contains("end_time is not after"),
        "{reported:?}"
    );
    assert_eq!(
        reported[3],
        (
            "T2",
            "omitted: none of its frequencies.txt rows could be expanded"
        )
    );
    assert_eq!(reported[4].0, "T9");
    assert!(reported[4].1.contains("no such trip"), "{reported:?}");
    // A row whose runs would outgrow the feed-wide budget refuses the
    // feed before any run is built.
    let error = read_zip_bytes(
        "frequency-budget",
        &minimal_feed_zip(
            "",
            "",
            &[(
                "frequencies.txt",
                "trip_id,start_time,end_time,headway_secs\nT1,00:00:00,1193046:00:00,1\n",
            )],
        ),
    )
    .unwrap_err();
    assert!(
        crate::error_chain(&error).contains("more than 100000000 stop times"),
        "{error}"
    );
    // A frequencies.txt that fails to parse fails the feed.
    let error = read_zip_bytes(
        "bad-frequencies",
        &minimal_feed_zip(
            "",
            "",
            &[(
                "frequencies.txt",
                "trip_id,start_time,end_time,headway_secs\nT1,06:00:00,07:00:00,soon\n",
            )],
        ),
    )
    .unwrap_err();
    assert!(crate::error_chain(&error).contains("frequencies.txt"));
}

/// Two trips over the minimal feed's stops, with the given trips.txt
/// and stop_times.txt tables.
fn two_trip_feed_zip(trips: &str, stop_times: &str) -> Vec<u8> {
    minimal_feed_zip(
        "",
        "",
        &[("trips.txt", trips), ("stop_times.txt", stop_times)],
    )
}

#[test]
fn drops_trips_whose_rows_fail_to_parse() {
    // A stop_times.txt row with a malformed time drops its trip whole
    // (both of T2's rows), keeps T1, and is reported with its line.
    let feed = read_zip_bytes(
        "bad-stop-time",
        &two_trip_feed_zip(
            "route_id,service_id,trip_id\nR1,SV,T1\nR1,SV,T2\n",
            "trip_id,arrival_time,departure_time,stop_id,stop_sequence\n\
             T1,08:00:00,08:00:00,S1,1\nT1,08:10:00,08:10:00,S2,2\n\
             T2,9:00,09:00:00,S1,1\nT2,09:10:00,09:10:00,S2,2\n",
        ),
    )
    .unwrap();
    let ids: Vec<&str> = feed.trips.iter().map(|trip| trip.id.as_str()).collect();
    assert_eq!(ids, ["T1"]);
    assert_eq!(feed.trips[0].stop_times.len(), 2);
    let report = &feed.dropped_rows[0];
    assert_eq!(feed.dropped_rows.len(), 1);
    assert_eq!(
        (
            report.file_name.as_str(),
            report.rows,
            report.first_line,
            report.trips_dropped
        ),
        ("stop_times.txt", 1, 4, 1)
    );
    assert!(
        report.first_reason.contains("line: 4"),
        "{}",
        report.first_reason
    );
    // A trips.txt row that fails drops the trip and its stop times, so
    // the parser never meets an orphan.
    let feed = read_zip_bytes(
        "bad-trip",
        &two_trip_feed_zip(
            "route_id,service_id,trip_id,direction_id\nR1,SV,T1,0\nR1,SV,T2,x\n",
            "trip_id,arrival_time,departure_time,stop_id,stop_sequence\n\
             T1,08:00:00,08:00:00,S1,1\nT1,08:10:00,08:10:00,S2,2\n\
             T2,09:00:00,09:00:00,S1,1\nT2,09:10:00,09:10:00,S2,2\n",
        ),
    )
    .unwrap();
    assert_eq!(feed.trips.len(), 1);
    assert_eq!(feed.dropped_rows[0].file_name, "trips.txt");
    assert_eq!(feed.dropped_rows[0].trips_dropped, 1);
}

#[test]
fn keeps_row_failures_the_cascade_cannot_trust() {
    // A missing required column fails every row; a failing row whose
    // trip id is blank, and a ragged failing row (a field short or an
    // extra one: the key cannot be read by column), keep the strict
    // error.
    let tables = [
        (
            "no-sequence",
            "trip_id,arrival_time,departure_time,stop_id\n\
             T1,08:00:00,08:00:00,S1\nT1,08:10:00,08:10:00,S2\n",
        ),
        (
            "blank-key",
            "trip_id,arrival_time,departure_time,stop_id,stop_sequence\n\
             T1,08:00:00,08:00:00,S1,1\nT1,08:10:00,08:10:00,S2,2\n,09:00:00,09:00:00,S1,x\n",
        ),
        (
            "field-short",
            "arrival_time,departure_time,stop_id,stop_sequence,trip_id\n\
             08:00:00,08:00:00,S1,1,T1\n08:10:00,08:10:00,S2,2,T1\n09:00:00,S1,1,T2\n",
        ),
        (
            "extra-field",
            "trip_id,arrival_time,departure_time,stop_id,stop_sequence\n\
             T1,08:00:00,08:00:00,S1,1\nT1,08:10:00,08:10:00,S2,2\nT2,9:00,09:00:00,S1,1,extra\n",
        ),
    ];
    for (tag, stop_times) in tables {
        let error = read_zip_bytes(
            tag,
            &two_trip_feed_zip(
                "route_id,service_id,trip_id\nR1,SV,T1\nR1,SV,T2\n",
                stop_times,
            ),
        )
        .unwrap_err();
        assert!(
            crate::error_chain(&error).contains("stop_times.txt"),
            "{tag}: {error}"
        );
    }
}
