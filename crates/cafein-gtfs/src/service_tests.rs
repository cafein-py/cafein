use super::*;
use crate::model::{Calendar, CalendarDate};

fn date(year: i32, month: u32, day: u32) -> NaiveDate {
    NaiveDate::from_ymd_opt(year, month, day).unwrap()
}

fn weekday_service(service_id: &str, weekdays: [bool; 7]) -> Calendar {
    Calendar {
        feed: 0,
        service_id: service_id.to_string(),
        weekdays,
        start_date: date(2022, 2, 1),
        end_date: date(2022, 2, 28),
    }
}

#[test]
fn resolves_weekly_patterns_and_exceptions() {
    let feed = Feed {
        calendars: vec![weekday_service(
            "weekdays",
            [true, true, true, true, true, false, false],
        )],
        calendar_dates: vec![
            CalendarDate {
                feed: 0,
                service_id: "weekdays".to_string(),
                date: date(2022, 2, 15),
                exception: Exception::Deleted,
            },
            CalendarDate {
                feed: 0,
                service_id: "extra".to_string(),
                date: date(2022, 2, 19),
                exception: Exception::Added,
            },
        ],
        ..Feed::default()
    };
    let services = ServiceCalendar::from_feed(&feed);
    assert_eq!(services.service_count(), 2);

    let weekdays = services.index(0, "weekdays").unwrap();
    // Monday the 14th runs; Tuesday the 15th is removed by exception.
    assert!(services.runs_on(weekdays, date(2022, 2, 14)));
    assert!(!services.runs_on(weekdays, date(2022, 2, 15)));
    // Saturday is off the weekly pattern; outside the range nothing runs.
    assert!(!services.runs_on(weekdays, date(2022, 2, 19)));
    assert!(!services.runs_on(weekdays, date(2022, 3, 1)));

    // A service defined only by an added exception runs on that day only.
    let extra = services.index(0, "extra").unwrap();
    assert!(services.runs_on(extra, date(2022, 2, 19)));
    assert!(!services.runs_on(extra, date(2022, 2, 20)));

    let active = services.active_on(date(2022, 2, 19));
    assert!(!active[weekdays as usize]);
    assert!(active[extra as usize]);
}

#[test]
fn services_repeat_across_feeds_without_collision() {
    let feed = Feed {
        calendars: vec![
            weekday_service("s", [true; 7]),
            Calendar {
                feed: 1,
                ..weekday_service("s", [false; 7])
            },
        ],
        ..Feed::default()
    };
    let services = ServiceCalendar::from_feed(&feed);
    assert_eq!(services.service_count(), 2);
    let first = services.index(0, "s").unwrap();
    let second = services.index(1, "s").unwrap();
    assert!(services.runs_on(first, date(2022, 2, 14)));
    assert!(!services.runs_on(second, date(2022, 2, 14)));
}

#[test]
fn trip_only_services_have_no_calendar_data() {
    let feed = Feed {
        trips: vec![crate::Trip {
            feed: 0,
            id: "t".to_string(),
            route: 0,
            service_id: "ghost".to_string(),
            direction_id: None,
            shape_id: None,
            headsign: None,
            stop_times: Vec::new(),
        }],
        ..Feed::default()
    };
    let services = ServiceCalendar::from_feed(&feed);
    let ghost = services.index(0, "ghost").unwrap();
    assert!(!services.has_calendar_data(ghost));
    assert!(!services.runs_on(ghost, date(2022, 2, 14)));
}
