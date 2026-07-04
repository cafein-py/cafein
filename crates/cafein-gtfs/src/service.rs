//! Resolving which services run on a given date.

use std::collections::hash_map::Entry;
use std::collections::HashMap;

use chrono::{Datelike, NaiveDate};

use crate::model::{Exception, Feed, FeedIndex};

/// Dense index of a service in a [`ServiceCalendar`], carried on timetable
/// trips as their service identifier.
pub type ServiceIndex = u32;

#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct WeeklyPattern {
    /// Monday through Sunday.
    weekdays: [bool; 7],
    start_date: NaiveDate,
    end_date: NaiveDate,
}

/// Maps `(feed, service_id)` pairs to dense indices and resolves which
/// services run on a date, combining `calendar.txt` weekly patterns with
/// `calendar_dates.txt` exceptions.
///
/// Every service referenced anywhere in the feed — by a calendar entry, an
/// exception, or a trip — gets an index; a service referenced only by trips
/// has no calendar data and never runs.
#[derive(Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct ServiceCalendar {
    index_by_id: HashMap<(FeedIndex, String), ServiceIndex>,
    weekly: Vec<Option<WeeklyPattern>>,
    /// Per service: `(date, added)` exceptions, sorted by date.
    exceptions: Vec<Vec<(NaiveDate, bool)>>,
}

impl ServiceCalendar {
    /// Collects every service referenced by `feed`.
    pub fn from_feed(feed: &Feed) -> ServiceCalendar {
        let mut services = ServiceCalendar::default();
        for calendar in &feed.calendars {
            let service = services.intern(calendar.feed, &calendar.service_id);
            services.weekly[service as usize] = Some(WeeklyPattern {
                weekdays: calendar.weekdays,
                start_date: calendar.start_date,
                end_date: calendar.end_date,
            });
        }
        for calendar_date in &feed.calendar_dates {
            let service = services.intern(calendar_date.feed, &calendar_date.service_id);
            services.exceptions[service as usize].push((
                calendar_date.date,
                calendar_date.exception == Exception::Added,
            ));
        }
        for trip in &feed.trips {
            services.intern(trip.feed, &trip.service_id);
        }
        for exceptions in &mut services.exceptions {
            exceptions.sort();
        }
        services
    }

    fn intern(&mut self, feed: FeedIndex, service_id: &str) -> ServiceIndex {
        let next = self.weekly.len() as ServiceIndex;
        match self.index_by_id.entry((feed, service_id.to_owned())) {
            Entry::Occupied(entry) => *entry.get(),
            Entry::Vacant(entry) => {
                entry.insert(next);
                self.weekly.push(None);
                self.exceptions.push(Vec::new());
                next
            }
        }
    }

    pub fn service_count(&self) -> u32 {
        self.weekly.len() as u32
    }

    /// The dense index of a service, if the feed references it anywhere.
    pub fn index(&self, feed: FeedIndex, service_id: &str) -> Option<ServiceIndex> {
        self.index_by_id
            .get(&(feed, service_id.to_owned()))
            .copied()
    }

    /// Whether `service` runs on `date`: an exception for the date wins;
    /// otherwise the weekly pattern decides within its date range.
    pub fn runs_on(&self, service: ServiceIndex, date: NaiveDate) -> bool {
        if let Some(&(_, added)) = self.exceptions[service as usize]
            .iter()
            .find(|(exception_date, _)| *exception_date == date)
        {
            return added;
        }
        match &self.weekly[service as usize] {
            Some(pattern) => {
                date >= pattern.start_date
                    && date <= pattern.end_date
                    && pattern.weekdays[date.weekday().num_days_from_monday() as usize]
            }
            None => false,
        }
    }

    /// One flag per service: whether it runs on `date`. Indexable by the
    /// service identifiers carried on timetable trips.
    pub fn active_on(&self, date: NaiveDate) -> Vec<bool> {
        (0..self.service_count())
            .map(|service| self.runs_on(service, date))
            .collect()
    }

    /// Whether the service has any calendar data; a service referenced only
    /// by trips never runs.
    pub fn has_calendar_data(&self, service: ServiceIndex) -> bool {
        self.weekly[service as usize].is_some() || !self.exceptions[service as usize].is_empty()
    }
}

#[cfg(test)]
mod tests {
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
}
