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
#[path = "service_tests.rs"]
mod tests;
