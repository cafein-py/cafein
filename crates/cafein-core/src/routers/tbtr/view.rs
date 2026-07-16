//! The service-day view: shifted previous-day lines over the
//! timetable's patterns.

use super::*;

/// A virtual trip of a [`DayView`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ViewTrip(pub u32);

/// The trip universe one query date sees, grouped into FIFO lines.
///
/// A line is a pattern's active trips on one day class: today's trips
/// with their stored times, or the previous day's over-midnight tails
/// with times shifted back a day. Within a line, departures at every
/// position are non-decreasing with rank (a subset of a FIFO chain
/// stays FIFO), so boarding searches stay binary.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct DayView {
    /// Per virtual trip: the backing timetable trip. Virtual trips are
    /// contiguous per line, in line order.
    trips: Vec<TripIdx>,
    /// Per virtual trip: its line.
    trip_lines: Vec<u32>,
    /// Per virtual trip: the first position boardable on the query
    /// day's clock (nonzero only on previous-day tails).
    first_boardable: Vec<u16>,
    /// Per line: the pattern its stops come from.
    line_patterns: Vec<PatternIdx>,
    /// Per line: subtracted from stored times to land on the query
    /// day's clock (0 today, 86 400 for the previous day).
    line_offsets: Vec<u32>,
    /// CSR offsets into `trips`, one per line plus a tail.
    line_trips_offsets: Vec<u32>,
    /// Per pattern: its today and previous-day lines, where active.
    pattern_lines: Vec<[Option<u32>; 2]>,
}

impl DayView {
    /// Every trip, one line per pattern, on the stored clock — the
    /// calendar-free view. Virtual trip indexes equal timetable trip
    /// indexes.
    pub fn universal(timetable: &Timetable) -> DayView {
        DayView::assemble(timetable, |_| Some(0), |_| false)
    }

    /// The trip universe of one query date: trips of the services
    /// active on the date, plus the previous day's active trips that
    /// still have boardable track past midnight, as shifted lines.
    pub fn for_date(
        timetable: &Timetable,
        active_services: &[bool],
        active_services_previous: &[bool],
    ) -> DayView {
        let runs = |mask: &[bool], trip: TripIdx| {
            mask.get(timetable.trip_service(trip) as usize)
                .copied()
                .unwrap_or(false)
        };
        DayView::assemble(
            timetable,
            |trip| {
                if runs(active_services, trip) {
                    Some(0)
                } else {
                    None
                }
            },
            |trip| runs(active_services_previous, trip),
        )
    }

    /// Builds the line structure: per pattern, a today line of the
    /// trips `today` admits, then a previous-day line of the trips
    /// `previous` admits that are still boardable after the shift.
    fn assemble(
        timetable: &Timetable,
        today: impl Fn(TripIdx) -> Option<u32>,
        previous: impl Fn(TripIdx) -> bool,
    ) -> DayView {
        let mut view = DayView {
            trips: Vec::new(),
            trip_lines: Vec::new(),
            first_boardable: Vec::new(),
            line_patterns: Vec::new(),
            line_offsets: Vec::new(),
            line_trips_offsets: vec![0],
            pattern_lines: vec![[None; 2]; timetable.pattern_count() as usize],
        };
        for pattern in (0..timetable.pattern_count()).map(PatternIdx) {
            let stops = timetable.pattern_stops(pattern).len();
            let mut today_line = None;
            let mut previous_line = None;
            let members: Vec<TripIdx> = timetable
                .pattern_trips(pattern)
                .filter(|&trip| today(trip).is_some())
                .collect();
            if !members.is_empty() {
                today_line = Some(view.push_line(pattern, 0, &members, |_| 0));
            }
            let members: Vec<(TripIdx, u16)> = timetable
                .pattern_trips(pattern)
                .filter(|&trip| previous(trip))
                .filter_map(|trip| {
                    let times = timetable.trip_stop_times(trip);
                    let boardable = times.partition_point(|time| time.departure < DAY_SECONDS);
                    // Still boardable with track ahead after the shift.
                    (boardable + 1 < stops).then_some((trip, boardable as u16))
                })
                .collect();
            if !members.is_empty() {
                let boardable: Vec<u16> = members.iter().map(|&(_, at)| at).collect();
                let trips: Vec<TripIdx> = members.into_iter().map(|(trip, _)| trip).collect();
                previous_line =
                    Some(view.push_line(pattern, DAY_SECONDS, &trips, |rank| boardable[rank]));
            }
            view.pattern_lines[pattern.0 as usize] = [today_line, previous_line];
        }
        view
    }

    fn push_line(
        &mut self,
        pattern: PatternIdx,
        offset: u32,
        members: &[TripIdx],
        first_boardable: impl Fn(usize) -> u16,
    ) -> u32 {
        let line = self.line_patterns.len() as u32;
        self.line_patterns.push(pattern);
        self.line_offsets.push(offset);
        for (rank, &trip) in members.iter().enumerate() {
            self.trips.push(trip);
            self.trip_lines.push(line);
            self.first_boardable.push(first_boardable(rank));
        }
        self.line_trips_offsets.push(self.trips.len() as u32);
        line
    }

    /// The number of virtual trips in the view.
    pub fn trip_count(&self) -> u32 {
        self.trips.len() as u32
    }

    pub fn line_count(&self) -> u32 {
        self.line_patterns.len() as u32
    }

    /// The backing timetable trip of a virtual trip.
    pub fn backing(&self, trip: ViewTrip) -> TripIdx {
        self.trips[trip.0 as usize]
    }

    /// Subtracted from the backing trip's stored times to land on the
    /// query day's clock.
    pub fn day_offset(&self, trip: ViewTrip) -> u32 {
        self.line_offsets[self.line_of(trip) as usize]
    }

    pub fn line_of(&self, trip: ViewTrip) -> u32 {
        self.trip_lines[trip.0 as usize]
    }

    pub fn line_pattern(&self, line: u32) -> PatternIdx {
        self.line_patterns[line as usize]
    }

    pub fn line_day_offset(&self, line: u32) -> u32 {
        self.line_offsets[line as usize]
    }

    /// The virtual trips of a line, in FIFO order.
    pub fn line_trips(&self, line: u32) -> std::ops::Range<u32> {
        self.line_trips_offsets[line as usize]..self.line_trips_offsets[line as usize + 1]
    }

    /// The today and previous-day lines of a pattern, where active.
    pub fn lines_of_pattern(&self, pattern: PatternIdx) -> [Option<u32>; 2] {
        self.pattern_lines[pattern.0 as usize]
    }

    /// The first position of a virtual trip boardable on the query
    /// day's clock.
    pub fn first_boardable(&self, trip: ViewTrip) -> u16 {
        self.first_boardable[trip.0 as usize]
    }

    /// The backing trip's stored stop times (shift by the day offset to
    /// reach the query day's clock).
    pub fn stored_times<'t>(&self, timetable: &'t Timetable, trip: ViewTrip) -> &'t [StopTime] {
        timetable.trip_stop_times(self.backing(trip))
    }
}
