//! RAPTOR: earliest-arrival routing for a single departure time, and its
//! range (rRAPTOR) extension for a departure window.
//!
//! Round-based: round `k` finds the earliest arrivals reachable with
//! exactly `k` rides. Within a pattern, the earliest catchable trip at a
//! stop position is found by binary search over departures, which is valid
//! at every position because the timetable's patterns are FIFO chains.
//!
//! The range query runs one pass per candidate departure time, in
//! decreasing order, on shared state: arrivals found for a later departure
//! stay feasible for every earlier one, so each pass explores only what
//! its departure improves, and journeys dominated by a later departure are
//! never emitted.

use crate::journey::{Journey, Leg};
use crate::router::{Request, TransitRouter};
use crate::timetable::{PatternIdx, StopIdx, Timetable, TripIdx};
use crate::transfers::Transfers;

/// The RAPTOR router.
pub struct Raptor;

const UNREACHED: u32 = u32::MAX;

/// How a stop's arrival time in a round was achieved.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Label {
    Unreached,
    /// Reached directly from the origin.
    Access,
    /// Alighted from a trip boarded at `board_position` of its pattern.
    Transit {
        trip: TripIdx,
        board_position: u16,
        alight_position: u16,
    },
    /// Walked from another stop reached by transit in the same round.
    Transfer {
        from_stop: StopIdx,
        duration: u32,
    },
}

impl TransitRouter for Raptor {
    fn route(
        &self,
        timetable: &Timetable,
        transfers: &Transfers,
        request: &Request,
    ) -> Vec<Journey> {
        Search::new(timetable, transfers, request).profile(&[request.departure])
    }
}

impl Raptor {
    /// Earliest arrival at every stop for a single departure.
    ///
    /// One run serves all destinations — the matrix primitive: matrices
    /// are computed origin by origin, never per OD pair. The request's
    /// egress list is not consulted; unreachable stops are `None`.
    pub fn one_to_all(
        &self,
        timetable: &Timetable,
        transfers: &Transfers,
        request: &Request,
    ) -> Vec<Option<u32>> {
        let mut search = Search::new(timetable, transfers, request);
        search.run(request.departure);
        search
            .best
            .last()
            .expect("search always has a round")
            .iter()
            .map(|&arrival| (arrival != UNREACHED).then_some(arrival))
            .collect()
    }

    /// Routes over a departure window: the Pareto set of journeys over
    /// (departure, arrival, rides) for departures within
    /// `[request.departure, request.departure + window)`.
    ///
    /// Each journey's departure is the latest time the origin can be left
    /// to catch it, capped at the window's final second — a journey that
    /// leaves within the window but waits for a ride beyond it is
    /// reported with that final second as its departure. So unlike
    /// [`TransitRouter::route`] — which answers "leaving exactly at the
    /// requested time" — the result enumerates the distinct departure
    /// choices the window offers, sorted by departure and then by ride
    /// count. A zero-length window has no departures.
    pub fn route_range(
        &self,
        timetable: &Timetable,
        transfers: &Transfers,
        request: &Request,
        window: u32,
    ) -> Vec<Journey> {
        let departures = departure_candidates(timetable, request, window);
        Search::new(timetable, transfers, request).profile(&departures)
    }
}

/// The origin departure times within `[request.departure,
/// request.departure + window)` at which some trip becomes catchable: one
/// candidate per active-service trip departure at an access stop, shifted
/// back by the stop's access duration. Descending, deduplicated.
fn departure_candidates(timetable: &Timetable, request: &Request, window: u32) -> Vec<u32> {
    // Widened so a window reaching past u32::MAX cannot clip candidates.
    let end = request.departure as u64 + window as u64;
    let mut candidates = Vec::new();
    for &(stop, duration) in &request.access {
        for pattern_stop in timetable.patterns_at_stop(stop) {
            let position = pattern_stop.position as usize;
            // Boarding at a pattern's last position is pointless.
            if position + 1 == timetable.pattern_stops(pattern_stop.pattern).len() {
                continue;
            }
            for trip in timetable.pattern_trips(pattern_stop.pattern) {
                let service = timetable.trip_service(trip) as usize;
                if !request
                    .active_services
                    .get(service)
                    .copied()
                    .unwrap_or(false)
                {
                    continue;
                }
                let trip_departure = timetable.trip_stop_times(trip)[position].departure;
                let Some(origin_departure) = trip_departure.checked_sub(duration) else {
                    continue;
                };
                if origin_departure >= request.departure && (origin_departure as u64) < end {
                    candidates.push(origin_departure);
                }
            }
        }
    }
    // The window's final second is always a candidate: it covers journeys
    // that leave within the window and wait for a ride beyond it.
    if window > 0 {
        candidates.push((end - 1).min(u32::MAX as u64) as u32);
    }
    candidates.sort_unstable_by(|left, right| right.cmp(left));
    candidates.dedup();
    candidates
}

/// RAPTOR state shared by the passes of one query.
struct Search<'a> {
    timetable: &'a Timetable,
    transfers: &'a Transfers,
    request: &'a Request,
    rounds: usize,
    /// Per-round arrival times; `tau[k][stop]` is the earliest arrival at
    /// `stop` with exactly `k` rides, over all departures processed so far.
    tau: Vec<Vec<u32>>,
    labels: Vec<Vec<Label>>,
    /// Prefix minimum of `tau`: `best[k][stop]` is the earliest arrival at
    /// `stop` with at most `k` rides, over all departures processed so
    /// far. Pruning at a round must not consult later rounds — a faster
    /// but more-rides arrival from a later departure does not dominate
    /// the fewer-ride option an earlier departure still offers.
    best: Vec<Vec<u32>>,
    marked: Vec<StopIdx>,
    is_marked: Vec<bool>,
    /// First marked position per pattern for the current round.
    queue_position: Vec<u16>,
    queued_patterns: Vec<PatternIdx>,
}

impl<'a> Search<'a> {
    fn new(timetable: &'a Timetable, transfers: &'a Transfers, request: &'a Request) -> Self {
        let stop_count = timetable.stop_count() as usize;
        let rounds = request.max_transfers as usize + 1;
        Search {
            timetable,
            transfers,
            request,
            rounds,
            tau: vec![vec![UNREACHED; stop_count]; rounds + 1],
            labels: vec![vec![Label::Unreached; stop_count]; rounds + 1],
            best: vec![vec![UNREACHED; stop_count]; rounds + 1],
            marked: Vec::new(),
            is_marked: vec![false; stop_count],
            queue_position: vec![u16::MAX; timetable.pattern_count() as usize],
            queued_patterns: Vec::new(),
        }
    }

    /// Runs one pass per departure — which must be strictly decreasing —
    /// and returns the journeys no later departure dominates, sorted by
    /// departure and then by ride count.
    fn profile(mut self, departures: &[u32]) -> Vec<Journey> {
        let mut journeys = Vec::new();
        // The best arrival emitted with at most `round` rides so far; a
        // pass may only emit strictly earlier arrivals, so journeys
        // dominated by a later departure never surface.
        let mut thresholds = vec![UNREACHED; self.rounds + 1];
        for &departure in departures {
            self.run(departure);
            self.collect(departure, &mut thresholds, &mut journeys);
        }
        journeys.sort_by_key(|journey| (journey.departure, journey.rides()));
        journeys
    }

    /// One RAPTOR pass from `departure`, improving the shared state.
    fn run(&mut self, departure: u32) {
        let timetable = self.timetable;
        let request = self.request;

        // Leftover marks from the previous pass describe stops whose
        // labels are already final for later departures; they carry no
        // work for this one.
        for stop in std::mem::take(&mut self.marked) {
            self.is_marked[stop.0 as usize] = false;
        }

        for &(stop, duration) in &request.access {
            let arrival = departure.saturating_add(duration);
            let index = stop.0 as usize;
            if arrival < self.tau[0][index] {
                self.tau[0][index] = arrival;
                self.labels[0][index] = Label::Access;
                for best in &mut self.best {
                    best[index] = best[index].min(arrival);
                }
                if !self.is_marked[index] {
                    self.is_marked[index] = true;
                    self.marked.push(stop);
                }
            }
        }

        for round in 1..=self.rounds {
            self.queued_patterns.clear();
            let mut marked = std::mem::take(&mut self.marked);
            for stop in marked.drain(..) {
                self.is_marked[stop.0 as usize] = false;
                for pattern_stop in timetable.patterns_at_stop(stop) {
                    let slot = &mut self.queue_position[pattern_stop.pattern.0 as usize];
                    if *slot == u16::MAX {
                        self.queued_patterns.push(pattern_stop.pattern);
                    }
                    if pattern_stop.position < *slot {
                        *slot = pattern_stop.position;
                    }
                }
            }
            self.marked = marked;

            for index in 0..self.queued_patterns.len() {
                let pattern = self.queued_patterns[index];
                let start_position = self.queue_position[pattern.0 as usize];
                self.queue_position[pattern.0 as usize] = u16::MAX;
                let stops = timetable.pattern_stops(pattern);
                let mut current: Option<(TripIdx, u16)> = None;

                for position in start_position as usize..stops.len() {
                    let stop = stops[position].0 as usize;

                    if let Some((trip, board_position)) = current {
                        let arrival = timetable.trip_stop_times(trip)[position].arrival;
                        if arrival < self.best[round][stop] {
                            self.tau[round][stop] = arrival;
                            for best in &mut self.best[round..] {
                                best[stop] = best[stop].min(arrival);
                            }
                            self.labels[round][stop] = Label::Transit {
                                trip,
                                board_position,
                                alight_position: position as u16,
                            };
                            if !self.is_marked[stop] {
                                self.is_marked[stop] = true;
                                self.marked.push(stops[position]);
                            }
                        }
                    }

                    // Try to catch an earlier trip from this stop, using the
                    // previous round's arrival. The arrival handling above
                    // has already recorded any improvement at this position;
                    // boarding at a pattern's last position is pointless
                    // because there is no later stop to alight at, and other
                    // patterns serving this stop are queued separately.
                    let reached = self.tau[round - 1][stop];
                    if reached == UNREACHED || position + 1 == stops.len() {
                        continue;
                    }
                    let can_catch_earlier = match current {
                        Some((trip, _)) => {
                            reached <= timetable.trip_stop_times(trip)[position].departure
                        }
                        None => true,
                    };
                    if can_catch_earlier {
                        if let Some(trip) =
                            earliest_trip(timetable, request, pattern, position, reached)
                        {
                            let replaces = match current {
                                Some((current_trip, _)) => trip.0 < current_trip.0,
                                None => true,
                            };
                            if replaces {
                                current = Some((trip, position as u16));
                            }
                        }
                    }
                }
            }

            // Relax one footpath hop from every stop improved by transit,
            // leaving from the transit arrivals as they stand now — a
            // transfer improving a marked stop must not chain into that
            // stop's own outgoing transfers within the round.
            let transit_marked: Vec<(StopIdx, u32)> = self
                .marked
                .iter()
                .map(|&stop| (stop, self.tau[round][stop.0 as usize]))
                .collect();
            for (stop, departure_at_stop) in transit_marked {
                for transfer in self.transfers.from_stop(stop) {
                    let arrival = departure_at_stop.saturating_add(transfer.duration);
                    let to = transfer.to.0 as usize;
                    if arrival < self.best[round][to] {
                        self.tau[round][to] = arrival;
                        for best in &mut self.best[round..] {
                            best[to] = best[to].min(arrival);
                        }
                        self.labels[round][to] = Label::Transfer {
                            from_stop: stop,
                            duration: transfer.duration,
                        };
                        if !self.is_marked[to] {
                            self.is_marked[to] = true;
                            self.marked.push(transfer.to);
                        }
                    }
                }
            }

            if self.marked.is_empty() {
                break;
            }
        }
    }

    /// Emits the pass's journeys: per round, the best egress arrival, if
    /// it strictly beats everything already emitted with no more rides.
    fn collect(&self, departure: u32, thresholds: &mut [u32], journeys: &mut Vec<Journey>) {
        for round in 1..=self.rounds {
            let mut best_egress: Option<(u32, StopIdx, u32)> = None;
            for &(stop, duration) in &self.request.egress {
                let at_stop = self.tau[round][stop.0 as usize];
                if at_stop == UNREACHED {
                    continue;
                }
                let arrival = at_stop.saturating_add(duration);
                if best_egress.is_none_or(|(current, _, _)| arrival < current) {
                    best_egress = Some((arrival, stop, duration));
                }
            }
            let Some((arrival, egress_stop, egress_duration)) = best_egress else {
                continue;
            };
            if arrival >= thresholds[round] {
                continue;
            }
            for threshold in &mut thresholds[round..] {
                *threshold = (*threshold).min(arrival);
            }
            journeys.push(self.reconstruct(departure, round, egress_stop, egress_duration));
        }
    }

    /// Walks the labels backwards from the egress stop.
    ///
    /// Labels written by later-departure passes stay valid here: their
    /// board times are at or after the (only ever improved) arrival of the
    /// previous round, so the reconstructed chain is time-consistent.
    fn reconstruct(
        &self,
        departure: u32,
        round: usize,
        egress_stop: StopIdx,
        egress_duration: u32,
    ) -> Journey {
        let timetable = self.timetable;
        let mut legs = Vec::new();
        let departure_at_stop = self.tau[round][egress_stop.0 as usize];
        legs.push(Leg::Egress {
            from_stop: egress_stop,
            departure: departure_at_stop,
            arrival: departure_at_stop + egress_duration,
        });

        let mut current_round = round;
        let mut stop = egress_stop;
        loop {
            match self.labels[current_round][stop.0 as usize] {
                Label::Transit {
                    trip,
                    board_position,
                    alight_position,
                } => {
                    let pattern = timetable.trip_pattern(trip);
                    let pattern_stops = timetable.pattern_stops(pattern);
                    let times = timetable.trip_stop_times(trip);
                    let board_stop = pattern_stops[board_position as usize];
                    legs.push(Leg::Transit {
                        trip,
                        board_stop,
                        alight_stop: stop,
                        board_position,
                        alight_position,
                        board_time: times[board_position as usize].departure,
                        alight_time: times[alight_position as usize].arrival,
                    });
                    stop = board_stop;
                    current_round -= 1;
                }
                Label::Transfer {
                    from_stop,
                    duration,
                } => {
                    let arrival = self.tau[current_round][stop.0 as usize];
                    legs.push(Leg::Transfer {
                        from_stop,
                        to_stop: stop,
                        departure: arrival - duration,
                        arrival,
                    });
                    stop = from_stop;
                }
                Label::Access => {
                    legs.push(Leg::Access {
                        to_stop: stop,
                        departure,
                        arrival: self.tau[0][stop.0 as usize],
                    });
                    break;
                }
                Label::Unreached => unreachable!("journey reconstruction hit an unreached label"),
            }
        }
        legs.reverse();

        Journey {
            departure,
            arrival: departure_at_stop + egress_duration,
            legs,
        }
    }
}

/// The earliest trip of `pattern` catchable at `position` from time
/// `reached`, skipping trips whose service is not active. Valid because
/// departures at every position are sorted within a FIFO pattern.
fn earliest_trip(
    timetable: &Timetable,
    request: &Request,
    pattern: PatternIdx,
    position: usize,
    reached: u32,
) -> Option<TripIdx> {
    let range = timetable.pattern_trip_range(pattern);
    let (mut low, mut high) = (range.start, range.end);
    while low < high {
        let mid = low + (high - low) / 2;
        if timetable.trip_stop_times(TripIdx(mid))[position].departure < reached {
            low = mid + 1;
        } else {
            high = mid;
        }
    }
    (low..range.end).map(TripIdx).find(|&trip| {
        let service = timetable.trip_service(trip) as usize;
        request
            .active_services
            .get(service)
            .copied()
            .unwrap_or(false)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::timetable::{StopTime, TimetableBuilder};

    fn time(at: u32) -> StopTime {
        StopTime {
            arrival: at,
            departure: at,
        }
    }

    /// Five stops. Pattern A rides 0→1→2, pattern B rides 1→3, and stop 2
    /// has a 50-second footpath to stop 4, from which pattern C rides 4→3.
    fn network() -> (Timetable, Transfers) {
        let mut builder = TimetableBuilder::new(5);
        let a = builder
            .add_pattern(&[StopIdx(0), StopIdx(1), StopIdx(2)], 0)
            .unwrap();
        let b = builder.add_pattern(&[StopIdx(1), StopIdx(3)], 1).unwrap();
        let c = builder.add_pattern(&[StopIdx(4), StopIdx(3)], 2).unwrap();
        // Two trips on A so boarding must pick the second when the first
        // has already left.
        builder
            .add_trip(a, vec![time(100), time(200), time(300)], 0, 0)
            .unwrap();
        builder
            .add_trip(a, vec![time(700), time(800), time(900)], 1, 0)
            .unwrap();
        // B departs stop 1 at 250, reachable from A's first trip (arr 200).
        builder
            .add_trip(b, vec![time(250), time(400)], 2, 0)
            .unwrap();
        // A later B trip on an inactive service would be wrong to board.
        builder
            .add_trip(b, vec![time(500), time(600)], 3, 1)
            .unwrap();
        // C departs stop 4 at 400; stop 4 is only reachable by footpath
        // from stop 2 (arr 300 + 50).
        builder
            .add_trip(c, vec![time(400), time(1000)], 4, 0)
            .unwrap();
        let timetable = builder.finish();
        let transfers = Transfers::from_edges(5, &[(StopIdx(2), StopIdx(4), 50)]).unwrap();
        (timetable, transfers)
    }

    fn request(from: StopIdx, to: StopIdx, departure: u32) -> Request {
        Request {
            departure,
            access: vec![(from, 0)],
            egress: vec![(to, 0)],
            active_services: vec![true, false],
            max_transfers: 3,
        }
    }

    #[test]
    fn routes_a_direct_ride() {
        let (timetable, transfers) = network();
        let journeys = Raptor.route(&timetable, &transfers, &request(StopIdx(0), StopIdx(2), 0));
        assert_eq!(journeys.len(), 1);
        let journey = &journeys[0];
        assert_eq!(journey.arrival, 300);
        assert_eq!(journey.rides(), 1);
        assert_eq!(
            journey.legs[1],
            Leg::Transit {
                trip: TripIdx(0),
                board_stop: StopIdx(0),
                alight_stop: StopIdx(2),
                board_position: 0,
                alight_position: 2,
                board_time: 100,
                alight_time: 300,
            }
        );
    }

    #[test]
    fn boards_the_next_trip_when_the_first_has_left() {
        let (timetable, transfers) = network();
        let journeys = Raptor.route(
            &timetable,
            &transfers,
            &request(StopIdx(0), StopIdx(2), 150),
        );
        assert_eq!(journeys.len(), 1);
        assert_eq!(journeys[0].arrival, 900);
    }

    #[test]
    fn transfers_at_a_shared_stop() {
        let (timetable, transfers) = network();
        let journeys = Raptor.route(&timetable, &transfers, &request(StopIdx(0), StopIdx(3), 0));
        // One ride cannot reach stop 3; two rides via stop 1 arrive at 400.
        assert_eq!(journeys.len(), 1);
        let journey = &journeys[0];
        assert_eq!(journey.arrival, 400);
        assert_eq!(journey.rides(), 2);
    }

    #[test]
    fn walks_a_footpath_after_a_ride() {
        let (timetable, transfers) = network();
        // Ride A to stop 2 (arr 300), walk the 50-second footpath to 4.
        let journeys = Raptor.route(&timetable, &transfers, &request(StopIdx(0), StopIdx(4), 0));
        assert_eq!(journeys.len(), 1);
        let journey = &journeys[0];
        assert_eq!(journey.arrival, 350);
        assert!(matches!(
            journey.legs[2],
            Leg::Transfer {
                from_stop: StopIdx(2),
                to_stop: StopIdx(4),
                departure: 300,
                arrival: 350,
            }
        ));
    }

    #[test]
    fn transfers_relax_a_single_hop_from_transit_arrivals() {
        // Footpaths 1→2 and 2→3 without a closing 1→3 edge: the walk out
        // of stop 2 must leave from its transit arrival (500), not chain
        // onto the walk that just improved stop 2 in the same round.
        let mut builder = TimetableBuilder::new(4);
        let to_a = builder.add_pattern(&[StopIdx(0), StopIdx(1)], 0).unwrap();
        let to_b = builder.add_pattern(&[StopIdx(0), StopIdx(2)], 1).unwrap();
        builder
            .add_trip(to_a, vec![time(0), time(100)], 0, 0)
            .unwrap();
        builder
            .add_trip(to_b, vec![time(0), time(500)], 1, 0)
            .unwrap();
        let timetable = builder.finish();
        let transfers = Transfers::from_edges(
            4,
            &[(StopIdx(1), StopIdx(2), 30), (StopIdx(2), StopIdx(3), 50)],
        )
        .unwrap();
        let journeys = Raptor.route(&timetable, &transfers, &request(StopIdx(0), StopIdx(3), 0));
        assert_eq!(journeys.len(), 1);
        assert_eq!(journeys[0].arrival, 550);
    }

    #[test]
    fn footpaths_from_the_origin_are_the_access_lists_job() {
        let (timetable, transfers) = network();
        // Stop 2 only rides A at its last position; the footpath 2→4 is
        // not relaxed from the origin itself, by contract.
        let journeys = Raptor.route(
            &timetable,
            &transfers,
            &request(StopIdx(2), StopIdx(3), 260),
        );
        assert_eq!(journeys.len(), 0);
    }

    #[test]
    fn skips_trips_of_inactive_services() {
        let (timetable, transfers) = network();
        // Departing at 260: B's active trip (dep 250) is gone; the service-1
        // trip at 500 exists but must not be boarded.
        let mut req = request(StopIdx(1), StopIdx(3), 260);
        let journeys = Raptor.route(&timetable, &transfers, &req);
        assert_eq!(journeys.len(), 0);
        // With service 1 active the 500 trip works.
        req.active_services = vec![true, true];
        let journeys = Raptor.route(&timetable, &transfers, &req);
        assert_eq!(journeys.len(), 1);
        assert_eq!(journeys[0].arrival, 600);
    }

    #[test]
    fn emits_the_pareto_set_over_rides_and_arrival() {
        let mut builder = TimetableBuilder::new(3);
        // A slow direct pattern and a faster two-ride alternative.
        let direct = builder.add_pattern(&[StopIdx(0), StopIdx(2)], 0).unwrap();
        let first = builder.add_pattern(&[StopIdx(0), StopIdx(1)], 1).unwrap();
        let second = builder.add_pattern(&[StopIdx(1), StopIdx(2)], 2).unwrap();
        builder
            .add_trip(direct, vec![time(0), time(1000)], 0, 0)
            .unwrap();
        builder
            .add_trip(first, vec![time(0), time(100)], 1, 0)
            .unwrap();
        builder
            .add_trip(second, vec![time(150), time(300)], 2, 0)
            .unwrap();
        let timetable = builder.finish();
        let transfers = Transfers::empty(3);
        let journeys = Raptor.route(
            &timetable,
            &transfers,
            &Request {
                departure: 0,
                access: vec![(StopIdx(0), 0)],
                egress: vec![(StopIdx(2), 0)],
                active_services: vec![true],
                max_transfers: 3,
            },
        );
        assert_eq!(journeys.len(), 2);
        assert_eq!((journeys[0].rides(), journeys[0].arrival), (1, 1000));
        assert_eq!((journeys[1].rides(), journeys[1].arrival), (2, 300));
    }

    #[test]
    fn chooses_between_access_and_egress_alternatives() {
        let (timetable, transfers) = network();
        // Origin can reach stop 0 slowly or stop 1 quickly; destination is
        // reachable from stop 2 or stop 3.
        let journeys = Raptor.route(
            &timetable,
            &transfers,
            &Request {
                departure: 0,
                access: vec![(StopIdx(0), 90), (StopIdx(1), 10)],
                egress: vec![(StopIdx(2), 500), (StopIdx(3), 20)],
                active_services: vec![true, false],
                max_transfers: 3,
            },
        );
        // Best: board B at stop 1 (reached at 10, dep 250), arrive 3 at
        // 400, egress 20 → 420 with one ride. Riding A from 0 to 2 then
        // egress 500 gives 800; two rides cannot beat 420.
        assert_eq!(journeys.len(), 1);
        let journey = &journeys[0];
        assert_eq!(journey.arrival, 420);
        assert_eq!(journey.rides(), 1);
        assert_eq!(
            journey.legs[0],
            Leg::Access {
                to_stop: StopIdx(1),
                departure: 0,
                arrival: 10,
            }
        );
    }

    #[test]
    fn terminal_stops_still_board_their_other_patterns() {
        // Stop 1 is the terminus of pattern X (0→1) and the start of
        // pattern Y (1→2); arriving at the terminus must still allow
        // boarding Y through its own pattern membership.
        let mut builder = TimetableBuilder::new(3);
        let x = builder.add_pattern(&[StopIdx(0), StopIdx(1)], 0).unwrap();
        let y = builder.add_pattern(&[StopIdx(1), StopIdx(2)], 1).unwrap();
        builder.add_trip(x, vec![time(0), time(100)], 0, 0).unwrap();
        builder
            .add_trip(y, vec![time(150), time(250)], 1, 0)
            .unwrap();
        let timetable = builder.finish();
        let transfers = Transfers::empty(3);
        let journeys = Raptor.route(
            &timetable,
            &transfers,
            &Request {
                departure: 0,
                access: vec![(StopIdx(0), 0)],
                egress: vec![(StopIdx(2), 0)],
                active_services: vec![true],
                max_transfers: 3,
            },
        );
        assert_eq!(journeys.len(), 1);
        assert_eq!(journeys[0].arrival, 250);
        assert_eq!(journeys[0].rides(), 2);
    }

    #[test]
    fn one_to_all_reports_earliest_arrivals_everywhere() {
        let (timetable, transfers) = network();
        let arrivals =
            Raptor.one_to_all(&timetable, &transfers, &request(StopIdx(0), StopIdx(0), 0));
        // Origin at the departure time; ride A to 1 and 2; B onward to 3;
        // the footpath reaches 4.
        assert_eq!(arrivals[0], Some(0));
        assert_eq!(arrivals[1], Some(200));
        assert_eq!(arrivals[2], Some(300));
        assert_eq!(arrivals[3], Some(400));
        assert_eq!(arrivals[4], Some(350));
        // Departing after the last useful trips, nothing is reachable
        // beyond the origin itself.
        let late = Raptor.one_to_all(&timetable, &transfers, &request(StopIdx(3), StopIdx(0), 0));
        assert_eq!(late[3], Some(0));
        assert_eq!(late[0], None);
    }

    #[test]
    fn respects_the_transfer_limit() {
        let (timetable, transfers) = network();
        let mut req = request(StopIdx(0), StopIdx(3), 0);
        req.max_transfers = 0;
        let journeys = Raptor.route(&timetable, &transfers, &req);
        assert_eq!(journeys.len(), 0);
    }

    /// One pattern 0→1 with three rides: 100→300, 200→400, 300→500.
    fn frequent_network() -> (Timetable, Transfers) {
        let mut builder = TimetableBuilder::new(2);
        let a = builder.add_pattern(&[StopIdx(0), StopIdx(1)], 0).unwrap();
        for departure in [100, 200, 300] {
            builder
                .add_trip(a, vec![time(departure), time(departure + 200)], 0, 0)
                .unwrap();
        }
        (builder.finish(), Transfers::empty(2))
    }

    #[test]
    fn range_emits_one_journey_per_feasible_departure() {
        let (timetable, transfers) = frequent_network();
        let journeys = Raptor.route_range(
            &timetable,
            &transfers,
            &request(StopIdx(0), StopIdx(1), 0),
            250,
        );
        // Departures 100 and 200 fall in [0, 250); each ride is the
        // latest-departure way to its arrival, so both survive. The
        // window's final second waits for the 300 ride.
        let profile: Vec<_> = journeys
            .iter()
            .map(|journey| (journey.departure, journey.arrival, journey.rides()))
            .collect();
        assert_eq!(profile, vec![(100, 300, 1), (200, 400, 1), (249, 500, 1)]);
        // Each journey departs the origin at its stated departure time.
        for journey in &journeys {
            assert_eq!(
                journey.legs[0],
                Leg::Access {
                    to_stop: StopIdx(0),
                    departure: journey.departure,
                    arrival: journey.departure,
                }
            );
        }
    }

    #[test]
    fn range_window_is_half_open() {
        let (timetable, transfers) = frequent_network();
        let journeys = Raptor.route_range(
            &timetable,
            &transfers,
            &request(StopIdx(0), StopIdx(1), 100),
            100,
        );
        // [100, 200) holds the 100 departure; the ride at 200 is only
        // reached by waiting from the window's final second.
        let profile: Vec<_> = journeys
            .iter()
            .map(|journey| (journey.departure, journey.arrival))
            .collect();
        assert_eq!(profile, vec![(100, 300), (199, 400)]);
        // A zero-length window has no departures at all.
        let none = Raptor.route_range(
            &timetable,
            &transfers,
            &request(StopIdx(0), StopIdx(1), 100),
            0,
        );
        assert!(none.is_empty());
    }

    #[test]
    fn range_waits_past_the_window_when_the_next_ride_is_later() {
        let (timetable, transfers) = frequent_network();
        // No ride departs within [0, 50), but leaving at its final second
        // and waiting catches the ride at 100.
        let journeys = Raptor.route_range(
            &timetable,
            &transfers,
            &request(StopIdx(0), StopIdx(1), 0),
            50,
        );
        let profile: Vec<_> = journeys
            .iter()
            .map(|journey| (journey.departure, journey.arrival, journey.rides()))
            .collect();
        assert_eq!(profile, vec![(49, 300, 1)]);
    }

    #[test]
    fn range_keeps_fewer_ride_options_from_earlier_departures() {
        // Departing at 200, a two-ride chain arrives at 320; departing at
        // 100, the direct ride arrives at 400. Neither dominates the
        // other — the direct journey needs fewer rides — so the faster
        // later pass must not prune the earlier pass's direct label.
        let mut builder = TimetableBuilder::new(3);
        let direct = builder.add_pattern(&[StopIdx(0), StopIdx(2)], 0).unwrap();
        let first = builder.add_pattern(&[StopIdx(0), StopIdx(1)], 1).unwrap();
        let second = builder.add_pattern(&[StopIdx(1), StopIdx(2)], 2).unwrap();
        builder
            .add_trip(direct, vec![time(100), time(400)], 0, 0)
            .unwrap();
        builder
            .add_trip(first, vec![time(200), time(240)], 1, 0)
            .unwrap();
        builder
            .add_trip(second, vec![time(250), time(320)], 2, 0)
            .unwrap();
        let timetable = builder.finish();
        let transfers = Transfers::empty(3);
        let journeys = Raptor.route_range(
            &timetable,
            &transfers,
            &request(StopIdx(0), StopIdx(2), 0),
            201,
        );
        let profile: Vec<_> = journeys
            .iter()
            .map(|journey| (journey.departure, journey.arrival, journey.rides()))
            .collect();
        assert_eq!(profile, vec![(100, 400, 1), (200, 320, 2)]);
    }

    #[test]
    fn range_drops_journeys_dominated_by_later_departures() {
        // A slow ride at 100 and an express at 150 that arrives earlier:
        // departing at 100 offers nothing the 150 departure does not beat.
        let mut builder = TimetableBuilder::new(2);
        let local = builder.add_pattern(&[StopIdx(0), StopIdx(1)], 0).unwrap();
        let express = builder.add_pattern(&[StopIdx(0), StopIdx(1)], 1).unwrap();
        builder
            .add_trip(local, vec![time(100), time(400)], 0, 0)
            .unwrap();
        builder
            .add_trip(express, vec![time(150), time(250)], 1, 0)
            .unwrap();
        let timetable = builder.finish();
        let transfers = Transfers::empty(2);
        let journeys = Raptor.route_range(
            &timetable,
            &transfers,
            &request(StopIdx(0), StopIdx(1), 0),
            200,
        );
        assert_eq!(journeys.len(), 1);
        assert_eq!((journeys[0].departure, journeys[0].arrival), (150, 250));
    }

    #[test]
    fn range_keeps_extra_rides_only_when_strictly_earlier() {
        // Departing at 200, one direct ride arrives at 500. Departing at
        // 100, a two-ride chain arrives at 300; the direct ride is also
        // catchable then but no longer beats anything.
        let mut builder = TimetableBuilder::new(3);
        let direct = builder.add_pattern(&[StopIdx(0), StopIdx(2)], 0).unwrap();
        let first = builder.add_pattern(&[StopIdx(0), StopIdx(1)], 1).unwrap();
        let second = builder.add_pattern(&[StopIdx(1), StopIdx(2)], 2).unwrap();
        builder
            .add_trip(direct, vec![time(200), time(500)], 0, 0)
            .unwrap();
        builder
            .add_trip(first, vec![time(100), time(150)], 1, 0)
            .unwrap();
        builder
            .add_trip(second, vec![time(160), time(300)], 2, 0)
            .unwrap();
        let timetable = builder.finish();
        let transfers = Transfers::empty(3);
        let journeys = Raptor.route_range(
            &timetable,
            &transfers,
            &request(StopIdx(0), StopIdx(2), 0),
            300,
        );
        let profile: Vec<_> = journeys
            .iter()
            .map(|journey| (journey.departure, journey.arrival, journey.rides()))
            .collect();
        assert_eq!(profile, vec![(100, 300, 2), (200, 500, 1)]);
    }

    #[test]
    fn range_shifts_candidates_by_the_access_duration() {
        let (timetable, transfers) = frequent_network();
        let journeys = Raptor.route_range(
            &timetable,
            &transfers,
            &Request {
                departure: 0,
                access: vec![(StopIdx(0), 50)],
                egress: vec![(StopIdx(1), 0)],
                active_services: vec![true],
                max_transfers: 3,
            },
            200,
        );
        // Catching the rides at 100 and 200 means leaving at 50 and 150;
        // the window's final second waits for the ride at 300.
        let departures: Vec<_> = journeys.iter().map(|journey| journey.departure).collect();
        assert_eq!(departures, vec![50, 150, 199]);
        assert_eq!(
            journeys[0].legs[0],
            Leg::Access {
                to_stop: StopIdx(0),
                departure: 50,
                arrival: 100,
            }
        );
    }

    #[test]
    fn range_skips_candidates_of_inactive_services() {
        let (timetable, transfers) = network();
        // B's service-1 trip departs stop 1 at 500 but never runs.
        let journeys = Raptor.route_range(
            &timetable,
            &transfers,
            &request(StopIdx(1), StopIdx(3), 0),
            600,
        );
        let departures: Vec<_> = journeys.iter().map(|journey| journey.departure).collect();
        assert_eq!(departures, vec![250]);
    }

    #[test]
    fn range_walks_footpaths_per_departure() {
        let (timetable, transfers) = network();
        // Only A's first trip (dep 100) reaches stop 4 in time for C: ride
        // to stop 2 (arr 300), walk 50 s, catch C at 400.
        let journeys = Raptor.route_range(
            &timetable,
            &transfers,
            &request(StopIdx(0), StopIdx(3), 0),
            800,
        );
        let profile: Vec<_> = journeys
            .iter()
            .map(|journey| (journey.departure, journey.arrival, journey.rides()))
            .collect();
        assert_eq!(profile, vec![(100, 400, 2)]);
    }
}
