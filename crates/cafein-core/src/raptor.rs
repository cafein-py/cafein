//! Plain RAPTOR: earliest-arrival routing for a single departure time.
//!
//! Round-based: round `k` finds the earliest arrivals reachable with
//! exactly `k` rides. Within a pattern, the earliest catchable trip at a
//! stop position is found by binary search over departures, which is valid
//! at every position because the timetable's patterns are FIFO chains.

use crate::journey::{Journey, Leg};
use crate::router::{Request, TransitRouter};
use crate::timetable::{PatternIdx, StopIdx, Timetable, TripIdx};
use crate::transfers::Transfers;

/// The plain RAPTOR router.
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
        let stop_count = timetable.stop_count() as usize;
        let rounds = request.max_transfers as usize + 1;

        // Per-round arrival times and labels; `best` holds the minimum over
        // all rounds so far for pruning.
        let mut tau = vec![vec![UNREACHED; stop_count]; rounds + 1];
        let mut labels = vec![vec![Label::Unreached; stop_count]; rounds + 1];
        let mut best = vec![UNREACHED; stop_count];
        let mut marked: Vec<StopIdx> = Vec::new();
        let mut is_marked = vec![false; stop_count];

        for &(stop, duration) in &request.access {
            let arrival = request.departure.saturating_add(duration);
            if arrival < tau[0][stop.0 as usize] {
                tau[0][stop.0 as usize] = arrival;
                labels[0][stop.0 as usize] = Label::Access;
                best[stop.0 as usize] = arrival;
                if !is_marked[stop.0 as usize] {
                    is_marked[stop.0 as usize] = true;
                    marked.push(stop);
                }
            }
        }

        // First marked position per pattern for the current round.
        let pattern_count = timetable.pattern_count() as usize;
        let mut queue_position = vec![u16::MAX; pattern_count];
        let mut queued_patterns: Vec<PatternIdx> = Vec::new();

        for round in 1..=rounds {
            queued_patterns.clear();
            for stop in marked.drain(..) {
                is_marked[stop.0 as usize] = false;
                for pattern_stop in timetable.patterns_at_stop(stop) {
                    let slot = &mut queue_position[pattern_stop.pattern.0 as usize];
                    if *slot == u16::MAX {
                        queued_patterns.push(pattern_stop.pattern);
                    }
                    if pattern_stop.position < *slot {
                        *slot = pattern_stop.position;
                    }
                }
            }

            for &pattern in &queued_patterns {
                let start_position = queue_position[pattern.0 as usize];
                queue_position[pattern.0 as usize] = u16::MAX;
                let stops = timetable.pattern_stops(pattern);
                let mut current: Option<(TripIdx, u16)> = None;

                for position in start_position as usize..stops.len() {
                    let stop = stops[position].0 as usize;

                    if let Some((trip, board_position)) = current {
                        let arrival = timetable.trip_stop_times(trip)[position].arrival;
                        if arrival < best[stop] {
                            tau[round][stop] = arrival;
                            best[stop] = arrival;
                            labels[round][stop] = Label::Transit {
                                trip,
                                board_position,
                                alight_position: position as u16,
                            };
                            if !is_marked[stop] {
                                is_marked[stop] = true;
                                marked.push(stops[position]);
                            }
                        }
                    }

                    // Try to catch an earlier trip from this stop, using the
                    // previous round's arrival. The arrival handling above
                    // has already recorded any improvement at this position;
                    // boarding at a pattern's last position is pointless
                    // because there is no later stop to alight at, and other
                    // patterns serving this stop are queued separately.
                    let reached = tau[round - 1][stop];
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

            // Relax one footpath hop from every stop improved by transit.
            let transit_marked: Vec<StopIdx> = marked.clone();
            for stop in transit_marked {
                let departure = tau[round][stop.0 as usize];
                for transfer in transfers.from_stop(stop) {
                    let arrival = departure.saturating_add(transfer.duration);
                    let to = transfer.to.0 as usize;
                    if arrival < best[to] {
                        tau[round][to] = arrival;
                        best[to] = arrival;
                        labels[round][to] = Label::Transfer {
                            from_stop: stop,
                            duration: transfer.duration,
                        };
                        if !is_marked[to] {
                            is_marked[to] = true;
                            marked.push(transfer.to);
                        }
                    }
                }
            }

            if marked.is_empty() {
                break;
            }
        }

        collect_journeys(timetable, request, &tau, &labels, rounds)
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

/// Emits the Pareto set over (arrival, rides): one journey per round whose
/// arrival strictly improves on all earlier rounds.
fn collect_journeys(
    timetable: &Timetable,
    request: &Request,
    tau: &[Vec<u32>],
    labels: &[Vec<Label>],
    rounds: usize,
) -> Vec<Journey> {
    let mut journeys = Vec::new();
    let mut best_arrival = UNREACHED;
    for round in 1..=rounds {
        let mut best_egress: Option<(u32, StopIdx, u32)> = None;
        for &(stop, duration) in &request.egress {
            let at_stop = tau[round][stop.0 as usize];
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
        if arrival >= best_arrival {
            continue;
        }
        best_arrival = arrival;
        journeys.push(reconstruct(
            timetable,
            request,
            tau,
            labels,
            round,
            egress_stop,
            egress_duration,
        ));
    }
    journeys
}

fn reconstruct(
    timetable: &Timetable,
    request: &Request,
    tau: &[Vec<u32>],
    labels: &[Vec<Label>],
    round: usize,
    egress_stop: StopIdx,
    egress_duration: u32,
) -> Journey {
    let mut legs = Vec::new();
    let departure_at_stop = tau[round][egress_stop.0 as usize];
    legs.push(Leg::Egress {
        from_stop: egress_stop,
        departure: departure_at_stop,
        arrival: departure_at_stop + egress_duration,
    });

    let mut current_round = round;
    let mut stop = egress_stop;
    loop {
        match labels[current_round][stop.0 as usize] {
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
                let arrival = tau[current_round][stop.0 as usize];
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
                    departure: request.departure,
                    arrival: tau[0][stop.0 as usize],
                });
                break;
            }
            Label::Unreached => unreachable!("journey reconstruction hit an unreached label"),
        }
    }
    legs.reverse();

    Journey {
        departure: request.departure,
        arrival: departure_at_stop + egress_duration,
        legs,
    }
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
    fn respects_the_transfer_limit() {
        let (timetable, transfers) = network();
        let mut req = request(StopIdx(0), StopIdx(3), 0);
        req.max_transfers = 0;
        let journeys = Raptor.route(&timetable, &transfers, &req);
        assert_eq!(journeys.len(), 0);
    }
}
