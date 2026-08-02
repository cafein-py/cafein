//! The possession-state search for own-vehicle carriage (stage 17b):
//! two label planes — **Carrying** (the vehicle rides along) and
//! **Free** (parked or left at the origin) — over the standard round
//! structure, with an ordered non-transit closure inside each round:
//! Carrying transfers, then parking, then Free walks. The phases form
//! a DAG (a carriage transfer cannot follow a park), so one ordered
//! pass reaches the round's fixed point.
//!
//! Carrying boards only trips whose carrying mask allows it and
//! relaxes the (unclosed) carriage set under the exact-phase rule —
//! the round's best transit arrival relaxes even when a faster
//! carriage-transfer label shadows it. Free is exactly the pedestrian
//! search: every trip, the walking closure, standard gates. Dominance
//! holds within a plane only; the caller folds the cross-state minimum
//! at egress.

use crate::routers::raptor::earliest_active_trip;
use crate::timetable::{StopIdx, Timetable, TripIdx};
use crate::transfers::Transfers;

const UNREACHED: u32 = u32::MAX;

/// The number of possession states.
pub const STATES: usize = 2;
/// The Carrying plane's index.
pub const CARRYING: usize = 0;
/// The Free plane's index.
pub const FREE: usize = 1;

/// The network-and-policy inputs one carriage query runs on.
pub struct CarriageInputs<'a> {
    pub timetable: &'a Timetable,
    /// The walking closure the Free plane relaxes (closed).
    pub walking: &'a Transfers,
    /// The carriage set the Carrying plane relaxes (unclosed).
    pub carriage: &'a Transfers,
    /// Per carriage-CSR-edge: whether the row is the vehicle's ride.
    pub ride_edge: &'a [bool],
    /// Whether each trip may be boarded while carrying — the GTFS
    /// tri-state resolved under the policy's unknown rule.
    pub carrying_mask: &'a [bool],
    /// Whether the vehicle may be parked at each stop.
    pub park_mask: &'a [bool],
    pub active_services: &'a [bool],
    pub active_services_previous: &'a [bool],
}

/// One carriage query: seeds per plane (carriage is optional, so the
/// Free plane seeds independently — leaving the vehicle at the origin)
/// and the round budget.
pub struct CarriageRequest {
    pub departure: u32,
    pub carrying_access: Vec<(StopIdx, u32)>,
    pub free_access: Vec<(StopIdx, u32)>,
    pub max_transfers: u8,
}

/// How a label was reached, per (round, stop, plane).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CarriageLabel {
    Unreached,
    /// Seeded from the access reduction of this plane.
    Access,
    /// Alighted from `trip`, boarded at `board_position`, in the same
    /// plane (state is preserved across a ride).
    Transit {
        trip: TripIdx,
        board_position: u16,
        alight_position: u16,
        day_offset: u32,
    },
    /// Relaxed a transfer row of this plane's set from `from_stop`;
    /// `ride` marks the carriage set's vehicle rows (mode = the
    /// carried vehicle's). The carried ride, when the source arrival
    /// was shadowed, travels inline exactly as in the exact phase.
    Transfer {
        from_stop: StopIdx,
        duration: u32,
        ride: bool,
        via_transit: Option<(TripIdx, u16, u16, u32)>,
        /// The walk left a park whose Free label may since have been
        /// overwritten: the park travels inline (the uniform-carry
        /// rule), so reconstruction crosses planes here.
        via_park: bool,
    },
    /// Parked at this stop: entered Free from Carrying's same-stop
    /// label this round.
    Park,
}

/// Per-plane, per-round label state.
struct Plane {
    tau: Vec<Vec<u32>>,
    best: Vec<Vec<u32>>,
    labels: Vec<Vec<CarriageLabel>>,
    marked: Vec<StopIdx>,
    is_marked: Vec<bool>,
}

impl Plane {
    fn new(stop_count: usize, rounds: usize) -> Plane {
        Plane {
            tau: vec![vec![UNREACHED; stop_count]; rounds + 1],
            best: vec![vec![UNREACHED; stop_count]; rounds + 1],
            labels: vec![vec![CarriageLabel::Unreached; stop_count]; rounds + 1],
            marked: Vec::new(),
            is_marked: vec![false; stop_count],
        }
    }

    fn mark(&mut self, stop: StopIdx) {
        if !self.is_marked[stop.0 as usize] {
            self.is_marked[stop.0 as usize] = true;
            self.marked.push(stop);
        }
    }

    /// Writes `arrival` at (round, stop) when it improves the plane's
    /// prefix best; marks the stop for the next transit scan.
    fn improve(&mut self, round: usize, stop: StopIdx, arrival: u32, label: CarriageLabel) -> bool {
        let index = stop.0 as usize;
        if arrival >= self.best[round][index] {
            return false;
        }
        self.tau[round][index] = arrival;
        for best in &mut self.best[round..] {
            best[index] = best[index].min(arrival);
        }
        self.labels[round][index] = label;
        self.mark(stop);
        true
    }
}

/// The full label state of one finished carriage search, for the
/// caller's egress folds and reconstruction.
pub struct CarriageSearch {
    rounds: usize,
    planes: [Plane; STATES],
}

impl CarriageSearch {
    /// The per-stop earliest arrival of one plane, over all rounds.
    pub fn arrivals(&self, plane: usize) -> Vec<Option<u32>> {
        self.planes[plane]
            .best
            .last()
            .expect("a search always has a round")
            .iter()
            .map(|&arrival| (arrival != UNREACHED).then_some(arrival))
            .collect()
    }

    /// The earliest (round, arrival) of `plane` at `stop`, if any.
    pub fn best_round(&self, plane: usize, stop: StopIdx) -> Option<(usize, u32)> {
        let planes = &self.planes[plane];
        let target = planes.best.last().expect("rounds")[stop.0 as usize];
        if target == UNREACHED {
            return None;
        }
        (0..=self.rounds)
            .find(|&round| planes.tau[round][stop.0 as usize] == target)
            .map(|round| (round, target))
    }

    /// The label at (plane, round, stop).
    pub fn label(&self, plane: usize, round: usize, stop: StopIdx) -> CarriageLabel {
        self.planes[plane].labels[round][stop.0 as usize]
    }

    /// The round-`round` arrival at (plane, stop).
    pub fn arrival_at(&self, plane: usize, round: usize, stop: StopIdx) -> Option<u32> {
        let arrival = self.planes[plane].tau[round][stop.0 as usize];
        (arrival != UNREACHED).then_some(arrival)
    }

    /// The best arrival within `round` rounds at (plane, stop).
    pub fn prefix_arrival(&self, plane: usize, round: usize, stop: StopIdx) -> Option<u32> {
        let arrival = self.planes[plane].best[round][stop.0 as usize];
        (arrival != UNREACHED).then_some(arrival)
    }

    /// The first round at or under `round` whose write equals the
    /// prefix best at (plane, stop) — the reconstruction entry for the
    /// (arrival, rides) frontier.
    pub fn achieving_round(&self, plane: usize, round: usize, stop: StopIdx) -> Option<usize> {
        let target = self.planes[plane].best[round][stop.0 as usize];
        if target == UNREACHED {
            return None;
        }
        (0..=round).find(|&r| self.planes[plane].tau[r][stop.0 as usize] == target)
    }

    /// The search's round count (`max_transfers + 1`).
    pub fn rounds(&self) -> usize {
        self.rounds
    }
}

/// Runs one carriage search to completion.
pub fn search(inputs: &CarriageInputs<'_>, request: &CarriageRequest) -> CarriageSearch {
    let timetable = inputs.timetable;
    let stop_count = timetable.stop_count() as usize;
    let rounds = request.max_transfers as usize + 1;
    let mut planes = [
        Plane::new(stop_count, rounds),
        Plane::new(stop_count, rounds),
    ];

    // Seeds. Carrying from the policy reduction; Free independently
    // (the vehicle stays at the origin), plus the park transition at
    // eligible seed stops.
    for &(stop, seconds) in &request.carrying_access {
        let Some(arrival) = request
            .departure
            .checked_add(seconds)
            .filter(|&at| at != UNREACHED)
        else {
            continue;
        };
        planes[CARRYING].improve(0, stop, arrival, CarriageLabel::Access);
    }
    for &(stop, seconds) in &request.free_access {
        let Some(arrival) = request
            .departure
            .checked_add(seconds)
            .filter(|&at| at != UNREACHED)
        else {
            continue;
        };
        planes[FREE].improve(0, stop, arrival, CarriageLabel::Access);
    }
    let mut parked_seeds: Vec<(StopIdx, u32)> = Vec::new();
    for index in 0..stop_count {
        if !inputs.park_mask[index] {
            continue;
        }
        let arrival = planes[CARRYING].tau[0][index];
        if arrival != UNREACHED {
            planes[FREE].improve(0, StopIdx(index as u32), arrival, CarriageLabel::Park);
            parked_seeds.push((StopIdx(index as u32), arrival));
        }
    }

    // The parked seeds are not walking-closed (the Free access rows
    // are, by their reduction — walking again from them would compose
    // closure rows past the budget): relax the closed walking set over
    // every parking candidate, shadowed or not — a better Free label
    // may itself be closure-derived and barred from walking again,
    // while the park walks fresh (the exact rule; the park travels
    // inline for reconstruction). Walk targets mark themselves; marks
    // stay for round one's scan.
    let mut seed_walks: Vec<(StopIdx, u32, CarriageLabel)> = Vec::new();
    for &(source, departure_at) in &parked_seeds {
        for edge in inputs.walking.from_stop(source) {
            let Some(arrival) = departure_at
                .checked_add(edge.duration)
                .filter(|&at| at != UNREACHED)
            else {
                continue;
            };
            seed_walks.push((
                edge.to,
                arrival,
                CarriageLabel::Transfer {
                    from_stop: source,
                    duration: edge.duration,
                    ride: false,
                    via_transit: None,
                    via_park: true,
                },
            ));
        }
    }
    for (stop, arrival, label) in seed_walks {
        planes[FREE].improve(0, stop, arrival, label);
    }

    // Exact-phase sidecars, per plane: the round's best transit
    // arrivals with their rides, shadowed or not — Carrying's feed the
    // unclosed carriage set, Free's the walking phase (a fresh ride
    // resets the one-transfer allowance even when an older
    // closure-derived label shadows it).
    let mut transit_arrival = [vec![UNREACHED; stop_count], vec![UNREACHED; stop_count]];
    let mut transit_ride: [Vec<(TripIdx, u16, u16, u32)>; 2] = [
        vec![(TripIdx(0), 0, 0, 0); stop_count],
        vec![(TripIdx(0), 0, 0, 0); stop_count],
    ];
    let mut transit_touched: [Vec<u32>; 2] = [Vec::new(), Vec::new()];

    for round in 1..=rounds {
        // Transit phase, per plane. The Carrying scan filters trips by
        // the carrying mask; the sidecars record the transit arrivals.
        for plane_index in [CARRYING, FREE] {
            for &stop in &transit_touched[plane_index] {
                transit_arrival[plane_index][stop as usize] = UNREACHED;
            }
            transit_touched[plane_index].clear();
        }
        for plane_index in [CARRYING, FREE] {
            let marked = std::mem::take(&mut planes[plane_index].marked);
            for &stop in &marked {
                planes[plane_index].is_marked[stop.0 as usize] = false;
            }
            let mask = (plane_index == CARRYING).then_some(inputs.carrying_mask);
            let mut writes: Vec<(StopIdx, u32, CarriageLabel)> = Vec::new();
            scan_patterns(
                inputs,
                round,
                &planes[plane_index],
                &marked,
                mask,
                &mut writes,
            );
            for (stop, arrival, label) in writes {
                if arrival < transit_arrival[plane_index][stop.0 as usize] {
                    if transit_arrival[plane_index][stop.0 as usize] == UNREACHED {
                        transit_touched[plane_index].push(stop.0);
                    }
                    transit_arrival[plane_index][stop.0 as usize] = arrival;
                    if let CarriageLabel::Transit {
                        trip,
                        board_position,
                        alight_position,
                        day_offset,
                    } = label
                    {
                        transit_ride[plane_index][stop.0 as usize] =
                            (trip, board_position, alight_position, day_offset);
                    }
                }
                planes[plane_index].improve(round, stop, arrival, label);
            }
        }

        // Carrying transfer phase: the unclosed carriage set relaxes
        // from every transit arrival of the round (the exact rule).
        let mut relaxations: Vec<(StopIdx, u32, CarriageLabel)> = Vec::new();
        for &source in &transit_touched[CARRYING] {
            let source_stop = StopIdx(source);
            let departure_at = transit_arrival[CARRYING][source as usize];
            let ride = transit_ride[CARRYING][source as usize];
            let range = inputs.carriage.edge_range(source_stop);
            for (offset, edge) in inputs.carriage.from_stop(source_stop).iter().enumerate() {
                let Some(arrival) = departure_at
                    .checked_add(edge.duration)
                    .filter(|&at| at != UNREACHED)
                else {
                    continue;
                };
                // The ride travels inline unconditionally: the source's
                // label slot is mutable within this phase, so
                // reconstruction must never consult it (the exact
                // phase's uniform-carry rule).
                relaxations.push((
                    edge.to,
                    arrival,
                    CarriageLabel::Transfer {
                        from_stop: source_stop,
                        duration: edge.duration,
                        ride: inputs.ride_edge[range.start + offset],
                        via_transit: Some(ride),
                        via_park: false,
                    },
                ));
            }
        }
        for (stop, arrival, label) in relaxations {
            planes[CARRYING].improve(round, stop, arrival, label);
        }

        // Park phase: every Carrying label the round produced may park
        // at an eligible stop, entering Free the same round. The full
        // candidate list stays as a sidecar — a shadowed park still
        // walks below, since its shadower may be closure-derived and
        // barred from walking again (the exact rule).
        let carrying_round: Vec<(usize, u32)> = (0..stop_count)
            .filter(|&index| inputs.park_mask[index])
            .map(|index| (index, planes[CARRYING].tau[round][index]))
            .filter(|&(_, arrival)| arrival != UNREACHED)
            .collect();
        for &(index, arrival) in &carrying_round {
            planes[FREE].improve(round, StopIdx(index as u32), arrival, CarriageLabel::Park);
        }

        // Free walking phase: the closed walking closure relaxes from
        // the sidecar of the round's Free transit arrivals (shadowed
        // ones included — the fresh ride resets the one-transfer
        // allowance) and from every parking candidate, each walk
        // inline-carrying its predecessor (the uniform-carry rule):
        // reconstruction must never chain closure rows through a
        // mutable or missing label.
        let mut walks: Vec<(StopIdx, u32, CarriageLabel)> = Vec::new();
        for &source in &transit_touched[FREE] {
            let source_stop = StopIdx(source);
            let departure_at = transit_arrival[FREE][source as usize];
            let ride = Some(transit_ride[FREE][source as usize]);
            for edge in inputs.walking.from_stop(source_stop) {
                let Some(arrival) = departure_at
                    .checked_add(edge.duration)
                    .filter(|&at| at != UNREACHED)
                else {
                    continue;
                };
                walks.push((
                    edge.to,
                    arrival,
                    CarriageLabel::Transfer {
                        from_stop: source_stop,
                        duration: edge.duration,
                        ride: false,
                        via_transit: ride,
                        via_park: false,
                    },
                ));
            }
        }
        for &(index, departure_at) in &carrying_round {
            let source = StopIdx(index as u32);
            for edge in inputs.walking.from_stop(source) {
                let Some(arrival) = departure_at
                    .checked_add(edge.duration)
                    .filter(|&at| at != UNREACHED)
                else {
                    continue;
                };
                walks.push((
                    edge.to,
                    arrival,
                    CarriageLabel::Transfer {
                        from_stop: source,
                        duration: edge.duration,
                        ride: false,
                        via_transit: None,
                        via_park: true,
                    },
                ));
            }
        }
        for (stop, arrival, label) in walks {
            planes[FREE].improve(round, stop, arrival, label);
        }

        if planes[CARRYING].marked.is_empty() && planes[FREE].marked.is_empty() {
            break;
        }
    }

    CarriageSearch { rounds, planes }
}

/// Scans the patterns serving the plane's marked stops, collecting the
/// round's alighting candidates; `mask` filters boardable trips for
/// the Carrying plane.
fn scan_patterns(
    inputs: &CarriageInputs<'_>,
    round: usize,
    plane: &Plane,
    marked: &[StopIdx],
    mask: Option<&[bool]>,
    writes: &mut Vec<(StopIdx, u32, CarriageLabel)>,
) {
    const DAY_SECONDS: u32 = 86_400;
    let timetable = inputs.timetable;
    let has_previous = inputs.active_services_previous.iter().any(|&active| active);
    let mut queue_position: std::collections::HashMap<u32, u16> = std::collections::HashMap::new();
    for &stop in marked {
        for pattern_stop in timetable.patterns_at_stop(stop) {
            let slot = queue_position
                .entry(pattern_stop.pattern.0)
                .or_insert(u16::MAX);
            if pattern_stop.position < *slot {
                *slot = pattern_stop.position;
            }
        }
    }
    // Deterministic scan order: equal-arrival ties resolve by pattern
    // index, never by a hash map's seeding.
    let mut queued: Vec<(u32, u16)> = queue_position.into_iter().collect();
    queued.sort_unstable();
    for (pattern, start_position) in queued {
        let pattern = crate::timetable::PatternIdx(pattern);
        let stops = timetable.pattern_stops(pattern);
        let mut currents: [Option<(TripIdx, u16)>; 2] = [None, None];
        for position in start_position as usize..stops.len() {
            for (current, day_offset) in currents.into_iter().zip([0, DAY_SECONDS]) {
                if let Some((trip, board_position)) = current {
                    let arrival = timetable.trip_stop_times(trip)[position]
                        .arrival
                        .saturating_sub(day_offset);
                    writes.push((
                        stops[position],
                        arrival,
                        CarriageLabel::Transit {
                            trip,
                            board_position,
                            alight_position: position as u16,
                            day_offset,
                        },
                    ));
                }
            }
            let reached = plane.tau[round - 1][stops[position].0 as usize];
            if reached == UNREACHED || position + 1 == stops.len() {
                continue;
            }
            for (stream, current) in currents.iter_mut().enumerate() {
                let active: &[bool] = if stream == 0 {
                    inputs.active_services
                } else if has_previous {
                    inputs.active_services_previous
                } else {
                    continue;
                };
                let threshold = if stream == 0 {
                    reached
                } else {
                    match reached.checked_add(DAY_SECONDS) {
                        Some(threshold) => threshold,
                        None => continue,
                    }
                };
                let can_catch_earlier = match *current {
                    Some((trip, _)) => {
                        threshold <= timetable.trip_stop_times(trip)[position].departure
                    }
                    None => true,
                };
                if !can_catch_earlier {
                    continue;
                }
                let Some(trip) =
                    boardable_trip(timetable, active, mask, pattern, position, threshold)
                else {
                    continue;
                };
                let replaces = match *current {
                    Some((current_trip, _)) => {
                        timetable.trip_stop_times(trip)[position].departure
                            < timetable.trip_stop_times(current_trip)[position].departure
                    }
                    None => true,
                };
                if replaces {
                    *current = Some((trip, position as u16));
                }
            }
        }
    }
}

/// The earliest boardable trip under the plane's mask: the FIFO binary
/// search, then a forward walk over active (and, while carrying,
/// mask-permitted) trips.
fn boardable_trip(
    timetable: &Timetable,
    active: &[bool],
    mask: Option<&[bool]>,
    pattern: crate::timetable::PatternIdx,
    position: usize,
    ready: u32,
) -> Option<TripIdx> {
    match mask {
        None => earliest_active_trip(timetable, active, None, pattern, position, ready),
        Some(mask) => {
            // The FIFO lower bound, then a forward scan requiring both
            // an active service and a permitting mask on every
            // candidate.
            let range = timetable.pattern_trip_range(pattern);
            let first = earliest_active_trip(timetable, active, None, pattern, position, ready)?;
            (first.0..range.end).map(TripIdx).find(|&trip| {
                active
                    .get(timetable.trip_service(trip) as usize)
                    .copied()
                    .unwrap_or(false)
                    && mask[trip.0 as usize]
            })
        }
    }
}

/// One leg of a reconstructed carriage journey. Kept apart from the
/// shared [`crate::journey::Leg`] contract: possession state and park
/// events are carriage-specific outputs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CarriageLeg {
    /// The seed of `plane` at `to_stop` (street legs rebuild outside).
    Access {
        plane: usize,
        to_stop: StopIdx,
        arrival: u32,
    },
    Transit {
        trip: TripIdx,
        board_stop: StopIdx,
        alight_stop: StopIdx,
        board_position: u16,
        alight_position: u16,
        board_time: u32,
        alight_time: u32,
        bike_aboard: bool,
    },
    /// A relaxed transfer row; `ride` marks the carried vehicle's row.
    Transfer {
        from_stop: StopIdx,
        to_stop: StopIdx,
        departure: u32,
        arrival: u32,
        ride: bool,
    },
    /// The vehicle parked at `stop` at `at`; the chain continues in
    /// the Carrying plane before this instant.
    Park { stop: StopIdx, at: u32 },
}

impl CarriageSearch {
    /// Walks the labels backwards from (`plane`, `round`, `stop`) into
    /// the journey's legs, in travel order. A `Park` label crosses
    /// into the Carrying plane at the same round and stop; an inline
    /// `via_transit` ride (a shadowed transit arrival's extension)
    /// emits its transit leg directly.
    pub fn reconstruct(
        &self,
        timetable: &Timetable,
        plane: usize,
        round: usize,
        stop: StopIdx,
    ) -> Vec<CarriageLeg> {
        let mut legs = Vec::new();
        let mut plane_index = plane;
        let mut current_round = round;
        let mut at = stop;
        loop {
            let arrival = self.planes[plane_index].tau[current_round][at.0 as usize];
            match self.planes[plane_index].labels[current_round][at.0 as usize] {
                CarriageLabel::Unreached => {
                    unreachable!("carriage reconstruction hit an unreached label")
                }
                CarriageLabel::Access => {
                    legs.push(CarriageLeg::Access {
                        plane: plane_index,
                        to_stop: at,
                        arrival,
                    });
                    break;
                }
                CarriageLabel::Park => {
                    legs.push(CarriageLeg::Park {
                        stop: at,
                        at: arrival,
                    });
                    plane_index = CARRYING;
                }
                CarriageLabel::Transit {
                    trip,
                    board_position,
                    alight_position,
                    day_offset,
                } => {
                    let pattern = timetable.trip_pattern(trip);
                    let pattern_stops = timetable.pattern_stops(pattern);
                    let times = timetable.trip_stop_times(trip);
                    let board_stop = pattern_stops[board_position as usize];
                    legs.push(CarriageLeg::Transit {
                        trip,
                        board_stop,
                        alight_stop: at,
                        board_position,
                        alight_position,
                        board_time: times[board_position as usize]
                            .departure
                            .saturating_sub(day_offset),
                        alight_time: times[alight_position as usize]
                            .arrival
                            .saturating_sub(day_offset),
                        bike_aboard: plane_index == CARRYING,
                    });
                    at = board_stop;
                    current_round -= 1;
                }
                CarriageLabel::Transfer {
                    from_stop,
                    duration,
                    ride,
                    via_transit,
                    via_park,
                } => {
                    legs.push(CarriageLeg::Transfer {
                        from_stop,
                        to_stop: at,
                        departure: arrival - duration,
                        arrival,
                        ride,
                    });
                    if via_park {
                        // The park travels inline: its Free label may
                        // have been overwritten by a cheaper walk, but
                        // the Carrying seed below it is immutable.
                        legs.push(CarriageLeg::Park {
                            stop: from_stop,
                            at: arrival - duration,
                        });
                        plane_index = CARRYING;
                        at = from_stop;
                        continue;
                    }
                    match via_transit {
                        Some((trip, board_position, alight_position, day_offset)) => {
                            let pattern = timetable.trip_pattern(trip);
                            let pattern_stops = timetable.pattern_stops(pattern);
                            let times = timetable.trip_stop_times(trip);
                            let board_stop = pattern_stops[board_position as usize];
                            legs.push(CarriageLeg::Transit {
                                trip,
                                board_stop,
                                alight_stop: from_stop,
                                board_position,
                                alight_position,
                                board_time: times[board_position as usize]
                                    .departure
                                    .saturating_sub(day_offset),
                                alight_time: times[alight_position as usize]
                                    .arrival
                                    .saturating_sub(day_offset),
                                bike_aboard: plane_index == CARRYING,
                            });
                            at = board_stop;
                            current_round -= 1;
                        }
                        None => {
                            at = from_stop;
                        }
                    }
                }
            }
        }
        legs.reverse();
        legs
    }
}

#[cfg(test)]
mod tests;
