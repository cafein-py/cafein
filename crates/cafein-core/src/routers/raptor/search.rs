//! The round-based search core: labels, one pass, range profiles,
//! and the label-chain reconstruction.

use super::*;

/// Appends the destination-to-origin canonical tokens of the chain
/// behind `labels[round][stop]` and returns its root departure. Reads
/// the same label chains `walk_costs` walks, in the time/topology
/// domain only.
pub(super) fn chain_tokens_into(
    labels: &[Vec<Label>],
    timetable: &Timetable,
    stop: usize,
    round: usize,
    out: &mut Vec<PathToken>,
) -> u32 {
    let mut at = stop;
    let mut r = round;
    loop {
        match labels[r][at] {
            Label::Transit {
                trip,
                board_position,
                alight_position,
                day_offset,
            } => {
                out.push(PathToken::Ride {
                    trip: trip.0,
                    day_offset,
                    board: board_position,
                    alight: alight_position,
                });
                let pattern = timetable.trip_pattern(trip);
                at = timetable.pattern_stops(pattern)[board_position as usize].0 as usize;
                r -= 1;
            }
            Label::Transfer {
                from_stop,
                duration,
            } => {
                out.push(PathToken::Walk {
                    from: from_stop.0,
                    to: at as u32,
                    duration,
                });
                at = from_stop.0 as usize;
            }
            Label::Access {
                departure,
                duration,
            } => {
                out.push(PathToken::Access {
                    stop: at as u32,
                    duration,
                });
                return departure;
            }
            Label::Unreached => unreachable!("canonical key walked an unreached label"),
        }
    }
}

/// Seconds in a service day: a previous-day trip's stored times are shifted
/// back by this to place it on the queried day's clock.
pub(super) const DAY_SECONDS: u32 = 86_400;

/// How a stop's arrival time in a round was achieved.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Label {
    Unreached,
    /// Reached directly from the origin, leaving at `departure` and
    /// walking `duration` — the chain root the canonical path key
    /// compares.
    Access {
        departure: u32,
        duration: u32,
    },
    /// Alighted from a trip boarded at `board_position` of its pattern.
    /// `day_offset` is subtracted from the trip's stored times to place
    /// them on the queried day (nonzero for a previous-day trip).
    Transit {
        trip: TripIdx,
        board_position: u16,
        alight_position: u16,
        day_offset: u32,
    },
    /// Walked from another stop reached by transit in the same round.
    Transfer {
        from_stop: StopIdx,
        duration: u32,
    },
}

/// RAPTOR state shared by the passes of one query.
pub(crate) struct Search<'a> {
    timetable: &'a Timetable,
    transfers: &'a Transfers,
    request: &'a Request,
    pub(super) rounds: usize,
    /// Per-round arrival times; `tau[k][stop]` is the earliest arrival at
    /// `stop` with exactly `k` rides, over all departures processed so far.
    tau: Vec<Vec<u32>>,
    labels: Vec<Vec<Label>>,
    /// Prefix minimum of `tau`: `best[k][stop]` is the earliest arrival at
    /// `stop` with at most `k` rides, over all departures processed so
    /// far. Pruning at a round must not consult later rounds — a faster
    /// but more-rides arrival from a later departure does not dominate
    /// the fewer-ride option an earlier departure still offers.
    pub(super) best: Vec<Vec<u32>>,
    marked: Vec<StopIdx>,
    is_marked: Vec<bool>,
    /// First marked position per pattern for the current round.
    queue_position: Vec<u16>,
    queued_patterns: Vec<PatternIdx>,
    /// Reusable scratch for canonical-key comparisons on exact ties.
    key_scratch_a: Vec<PathToken>,
    key_scratch_b: Vec<PathToken>,
}

impl<'a> Search<'a> {
    pub(crate) fn new(
        timetable: &'a Timetable,
        transfers: &'a Transfers,
        request: &'a Request,
    ) -> Self {
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
            key_scratch_a: Vec::new(),
            key_scratch_b: Vec::new(),
        }
    }

    /// Clears the per-query state for a new request, reusing the
    /// allocated buffers. The request must keep the round count: callers
    /// pooling a search across origins hold `max_transfers` fixed.
    pub(super) fn reset(&mut self, request: &'a Request) {
        debug_assert_eq!(self.rounds, request.max_transfers as usize + 1);
        self.request = request;
        for tau in &mut self.tau {
            tau.fill(UNREACHED);
        }
        for labels in &mut self.labels {
            labels.fill(Label::Unreached);
        }
        for best in &mut self.best {
            best.fill(UNREACHED);
        }
        self.marked.clear();
        self.is_marked.fill(false);
        // `queue_position` needs no reset: every pass restores it to
        // u16::MAX as patterns are dequeued.
    }

    /// The earliest arrival at every stop over all processed departures,
    /// with any number of rides; unreachable stops are `None`.
    pub(super) fn arrivals(&self) -> Vec<Option<u32>> {
        self.best
            .last()
            .expect("search always has a round")
            .iter()
            .map(|&arrival| (arrival != UNREACHED).then_some(arrival))
            .collect()
    }

    /// The fastest journey's aggregated costs to a destination point
    /// over its egress links; `None` when no link's stop is reachable.
    pub(super) fn costs_to_point(
        &self,
        point: u32,
        egress: &[(StopIdx, u32, f64)],
        departure: u32,
        inputs: &CostInputs<'_>,
        access_meters: &HashMap<StopIdx, f64>,
    ) -> Option<CostRow> {
        let best = self.best.last().expect("search always has a round");
        let mut chosen: Option<(u32, StopIdx, f64)> = None;
        for &(stop, seconds, meters) in egress {
            let at_stop = best[stop.0 as usize];
            if at_stop == UNREACHED {
                continue;
            }
            let Some(arrival) = at_stop.checked_add(seconds).filter(|&at| at != UNREACHED) else {
                continue;
            };
            if chosen.is_none_or(|(current, _, _)| arrival < current) {
                chosen = Some((arrival, stop, meters));
            }
        }
        let (arrival, stop, egress_meters) = chosen?;
        let mut row = self.costs_to(stop, departure, inputs, Some(access_meters))?;
        row.to = point;
        row.seconds = arrival - departure;
        row.walk_meters += egress_meters;
        Some(row)
    }

    /// The fastest journey's aggregated costs to `stop`, walking the
    /// label chain of the round that achieves the earliest arrival;
    /// `None` when the stop is unreachable.
    pub(super) fn costs_to(
        &self,
        stop: StopIdx,
        departure: u32,
        inputs: &CostInputs<'_>,
        access_meters: Option<&HashMap<StopIdx, f64>>,
    ) -> Option<CostRow> {
        let mut best_round = 0;
        let mut best_arrival = UNREACHED;
        for round in 0..=self.rounds {
            let arrival = self.tau[round][stop.0 as usize];
            if arrival < best_arrival {
                best_arrival = arrival;
                best_round = round;
            }
        }
        if best_arrival == UNREACHED {
            return None;
        }
        let mut row = self.walk_costs(stop, best_round, inputs, access_meters);
        row.seconds = best_arrival - departure;
        Some(row)
    }

    /// The aggregated costs of the journey behind `tau[round][stop]`,
    /// walking its label chain. The caller has checked the label is
    /// reached; `seconds` is left at zero for the caller to fill (the
    /// travel time depends on which departure the caller charges).
    fn walk_costs(
        &self,
        stop: StopIdx,
        best_round: usize,
        inputs: &CostInputs<'_>,
        access_meters: Option<&HashMap<StopIdx, f64>>,
    ) -> CostRow {
        let timetable = self.timetable;
        let mut round = best_round;
        let mut at = stop;
        let mut rides = 0u32;
        let mut transit_meters = 0.0;
        let mut walk_meters = 0.0;
        let mut grams = 0.0;
        let mut resolved = true;
        let mut legs: Vec<(TripIdx, u16, u16)> = Vec::new();
        let mut fare_legs: Vec<FareLeg> = Vec::new();
        loop {
            match self.labels[round][at.0 as usize] {
                Label::Transit {
                    trip,
                    board_position,
                    alight_position,
                    day_offset,
                } => {
                    rides += 1;
                    let meters = inputs
                        .geometry
                        .leg_distance(trip, board_position, alight_position)
                        as f64;
                    transit_meters += meters;
                    let factor = inputs.factors[trip.0 as usize];
                    if factor.is_finite() {
                        grams += meters / 1000.0 * factor;
                    } else {
                        resolved = false;
                    }
                    if inputs.with_geometry {
                        legs.push((trip, board_position, alight_position));
                    }
                    let pattern = timetable.trip_pattern(trip);
                    let board_stop = timetable.pattern_stops(pattern)[board_position as usize];
                    if inputs.fares.is_some() {
                        fare_legs.push(FareLeg {
                            route: timetable.pattern_route(pattern),
                            board_stop: board_stop.0,
                            alight_stop: at.0,
                            board_time: timetable.trip_stop_times(trip)[board_position as usize]
                                .departure
                                .saturating_sub(day_offset),
                        });
                    }
                    at = board_stop;
                    round -= 1;
                }
                Label::Transfer {
                    from_stop,
                    duration: _,
                } => {
                    // Transfers are deduplicated per stop pair: the one
                    // edge found is the one routing relaxed.
                    walk_meters += self
                        .transfers
                        .from_stop(from_stop)
                        .iter()
                        .find(|transfer| transfer.to == at)
                        .map(|transfer| transfer.meters)
                        .unwrap_or(0.0);
                    at = from_stop;
                }
                Label::Access { .. } => {
                    if let Some(access) = access_meters {
                        walk_meters += access.get(&at).copied().unwrap_or(0.0);
                    }
                    break;
                }
                Label::Unreached => unreachable!("cost reconstruction hit an unreached label"),
            }
        }
        let geometry = match (inputs.with_geometry, inputs.leg_geometry) {
            (true, Some(shapes)) => {
                let parts: Vec<Vec<(f64, f64)>> = legs
                    .iter()
                    .rev()
                    .map(|&(trip, board, alight)| shapes.leg_coordinates(trip, board, alight))
                    .collect();
                Some(wkb_multi_line_string(&parts))
            }
            _ => None,
        };
        let fare = match inputs.fares {
            Some(tables) => {
                // The chain walks back to front; pricing needs rides in
                // order.
                fare_legs.reverse();
                tables.price(&fare_legs)
            }
            None => f64::NAN,
        };
        CostRow {
            to: stop.0,
            seconds: 0,
            rides,
            transit_meters,
            walk_meters,
            emission_grams: if resolved { grams } else { f64::NAN },
            fare,
            geometry,
        }
    }

    /// Folds a pass's improved arrivals into each destination's best
    /// candidate on the objective: the lowest-key journey within the
    /// travel-time budget seen so far, ties resolved toward the
    /// shorter travel time.
    ///
    /// `thresholds` carries the best arrival per (destination, round)
    /// across the descending departures; an arrival that does not
    /// improve its threshold left `tau` unchanged, so it has no label
    /// chain to reconstruct — the candidates are exactly the profile's
    /// (departure, arrival, rides)-Pareto set, the same set
    /// `journey_frontier` sees. NaN keys (an unresolved emission
    /// factor, an unpriceable journey) never qualify.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn fold_best(
        &self,
        departure: u32,
        destinations: &[StopIdx],
        budget: Option<u32>,
        inputs: &CostInputs<'_>,
        access_meters: Option<&HashMap<StopIdx, f64>>,
        objective: Objective,
        thresholds: &mut [u32],
        best: &mut [Option<CostRow>],
    ) {
        for (slot, &stop) in destinations.iter().enumerate() {
            let thresholds = &mut thresholds[slot * (self.rounds + 1)..][..self.rounds + 1];
            // At-most-k-ride thresholds — `collect`'s admission rule —
            // so the ride candidates match the profile's. Round 0 joins
            // as the zero-ride floor: the access-seeded stops (for a
            // stop matrix, the origin itself at 0 s and 0 g — footpaths
            // only relax after a ride, as in the time-optimising
            // matrix); for points the caller's direct-walk overlay
            // plays that part instead.
            for round in 0..=self.rounds {
                let arrival = self.tau[round][stop.0 as usize];
                if arrival >= thresholds[round] {
                    continue;
                }
                for threshold in &mut thresholds[round..] {
                    *threshold = (*threshold).min(arrival);
                }
                let seconds = arrival - departure;
                if budget.is_some_and(|budget| seconds > budget) {
                    continue;
                }
                let mut row = self.walk_costs(stop, round, inputs, access_meters);
                row.seconds = seconds;
                fold_better(&mut best[slot], row, objective);
            }
        }
    }

    /// [`Search::fold_best`] joined through each destination point's
    /// egress links: a point's per-round arrival is the minimum over its
    /// links of the stop arrival plus the egress walk.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn fold_best_points(
        &self,
        departure: u32,
        egress: &[Vec<(StopIdx, u32, f64)>],
        budget: Option<u32>,
        inputs: &CostInputs<'_>,
        access_meters: &HashMap<StopIdx, f64>,
        objective: Objective,
        thresholds: &mut [u32],
        best: &mut [Option<CostRow>],
    ) {
        for (point, links) in egress.iter().enumerate() {
            let thresholds = &mut thresholds[point * (self.rounds + 1)..][..self.rounds + 1];
            // Rounds from 1, at-most-k-ride thresholds over the joined
            // point arrival — exactly `collect`'s admission rule (the
            // per-stop pruning inside `tau` does not survive the link
            // join, so the suffix update is what enforces dominance by
            // fewer-ride journeys here). The walking-only alternative
            // arrives via the caller's direct-walk overlay, as it does
            // for `journey_frontier`.
            for round in 1..=self.rounds {
                let mut chosen: Option<(u32, StopIdx, f64)> = None;
                for &(stop, seconds, meters) in links {
                    let at_stop = self.tau[round][stop.0 as usize];
                    if at_stop == UNREACHED {
                        continue;
                    }
                    let Some(arrival) = at_stop.checked_add(seconds).filter(|&at| at != UNREACHED)
                    else {
                        continue;
                    };
                    if chosen.is_none_or(|(current, _, _)| arrival < current) {
                        chosen = Some((arrival, stop, meters));
                    }
                }
                let Some((arrival, stop, egress_meters)) = chosen else {
                    continue;
                };
                if arrival >= thresholds[round] {
                    continue;
                }
                for threshold in &mut thresholds[round..] {
                    *threshold = (*threshold).min(arrival);
                }
                let seconds = arrival - departure;
                if budget.is_some_and(|budget| seconds > budget) {
                    continue;
                }
                let mut row = self.walk_costs(stop, round, inputs, Some(access_meters));
                row.to = point as u32;
                row.seconds = seconds;
                row.walk_meters += egress_meters;
                fold_better(&mut best[point], row, objective);
            }
        }
    }

    /// Runs one pass per departure — which must be strictly decreasing —
    /// and returns the journeys no later departure dominates, sorted by
    /// departure and then by ride count.
    pub(super) fn profile(mut self, departures: &[u32]) -> Vec<Journey> {
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

    /// Records an alighting from `trip` at `position`: a strict
    /// improvement writes through; an exact same-round tie keeps the
    /// canonical chain and re-marks the stop so the replacement
    /// propagates.
    #[allow(clippy::too_many_arguments)]
    fn alight_at(
        &mut self,
        timetable: &Timetable,
        round: usize,
        stops: &[StopIdx],
        position: usize,
        trip: TripIdx,
        board_position: u16,
        day_offset: u32,
    ) {
        let stop = stops[position].0 as usize;
        let arrival = timetable.trip_stop_times(trip)[position]
            .arrival
            .saturating_sub(day_offset);
        if arrival < self.best[round][stop] {
            self.tau[round][stop] = arrival;
            for best in &mut self.best[round..] {
                best[stop] = best[stop].min(arrival);
            }
            self.labels[round][stop] = Label::Transit {
                trip,
                board_position,
                alight_position: position as u16,
                day_offset,
            };
            if !self.is_marked[stop] {
                self.is_marked[stop] = true;
                self.marked.push(stops[position]);
            }
        } else if arrival == self.tau[round][stop] && arrival != UNREACHED {
            let mut challenger = std::mem::take(&mut self.key_scratch_a);
            let mut incumbent = std::mem::take(&mut self.key_scratch_b);
            challenger.clear();
            challenger.push(PathToken::Ride {
                trip: trip.0,
                day_offset,
                board: board_position,
                alight: position as u16,
            });
            let root = chain_tokens_into(
                &self.labels,
                timetable,
                stops[board_position as usize].0 as usize,
                round - 1,
                &mut challenger,
            );
            incumbent.clear();
            let incumbent_root =
                chain_tokens_into(&self.labels, timetable, stop, round, &mut incumbent);
            if challenger_wins(root, &challenger, incumbent_root, &incumbent) {
                self.labels[round][stop] = Label::Transit {
                    trip,
                    board_position,
                    alight_position: position as u16,
                    day_offset,
                };
                if !self.is_marked[stop] {
                    self.is_marked[stop] = true;
                    self.marked.push(stops[position]);
                }
            }
            self.key_scratch_a = challenger;
            self.key_scratch_b = incumbent;
        }
    }

    /// One RAPTOR pass from `departure`, improving the shared state.
    pub(crate) fn run(&mut self, departure: u32) {
        let timetable = self.timetable;
        let request = self.request;
        let has_previous = request
            .active_services_previous
            .iter()
            .any(|&active| active);

        // Leftover marks from the previous pass describe stops whose
        // labels are already final for later departures; they carry no
        // work for this one.
        for stop in std::mem::take(&mut self.marked) {
            self.is_marked[stop.0 as usize] = false;
        }

        for &(stop, duration) in &request.access {
            // Skip access whose arrival cannot be represented below the
            // UNREACHED sentinel; wrapping or saturating would corrupt it.
            let Some(arrival) = departure
                .checked_add(duration)
                .filter(|&at| at != UNREACHED)
            else {
                continue;
            };
            let index = stop.0 as usize;
            if arrival < self.tau[0][index] {
                self.tau[0][index] = arrival;
                self.labels[0][index] = Label::Access {
                    departure,
                    duration,
                };
                for best in &mut self.best {
                    best[index] = best[index].min(arrival);
                }
                if !self.is_marked[index] {
                    self.is_marked[index] = true;
                    self.marked.push(stop);
                }
            } else if arrival == self.tau[0][index] && arrival != UNREACHED {
                let mut challenger = std::mem::take(&mut self.key_scratch_a);
                let mut incumbent = std::mem::take(&mut self.key_scratch_b);
                challenger.clear();
                challenger.push(PathToken::Access {
                    stop: stop.0,
                    duration,
                });
                incumbent.clear();
                let incumbent_root =
                    chain_tokens_into(&self.labels, timetable, index, 0, &mut incumbent);
                if challenger_wins(departure, &challenger, incumbent_root, &incumbent) {
                    self.labels[0][index] = Label::Access {
                        departure,
                        duration,
                    };
                    if !self.is_marked[index] {
                        self.is_marked[index] = true;
                        self.marked.push(stop);
                    }
                }
                self.key_scratch_a = challenger;
                self.key_scratch_b = incumbent;
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
                let mut currents: [Option<(TripIdx, u16)>; 2] = [None, None];

                for position in start_position as usize..stops.len() {
                    let stop = stops[position].0 as usize;

                    for (current, day_offset) in currents.into_iter().zip([0, DAY_SECONDS]) {
                        if let Some((trip, board_position)) = current {
                            self.alight_at(
                                timetable,
                                round,
                                stops,
                                position,
                                trip,
                                board_position,
                                day_offset,
                            );
                        }
                    }

                    // Try to catch an earlier trip from this stop, using the
                    // previous round's arrival. The arrival handling above
                    // has already recorded any improvement at this position;
                    // boarding at a pattern's last position is pointless
                    // because there is no later stop to alight at, and other
                    // patterns serving this stop are queued separately. The
                    // day streams board independently: merged across days
                    // departures are not FIFO — yesterday's tail can depart
                    // later here yet arrive earlier downstream — so each
                    // stream rides its own earliest trip and the arrival
                    // writes settle the competition.
                    let reached = self.tau[round - 1][stop];
                    if reached == UNREACHED || position + 1 == stops.len() {
                        continue;
                    }
                    for (stream, current) in currents.iter_mut().enumerate() {
                        let active: &[bool] = if stream == 0 {
                            &request.active_services
                        } else if has_previous {
                            &request.active_services_previous
                        } else {
                            continue;
                        };
                        // A previous-day trip stored at `t` runs at
                        // `t - DAY_SECONDS`, so it is boardable from
                        // `reached` when `t >= reached + DAY_SECONDS`.
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
                            earliest_active_trip(timetable, active, pattern, position, threshold)
                        else {
                            continue;
                        };
                        // Stored times order a single day's FIFO stream.
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
                    let Some(arrival) = departure_at_stop
                        .checked_add(transfer.duration)
                        .filter(|&at| at != UNREACHED)
                    else {
                        continue;
                    };
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
                    } else if arrival == self.tau[round][to] && arrival != UNREACHED {
                        let mut challenger = std::mem::take(&mut self.key_scratch_a);
                        let mut incumbent = std::mem::take(&mut self.key_scratch_b);
                        challenger.clear();
                        challenger.push(PathToken::Walk {
                            from: stop.0,
                            to: transfer.to.0,
                            duration: transfer.duration,
                        });
                        let root = chain_tokens_into(
                            &self.labels,
                            timetable,
                            stop.0 as usize,
                            round,
                            &mut challenger,
                        );
                        incumbent.clear();
                        let incumbent_root =
                            chain_tokens_into(&self.labels, timetable, to, round, &mut incumbent);
                        if challenger_wins(root, &challenger, incumbent_root, &incumbent) {
                            self.labels[round][to] = Label::Transfer {
                                from_stop: stop,
                                duration: transfer.duration,
                            };
                            if !self.is_marked[to] {
                                self.is_marked[to] = true;
                                self.marked.push(transfer.to);
                            }
                        }
                        self.key_scratch_a = challenger;
                        self.key_scratch_b = incumbent;
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
                let Some(arrival) = at_stop.checked_add(duration).filter(|&at| at != UNREACHED)
                else {
                    continue;
                };
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
        // Cannot saturate: collect only emits in-range egress arrivals.
        legs.push(Leg::Egress {
            from_stop: egress_stop,
            departure: departure_at_stop,
            arrival: departure_at_stop.saturating_add(egress_duration),
        });

        let mut current_round = round;
        let mut stop = egress_stop;
        loop {
            match self.labels[current_round][stop.0 as usize] {
                Label::Transit {
                    trip,
                    board_position,
                    alight_position,
                    day_offset,
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
                        board_time: times[board_position as usize]
                            .departure
                            .saturating_sub(day_offset),
                        alight_time: times[alight_position as usize]
                            .arrival
                            .saturating_sub(day_offset),
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
                Label::Access { .. } => {
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
            arrival: departure_at_stop.saturating_add(egress_duration),
            legs,
        }
    }
}

/// The earliest trip of `pattern` departing `position` at or after `reached`
/// whose service is `active`. Valid because departures at every position are
/// sorted within a FIFO pattern.
pub(super) fn earliest_active_trip(
    timetable: &Timetable,
    active: &[bool],
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
        active
            .get(timetable.trip_service(trip) as usize)
            .copied()
            .unwrap_or(false)
    })
}
