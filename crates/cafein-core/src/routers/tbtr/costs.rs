//! Cost-row reconstruction from winner chains and the four matrix
//! cost methods.

use super::*;

impl<'a> TbtrEngine<'a> {
    /// The fastest journey's aggregated costs to `stop`, mirroring
    /// `Search::costs_to`: rounds scan ascending, only a strictly
    /// earlier arrival replaces the winner — fewest rides on ties.
    fn matrix_costs_to(
        &self,
        state: &MatrixState,
        stop: StopIdx,
        departure: u32,
        inputs: &CostInputs<'_>,
        access_meters: Option<&HashMap<StopIdx, f64>>,
    ) -> Option<CostRow> {
        let mut best_round = 0;
        let mut best_arrival = UNREACHED;
        for round in 0..=state.rounds {
            let arrival = state.tau_at(stop, round);
            if arrival < best_arrival {
                best_arrival = arrival;
                best_round = round;
            }
        }
        if best_arrival == UNREACHED {
            return None;
        }
        let mut row = self.matrix_cost_row(state, stop, best_round, inputs, access_meters);
        row.seconds = best_arrival - departure;
        Some(row)
    }

    /// The aggregated costs of the journey behind the winner at
    /// `(stop, round)`, walking its chain destination → origin in the
    /// same order (and with the same floating-point accumulation
    /// sequence) as `Raptor::walk_costs`, so identical journeys yield
    /// bit-identical rows.
    fn matrix_cost_row(
        &self,
        state: &MatrixState,
        stop: StopIdx,
        round: usize,
        inputs: &CostInputs<'_>,
        access_meters: Option<&HashMap<StopIdx, f64>>,
    ) -> CostRow {
        let mut rides = 0u32;
        let mut transit_meters = 0.0;
        let mut walk_meters = 0.0;
        let mut grams = 0.0;
        let mut resolved = true;
        let mut legs: Vec<(TripIdx, u16, u16)> = Vec::new();
        let mut fare_legs: Vec<crate::fares::FareLeg> = Vec::new();
        let slot = stop.0 as usize * (state.rounds + 1) + round;
        // Resolve the destination-side walk, if any, then follow the
        // segment chain of the alighted arena entry.
        let (mut segment, mut alight_position, mut at) = match state.winners[slot] {
            StopWinner::Unreached => {
                unreachable!("cost reconstruction hit an unreached winner")
            }
            StopWinner::Access { .. } => {
                if let Some(access) = access_meters {
                    walk_meters += access.get(&stop).copied().unwrap_or(0.0);
                }
                return self.finish_cost_row(
                    stop.0,
                    rides,
                    transit_meters,
                    walk_meters,
                    grams,
                    resolved,
                    legs,
                    fare_legs,
                    inputs,
                );
            }
            StopWinner::Alight { segment, alight } => (segment, alight, stop),
            StopWinner::Walked {
                segment,
                alight,
                from,
            } => {
                walk_meters += self
                    .footpaths
                    .from_stop(from)
                    .iter()
                    .find(|transfer| transfer.to == stop)
                    .map(|transfer| transfer.meters)
                    .unwrap_or(0.0);
                (segment, alight, from)
            }
        };
        loop {
            let entry = &state.arena[segment as usize];
            let backing = self.view.backing(entry.trip);
            let offset = self.view.day_offset(entry.trip);
            rides += 1;
            let meters = inputs
                .geometry
                .leg_distance(backing, entry.board, alight_position)
                as f64;
            transit_meters += meters;
            let factor = inputs.factors[backing.0 as usize];
            if factor.is_finite() {
                grams += meters / 1000.0 * factor;
            } else {
                resolved = false;
            }
            if inputs.with_geometry {
                legs.push((backing, entry.board, alight_position));
            }
            let pattern = self.timetable.trip_pattern(backing);
            let board_stop = self.timetable.pattern_stops(pattern)[entry.board as usize];
            if inputs.fares.is_some() {
                fare_legs.push(crate::fares::FareLeg {
                    route: self.timetable.pattern_route(pattern),
                    board_stop: board_stop.0,
                    alight_stop: at.0,
                    board_time: self.timetable.trip_stop_times(backing)[entry.board as usize]
                        .departure
                        .saturating_sub(offset),
                });
            }
            match entry.origin {
                SegmentOrigin::Access { stop: origin, .. } => {
                    if board_stop != origin {
                        walk_meters += self
                            .footpaths
                            .from_stop(origin)
                            .iter()
                            .find(|transfer| transfer.to == board_stop)
                            .map(|transfer| transfer.meters)
                            .unwrap_or(0.0);
                    }
                    if let Some(access) = access_meters {
                        walk_meters += access.get(&origin).copied().unwrap_or(0.0);
                    }
                    break;
                }
                SegmentOrigin::Transfer { parent, alight } => {
                    let parent_entry = &state.arena[parent as usize];
                    let parent_line = self.view.line_of(parent_entry.trip);
                    let parent_stops = self
                        .timetable
                        .pattern_stops(self.view.line_pattern(parent_line));
                    let parent_stop = parent_stops[alight as usize];
                    if parent_stop != board_stop {
                        walk_meters += self
                            .footpaths
                            .from_stop(parent_stop)
                            .iter()
                            .find(|transfer| transfer.to == board_stop)
                            .map(|transfer| transfer.meters)
                            .unwrap_or(0.0);
                    }
                    at = parent_stop;
                    alight_position = alight;
                    segment = parent;
                }
            }
        }
        self.finish_cost_row(
            stop.0,
            rides,
            transit_meters,
            walk_meters,
            grams,
            resolved,
            legs,
            fare_legs,
            inputs,
        )
    }

    /// Assembles the row tail exactly as `Raptor::walk_costs` does:
    /// geometry legs reversed into ride order, fares priced in ride
    /// order, NaN-poisoned emissions.
    #[allow(clippy::too_many_arguments)]
    fn finish_cost_row(
        &self,
        to: u32,
        rides: u32,
        transit_meters: f64,
        walk_meters: f64,
        grams: f64,
        resolved: bool,
        legs: Vec<(TripIdx, u16, u16)>,
        mut fare_legs: Vec<crate::fares::FareLeg>,
        inputs: &CostInputs<'_>,
    ) -> CostRow {
        let geometry = match (inputs.with_geometry, inputs.leg_geometry) {
            (true, Some(shapes)) => {
                let parts: Vec<Vec<(f64, f64)>> = legs
                    .iter()
                    .rev()
                    .map(|&(trip, board, alight)| shapes.leg_coordinates(trip, board, alight))
                    .collect();
                Some(crate::geometry::wkb_multi_line_string(&parts))
            }
            _ => None,
        };
        let fare = match inputs.fares {
            Some(tables) => {
                fare_legs.reverse();
                tables.price(&fare_legs)
            }
            None => f64::NAN,
        };
        CostRow {
            to,
            seconds: 0,
            rides,
            transit_meters,
            walk_meters,
            emission_grams: if resolved { grams } else { f64::NAN },
            fare,
            geometry,
        }
    }

    /// The fastest journey's aggregated costs from each request to each
    /// destination — the TBTR counterpart of `Raptor::cost_matrix`,
    /// fanned out over the origins with pooled per-worker state.
    pub fn cost_matrix(
        &self,
        inputs: &CostInputs<'_>,
        requests: &[Request],
        destinations: &[StopIdx],
    ) -> Vec<Vec<CostRow>> {
        requests
            .par_iter()
            .map_init(
                || None,
                |pooled: &mut Option<MatrixState>, request| {
                    let state = match pooled {
                        Some(state) if state.rounds == request.max_transfers as usize + 1 => {
                            state.reset(self);
                            state
                        }
                        _ => pooled.insert(MatrixState::new(self, request.max_transfers)),
                    };
                    self.matrix_pass(request.departure, &request.access, state);
                    destinations
                        .iter()
                        .filter_map(|&stop| {
                            self.matrix_costs_to(state, stop, request.departure, inputs, None)
                        })
                        .collect()
                },
            )
            .collect()
    }

    /// The objective-best journey's costs within a travel-time budget,
    /// per destination, over a departure window — the TBTR counterpart
    /// of `Raptor::least_cost_matrix`, with `Search::fold_best`'s exact
    /// admission order: per-round suffix thresholds first (strict), the
    /// budget second, reconstruction third, the objective fold last.
    #[allow(clippy::too_many_arguments)]
    pub fn least_cost_matrix(
        &self,
        inputs: &CostInputs<'_>,
        requests: &[Request],
        destinations: &[StopIdx],
        window: u32,
        budget: Option<u32>,
        objective: crate::raptor::Objective,
    ) -> Vec<Vec<CostRow>> {
        requests
            .par_iter()
            .map_init(
                || None,
                |pooled: &mut Option<MatrixState>, request| {
                    let state = match pooled {
                        Some(state) if state.rounds == request.max_transfers as usize + 1 => {
                            state.reset(self);
                            state
                        }
                        _ => pooled.insert(MatrixState::new(self, request.max_transfers)),
                    };
                    let departures =
                        crate::raptor::departure_candidates(self.timetable, request, window);
                    let mut thresholds = vec![UNREACHED; destinations.len() * (state.rounds + 1)];
                    let mut best: Vec<Option<CostRow>> = vec![None; destinations.len()];
                    for &departure in &departures {
                        self.matrix_pass(departure, &request.access, state);
                        self.fold_matrix_best(
                            state,
                            departure,
                            destinations,
                            budget,
                            inputs,
                            None,
                            objective,
                            &mut thresholds,
                            &mut best,
                        );
                    }
                    best.into_iter().flatten().collect()
                },
            )
            .collect()
    }

    /// One pass's fold onto the per-destination bests — the mirror of
    /// `Search::fold_best`. Stale slots from later departures fail the
    /// strict threshold and are never reconstructed, so each fold reads
    /// a consistent snapshot of what its pass improved.
    #[allow(clippy::too_many_arguments)]
    fn fold_matrix_best(
        &self,
        state: &MatrixState,
        departure: u32,
        destinations: &[StopIdx],
        budget: Option<u32>,
        inputs: &CostInputs<'_>,
        access_meters: Option<&HashMap<StopIdx, f64>>,
        objective: crate::raptor::Objective,
        thresholds: &mut [u32],
        best: &mut [Option<CostRow>],
    ) {
        for (slot, &stop) in destinations.iter().enumerate() {
            let thresholds = &mut thresholds[slot * (state.rounds + 1)..][..state.rounds + 1];
            for round in 0..=state.rounds {
                let arrival = state.tau_at(stop, round);
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
                let mut row = self.matrix_cost_row(state, stop, round, inputs, access_meters);
                row.seconds = seconds;
                crate::raptor::fold_better(&mut best[slot], row, objective);
            }
        }
    }

    /// The fastest journey's costs to each destination *point*, joined
    /// through egress link tables — the TBTR counterpart of
    /// `Raptor::cost_matrix_to_points`, including `costs_to_point`'s
    /// link election: the first strictly smaller joined arrival wins,
    /// with no ride-count comparison across equal links.
    pub fn cost_matrix_to_points(
        &self,
        inputs: &CostInputs<'_>,
        requests: &[Request],
        access_meters: &[HashMap<StopIdx, f64>],
        egress: &[Vec<(StopIdx, u32, f64)>],
    ) -> Vec<Vec<CostRow>> {
        assert_eq!(requests.len(), access_meters.len());
        requests
            .par_iter()
            .zip(access_meters.par_iter())
            .map_init(
                || None,
                |pooled: &mut Option<MatrixState>, (request, access)| {
                    let state = match pooled {
                        Some(state) if state.rounds == request.max_transfers as usize + 1 => {
                            state.reset(self);
                            state
                        }
                        _ => pooled.insert(MatrixState::new(self, request.max_transfers)),
                    };
                    self.matrix_pass(request.departure, &request.access, state);
                    egress
                        .iter()
                        .enumerate()
                        .filter_map(|(point, links)| {
                            self.matrix_costs_to_point(
                                state,
                                point as u32,
                                links,
                                request.departure,
                                inputs,
                                access,
                            )
                        })
                        .collect()
                },
            )
            .collect()
    }

    /// `Raptor::costs_to_point`'s mirror over the matrix state.
    fn matrix_costs_to_point(
        &self,
        state: &MatrixState,
        point: u32,
        egress: &[(StopIdx, u32, f64)],
        departure: u32,
        inputs: &CostInputs<'_>,
        access_meters: &HashMap<StopIdx, f64>,
    ) -> Option<CostRow> {
        let mut chosen: Option<(u32, StopIdx, f64)> = None;
        for &(stop, seconds, meters) in egress {
            let at_stop = (0..=state.rounds)
                .map(|round| state.tau_at(stop, round))
                .min()
                .expect("a matrix state always has a round");
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
        let mut row = self.matrix_costs_to(state, stop, departure, inputs, Some(access_meters))?;
        row.to = point;
        row.seconds = arrival - departure;
        row.walk_meters += egress_meters;
        Some(row)
    }

    /// `Raptor::least_cost_matrix_to_points`'s TBTR counterpart.
    #[allow(clippy::too_many_arguments)]
    pub fn least_cost_matrix_to_points(
        &self,
        inputs: &CostInputs<'_>,
        requests: &[Request],
        access_meters: &[HashMap<StopIdx, f64>],
        egress: &[Vec<(StopIdx, u32, f64)>],
        window: u32,
        budget: Option<u32>,
        objective: crate::raptor::Objective,
    ) -> Vec<Vec<CostRow>> {
        assert_eq!(requests.len(), access_meters.len());
        requests
            .par_iter()
            .zip(access_meters.par_iter())
            .map_init(
                || None,
                |pooled: &mut Option<MatrixState>, (request, access)| {
                    let state = match pooled {
                        Some(state) if state.rounds == request.max_transfers as usize + 1 => {
                            state.reset(self);
                            state
                        }
                        _ => pooled.insert(MatrixState::new(self, request.max_transfers)),
                    };
                    let departures =
                        crate::raptor::departure_candidates(self.timetable, request, window);
                    let mut thresholds = vec![UNREACHED; egress.len() * (state.rounds + 1)];
                    let mut best: Vec<Option<CostRow>> = vec![None; egress.len()];
                    for &departure in &departures {
                        self.matrix_pass(departure, &request.access, state);
                        self.fold_matrix_best_points(
                            state,
                            departure,
                            egress,
                            budget,
                            inputs,
                            access,
                            objective,
                            &mut thresholds,
                            &mut best,
                        );
                    }
                    best.into_iter().flatten().collect()
                },
            )
            .collect()
    }

    /// `Search::fold_best_points`' mirror: rounds from 1 (the access
    /// floor is the caller's direct-walk overlay), the first strictly
    /// smaller joined link wins each round, thresholds and budget in
    /// the same order.
    #[allow(clippy::too_many_arguments)]
    fn fold_matrix_best_points(
        &self,
        state: &MatrixState,
        departure: u32,
        egress: &[Vec<(StopIdx, u32, f64)>],
        budget: Option<u32>,
        inputs: &CostInputs<'_>,
        access_meters: &HashMap<StopIdx, f64>,
        objective: crate::raptor::Objective,
        thresholds: &mut [u32],
        best: &mut [Option<CostRow>],
    ) {
        for (point, links) in egress.iter().enumerate() {
            let thresholds = &mut thresholds[point * (state.rounds + 1)..][..state.rounds + 1];
            for round in 1..=state.rounds {
                let mut chosen: Option<(u32, StopIdx, f64)> = None;
                for &(stop, seconds, meters) in links {
                    let at_stop = state.tau_at(stop, round);
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
                let mut row = self.matrix_cost_row(state, stop, round, inputs, Some(access_meters));
                row.to = point as u32;
                row.seconds = seconds;
                row.walk_meters += egress_meters;
                crate::raptor::fold_better(&mut best[point], row, objective);
            }
        }
    }
}
