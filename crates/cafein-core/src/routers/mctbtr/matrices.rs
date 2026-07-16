//! The batched products: the frontier and least-emissions matrices.

use super::*;

impl<'a> McTbtrEngine<'a> {
    /// The batched Pareto frontiers over this engine's candidate set:
    /// per request × destination slot, the (departure, arrival,
    /// emissions bucket) Pareto journeys of the departure window —
    /// each cell exactly the one-pair `route_range` set, the McTBTR
    /// counterpart of `mcraptor::frontier_matrix`. Stop mode takes the
    /// destination stops; door-to-door mode (`egress_active`) a
    /// per-stop final-egress map over `slot_count` destination points,
    /// the walking-only journey being the caller's overlay. One engine
    /// build (one transfer set) serves every origin, fanned out with
    /// rayon.
    #[allow(clippy::too_many_arguments)]
    pub fn frontier_matrix(
        &self,
        requests: &[Request],
        destinations: &[StopIdx],
        egress: &[Vec<(u32, u32, f64)>],
        egress_active: bool,
        slot_count: usize,
        window: u32,
        bucket: f64,
    ) -> Vec<Vec<Vec<Journey>>> {
        if egress_active {
            assert_eq!(
                egress.len(),
                self.timetable.stop_count() as usize,
                "the egress map must be per stop"
            );
        } else {
            assert_eq!(
                destinations.len(),
                slot_count,
                "stop mode takes one slot per destination stop"
            );
        }
        // A stop holds one slot, so repeated destination stops share the
        // first occurrence's slot and their cells are re-expanded to the
        // requested order after assembly.
        let mut slots = vec![0u32; self.timetable.stop_count() as usize];
        let mut cell_of: Vec<usize> = Vec::with_capacity(destinations.len());
        let mut unique = 0usize;
        for &stop in destinations {
            let slot = slots[stop.0 as usize];
            if slot == 0 {
                unique += 1;
                slots[stop.0 as usize] = unique as u32;
                cell_of.push(unique - 1);
            } else {
                cell_of.push(slot as usize - 1);
            }
        }
        let bag_count = if egress_active { slot_count } else { unique };
        let collected: Vec<(Vec<Vec<Journey>>, SearchStats)> = requests
            .par_iter()
            .map(|request| {
                let departures = departure_candidates(self.timetable, request, window);
                let mut bags: Vec<Vec<Arrived>> = vec![Vec::new(); bag_count];
                let mut sink = Some(FrontierSink {
                    slots: &slots,
                    egress,
                    egress_active,
                    bags: &mut bags,
                });
                let mut stats = SearchStats::default();
                let (arena, _) = self.passes(
                    request,
                    &departures,
                    bucket,
                    &mut None,
                    &mut sink,
                    &mut stats,
                );
                let cells: Vec<Vec<Journey>> = bags
                    .iter()
                    .map(|bag| {
                        let mut journeys: Vec<Journey> = bag
                            .iter()
                            .map(|arrived| self.assemble(arrived, &arena))
                            .collect();
                        journeys.sort_by_key(|journey| {
                            (journey.departure, journey.arrival, journey.rides())
                        });
                        journeys
                    })
                    .collect();
                let cells = if egress_active || unique == destinations.len() {
                    cells
                } else {
                    cell_of.iter().map(|&cell| cells[cell].clone()).collect()
                };
                (cells, stats)
            })
            .collect();
        let mut reduced = SearchStats::default();
        let mut result = Vec::with_capacity(collected.len());
        for (cells, stats) in collected {
            reduced.absorb(&stats);
            result.push(cells);
        }
        reduced.report("frontier_matrix");
        result
    }

    /// The least-emissions cost matrix over this engine's candidate
    /// set: per origin–destination cell, the cleanest journey (ties
    /// toward the shorter travel time) among the (departure, arrival,
    /// emissions bucket) Pareto candidates of the departure window,
    /// optionally within a travel-time `budget`. The McTBTR
    /// counterpart of the McRAPTOR matrix: one engine build serves
    /// every origin, fanned out with rayon.
    pub fn least_emissions_matrix(
        &self,
        inputs: &CostInputs<'_>,
        requests: &[Request],
        destinations: &[StopIdx],
        window: u32,
        budget: Option<u32>,
        bucket: f64,
    ) -> Vec<Vec<CostRow>> {
        let mut slots = vec![0u32; self.timetable.stop_count() as usize];
        for (index, stop) in destinations.iter().enumerate() {
            slots[stop.0 as usize] = index as u32 + 1;
        }
        requests
            .par_iter()
            .map(|request| {
                let departures = departure_candidates(self.timetable, request, window);
                let mut best: Vec<Option<Winner>> = vec![None; destinations.len()];
                let mut fold = Some(MatrixSink {
                    slots: &slots,
                    budget,
                    best: &mut best,
                });
                let (arena_out, _) = self.passes(
                    request,
                    &departures,
                    bucket,
                    &mut fold,
                    &mut None,
                    &mut SearchStats::default(),
                );
                best.into_iter()
                    .enumerate()
                    .filter_map(|(slot, winner)| {
                        winner.map(|winner| {
                            self.cost_row(inputs, &winner, &arena_out, destinations[slot])
                        })
                    })
                    .collect()
            })
            .collect()
    }

    /// Walks a matrix winner's chain into a cost row, mirroring the
    /// McRAPTOR matrix reconstruction; the sentinel leaf is the
    /// origin's zero-ride floor.
    fn cost_row(
        &self,
        inputs: &CostInputs<'_>,
        winner: &Winner,
        arena: &[Segment],
        to: StopIdx,
    ) -> CostRow {
        let mut rides = 0u32;
        let mut transit_meters = 0.0;
        let mut walk_meters = 0.0;
        let mut legs: Vec<(TripIdx, u16, u16)> = Vec::new();
        let mut fare_legs: Vec<FareLeg> = Vec::new();
        if winner.leaf != ACCESS_LEAF {
            let mut segment = &arena[winner.leaf as usize];
            let mut alight = winner.alight;
            if winner.walked {
                let stops = self
                    .timetable
                    .pattern_stops(self.view.line_pattern(self.view.line_of(segment.trip)));
                walk_meters += self
                    .footpaths
                    .from_stop(stops[alight as usize])
                    .iter()
                    .find(|footpath| footpath.to == to)
                    .map(|footpath| footpath.meters)
                    .unwrap_or(0.0);
            }
            loop {
                let trip = segment.trip;
                let backing = self.view.backing(trip);
                rides += 1;
                transit_meters +=
                    inputs.geometry.leg_distance(backing, segment.board, alight) as f64;
                if inputs.with_geometry {
                    legs.push((backing, segment.board, alight));
                }
                if inputs.fares.is_some() {
                    let pattern = self.timetable.trip_pattern(backing);
                    let stops = self.timetable.pattern_stops(pattern);
                    fare_legs.push(FareLeg {
                        route: self.timetable.pattern_route(pattern),
                        board_stop: stops[segment.board as usize].0,
                        alight_stop: stops[alight as usize].0,
                        board_time: self.timetable.trip_stop_times(backing)[segment.board as usize]
                            .departure
                            .saturating_sub(self.view.day_offset(trip)),
                    });
                }
                let board_stop = self
                    .timetable
                    .pattern_stops(self.view.line_pattern(self.view.line_of(trip)))
                    [segment.board as usize];
                match segment.origin {
                    SegOrigin::Access { .. } => break,
                    SegOrigin::Transfer { parent, alight: at }
                    | SegOrigin::Walked {
                        parent, alight: at, ..
                    } => {
                        let parent_segment = &arena[parent as usize];
                        let parent_stops = self.timetable.pattern_stops(
                            self.view
                                .line_pattern(self.view.line_of(parent_segment.trip)),
                        );
                        let parent_stop = parent_stops[at as usize];
                        if parent_stop != board_stop {
                            walk_meters += self
                                .footpaths
                                .from_stop(parent_stop)
                                .iter()
                                .find(|footpath| footpath.to == board_stop)
                                .map(|footpath| footpath.meters)
                                .unwrap_or(0.0);
                        }
                        alight = at;
                        segment = parent_segment;
                    }
                }
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
                fare_legs.reverse();
                tables.price(&fare_legs)
            }
            None => f64::NAN,
        };
        CostRow {
            to: to.0,
            seconds: winner.seconds,
            rides,
            transit_meters,
            walk_meters,
            emission_grams: winner.grams,
            fare,
            geometry,
        }
    }

    /// Walks a winning segment chain back into the journey contract.
    pub(super) fn assemble(&self, arrived: &Arrived, arena: &[Segment]) -> Journey {
        let mut legs = Vec::new();
        let mut segment = &arena[arrived.leaf as usize];
        let mut alight = arrived.alight;
        let leaf_stops = self
            .timetable
            .pattern_stops(self.view.line_pattern(self.view.line_of(segment.trip)));
        let leaf_times = self.view.stored_times(self.timetable, segment.trip);
        let leaf_offset = self.view.day_offset(segment.trip);
        let alight_arrival = leaf_times[alight as usize].arrival - leaf_offset;
        match arrived.walk {
            Some((stop, duration)) => {
                let reached = alight_arrival.saturating_add(duration);
                legs.push(Leg::Egress {
                    from_stop: stop,
                    departure: reached,
                    arrival: arrived.arrival,
                });
                legs.push(Leg::Transfer {
                    from_stop: leaf_stops[alight as usize],
                    to_stop: stop,
                    departure: alight_arrival,
                    arrival: reached,
                });
            }
            None => {
                legs.push(Leg::Egress {
                    from_stop: leaf_stops[alight as usize],
                    departure: alight_arrival,
                    arrival: arrived.arrival,
                });
            }
        }
        loop {
            let trip = segment.trip;
            let line = self.view.line_of(trip);
            let offset = self.view.line_day_offset(line);
            let stops = self.timetable.pattern_stops(self.view.line_pattern(line));
            let times = self.view.stored_times(self.timetable, trip);
            legs.push(Leg::Transit {
                trip: self.view.backing(trip),
                board_stop: stops[segment.board as usize],
                alight_stop: stops[alight as usize],
                board_position: segment.board,
                alight_position: alight,
                board_time: times[segment.board as usize].departure - offset,
                alight_time: times[alight as usize].arrival - offset,
            });
            let board_stop = stops[segment.board as usize];
            match segment.origin {
                SegOrigin::Access { stop, seconds } => {
                    legs.push(Leg::Access {
                        to_stop: stop,
                        departure: segment.departure,
                        arrival: segment.departure.saturating_add(seconds),
                    });
                    break;
                }
                SegOrigin::Transfer { parent, alight: at }
                | SegOrigin::Walked {
                    parent, alight: at, ..
                } => {
                    let parent_segment = &arena[parent as usize];
                    let parent_line = self.view.line_of(parent_segment.trip);
                    let parent_stops = self
                        .timetable
                        .pattern_stops(self.view.line_pattern(parent_line));
                    let parent_stop = parent_stops[at as usize];
                    if parent_stop != board_stop {
                        let left = self.view.stored_times(self.timetable, parent_segment.trip)
                            [at as usize]
                            .arrival
                            - self.view.line_day_offset(parent_line);
                        let duration = match segment.origin {
                            SegOrigin::Walked { duration, .. } => duration,
                            _ => self
                                .footpaths
                                .from_stop(parent_stop)
                                .iter()
                                .find(|footpath| footpath.to == board_stop)
                                .map(|footpath| footpath.duration)
                                .unwrap_or(0),
                        };
                        legs.push(Leg::Transfer {
                            from_stop: parent_stop,
                            to_stop: board_stop,
                            departure: left,
                            arrival: left.saturating_add(duration),
                        });
                    }
                    alight = at;
                    segment = parent_segment;
                }
            }
        }
        legs.reverse();
        Journey {
            departure: arrived.departure,
            arrival: arrived.arrival,
            legs,
        }
    }
}
