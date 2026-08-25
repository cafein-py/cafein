//! The carriage query entry points (stage 17b): possession-state
//! searches over the two planes, seeds pre-reduced by Python exactly
//! as the street-policy paths pre-reduce theirs.

use super::*;
use cafein_core::routers::carriage::{
    search, CarriageInputs, CarriageLeg, CarriageRequest, CARRYING, FREE,
};

impl TransportNetwork {
    /// The carriage inputs a query borrows: the Carrying plane's set
    /// (the carriage set under a matching ``transfers=`` binding, else
    /// the walking closure), its ride flags, the per-trip carrying
    /// mask under the policy's unknown rule, and the park mask.
    fn carriage_query<'a>(
        &'a self,
        transfer_mode: Option<&(String, f64)>,
        unknown_rule: &str,
        park_stops: Option<&[String]>,
        empty_flags: &'a mut Vec<bool>,
    ) -> PyResult<(CarriageInputs<'a>, Vec<bool>, Vec<bool>)> {
        let (carriage_set, ride_edge): (&Transfers, &[bool]) = match transfer_mode {
            Some((mode, budget)) => match &self.carriage_transfers {
                Some(held) if held.mode == *mode && held.budget == *budget => {
                    (&held.set, &held.ride_edge)
                }
                Some(held) => {
                    return Err(PyValueError::new_err(format!(
                        "the computed carriage set is bound to ('{}', {} s), not \
                         ('{mode}', {budget} s); recompute with \
                         compute_carriage_transfers",
                        held.mode, held.budget
                    )))
                }
                None => {
                    return Err(PyValueError::new_err(
                        "street_policy transfers= with a carried vehicle needs \
                         the carriage set; compute it with \
                         compute_carriage_transfers(mode, budget)",
                    ))
                }
            },
            None => {
                empty_flags.resize(self.transfers.edge_count(), false);
                (&self.transfers, empty_flags.as_slice())
            }
        };
        let allow_unknown = match unknown_rule {
            "forbid" => false,
            "allow" => true,
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown_bike_trips must be 'forbid' or 'allow', not {other:?}"
                )))
            }
        };
        let carrying_mask: Vec<bool> = (0..self.build.timetable.trip_count())
            .map(|trip| {
                let source = self.build.timetable.trip_source(TripIdx(trip));
                match self.feed.trips[source as usize].bikes_allowed {
                    Some(allowed) => allowed,
                    None => allow_unknown,
                }
            })
            .collect();
        let stop_count = self.build.timetable.stop_count() as usize;
        let park_mask = match park_stops {
            None => vec![true; stop_count],
            Some(stops) => {
                let mut mask = vec![false; stop_count];
                for stop in stops {
                    mask[self.resolve_stop(stop)?.0 as usize] = true;
                }
                mask
            }
        };
        Ok((
            CarriageInputs {
                timetable: &self.build.timetable,
                walking: &self.transfers,
                carriage: carriage_set,
                ride_edge,
                // Filled by the caller: masks move in after the borrow
                // dance below.
                carrying_mask: &[],
                park_mask: &[],
                active_services: &[],
                active_services_previous: &[],
            },
            carrying_mask,
            park_mask,
        ))
    }
}

#[pymethods]
impl TransportNetwork {
    /// Earliest arrival at every reachable stop under a carriage
    /// policy: the two-plane possession-state search, the reported
    /// time the cross-plane minimum. Seeds arrive pre-reduced per
    /// plane (Carrying from the policy reduction, Free from the
    /// walking-only reduction — carriage is optional). Internal.
    #[pyo3(signature = (carrying_access, free_access, date, departure, max_transfers,
                        unknown_rule, park_stops = None, transfer_mode = None))]
    #[allow(clippy::too_many_arguments)]
    fn _carriage_travel_times(
        &self,
        py: Python<'_>,
        carrying_access: Vec<(String, u32, bool)>,
        free_access: Vec<(String, u32)>,
        date: &str,
        departure: &str,
        max_transfers: u8,
        unknown_rule: &str,
        park_stops: Option<Vec<String>>,
        transfer_mode: Option<(String, f64)>,
    ) -> PyResult<Py<PyDict>> {
        let departure = parse_time(departure)?;
        let resolve = |offsets: &[(String, u32)]| {
            offsets
                .iter()
                .map(|(stop, seconds)| Ok((self.resolve_stop(stop)?, *seconds)))
                .collect::<PyResult<Vec<_>>>()
        };
        let resolve_carrying = |offsets: &[(String, u32, bool)]| {
            offsets
                .iter()
                .map(|(stop, seconds, walked)| Ok((self.resolve_stop(stop)?, *seconds, *walked)))
                .collect::<PyResult<Vec<_>>>()
        };
        let request = CarriageRequest {
            departure,
            carrying_access: resolve_carrying(&carrying_access)?,
            free_access: resolve(&free_access)?,
            max_transfers,
        };
        let active_services = self.active_services(date)?;
        let active_services_previous = self.active_services_previous(date)?;
        let mut empty_flags = Vec::new();
        let (mut inputs, carrying_mask, park_mask) = self.carriage_query(
            transfer_mode.as_ref(),
            unknown_rule,
            park_stops.as_deref(),
            &mut empty_flags,
        )?;
        inputs.carrying_mask = &carrying_mask;
        inputs.park_mask = &park_mask;
        inputs.active_services = &active_services;
        inputs.active_services_previous = &active_services_previous;
        let result = py.allow_threads(|| search(&inputs, &request));
        let carrying = result.arrivals(CARRYING);
        let free = result.arrivals(FREE);
        let dict = PyDict::new(py);
        for stop in 0..self.build.timetable.stop_count() as usize {
            let best = match (carrying[stop], free[stop]) {
                (Some(a), Some(b)) => Some(a.min(b)),
                (a, b) => a.or(b),
            };
            if let Some(arrival) = best {
                dict.set_item(
                    self.public_stop_id(StopIdx(stop as u32)),
                    arrival - departure,
                )?;
            }
        }
        Ok(dict.unbind())
    }

    /// The carriage journeys on the ``(arrival, rides)`` frontier: per
    /// round the planes fold their egress offsets (Carrying may cycle
    /// out, Free walks), a round joins the frontier when its cross-
    /// plane best strictly beats every earlier round, and each kept
    /// chain reconstructs into ``route_between_stops``-shaped legs
    /// plus the carriage extras (``bike_aboard`` transit flags,
    /// ``park`` events, ``ride`` transfers in the carried mode).
    /// Internal.
    #[pyo3(signature = (carrying_access, free_access, carrying_egress, free_egress,
                        date, departure, max_transfers, unknown_rule, geometries,
                        park_stops = None, transfer_mode = None))]
    #[allow(clippy::too_many_arguments)]
    fn _carriage_route(
        &self,
        py: Python<'_>,
        carrying_access: Vec<(String, u32, bool)>,
        free_access: Vec<(String, u32)>,
        carrying_egress: Vec<(String, u32)>,
        free_egress: Vec<(String, u32)>,
        date: &str,
        departure: &str,
        max_transfers: u8,
        unknown_rule: &str,
        geometries: bool,
        park_stops: Option<Vec<String>>,
        transfer_mode: Option<(String, f64)>,
    ) -> PyResult<Py<PyList>> {
        let departure = parse_time(departure)?;
        let resolve = |offsets: &[(String, u32)]| {
            offsets
                .iter()
                .map(|(stop, seconds)| Ok((self.resolve_stop(stop)?, *seconds)))
                .collect::<PyResult<Vec<_>>>()
        };
        let resolve_carrying = |offsets: &[(String, u32, bool)]| {
            offsets
                .iter()
                .map(|(stop, seconds, walked)| Ok((self.resolve_stop(stop)?, *seconds, *walked)))
                .collect::<PyResult<Vec<_>>>()
        };
        let request = CarriageRequest {
            departure,
            carrying_access: resolve_carrying(&carrying_access)?,
            free_access: resolve(&free_access)?,
            max_transfers,
        };
        let egress = [resolve(&carrying_egress)?, resolve(&free_egress)?];
        let active_services = self.active_services(date)?;
        let active_services_previous = self.active_services_previous(date)?;
        let mut empty_flags = Vec::new();
        let (mut inputs, carrying_mask, park_mask) = self.carriage_query(
            transfer_mode.as_ref(),
            unknown_rule,
            park_stops.as_deref(),
            &mut empty_flags,
        )?;
        inputs.carrying_mask = &carrying_mask;
        inputs.park_mask = &park_mask;
        inputs.active_services = &active_services;
        inputs.active_services_previous = &active_services_previous;
        let result = py.allow_threads(|| search(&inputs, &request));
        // The (arrival, rides) frontier by round: a round is kept when
        // its cross-plane egress best strictly beats every earlier
        // round's, and reconstructs at the round its label was set.
        let mut frontier: Vec<(u32, usize, usize, StopIdx, u32)> = Vec::new();
        let mut previous = u32::MAX;
        for round in 0..=result.rounds() {
            let mut candidate: Option<(u32, usize, StopIdx, u32)> = None;
            for plane in [CARRYING, FREE] {
                for &(stop, seconds) in &egress[plane] {
                    let Some(at_stop) = result.prefix_arrival(plane, round, stop) else {
                        continue;
                    };
                    let Some(arrival) = at_stop.checked_add(seconds).filter(|&at| at != u32::MAX)
                    else {
                        continue;
                    };
                    if candidate.is_none_or(|(current, ..)| arrival < current) {
                        candidate = Some((arrival, plane, stop, seconds));
                    }
                }
            }
            let Some((arrival, plane, stop, seconds)) = candidate else {
                continue;
            };
            if arrival >= previous {
                continue;
            }
            previous = arrival;
            let achieved = result
                .achieving_round(plane, round, stop)
                .expect("a prefix arrival always has an achieving round");
            frontier.push((arrival, plane, achieved, stop, seconds));
        }
        let journeys = PyList::empty(py);
        for (arrival, plane, round, stop, egress_seconds) in frontier {
            let legs = result.reconstruct(&self.build.timetable, plane, round, stop);
            let out = PyList::empty(py);
            let mut rides = 0u32;
            for leg in &legs {
                let item = PyDict::new(py);
                match *leg {
                    CarriageLeg::Access {
                        plane: leg_plane,
                        to_stop,
                        arrival,
                    } => {
                        item.set_item("type", "access")?;
                        item.set_item("carrying", leg_plane == CARRYING)?;
                        item.set_item("to_stop", self.public_stop_id(to_stop))?;
                        item.set_item("departure_s", departure)?;
                        item.set_item("arrival_s", arrival)?;
                    }
                    CarriageLeg::Park { stop, at } => {
                        item.set_item("type", "park")?;
                        item.set_item("stop", self.public_stop_id(stop))?;
                        item.set_item("departure_s", at)?;
                        item.set_item("arrival_s", at)?;
                    }
                    CarriageLeg::Transit {
                        trip,
                        board_stop,
                        alight_stop,
                        board_position,
                        alight_position,
                        board_time,
                        alight_time,
                        bike_aboard,
                    } => {
                        rides += 1;
                        let source_trip =
                            &self.feed.trips[self.build.timetable.trip_source(trip) as usize];
                        item.set_item("type", "transit")?;
                        item.set_item(
                            "trip_id",
                            self.public_id(source_trip.feed, &source_trip.id),
                        )?;
                        let route = &self.feed.routes[source_trip.route as usize];
                        item.set_item("route_id", self.public_id(route.feed, &route.id))?;
                        item.set_item("route_short_name", route.short_name.as_deref())?;
                        item.set_item("board_stop", self.public_stop_id(board_stop))?;
                        item.set_item("alight_stop", self.public_stop_id(alight_stop))?;
                        item.set_item("departure_s", board_time)?;
                        item.set_item("arrival_s", alight_time)?;
                        item.set_item("bike_aboard", bike_aboard)?;
                        match &self.geometry {
                            Some(geometry) => {
                                item.set_item(
                                    "distance_m",
                                    geometry.leg_distance(trip, board_position, alight_position)
                                        as f64,
                                )?;
                                item.set_item(
                                    "distance_provenance",
                                    provenance_name(geometry.provenance(trip)),
                                )?;
                            }
                            None => {
                                item.set_item("distance_m", py.None())?;
                                item.set_item("distance_provenance", py.None())?;
                            }
                        }
                        let geometry =
                            self.leg_geometry
                                .as_ref()
                                .filter(|_| geometries)
                                .map(|geometry| {
                                    wkb_line_string(
                                        py,
                                        &geometry.leg_coordinates(
                                            trip,
                                            board_position,
                                            alight_position,
                                        ),
                                    )
                                });
                        item.set_item("geometry", geometry)?;
                    }
                    CarriageLeg::Transfer {
                        from_stop,
                        to_stop,
                        departure,
                        arrival,
                        ride,
                    } => {
                        item.set_item("type", "transfer")?;
                        item.set_item("ride", ride)?;
                        item.set_item("from_stop", self.public_stop_id(from_stop))?;
                        item.set_item("to_stop", self.public_stop_id(to_stop))?;
                        item.set_item("departure_s", departure)?;
                        item.set_item("arrival_s", arrival)?;
                    }
                }
                out.append(item)?;
            }
            let egress_leg = PyDict::new(py);
            egress_leg.set_item("type", "egress")?;
            egress_leg.set_item("carrying", plane == CARRYING)?;
            egress_leg.set_item("from_stop", self.public_stop_id(stop))?;
            egress_leg.set_item("departure_s", arrival - egress_seconds)?;
            egress_leg.set_item("arrival_s", arrival)?;
            out.append(egress_leg)?;
            let journey = PyDict::new(py);
            journey.set_item("departure_s", departure)?;
            journey.set_item("arrival_s", arrival)?;
            journey.set_item("rides", rides)?;
            journey.set_item("legs", out)?;
            journeys.append(journey)?;
        }
        Ok(journeys.unbind())
    }
}

#[pymethods]
impl TransportNetwork {
    /// The carriage time matrix core: per-origin pre-reduced plane
    /// seeds and per-destination pre-reduced plane egress offsets in,
    /// the flat origins-major seconds out (``None`` = unreachable).
    /// One possession-state search per origin through the rayon
    /// fan-out, then the per-plane egress fold per destination — a
    /// carried egress folds from the Carrying plane only (a parked
    /// chain lives in Free), a walking egress from Free. Internal.
    #[pyo3(signature = (carrying_rows, free_rows, carrying_egress_rows, free_egress_rows,
                        date, departure, max_transfers, unknown_rule,
                        park_stops = None, transfer_mode = None))]
    #[allow(clippy::too_many_arguments)]
    fn _carriage_time_matrix(
        &self,
        py: Python<'_>,
        carrying_rows: Vec<Vec<(String, u32, bool)>>,
        free_rows: Vec<Vec<(String, u32)>>,
        carrying_egress_rows: Vec<Vec<(String, u32)>>,
        free_egress_rows: Vec<Vec<(String, u32)>>,
        date: &str,
        departure: &str,
        max_transfers: u8,
        unknown_rule: &str,
        park_stops: Option<Vec<String>>,
        transfer_mode: Option<(String, f64)>,
    ) -> PyResult<Vec<Vec<Option<u32>>>> {
        if carrying_rows.len() != free_rows.len()
            || carrying_egress_rows.len() != free_egress_rows.len()
        {
            return Err(PyValueError::new_err(
                "the per-plane row lists must pair up per point",
            ));
        }
        let departure = parse_time(departure)?;
        let resolve = |offsets: &[(String, u32)]| {
            offsets
                .iter()
                .map(|(stop, seconds)| Ok((self.resolve_stop(stop)?, *seconds)))
                .collect::<PyResult<Vec<_>>>()
        };
        let resolve_carrying = |offsets: &[(String, u32, bool)]| {
            offsets
                .iter()
                .map(|(stop, seconds, walked)| Ok((self.resolve_stop(stop)?, *seconds, *walked)))
                .collect::<PyResult<Vec<_>>>()
        };
        let mut requests = Vec::with_capacity(carrying_rows.len());
        for (carrying, free) in carrying_rows.iter().zip(&free_rows) {
            requests.push(CarriageRequest {
                departure,
                carrying_access: resolve_carrying(carrying)?,
                free_access: resolve(free)?,
                max_transfers,
            });
        }
        let mut egress: Vec<[Vec<(StopIdx, u32)>; 2]> =
            Vec::with_capacity(carrying_egress_rows.len());
        for (carrying, free) in carrying_egress_rows.iter().zip(&free_egress_rows) {
            egress.push([resolve(carrying)?, resolve(free)?]);
        }
        let active_services = self.active_services(date)?;
        let active_services_previous = self.active_services_previous(date)?;
        let mut empty_flags = Vec::new();
        let (mut inputs, carrying_mask, park_mask) = self.carriage_query(
            transfer_mode.as_ref(),
            unknown_rule,
            park_stops.as_deref(),
            &mut empty_flags,
        )?;
        inputs.carrying_mask = &carrying_mask;
        inputs.park_mask = &park_mask;
        inputs.active_services = &active_services;
        inputs.active_services_previous = &active_services_previous;
        let matrix = py.allow_threads(|| {
            use rayon::prelude::*;

            let ticker = crate::logging::ProgressTicker::new("carriage matrix", requests.len());
            requests
                .par_iter()
                .map(|request| {
                    let result = search(&inputs, request);
                    let arrivals = [result.arrivals(CARRYING), result.arrivals(FREE)];
                    egress
                        .iter()
                        .map(|links| {
                            let mut best = u32::MAX;
                            for plane in [CARRYING, FREE] {
                                for &(stop, seconds) in &links[plane] {
                                    let Some(at_stop) = arrivals[plane][stop.0 as usize] else {
                                        continue;
                                    };
                                    let Some(arrival) =
                                        at_stop.checked_add(seconds).filter(|&at| at != u32::MAX)
                                    else {
                                        continue;
                                    };
                                    best = best.min(arrival);
                                }
                            }
                            (best != u32::MAX).then(|| best - departure)
                        })
                        .collect()
                })
                .map(|row| {
                    ticker.tick();
                    row
                })
                .collect()
        });
        Ok(matrix)
    }
}
