//! The accessibility primitive's Python entries: per-origin
//! decay-weighted opportunity sums over transit (stop destinations)
//! and street (point destinations) costs.

use super::*;

use cafein_core::access::{nearest, opportunity_sums, Decay};
use numpy::{PyArray1, PyReadonlyArray2};
use rayon::prelude::*;

/// The parsed decay for user input: one optional parameter whose
/// meaning depends on the family. Errors name the knob.
pub(super) fn parse_decay(decay: &str, decay_param: Option<f64>) -> PyResult<Decay> {
    let parameter = |name: &str| -> PyResult<f64> {
        let Some(value) = decay_param else {
            return Err(PyValueError::new_err(format!(
                "decay={decay:?} requires decay_param ({name}), a positive finite number"
            )));
        };
        if !value.is_finite() || value <= 0.0 {
            return Err(PyValueError::new_err(format!(
                "decay_param ({name}) must be a positive finite number, not {value:?}"
            )));
        }
        Ok(value)
    };
    match decay {
        "step" => {
            if decay_param.is_some() {
                return Err(PyValueError::new_err("decay='step' takes no decay_param"));
            }
            Ok(Decay::Step)
        }
        "linear" => Ok(Decay::Linear {
            width: parameter("width")?,
        }),
        "exponential" => Ok(Decay::Exponential {
            half_life: parameter("half_life")?,
        }),
        "logistic" => Ok(Decay::Logistic {
            scale: parameter("scale")?,
        }),
        unknown => Err(PyValueError::new_err(format!(
            "unknown decay {unknown:?}: the decay functions are step, linear, \
             exponential, logistic"
        ))),
    }
}

/// Budgets and the opportunity matrix, validated against the
/// destination count. Shared by both network kinds.
pub(super) fn validated_aggregation(
    destinations: usize,
    opportunities: &[f64],
    fields: usize,
    budgets: &[f64],
) -> PyResult<()> {
    if fields == 0 {
        return Err(PyValueError::new_err("fields must be at least 1"));
    }
    let expected = destinations
        .checked_mul(fields)
        .ok_or_else(|| PyValueError::new_err("destinations * fields overflows"))?;
    budgets
        .len()
        .checked_mul(fields)
        .ok_or_else(|| PyValueError::new_err("budgets * fields overflows"))?;
    if opportunities.len() != expected {
        return Err(PyValueError::new_err(format!(
            "opportunities carries {} values, expected destinations * fields = {}",
            opportunities.len(),
            expected
        )));
    }
    if budgets.is_empty() {
        return Err(PyValueError::new_err(
            "budgets must name at least one cutoff",
        ));
    }
    for budget in budgets {
        if !budget.is_finite() || *budget <= 0.0 {
            return Err(PyValueError::new_err(format!(
                "budgets must be positive finite numbers, not {budget:?}"
            )));
        }
    }
    for opportunity in opportunities {
        if !opportunity.is_finite() || *opportunity < 0.0 {
            return Err(PyValueError::new_err(format!(
                "opportunity values must be finite and non-negative, not {opportunity:?}"
            )));
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn aggregated_rows<'py>(
    py: Python<'py>,
    rows: Vec<Vec<Option<f64>>>,
    count: usize,
    destinations: usize,
    opportunities: Vec<f64>,
    fields: usize,
    budgets: Vec<f64>,
    decay: &str,
    decay_param: Option<f64>,
    workers: Option<usize>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let decay = parse_decay(decay, decay_param)?;
    validated_aggregation(destinations, &opportunities, fields, &budgets)?;
    let width = budgets.len() * fields;
    let flat: Vec<f64> = py.allow_threads(|| {
        crate::workers::with_workers("accessibility aggregation", workers, || {
            rows.par_iter()
                .flat_map_iter(|costs| {
                    opportunity_sums(costs, &opportunities, fields, &budgets, &decay)
                })
                .collect()
        })
    });
    flat.into_pyarray(py)
        .reshape([count, width])
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

/// Decay-weighted opportunity sums over an already-computed time
/// matrix (`u32` cost cells, `u32::MAX` = unreached): row-major
/// `[origin][budget * fields]`. Serves every resolution path that
/// produces a cost matrix, so the weight formulas exist once.
#[pyfunction]
#[pyo3(signature = (matrix, opportunities, fields, budgets, decay, decay_param, workers=None))]
#[allow(clippy::too_many_arguments)]
pub(super) fn aggregate_opportunity_sums<'py>(
    py: Python<'py>,
    matrix: PyReadonlyArray2<'py, u32>,
    opportunities: Vec<f64>,
    fields: usize,
    budgets: Vec<f64>,
    decay: &str,
    decay_param: Option<f64>,
    workers: Option<usize>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let matrix = matrix.as_array().to_owned();
    let (count, destinations) = matrix.dim();
    let rows: Vec<Vec<Option<f64>>> = matrix
        .outer_iter()
        .map(|row| {
            row.iter()
                .map(|&cell| (cell != u32::MAX).then_some(f64::from(cell)))
                .collect()
        })
        .collect();
    aggregated_rows(
        py,
        rows,
        count,
        destinations,
        opportunities,
        fields,
        budgets,
        decay,
        decay_param,
        workers,
    )
}

/// The float twin for cost axes whose cells are `f64` (grams CO2e,
/// currency units, metres): a non-finite cell is unreached.
#[pyfunction]
#[pyo3(signature = (matrix, opportunities, fields, budgets, decay, decay_param, workers=None))]
#[allow(clippy::too_many_arguments)]
pub(super) fn aggregate_opportunity_sums_f64<'py>(
    py: Python<'py>,
    matrix: PyReadonlyArray2<'py, f64>,
    opportunities: Vec<f64>,
    fields: usize,
    budgets: Vec<f64>,
    decay: &str,
    decay_param: Option<f64>,
    workers: Option<usize>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let matrix = matrix.as_array().to_owned();
    let (count, destinations) = matrix.dim();
    let rows: Vec<Vec<Option<f64>>> = matrix
        .outer_iter()
        .map(|row| {
            row.iter()
                .map(|&cell| cell.is_finite().then_some(cell))
                .collect()
        })
        .collect();
    aggregated_rows(
        py,
        rows,
        count,
        destinations,
        opportunities,
        fields,
        budgets,
        decay,
        decay_param,
        workers,
    )
}

/// Padded `[origin, rank]` twins: destination column indices (`-1` =
/// absent rank) and costs (`NaN` = absent).
type NearestArrays<'py> = (Bound<'py, PyArray2<i64>>, Bound<'py, PyArray2<f64>>);

fn nearest_rows<'py>(
    py: Python<'py>,
    rows: Vec<Vec<Option<f64>>>,
    count: usize,
    columns: usize,
    k: usize,
    max_cost: Option<f64>,
    workers: Option<usize>,
) -> PyResult<NearestArrays<'py>> {
    if k == 0 {
        return Err(PyValueError::new_err("k must be at least 1"));
    }
    // Ranks past the column count are always absent, so clamping keeps
    // the output complete while bounding the allocation by the matrix
    // the caller already holds — count * k cannot overflow. A
    // zero-column matrix yields correctly shaped empty twins.
    let k = k.min(columns);
    if k == 0 {
        let indices = Vec::<i64>::new()
            .into_pyarray(py)
            .reshape([count, 0])
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        let costs = Vec::<f64>::new()
            .into_pyarray(py)
            .reshape([count, 0])
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        return Ok((indices, costs));
    }
    // None: the engine's natural horizon — every reached destination
    // competes.
    let max_cost = match max_cost {
        None => f64::INFINITY,
        Some(value) if value.is_finite() && value > 0.0 => value,
        Some(value) => {
            return Err(PyValueError::new_err(format!(
                "max_cost must be a positive finite number, not {value:?}"
            )))
        }
    };
    let ranked: Vec<Vec<(usize, f64)>> = py.allow_threads(|| {
        crate::workers::with_workers("nearest aggregation", workers, || {
            rows.par_iter()
                .map(|costs| nearest(costs, k, max_cost))
                .collect()
        })
    });
    // Padded [count, k] twins: index -1 / cost NaN mark an absent rank.
    let mut indices = vec![-1_i64; count * k];
    let mut costs = vec![f64::NAN; count * k];
    for (origin, pairs) in ranked.iter().enumerate() {
        for (rank, (destination, cost)) in pairs.iter().enumerate() {
            indices[origin * k + rank] = *destination as i64;
            costs[origin * k + rank] = *cost;
        }
    }
    let indices = indices
        .into_pyarray(py)
        .reshape([count, k])
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    let costs = costs
        .into_pyarray(py)
        .reshape([count, k])
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok((indices, costs))
}

/// The `k` nearest destinations per origin over an already-computed
/// time matrix (`u32` cost cells, `u32::MAX` = unreached), within the
/// `max_cost` horizon: padded `[origin, rank]` twins of destination
/// column indices (`-1` = absent rank) and costs (`NaN` = absent).
/// Ties break deterministically by (cost, column index).
#[pyfunction]
#[pyo3(signature = (matrix, k, max_cost=None, workers=None))]
pub(super) fn aggregate_nearest<'py>(
    py: Python<'py>,
    matrix: PyReadonlyArray2<'py, u32>,
    k: usize,
    max_cost: Option<f64>,
    workers: Option<usize>,
) -> PyResult<NearestArrays<'py>> {
    let matrix = matrix.as_array().to_owned();
    let (count, columns) = matrix.dim();
    let rows: Vec<Vec<Option<f64>>> = matrix
        .outer_iter()
        .map(|row| {
            row.iter()
                .map(|&cell| (cell != u32::MAX).then_some(f64::from(cell)))
                .collect()
        })
        .collect();
    nearest_rows(py, rows, count, columns, k, max_cost, workers)
}

/// The float twin for cost axes whose cells are `f64` (grams CO2e,
/// currency units, metres): a non-finite cell is unreached.
#[pyfunction]
#[pyo3(signature = (matrix, k, max_cost=None, workers=None))]
pub(super) fn aggregate_nearest_f64<'py>(
    py: Python<'py>,
    matrix: PyReadonlyArray2<'py, f64>,
    k: usize,
    max_cost: Option<f64>,
    workers: Option<usize>,
) -> PyResult<NearestArrays<'py>> {
    let matrix = matrix.as_array().to_owned();
    let (count, columns) = matrix.dim();
    let rows: Vec<Vec<Option<f64>>> = matrix
        .outer_iter()
        .map(|row| {
            row.iter()
                .map(|&cell| cell.is_finite().then_some(cell))
                .collect()
        })
        .collect();
    nearest_rows(py, rows, count, columns, k, max_cost, workers)
}

/// A catchment field as `(latitudes, longitudes, values)` arrays.
type FieldArrays<'py> = (
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
);

#[pymethods]
impl TransportNetwork {
    /// The profile-aware catchment field: the multi-seed directed
    /// spread over the multimodal graph under `mode`'s compiled
    /// profile, seeded at the origin (profile-aware snap) and at each
    /// stop's mode link with its transit arrival cost. The wheelchair
    /// arm of `_catchment_walk_field`. Internal.
    fn _catchment_directed_field<'py>(
        &self,
        py: Python<'py>,
        origin: (f64, f64),
        stop_costs: Vec<(u32, f64)>,
        mode: &str,
        cutoff_seconds: f64,
        max_snap_distance: f64,
    ) -> PyResult<FieldArrays<'py>> {
        let profile = self.multimodal_profile(mode)?;
        let network = self.multimodal.as_ref().ok_or_else(|| {
            PyValueError::new_err(
                "no multimodal street graph is installed; build with street_modes=",
            )
        })?;
        let links = self.mode_link_targets(network, profile.definition.mode.bit());
        for &(stop, _) in &stop_costs {
            if stop as usize >= links.len() {
                return Err(PyValueError::new_err(format!(
                    "stop index {stop} is outside the network's stop range"
                )));
            }
        }
        let field: Vec<(f64, f64, f64)> = py.allow_threads(|| {
            let mut seeds = Vec::with_capacity(stop_costs.len() + 1);
            let (latitude, longitude) = origin;
            if let Some(snap) =
                network.snap_for_profile(latitude, longitude, max_snap_distance, &profile)
            {
                seeds.push((snap, 0.0));
            }
            for &(stop, seconds) in &stop_costs {
                if let Some(snap) = links[stop as usize] {
                    seeds.push((snap, seconds));
                }
            }
            let positions = network.vertex_positions();
            network
                .directed_field(&seeds, &profile, cutoff_seconds)
                .into_iter()
                .filter_map(|(vertex, seconds)| {
                    let (lon, lat) = positions[vertex as usize];
                    (lon.is_finite() && lat.is_finite()).then_some((lat, lon, seconds))
                })
                .collect()
        });
        let lats: Vec<f64> = field.iter().map(|&(lat, _, _)| lat).collect();
        let lons: Vec<f64> = field.iter().map(|&(_, lon, _)| lon).collect();
        let seconds: Vec<f64> = field.iter().map(|&(_, _, s)| s).collect();
        Ok((
            lats.into_pyarray(py),
            lons.into_pyarray(py),
            seconds.into_pyarray(py),
        ))
    }

    /// The door-to-door walking field of a catchment: reached street
    /// vertices as `(latitudes, longitudes, seconds)` arrays, seeded
    /// from the snapped origin at zero seconds and every listed
    /// stop's street links at that stop's arrival seconds, spread to
    /// `cutoff_seconds`. Stops are global indices; an unsnappable
    /// origin still spreads from its reached stops.
    #[pyo3(signature = (origin, stop_costs, walking_speed, cutoff_seconds, max_snap_distance))]
    fn _catchment_walk_field<'py>(
        &self,
        py: Python<'py>,
        origin: (f64, f64),
        stop_costs: Vec<(u32, f64)>,
        walking_speed: f64,
        cutoff_seconds: f64,
        max_snap_distance: f64,
    ) -> PyResult<FieldArrays<'py>> {
        use cafein_core::timetable::StopIdx;

        let streets = self.installed_streets()?;
        let field: Vec<(f64, f64, f64)> = py.allow_threads(|| {
            let mut seeds = Vec::with_capacity(stop_costs.len() + 1);
            let (latitude, longitude) = origin;
            if let Some(snap) = streets.snap(latitude, longitude, max_snap_distance) {
                seeds.push((snap, 0.0));
            }
            for &(stop, seconds) in &stop_costs {
                for snap in streets.stop_snaps(StopIdx(stop)) {
                    seeds.push((snap, seconds));
                }
            }
            let positions = streets.vertex_positions();
            streets
                .walk_field(&seeds, walking_speed, cutoff_seconds)
                .into_iter()
                .filter_map(|(vertex, seconds)| {
                    let (lon, lat) = positions[vertex as usize];
                    (lon.is_finite() && lat.is_finite()).then_some((lat, lon, seconds))
                })
                .collect()
        });
        let lats: Vec<f64> = field.iter().map(|&(lat, _, _)| lat).collect();
        let lons: Vec<f64> = field.iter().map(|&(_, lon, _)| lon).collect();
        let seconds: Vec<f64> = field.iter().map(|&(_, _, s)| s).collect();
        Ok((
            lats.into_pyarray(py),
            lons.into_pyarray(py),
            seconds.into_pyarray(py),
        ))
    }

    /// The arrive-by catchment's stop reaches: one reverse one-to-all
    /// from the egress seeds, each origin stop reduced to the
    /// complete-journey order's winner — `(stop index, latest
    /// departure, rides, achieved arrival)`, seconds past the service
    /// day's start. Internal.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (egress, date, deadline, max_transfers, exclude_routes, exclude_trips, exclude_stops, workers=None))]
    fn _arrive_by_reaches(
        &self,
        py: Python<'_>,
        egress: Vec<(String, u32)>,
        date: &str,
        deadline: &str,
        max_transfers: u8,
        exclude_routes: Vec<String>,
        exclude_trips: Vec<String>,
        exclude_stops: Vec<String>,
        workers: Option<usize>,
    ) -> PyResult<Vec<(u32, u32, u32, u32)>> {
        let deadline = parse_time(deadline)?;
        let exclusions = self.exclusion_masks(&exclude_routes, &exclude_trips, &exclude_stops)?;
        let egress = egress
            .iter()
            .map(|(stop, seconds)| Ok((self.resolve_stop(stop)?, *seconds)))
            .collect::<PyResult<Vec<_>>>()?;
        let request = Request {
            departure: deadline,
            access: Vec::new(),
            egress,
            active_services: self.active_services(date)?,
            active_services_previous: self.active_services_previous(date)?,
            max_transfers,
            exclusions,
        };
        Ok(py.allow_threads(|| {
            crate::workers::with_workers("arrive_by_reaches", workers, || {
                let reversed = ReversedTransfers::build(&self.transfers);
                let states =
                    reverse::reverse_one_to_all(&self.build.timetable, &reversed, &request);
                let mut reaches = Vec::new();
                for (stop, stop_states) in states.iter().enumerate() {
                    let mut best: Option<(u32, u32, u32)> = None;
                    for &(round, departure, achieved) in stop_states {
                        crate::options::arrive_by_winner(
                            &mut best,
                            (departure, round as u32, achieved),
                        );
                    }
                    if let Some((departure, rides, achieved)) = best {
                        reaches.push((stop as u32, departure, rides, achieved));
                    }
                }
                reaches
            })
        }))
    }

    /// The nearest-of-set fast path, stop form: ONE tagged reverse
    /// seeded with every destination stop answers, per origin stop,
    /// which destination its latest-departure journey reaches and at
    /// what duration — `(duration, destination index)`, `None` where
    /// nothing is reachable. Destinations answer themselves at zero.
    /// Behavior-invariant with the per-destination fan-out; ties keep
    /// the incumbent, so attribution is deterministic. Internal.
    #[allow(clippy::type_complexity)]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (from_stops, to_stops, date, deadline, max_transfers, exclude_routes, exclude_trips, exclude_stops, workers=None))]
    fn _arrive_by_nearest(
        &self,
        py: Python<'_>,
        from_stops: Vec<String>,
        to_stops: Vec<String>,
        date: &str,
        deadline: &str,
        max_transfers: u8,
        exclude_routes: Vec<String>,
        exclude_trips: Vec<String>,
        exclude_stops: Vec<String>,
        workers: Option<usize>,
    ) -> PyResult<Vec<Option<(u32, u32)>>> {
        let origins: Vec<StopIdx> = from_stops
            .iter()
            .map(|stop| self.resolve_stop(stop))
            .collect::<PyResult<_>>()?;
        let destinations: Vec<StopIdx> = to_stops
            .iter()
            .map(|stop| self.resolve_stop(stop))
            .collect::<PyResult<_>>()?;
        let deadline = parse_time(deadline)?;
        let exclusions = self.exclusion_masks(&exclude_routes, &exclude_trips, &exclude_stops)?;
        let request = Request {
            departure: deadline,
            access: Vec::new(),
            egress: destinations.iter().map(|&stop| (stop, 0)).collect(),
            active_services: self.active_services(date)?,
            active_services_previous: self.active_services_previous(date)?,
            max_transfers,
            exclusions: exclusions.clone(),
        };
        Ok(py.allow_threads(|| {
            crate::workers::with_workers("arrive_by_nearest", workers, || {
                let reversed = ReversedTransfers::build(&self.transfers);
                let states = reverse::reverse_one_to_all_tagged(
                    &self.build.timetable,
                    &reversed,
                    &request,
                    None,
                );
                origins
                    .iter()
                    .map(|&origin| {
                        // Each destination's own arrive-by winner first —
                        // the fan-out's cell — then the nearest by that
                        // winner's duration, ties to the lowest position.
                        let mut per_seed: Vec<Option<(u32, u32, u32)>> =
                            vec![None; destinations.len()];
                        for &(round, departure, achieved, seed) in &states[origin.0 as usize] {
                            crate::options::arrive_by_winner(
                                &mut per_seed[seed as usize],
                                (departure, round as u32, achieved),
                            );
                        }
                        for (seed, &destination) in destinations.iter().enumerate() {
                            if destination != origin {
                                continue;
                            }
                            if exclusions
                                .as_deref()
                                .is_some_and(|excluded| excluded.excludes_stop(destination))
                            {
                                continue;
                            }
                            crate::options::arrive_by_winner(
                                &mut per_seed[seed],
                                (deadline, 0, deadline),
                            );
                        }
                        let mut nearest: Option<(u32, u32)> = None;
                        for (seed, winner) in per_seed.iter().enumerate() {
                            if let Some((departure, _, achieved)) = winner {
                                let duration = achieved - departure;
                                if nearest.is_none_or(|(held, _)| duration < held) {
                                    nearest = Some((duration, seed as u32));
                                }
                            }
                        }
                        nearest
                    })
                    .collect()
            })
        }))
    }

    /// The nearest-of-set fast path, point form: the union of every
    /// destination's street links seeds one tagged reverse, each link
    /// tagged by its destination; per origin point the winner composes
    /// its access links over the tagged states beside every
    /// destination's direct-walk candidate placed to arrive exactly at
    /// the deadline. Returns per origin `(duration, destination
    /// index)` — `u32::MAX` duration where nothing is reachable — plus
    /// the unsnapped indices. Internal.
    #[pyo3(signature = (origins, destinations, date, deadline, max_transfers = 7, exclude_routes = vec![], exclude_trips = vec![], exclude_stops = vec![], walking_speed_kmph = 3.6, max_walking_time = 7200.0, max_snap_distance = 1600.0, workers=None))]
    #[allow(clippy::too_many_arguments)]
    fn _arrive_by_nearest_points(
        &self,
        py: Python<'_>,
        origins: Vec<(f64, f64)>,
        destinations: Vec<(f64, f64)>,
        date: &str,
        deadline: &str,
        max_transfers: u8,
        exclude_routes: Vec<String>,
        exclude_trips: Vec<String>,
        exclude_stops: Vec<String>,
        walking_speed_kmph: f64,
        max_walking_time: f64,
        max_snap_distance: f64,
        workers: Option<usize>,
    ) -> PyResult<Py<PyDict>> {
        let exclusions = self.exclusion_masks(&exclude_routes, &exclude_trips, &exclude_stops)?;
        let streets = self.installed_streets()?;
        let speed =
            validated_walking_speed(walking_speed_kmph, max_walking_time, max_snap_distance)?;
        validate_points(&origins)?;
        validate_points(&destinations)?;
        let deadline = parse_time(deadline)?;
        let active_services = self.active_services(date)?;
        let active_services_previous = self.active_services_previous(date)?;
        let (cells, unsnapped_from, unsnapped_to) = py.allow_threads(|| {
            crate::workers::with_workers("arrive_by_nearest_points", workers, || {
                let mut linked = streets.link_pointsets(
                    &[&origins[..], &destinations[..]],
                    speed,
                    max_walking_time,
                    max_snap_distance,
                );
                let destination_links = linked.pop().unwrap();
                let origin_links = linked.pop().unwrap();
                let unsnapped_from = unsnapped(&origin_links);
                let unsnapped_to = unsnapped(&destination_links);
                // The union egress: every destination's links, tagged by
                // the destination through the egress index mapping.
                let mut egress: Vec<(StopIdx, u32)> = Vec::new();
                let mut egress_owner: Vec<u32> = Vec::new();
                for (destination, links) in destination_links.iter().enumerate() {
                    for &(stop, seconds) in
                        crate::request_offsets(links.as_deref().unwrap_or(&[])).iter()
                    {
                        egress.push((stop, seconds));
                        egress_owner.push(destination as u32);
                    }
                }
                let request = Request {
                    departure: deadline,
                    access: Vec::new(),
                    egress,
                    active_services: active_services.clone(),
                    active_services_previous: active_services_previous.clone(),
                    max_transfers,
                    exclusions: exclusions.clone(),
                };
                let reversed = ReversedTransfers::build(&self.transfers);
                // The owner map groups isolation by destination: every
                // link of a destination shares one seed.
                let states = reverse::reverse_one_to_all_tagged(
                    &self.build.timetable,
                    &reversed,
                    &request,
                    Some(&egress_owner),
                );
                let access = egress_tables(&origin_links);
                let walk = streets.walk_matrix(
                    &origins,
                    &destinations,
                    speed,
                    max_walking_time,
                    max_snap_distance,
                );
                let cells: Vec<(u32, u32)> = access
                    .iter()
                    .enumerate()
                    .map(|(row, links)| {
                        // Each destination's own arrive-by winner first —
                        // its links' composed states beside its direct
                        // walk — then the nearest by winner duration,
                        // ties to the lowest position.
                        let mut per_destination: Vec<Option<(u32, u32, u32)>> =
                            vec![None; destinations.len()];
                        for &(stop, seconds, _) in links {
                            for &(round, departure, achieved, seed) in &states[stop.0 as usize] {
                                let Some(composed) = departure.checked_sub(seconds) else {
                                    continue;
                                };
                                crate::options::arrive_by_winner(
                                    &mut per_destination[seed as usize],
                                    (composed, round as u32, achieved),
                                );
                            }
                        }
                        for (destination, cell) in walk[row].iter().enumerate() {
                            if let Some((walk_seconds, _)) = cell {
                                if let Some(placed) = deadline.checked_sub(*walk_seconds) {
                                    crate::options::arrive_by_winner(
                                        &mut per_destination[destination],
                                        (placed, 0, deadline),
                                    );
                                }
                            }
                        }
                        let mut nearest: Option<(u32, u32)> = None;
                        for (destination, winner) in per_destination.iter().enumerate() {
                            if let Some((departure, _, achieved)) = winner {
                                let duration = achieved - departure;
                                if nearest.is_none_or(|(held, _)| duration < held) {
                                    nearest = Some((duration, destination as u32));
                                }
                            }
                        }
                        nearest.unwrap_or((u32::MAX, 0))
                    })
                    .collect();
                (cells, unsnapped_from, unsnapped_to)
            })
        });
        let result = PyDict::new(py);
        result.set_item(
            "costs",
            cells.iter().map(|&(cost, _)| cost).collect::<Vec<u32>>(),
        )?;
        result.set_item(
            "seeds",
            cells.iter().map(|&(_, seed)| seed).collect::<Vec<u32>>(),
        )?;
        result.set_item("unsnapped_from", unsnapped_from.into_pyarray(py))?;
        result.set_item("unsnapped_to", unsnapped_to.into_pyarray(py))?;
        Ok(result.unbind())
    }

    /// The windowed arrive-by stop reaches: per origin stop, the
    /// winner whose duration is the nearest-rank `percentile` of that
    /// stop's per-mark duration distribution — deadlines profiling at
    /// every minute mark of `[arrival, arrival + window)` from one
    /// final-mark reverse run. Stops unreachable at the percentile
    /// are absent. Internal.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (egress, date, arrival, window, percentile, max_transfers, exclude_routes, exclude_trips, exclude_stops, workers=None))]
    fn _arrive_by_percentile_reaches(
        &self,
        py: Python<'_>,
        egress: Vec<(String, u32)>,
        date: &str,
        arrival: &str,
        window: u32,
        percentile: f64,
        max_transfers: u8,
        exclude_routes: Vec<String>,
        exclude_trips: Vec<String>,
        exclude_stops: Vec<String>,
        workers: Option<usize>,
    ) -> PyResult<Vec<(u32, u32, u32, u32)>> {
        if window == 0 {
            return Err(PyValueError::new_err("window must be at least 1 second"));
        }
        let start = parse_time(arrival)?;
        let marks = crate::options::arrival_marks(start, window)?;
        let deadline = *marks.last().expect("a positive window holds a mark");
        let exclusions = self.exclusion_masks(&exclude_routes, &exclude_trips, &exclude_stops)?;
        let egress = egress
            .iter()
            .map(|(stop, seconds)| Ok((self.resolve_stop(stop)?, *seconds)))
            .collect::<PyResult<Vec<_>>>()?;
        let request = Request {
            departure: deadline,
            access: Vec::new(),
            egress,
            active_services: self.active_services(date)?,
            active_services_previous: self.active_services_previous(date)?,
            max_transfers,
            exclusions,
        };
        Ok(py.allow_threads(|| {
            crate::workers::with_workers("arrive_by_percentile_reaches", workers, || {
                let reversed = ReversedTransfers::build(&self.transfers);
                let states =
                    reverse::reverse_one_to_all(&self.build.timetable, &reversed, &request);
                let mut reaches = Vec::new();
                for (stop, stop_states) in states.iter().enumerate() {
                    if stop_states.is_empty() {
                        continue;
                    }
                    let profile = reverse::reverse_profile_states(stop_states, &marks);
                    // Per mark the cross-round winner; the percentile picks
                    // among the marks by duration, its winner seeding the
                    // field with that mark's (departure, rides, achieved).
                    let mut per_mark: Vec<(u32, (u32, u32, u32))> = profile
                        .iter()
                        .filter_map(|winners| {
                            let mut best: Option<(u32, u32, u32)> = None;
                            for &(round, departure, achieved) in winners {
                                crate::options::arrive_by_winner(
                                    &mut best,
                                    (departure, round as u32, achieved),
                                );
                            }
                            best.map(|winner| (winner.2 - winner.0, winner))
                        })
                        .collect();
                    if per_mark.is_empty() {
                        continue;
                    }
                    // Unreachable marks rank as unreachable: pad to the full
                    // mark count so the percentile sees the true distribution.
                    let mut durations: Vec<u32> = per_mark.iter().map(|&(d, _)| d).collect();
                    durations.resize(marks.len(), u32::MAX);
                    durations.sort_unstable();
                    let rank = crate::cafein_core_nearest_rank(&durations, percentile);
                    if rank == u32::MAX {
                        continue;
                    }
                    per_mark.sort_by_key(|&(duration, _)| duration);
                    let &(_, (departure, rides, achieved)) = per_mark
                        .iter()
                        .find(|&&(duration, _)| duration == rank)
                        .expect("the rank comes from the same distribution");
                    reaches.push((stop as u32, departure, rides, achieved));
                }
                reaches
            })
        }))
    }

    /// The arrive-by walking field of a catchment: reached street
    /// vertices as `(latitudes, longitudes, durations)` arrays. Seeds
    /// carry the before-deadline key (`deadline − departure`), rides,
    /// and slack (`deadline − achieved`); the destination point seeds
    /// at zero on all three (walking straight there arrives exactly at
    /// the deadline). `cutoff_seconds` is the extended bound — budget
    /// plus the maximum retained seed slack — and the returned value
    /// per vertex is its winning label's own duration (key − slack),
    /// which the caller judges against the budgets. Internal.
    #[pyo3(signature = (origin, stop_seeds, walking_speed, cutoff_seconds, max_snap_distance, workers=None))]
    #[allow(clippy::too_many_arguments)]
    fn _arrive_by_catchment_walk_field<'py>(
        &self,
        py: Python<'py>,
        origin: (f64, f64),
        stop_seeds: Vec<(u32, f64, u32, f64)>,
        walking_speed: f64,
        cutoff_seconds: f64,
        max_snap_distance: f64,
        workers: Option<usize>,
    ) -> PyResult<FieldArrays<'py>> {
        use cafein_core::timetable::StopIdx;

        let streets = self.installed_streets()?;
        let field: Vec<(f64, f64, f64)> = py.allow_threads(|| {
            crate::workers::with_workers("arrive_by_catchment_walk_field", workers, || {
                let mut seeds = Vec::with_capacity(stop_seeds.len() + 1);
                let (latitude, longitude) = origin;
                if let Some(snap) = streets.snap(latitude, longitude, max_snap_distance) {
                    seeds.push((snap, 0.0, 0, 0.0));
                }
                for &(stop, key, rides, slack) in &stop_seeds {
                    for snap in streets.stop_snaps(StopIdx(stop)) {
                        seeds.push((snap, key, rides, slack));
                    }
                }
                let positions = streets.vertex_positions();
                streets
                    .arrive_by_walk_field(&seeds, walking_speed, cutoff_seconds)
                    .into_iter()
                    .filter_map(|(vertex, key, _, slack)| {
                        let (lon, lat) = positions[vertex as usize];
                        (lon.is_finite() && lat.is_finite()).then_some((
                            lat,
                            lon,
                            (key - slack).max(0.0),
                        ))
                    })
                    .collect()
            })
        });
        let lats: Vec<f64> = field.iter().map(|&(lat, _, _)| lat).collect();
        let lons: Vec<f64> = field.iter().map(|&(_, lon, _)| lon).collect();
        let seconds: Vec<f64> = field.iter().map(|&(_, _, s)| s).collect();
        Ok((
            lats.into_pyarray(py),
            lons.into_pyarray(py),
            seconds.into_pyarray(py),
        ))
    }

    /// The canonical global stop index per id, through the same
    /// resolver every routing entry uses (qualified ids, ambiguity
    /// errors, unknown-id KeyErrors included).
    fn _stop_indices(&self, stops: Vec<String>) -> PyResult<Vec<u32>> {
        stops
            .iter()
            .map(|stop| self.resolve_stop(stop).map(|index| index.0))
            .collect()
    }

    /// Decay-weighted opportunity sums from stops, on the time axis.
    ///
    /// Costs are exactly `travel_time_matrix`'s (one shared engine
    /// dispatch), read at the destination stops; the result is
    /// row-major `[origin][budget * fields]` seconds-axis sums.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (from_stops, destination_stops, opportunities, fields, budgets,
                        decay, decay_param, date, departure, max_transfers, router,
                        exclude_routes, exclude_trips, exclude_stops,
                        walking_speed_kmph, max_walking_time, max_snap_distance, workers=None))]
    fn _accessibility_from_stops<'py>(
        &self,
        py: Python<'py>,
        from_stops: Vec<String>,
        destination_stops: Vec<String>,
        opportunities: Vec<f64>,
        fields: usize,
        budgets: Vec<f64>,
        decay: &str,
        decay_param: Option<f64>,
        date: &str,
        departure: &str,
        max_transfers: u8,
        router: &str,
        exclude_routes: Vec<String>,
        exclude_trips: Vec<String>,
        exclude_stops: Vec<String>,
        walking_speed_kmph: f64,
        max_walking_time: f64,
        max_snap_distance: f64,
        workers: Option<usize>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let decay = parse_decay(decay, decay_param)?;
        validated_aggregation(destination_stops.len(), &opportunities, fields, &budgets)?;
        let targets: Vec<StopIdx> = destination_stops
            .iter()
            .map(|stop| self.resolve_stop(stop))
            .collect::<PyResult<_>>()?;
        let (rows, departure) = self.resolved_time_rows(
            py,
            &from_stops,
            date,
            departure,
            max_transfers,
            router,
            &exclude_routes,
            &exclude_trips,
            &exclude_stops,
            walking_speed_kmph,
            max_walking_time,
            max_snap_distance,
            workers,
        )?;
        let count = rows.len();
        let width = budgets.len() * fields;
        let flat: Vec<f64> = py.allow_threads(|| {
            crate::workers::with_workers("accessibility", workers, || {
                rows.par_iter()
                    .flat_map_iter(|row| {
                        let costs: Vec<Option<f64>> = targets
                            .iter()
                            .map(|&target| {
                                row[target.0 as usize].map(|arrival| (arrival - departure) as f64)
                            })
                            .collect();
                        opportunity_sums(&costs, &opportunities, fields, &budgets, &decay)
                    })
                    .collect()
            })
        });
        flat.into_pyarray(py)
            .reshape([count, width])
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }
}
