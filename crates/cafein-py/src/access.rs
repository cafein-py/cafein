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
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let decay = parse_decay(decay, decay_param)?;
    validated_aggregation(destinations, &opportunities, fields, &budgets)?;
    let width = budgets.len() * fields;
    let flat: Vec<f64> = py.allow_threads(|| {
        rows.par_iter()
            .flat_map_iter(|costs| {
                opportunity_sums(costs, &opportunities, fields, &budgets, &decay)
            })
            .collect()
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
#[pyo3(signature = (matrix, opportunities, fields, budgets, decay, decay_param))]
pub(super) fn aggregate_opportunity_sums<'py>(
    py: Python<'py>,
    matrix: PyReadonlyArray2<'py, u32>,
    opportunities: Vec<f64>,
    fields: usize,
    budgets: Vec<f64>,
    decay: &str,
    decay_param: Option<f64>,
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
    )
}

/// The float twin for cost axes whose cells are `f64` (grams CO2e,
/// currency units, metres): a non-finite cell is unreached.
#[pyfunction]
#[pyo3(signature = (matrix, opportunities, fields, budgets, decay, decay_param))]
pub(super) fn aggregate_opportunity_sums_f64<'py>(
    py: Python<'py>,
    matrix: PyReadonlyArray2<'py, f64>,
    opportunities: Vec<f64>,
    fields: usize,
    budgets: Vec<f64>,
    decay: &str,
    decay_param: Option<f64>,
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
        rows.par_iter()
            .map(|costs| nearest(costs, k, max_cost))
            .collect()
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
#[pyo3(signature = (matrix, k, max_cost=None))]
pub(super) fn aggregate_nearest<'py>(
    py: Python<'py>,
    matrix: PyReadonlyArray2<'py, u32>,
    k: usize,
    max_cost: Option<f64>,
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
    nearest_rows(py, rows, count, columns, k, max_cost)
}

/// The float twin for cost axes whose cells are `f64` (grams CO2e,
/// currency units, metres): a non-finite cell is unreached.
#[pyfunction]
#[pyo3(signature = (matrix, k, max_cost=None))]
pub(super) fn aggregate_nearest_f64<'py>(
    py: Python<'py>,
    matrix: PyReadonlyArray2<'py, f64>,
    k: usize,
    max_cost: Option<f64>,
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
    nearest_rows(py, rows, count, columns, k, max_cost)
}

/// A catchment field as `(latitudes, longitudes, values)` arrays.
type FieldArrays<'py> = (
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
);

#[pymethods]
impl TransportNetwork {
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
                        walking_speed_kmph, max_walking_time, max_snap_distance))]
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
        )?;
        let count = rows.len();
        let width = budgets.len() * fields;
        let flat: Vec<f64> = py.allow_threads(|| {
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
        });
        flat.into_pyarray(py)
            .reshape([count, width])
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }
}
