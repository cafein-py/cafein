//! The accessibility primitive's Python entries: per-origin
//! decay-weighted opportunity sums over transit (stop destinations)
//! and street (point destinations) costs.

use super::*;

use cafein_core::access::{opportunity_sums, Decay};
use numpy::PyReadonlyArray2;
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
    let decay = parse_decay(decay, decay_param)?;
    let matrix = matrix.as_array().to_owned();
    let (count, destinations) = matrix.dim();
    validated_aggregation(destinations, &opportunities, fields, &budgets)?;
    let width = budgets.len() * fields;
    let flat: Vec<f64> = py.allow_threads(|| {
        let rows: Vec<Vec<Option<f64>>> = matrix
            .outer_iter()
            .map(|row| {
                row.iter()
                    .map(|&cell| (cell != u32::MAX).then_some(f64::from(cell)))
                    .collect()
            })
            .collect();
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

#[pymethods]
impl TransportNetwork {
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
