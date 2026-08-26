//! The accessibility primitive: per-origin aggregation over
//! destination costs.
//!
//! Engine-agnostic: callers hand in one origin's per-destination costs
//! on any axis (seconds, metres, grams, cents — unified as `f64`,
//! `None` = unreached) and get back one of three aggregations:
//! decay-weighted opportunity sums (budget-bounded), the k nearest
//! destinations (k-bounded within a finite horizon), or the reached
//! set under a budget. The engines and the parallel fan-out live with
//! the callers; nothing here allocates per destination beyond the
//! requested output.

/// A decay weight function over cost, hard truncated at the budget —
/// `weight(c, b) = 0` for `c > b` — except `Linear`, the ramp, which
/// keeps its weight to `b + width/2`. Within the support:
///
/// - `Step`: 1
/// - `Linear { width }`: `clip((b + width/2 - c) / width, 0, 1)`
/// - `LinearCutoff`: `1 - c / b`
/// - `Exponential { half_life }`: `exp(-ln(2) * c / half_life)`
/// - `Logistic { scale }`: `1 / (1 + exp((c - b) / scale))`
///
/// Parameters must be positive and finite; the boundary that parses
/// user input enforces that before anything reaches here.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Decay {
    Step,
    Linear { width: f64 },
    LinearCutoff,
    Exponential { half_life: f64 },
    Logistic { scale: f64 },
}

impl Decay {
    /// The cost beyond which the family weighs nothing under `budget`.
    pub fn support(&self, budget: f64) -> f64 {
        match *self {
            Decay::Linear { width } => budget + width / 2.0,
            _ => budget,
        }
    }

    /// The weight of a destination at `cost` under `budget`; a
    /// non-finite cost weighs nothing (it cannot lie within a finite
    /// support).
    pub fn weight(&self, cost: f64, budget: f64) -> f64 {
        if !cost.is_finite() || cost > self.support(budget) {
            return 0.0;
        }
        match *self {
            Decay::Step => 1.0,
            Decay::Linear { width } => ((budget - cost) / width + 0.5).clamp(0.0, 1.0),
            Decay::LinearCutoff => (1.0 - cost / budget).clamp(0.0, 1.0),
            Decay::Exponential { half_life } => (-std::f64::consts::LN_2 * cost / half_life).exp(),
            Decay::Logistic { scale } => 1.0 / (1.0 + ((cost - budget) / scale).exp()),
        }
    }
}

/// Mode (a): decay-weighted opportunity sums for one origin.
///
/// `opportunities` is row-major `[destination][field]` with
/// `fields * costs.len()` entries; the result is row-major
/// `[budget][field]` with `budgets.len() * fields` sums. Costs and
/// budgets share one axis unit; an unreached destination contributes
/// nothing at any budget.
pub fn opportunity_sums(
    costs: &[Option<f64>],
    opportunities: &[f64],
    fields: usize,
    budgets: &[f64],
    decay: &Decay,
) -> Vec<f64> {
    debug_assert_eq!(opportunities.len(), costs.len() * fields);
    let mut sums = vec![0.0; budgets.len() * fields];
    for (destination, cost) in costs.iter().enumerate() {
        let Some(cost) = *cost else { continue };
        if !cost.is_finite() {
            continue;
        }
        let row = &opportunities[destination * fields..(destination + 1) * fields];
        for (bucket, budget) in budgets.iter().enumerate() {
            let weight = decay.weight(cost, *budget);
            if weight == 0.0 {
                continue;
            }
            let out = &mut sums[bucket * fields..(bucket + 1) * fields];
            for (field, opportunity) in row.iter().enumerate() {
                out[field] += weight * opportunity;
            }
        }
    }
    sums
}

/// Mode (b): the `k` nearest destinations within the `max_cost`
/// horizon, as `(destination index, cost)` in deterministic
/// `(cost, index)` order. Unreached destinations and those beyond the
/// horizon are absent, so fewer than `k` pairs can come back.
pub fn nearest(costs: &[Option<f64>], k: usize, max_cost: f64) -> Vec<(usize, f64)> {
    let limit = k.min(costs.len());
    if limit == 0 {
        return Vec::new();
    }
    // A bounded insertion keeps this O(n * k) with k tiny (closest few
    // schools), no allocation beyond the result.
    let mut best: Vec<(usize, f64)> = Vec::with_capacity(limit);
    for (destination, cost) in costs.iter().enumerate() {
        let Some(cost) = *cost else { continue };
        if !cost.is_finite() || cost > max_cost {
            continue;
        }
        let position = best.partition_point(|(index, held)| {
            *held < cost || (*held == cost && *index < destination)
        });
        if position < limit {
            if best.len() == limit {
                best.pop();
            }
            best.insert(position, (destination, cost));
        }
    }
    best
}

/// Mode (c): the destination indices whose cost is within `budget`,
/// in index order.
pub fn reached(costs: &[Option<f64>], budget: f64) -> Vec<usize> {
    costs
        .iter()
        .enumerate()
        .filter_map(|(destination, cost)| match cost {
            Some(cost) if cost.is_finite() && *cost <= budget => Some(destination),
            _ => None,
        })
        .collect()
}

#[cfg(test)]
#[path = "access_tests.rs"]
mod tests;
