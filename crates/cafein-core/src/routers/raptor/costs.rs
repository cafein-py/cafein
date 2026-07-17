//! The cost-row contract: aggregated per-journey costs and their
//! fold rules, shared by every engine's cost path.

use super::*;

/// Aggregated costs of the fastest journey to one destination.
#[derive(Debug, Clone, PartialEq)]
pub struct CostRow {
    /// The destination's index: a stop index for stop matrices, a
    /// destination-point index for pointset matrices.
    pub to: u32,
    /// Travel time in seconds from the requested departure.
    pub seconds: u32,
    /// Number of transit legs; 0 for a destination reached on foot.
    pub rides: u32,
    /// Distance ridden on transit, in meters.
    pub transit_meters: f64,
    /// Distance walked on transfers and access links, in meters.
    pub walk_meters: f64,
    /// Grams CO₂e over the ridden legs; NaN when a ridden trip has no
    /// emission factor.
    pub emission_grams: f64,
    /// The journey's fare under the fare tables; NaN when the journey
    /// cannot be priced, or when no tables were given.
    pub fare: f64,
    /// The ridden legs' geometry as a WKB MultiLineString, when asked
    /// for and leg geometries are installed.
    pub geometry: Option<Vec<u8>>,
}

/// Everything the cost reconstruction reads besides the search state.
pub struct CostInputs<'a> {
    /// Per-trip cumulative distances (drives meters and emissions).
    pub geometry: &'a TripGeometry,
    /// Grams CO₂e per passenger-kilometer per trip, indexed by trip;
    /// NaN marks a trip without a resolved factor.
    pub factors: &'a [f64],
    /// Leg polylines; required to emit geometries.
    pub leg_geometry: Option<&'a LegGeometry>,
    /// Emit each row's WKB MultiLineString.
    pub with_geometry: bool,
    /// Fare tables to price each row's journey with; `None` leaves
    /// fares NaN.
    pub fares: Option<&'a FareTables>,
}

pub(super) const UNREACHED: u32 = u32::MAX;

/// What the windowed candidate fold minimises.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Objective {
    /// Grams CO₂e; unresolved (NaN) emissions never qualify.
    Emissions,
    /// The journey fare; unpriceable (NaN) journeys never qualify.
    Fare,
}

impl Objective {
    fn key(self, row: &CostRow) -> f64 {
        match self {
            Objective::Emissions => row.emission_grams,
            Objective::Fare => row.fare,
        }
    }
}

/// Keeps the better of an existing candidate and a challenger on the
/// objective: a lower key wins, equal keys resolve toward the shorter
/// travel time. NaN keys never qualify.
pub(crate) fn fold_better(
    current: &mut Option<CostRow>,
    challenger: CostRow,
    objective: Objective,
) {
    let key = objective.key(&challenger);
    if key.is_nan() {
        return;
    }
    let better = match current {
        None => true,
        Some(row) => {
            key < objective.key(row)
                || (key == objective.key(row) && challenger.seconds < row.seconds)
        }
    };
    if better {
        *current = Some(challenger);
    }
}
