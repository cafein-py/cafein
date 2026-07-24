//! Street travel profiles and their compilation to per-arc costs.
//!
//! A profile turns the semantic street attributes (produced by the Python
//! extraction in `python/cafein/_osm.py`, stored as the format-12 arrays)
//! into a flat directional cost: one millisecond value per directed arc,
//! `u32::MAX` for an arc the mode may not use. Compilation happens outside the
//! routing hot loop; a query then reads `arc_millis` directly.
//!
//! The mode bits, facility flags, and class-code counts below mirror the
//! contract in `_osm.py` — the raw `u8`/`u16` attribute arrays cross the
//! language boundary as those integers, so the two definitions must stay in
//! sync.

use super::*;

/// Mode permission bits, mirroring `_osm.py` — one bit per street mode in the
/// `adj_access` array. An e-bike reuses the bicycle bit.
pub const MODE_WALK: u8 = 1 << 0;
pub const MODE_BICYCLE: u8 = 1 << 1;
pub const MODE_E_SCOOTER: u8 = 1 << 2;

// The per-edge `edge_flags` bits, mirroring `_osm.py` exactly. `FLAG_DISMOUNT`
// is the only one the profile compiler reads today (it lowers the bicycle to
// walk speed); the rest are the class/facility flags the extraction sets, kept
// here so the whole `edge_flags` bit contract has a Rust-side counterpart and
// cannot drift silently. `bicycle=dismount` is a way-level tag applying to both
// directions, so it lives in the per-edge `edge_flags` (not the directional
// `adj_facility`).
pub const FLAG_DISMOUNT: u16 = 1 << 0;
pub const FLAG_BRIDGE: u16 = 1 << 1;
pub const FLAG_TUNNEL: u16 = 1 << 2;
pub const FLAG_INDOOR: u16 = 1 << 3;
pub const FLAG_STEPS: u16 = 1 << 4;
pub const FLAG_SEGREGATED: u16 = 1 << 5;
pub const FLAG_LIT: u16 = 1 << 6;

/// The class-code table sizes, mirroring `_osm.py`'s `HIGHWAY_CODES`,
/// `SURFACE_CODES`, and `SMOOTHNESS_CODES`. A profile carries one multiplier
/// per code.
pub const HIGHWAY_CODE_COUNT: usize = 27;
pub const SURFACE_CODE_COUNT: usize = 17;
pub const SMOOTHNESS_CODE_COUNT: usize = 9;

/// The street mode a profile routes — exactly one. It maps to a single
/// `adj_access` permission bit, so a profile can never match arcs by any-of
/// several modes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreetMode {
    Walk,
    Bicycle,
    EScooter,
}

impl StreetMode {
    /// The single `adj_access` bit this mode occupies.
    pub fn bit(self) -> u8 {
        match self {
            StreetMode::Walk => MODE_WALK,
            StreetMode::Bicycle => MODE_BICYCLE,
            StreetMode::EScooter => MODE_E_SCOOTER,
        }
    }
}

/// Why a [`StreetProfileDefinition`] cannot be compiled.
#[derive(Debug, PartialEq, Eq)]
pub enum ProfileError {
    /// A speed (base, connector, or dismount) is not finite and positive.
    NonPositiveSpeed,
    /// A multiplier table has the wrong length, or an entry is not finite and
    /// positive.
    InvalidMultipliers,
    /// `max_speed` is below the greatest speed an arc can attain, which would
    /// make the A* heuristic that divides by it inadmissible.
    MaxSpeedTooLow,
    /// A non-walk profile has no `adj_access` to route by — the network was
    /// built without the multimodal attributes.
    MissingAttributes,
    /// A validated definition produced an arc cost at or above the forbidden
    /// sentinel (`u32::MAX`), which cannot be represented; the definition is
    /// physically implausible (an extreme speed or multiplier).
    ArcCostOverflow,
}

impl std::fmt::Display for ProfileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ProfileError::NonPositiveSpeed => {
                write!(f, "a profile speed is not finite and positive")
            }
            ProfileError::InvalidMultipliers => {
                write!(f, "a profile multiplier table is malformed")
            }
            ProfileError::MaxSpeedTooLow => {
                write!(f, "max_speed is below the greatest attainable speed")
            }
            ProfileError::MissingAttributes => {
                write!(f, "a non-walk profile needs installed street attributes")
            }
            ProfileError::ArcCostOverflow => {
                write!(f, "a profile produced an unrepresentable arc cost")
            }
        }
    }
}

impl std::error::Error for ProfileError {}

/// How a street mode traverses the network: the mode it routes and the speed
/// model that turns an arc's attributes into a traversal time.
///
/// Speeds are metres per second. The per-class multipliers scale the base
/// speed by the arc's highway class, surface, and smoothness; the built-in
/// profiles are flat (all multipliers `1.0`), but the machinery supports
/// user-defined slower/faster classes. `PartialEq` lets a persisted compiled
/// profile bind by exact-definition equality (the `same_factors` principle),
/// and distinguishes profiles that route identically but carry a different
/// `vehicle_class` for later emissions.
#[derive(Debug, Clone, PartialEq)]
pub struct StreetProfileDefinition {
    pub name: String,
    /// The mode this profile routes.
    pub mode: StreetMode,
    /// The vehicle class this profile's legs are attributed to for emissions
    /// and monetary cost (resolved later, Python-side).
    pub vehicle_class: String,
    /// Base speed on a default edge, m/s.
    pub base_speed: f64,
    /// Speed on off-network connectors (the snap approach), m/s.
    pub connector_speed: f64,
    /// An upper bound on any effective on-network speed, m/s — the admissible
    /// A* heuristic divides straight-line distance by this, so it must be at
    /// least the greatest speed any arc can attain.
    pub max_speed: f64,
    /// Speed on a `bicycle=dismount` arc (walk speed), m/s.
    pub dismount_speed: f64,
    /// Per-highway-class speed multiplier, indexed by the highway code
    /// (length `HIGHWAY_CODE_COUNT`).
    pub highway_multipliers: Vec<f64>,
    /// Per-surface speed multiplier, indexed by the surface code
    /// (length `SURFACE_CODE_COUNT`).
    pub surface_multipliers: Vec<f64>,
    /// Per-smoothness speed multiplier, indexed by the smoothness code
    /// (length `SMOOTHNESS_CODE_COUNT`).
    pub smoothness_multipliers: Vec<f64>,
}

impl StreetProfileDefinition {
    /// A flat profile: the base speed applies on every class (all multipliers
    /// `1.0`), so `max_speed` equals the base speed. Connector and dismount
    /// run at walking speed.
    pub fn flat(
        name: &str,
        mode: StreetMode,
        vehicle_class: &str,
        base_speed: f64,
    ) -> StreetProfileDefinition {
        StreetProfileDefinition {
            name: name.to_string(),
            mode,
            vehicle_class: vehicle_class.to_string(),
            base_speed,
            connector_speed: WALK_SPEED,
            max_speed: base_speed,
            dismount_speed: WALK_SPEED,
            highway_multipliers: vec![1.0; HIGHWAY_CODE_COUNT],
            surface_multipliers: vec![1.0; SURFACE_CODE_COUNT],
            smoothness_multipliers: vec![1.0; SMOOTHNESS_CODE_COUNT],
        }
    }

    /// The default walking profile, 3.6 km/h (matching the time engines).
    pub fn walk() -> StreetProfileDefinition {
        StreetProfileDefinition::flat("walk", StreetMode::Walk, "walk", WALK_SPEED)
    }

    /// The default bicycle profile, 14.4 km/h (R5's 4 m/s default).
    pub fn bicycle() -> StreetProfileDefinition {
        StreetProfileDefinition::flat("bicycle", StreetMode::Bicycle, "bicycle", 4.0)
    }

    /// The default e-bike profile: bicycle permissions and speed for now, but a
    /// distinct vehicle class (its emissions differ).
    pub fn e_bike() -> StreetProfileDefinition {
        StreetProfileDefinition::flat("e_bike", StreetMode::Bicycle, "e_bike", 4.0)
    }

    /// The default e-scooter profile, 15 km/h (a modelling speed, not a legal
    /// maximum); permissions are bicycle-like.
    pub fn e_scooter() -> StreetProfileDefinition {
        StreetProfileDefinition::flat("e_scooter", StreetMode::EScooter, "e_scooter", 15.0 / 3.6)
    }

    /// Checks the numeric invariants the compiler and the later A* heuristic
    /// rely on: finite positive speeds, correctly sized multiplier tables with
    /// finite positive entries, and a `max_speed` at least the greatest
    /// attainable on-network speed.
    pub fn validate(&self) -> Result<(), ProfileError> {
        let positive = |value: f64| value.is_finite() && value > 0.0;
        if !positive(self.base_speed)
            || !positive(self.connector_speed)
            || !positive(self.dismount_speed)
        {
            return Err(ProfileError::NonPositiveSpeed);
        }
        let table_ok = |table: &[f64], len: usize| {
            table.len() == len && table.iter().all(|&value| positive(value))
        };
        if !table_ok(&self.highway_multipliers, HIGHWAY_CODE_COUNT)
            || !table_ok(&self.surface_multipliers, SURFACE_CODE_COUNT)
            || !table_ok(&self.smoothness_multipliers, SMOOTHNESS_CODE_COUNT)
        {
            return Err(ProfileError::InvalidMultipliers);
        }
        // The greatest attainable on-network speed. `multiplied` is the base
        // speed scaled by each class table's true peak (the tables are
        // validated non-empty and all-positive above, so a `0.0`-seeded max is
        // the real maximum). A walking profile also compiles over an
        // attribute-free graph, where the multipliers are not applied and every
        // arc runs at `base_speed`, so the bound must include it. A non-walk
        // profile always has attributes and can additionally hit the dismount
        // speed.
        let peak = |table: &[f64]| table.iter().cloned().fold(0.0_f64, f64::max);
        let multiplied = self.base_speed
            * peak(&self.highway_multipliers)
            * peak(&self.surface_multipliers)
            * peak(&self.smoothness_multipliers);
        let attainable = if self.mode == StreetMode::Walk {
            self.base_speed.max(multiplied)
        } else {
            multiplied.max(self.dismount_speed)
        };
        if !self.max_speed.is_finite() || self.max_speed < attainable {
            return Err(ProfileError::MaxSpeedTooLow);
        }
        Ok(())
    }
}

/// The default walking speed in m/s (3.6 km/h), shared by the connector and
/// dismount defaults.
const WALK_SPEED: f64 = 1.0;

/// A profile compiled against a street network: the definition it was built
/// from and the per-arc traversal cost in milliseconds (`u32::MAX` = the arc
/// is forbidden for this profile).
#[derive(Debug, Clone)]
pub struct CompiledStreetProfile {
    pub definition: StreetProfileDefinition,
    /// One entry per directed arc (adjacency slot), aligned with `adj_meters`.
    pub arc_millis: Vec<u32>,
}

impl StreetNetwork {
    /// Compiles `definition` into the per-arc millisecond costs for this
    /// network's graph, after validating it. An arc is forbidden (`u32::MAX`)
    /// when the profile's mode bit is absent from its `adj_access`, else it
    /// costs `ceil(length / effective_speed * 1000)` where the effective speed
    /// scales the base speed by the arc edge's class multipliers, falling to
    /// the dismount speed on a `FLAG_DISMOUNT` arc.
    ///
    /// A graph without installed attributes carries no per-mode permissions,
    /// so only the walking profile compiles over it (every arc permitted at
    /// the base speed); a non-walk profile has nothing to route by and returns
    /// [`ProfileError::MissingAttributes`].
    pub fn compile_profile(
        &self,
        definition: &StreetProfileDefinition,
    ) -> Result<CompiledStreetProfile, ProfileError> {
        definition.validate()?;
        let mode = definition.mode.bit();
        let meters = self.arrays().adj_meters();
        let edges = self.arrays().adj_edges();
        let attributes = self.street_attributes();
        // A non-walk profile has no permissions to route by without the
        // attributes — surface that rather than returning an all-forbidden set.
        if attributes.is_none() && definition.mode != StreetMode::Walk {
            return Err(ProfileError::MissingAttributes);
        }
        let mut arc_millis = vec![u32::MAX; meters.len()];
        for slot in 0..meters.len() {
            let permitted = match attributes {
                Some(attributes) => attributes.adj_access[slot] & mode != 0,
                None => true, // attribute-free walking graph: every arc walkable
            };
            if !permitted {
                continue;
            }
            let mut speed = definition.base_speed;
            if let Some(attributes) = attributes {
                let edge = edges[slot] as usize;
                speed *= multiplier(
                    &definition.highway_multipliers,
                    attributes.edge_highway[edge],
                );
                speed *= multiplier(
                    &definition.surface_multipliers,
                    attributes.edge_surface[edge],
                );
                speed *= multiplier(
                    &definition.smoothness_multipliers,
                    attributes.edge_smoothness[edge],
                );
                let dismount = attributes.edge_flags[edge] & FLAG_DISMOUNT != 0
                    && definition.mode != StreetMode::Walk;
                if dismount {
                    speed = definition.dismount_speed;
                }
            }
            // Validated speeds keep this finite and positive.
            let millis = (meters[slot] / speed * 1000.0).ceil();
            // `u32::MAX` is the forbidden sentinel; a cost reaching it cannot be
            // stored, so a (physically implausible) definition that produces one
            // is rejected rather than silently understated.
            if millis >= u32::MAX as f64 {
                return Err(ProfileError::ArcCostOverflow);
            }
            // A positive-length arc costs at least 1 ms — an extreme speed or a
            // tiny length can underflow the product to `0.0`, which must not
            // read as instantaneous. A genuine zero-length arc stays 0.
            arc_millis[slot] = if meters[slot] > 0.0 {
                (millis as u32).max(1)
            } else {
                millis as u32
            };
        }
        Ok(CompiledStreetProfile {
            definition: definition.clone(),
            arc_millis,
        })
    }
}

/// A class multiplier by code. Every attribute path validates codes against
/// the table sizes (`check_street_attributes`) and every profile validates its
/// table lengths, so the code is always in range — indexed directly rather
/// than silently defaulting an out-of-range (ABI-drifted) code to neutral.
fn multiplier(multipliers: &[f64], code: u8) -> f64 {
    multipliers[code as usize]
}
