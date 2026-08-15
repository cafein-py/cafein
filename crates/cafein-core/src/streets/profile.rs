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
pub const MODE_CAR: u8 = 1 << 3;
pub const MODE_WHEELCHAIR: u8 = 1 << 4;

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
pub const FLAG_ROUNDABOUT: u16 = 1 << 7;

/// The class-code table sizes, mirroring `_osm.py`'s `HIGHWAY_CODES`,
/// `SURFACE_CODES`, and `SMOOTHNESS_CODES`. A profile carries one multiplier
/// per code.
pub const HIGHWAY_CODE_COUNT: usize = 27;
pub const SURFACE_CODE_COUNT: usize = 17;
pub const SMOOTHNESS_CODE_COUNT: usize = 9;

/// Which highway codes are the ramp category (`*_link`), by `edge_highway`
/// code — mirroring `_osm.py`'s `HIGHWAY_CODES` order: motorway_link (2),
/// trunk_link (4), primary_link (6), secondary_link (8), tertiary_link (10).
pub const RAMP_HIGHWAY: [bool; HIGHWAY_CODE_COUNT] = {
    let mut ramps = [false; HIGHWAY_CODE_COUNT];
    ramps[2] = true;
    ramps[4] = true;
    ramps[6] = true;
    ramps[8] = true;
    ramps[10] = true;
    ramps
};

/// The speed bound splitting the delay model's high- and low-speed branches
/// (ramp shares and the junction-free multipliers), km/h.
pub const CAR_HIGH_SPEED_KMH: f64 = 70.0;

/// The default car profile's `max_speed`, km/h. Compiled arc speeds are the
/// persisted values exactly — the goal-directed bound is measured from the
/// compiled costs — so this only backs that bound's degenerate fallback and
/// the definition validation.
pub const CAR_SPEED_CEILING_KMH: f64 = 250.0;

/// The junction head classes the car delay model reads, mirroring
/// `_osm.py`'s `JUNCTION_*` values.
pub const JUNCTION_TOPOLOGICAL: u8 = 1;
pub const JUNCTION_PRIORITY: u8 = 2;
pub const JUNCTION_SIGNALS: u8 = 3;
pub const JUNCTION_RAMP: u8 = 4;

/// The spike clamp on a sub-segment slope: ±100 % grade. Steeper is a DEM
/// artifact, not a street.
pub const MAX_SLOPE: f64 = 1.0;

/// The floor on a sub-segment's slope multiplier `1 + f(s)`, so no downhill
/// factor can produce a non-positive cost.
pub const MIN_SLOPE_MULTIPLIER: f64 = 0.1;

/// The street mode a profile routes — exactly one. It maps to a single
/// `adj_access` permission bit, so a profile can never match arcs by any-of
/// several modes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreetMode {
    Walk,
    Bicycle,
    EScooter,
    Car,
    Wheelchair,
}

impl StreetMode {
    /// The single `adj_access` bit this mode occupies.
    pub fn bit(self) -> u8 {
        match self {
            StreetMode::Walk => MODE_WALK,
            StreetMode::Bicycle => MODE_BICYCLE,
            StreetMode::EScooter => MODE_E_SCOOTER,
            StreetMode::Car => MODE_CAR,
            StreetMode::Wheelchair => MODE_WHEELCHAIR,
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
    /// `max_speed` is below the greatest speed an arc can attain, so it
    /// would not be the upper bound it declares itself to be.
    MaxSpeedTooLow,
    /// A slope factor is not finite and non-negative.
    InvalidSlopeFactors,
    /// A non-walk profile has no `adj_access` to route by — the network was
    /// built without the multimodal attributes.
    MissingAttributes,
    /// The car delay model is malformed (wrong table lengths, out-of-range
    /// group indexes, or non-finite/negative numbers), or a non-car profile
    /// carries one.
    InvalidCarModel,
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
            ProfileError::InvalidSlopeFactors => {
                write!(f, "a slope factor is not finite and non-negative")
            }
            ProfileError::MaxSpeedTooLow => {
                write!(f, "max_speed is below the greatest attainable speed")
            }
            ProfileError::MissingAttributes => {
                write!(f, "a non-walk profile needs installed street attributes")
            }
            ProfileError::InvalidCarModel => {
                write!(f, "the car delay model is malformed")
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
/// profiles keep those tables neutral (all `1.0`), though the machinery
/// supports user-defined slower/faster classes. The slope factors are
/// separate — the built-in bicycle carries a nonzero slope model. `PartialEq` lets a persisted compiled
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
    /// A validated upper bound on any effective on-network speed, m/s — it
    /// must be at least the greatest speed any arc can attain (slope credit
    /// included). The goal-directed search normally divides by the tighter
    /// measured [`CompiledStreetProfile::max_effective_speed`]; this bound
    /// backs the validation and serves as its fallback.
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
    /// Cost factor on a climbing sub-segment: its multiplier is
    /// `1 + slope_uphill·s` for slope fraction `s > 0`. Zero disables the
    /// uphill penalty.
    pub slope_uphill: f64,
    /// Cost factor on a descending sub-segment: `1 + slope_downhill·s` with
    /// `s < 0` — a bounded credit. Zero disables it.
    pub slope_downhill: f64,
    /// The car delay model, resolved to one period's flat numbers. `None`
    /// compiles free-flow — the default regime; only a car profile may carry
    /// `Some`. The car mode reads the per-arc driving speeds either way; this
    /// controls the junction penalties and multipliers alone.
    pub car: Option<CarCostModel>,
}

/// The intersection-delay model's numbers for one period, resolved flat by
/// the Python layer (the shipped Jaakkola 2013 values merged with any
/// `delay_model=` override). Everything an arc's delay needs is here plus
/// the arc's own persisted attributes, so compilation stays edge-local.
#[derive(Debug, Clone, PartialEq)]
pub struct CarCostModel {
    /// Crossing penalty `b` in seconds by road-class group
    /// (`[groups 1–2, group 3, groups 4–6]`) for the resolved period.
    pub group_seconds: [f64; 3],
    /// Highway code → index into `group_seconds`, length
    /// `HIGHWAY_CODE_COUNT`.
    pub groups: Vec<u8>,
    /// The share of its own `b` a ramp element charges per junction
    /// endpoint at or above [`CAR_HIGH_SPEED_KMH`].
    pub ramp_share_high: f64,
    /// The below-threshold ramp share (the calibration's ½, every period).
    pub ramp_share_low: f64,
    /// The multiplier on a junction-free ramp element at or above the
    /// threshold.
    pub ramp_multiplier: f64,
    /// The multiplier on a junction-free non-ramp element at or above the
    /// threshold (1.0 in the midday period).
    pub congestion_multiplier: f64,
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
            slope_uphill: 0.0,
            slope_downhill: 0.0,
            car: None,
        }
    }

    /// The default walking profile, 3.6 km/h (matching the time engines).
    pub fn walk() -> StreetProfileDefinition {
        StreetProfileDefinition::flat("walk", StreetMode::Walk, "walk", WALK_SPEED)
    }

    /// The default wheelchair profile: walk speed over the wheelchair
    /// permission bit — walkable arcs without stairs, per the
    /// extraction's tag rules.
    pub fn wheelchair() -> StreetProfileDefinition {
        StreetProfileDefinition::flat(
            "wheelchair",
            StreetMode::Wheelchair,
            "wheelchair",
            WALK_SPEED,
        )
    }

    /// The default bicycle profile, 14.4 km/h (R5's 4 m/s default), with the
    /// slope model `w = d·(1 + f(s))`: `f(s) = s` uphill, `0.3·s` downhill.
    /// The downhill credit can raise an arc's effective speed to
    /// `base / (1 − 0.3·MAX_SLOPE)`, so `max_speed` covers that bound.
    pub fn bicycle() -> StreetProfileDefinition {
        let mut definition =
            StreetProfileDefinition::flat("bicycle", StreetMode::Bicycle, "bicycle", 4.0);
        definition.slope_uphill = 1.0;
        definition.slope_downhill = 0.3;
        definition.max_speed = definition.base_speed / (1.0 - 0.3 * MAX_SLOPE);
        definition
    }

    /// The default e-bike profile: bicycle permissions and speed for now, but a
    /// distinct vehicle class (its emissions differ). Slope stays off — the
    /// documented model is for the conventional bicycle, and an
    /// assist-flattened curve would be an invented number.
    pub fn e_bike() -> StreetProfileDefinition {
        StreetProfileDefinition::flat("e_bike", StreetMode::Bicycle, "e_bike", 4.0)
    }

    /// The default e-scooter profile, 15 km/h (a modelling speed, not a legal
    /// maximum); permissions are bicycle-like.
    pub fn e_scooter() -> StreetProfileDefinition {
        StreetProfileDefinition::flat("e_scooter", StreetMode::EScooter, "e_scooter", 15.0 / 3.6)
    }

    /// The default car profile: free-flow (no delay model), each arc at
    /// exactly its own persisted driving speed — `base_speed` and the class
    /// tables are unused, and no ceiling alters a tagged speed. The
    /// goal-directed bound is measured from the compiled costs themselves;
    /// `max_speed` only backs its degenerate fallback and the validation.
    /// The connector stays at walking speed: you walk to the car.
    pub fn car(model: Option<CarCostModel>) -> StreetProfileDefinition {
        let mut definition = StreetProfileDefinition::flat("car", StreetMode::Car, "ICE", 1.0);
        definition.max_speed = CAR_SPEED_CEILING_KMH / 3.6;
        definition.car = model;
        definition
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
        if self.car.is_some() && self.mode != StreetMode::Car {
            return Err(ProfileError::InvalidCarModel);
        }
        if let Some(model) = &self.car {
            let factor = |value: f64| value.is_finite() && value >= 0.0;
            let numbers_ok = model.group_seconds.iter().all(|&b| factor(b))
                && factor(model.ramp_share_high)
                && factor(model.ramp_share_low)
                && factor(model.ramp_multiplier)
                && factor(model.congestion_multiplier);
            let groups_ok = model.groups.len() == HIGHWAY_CODE_COUNT
                && model.groups.iter().all(|&group| (group as usize) < 3);
            if !numbers_ok || !groups_ok {
                return Err(ProfileError::InvalidCarModel);
            }
        }
        if self.mode == StreetMode::Car {
            // The car routes on the per-arc persisted speeds; the base speed
            // and the class tables are unused, and `max_speed` is the clamp
            // the compiler applies to every arc speed, so it only needs to be
            // a positive finite ceiling.
            return if positive(self.max_speed) {
                Ok(())
            } else {
                Err(ProfileError::MaxSpeedTooLow)
            };
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
        let factor_ok = |value: f64| value.is_finite() && value >= 0.0;
        if !factor_ok(self.slope_uphill) || !factor_ok(self.slope_downhill) {
            return Err(ProfileError::InvalidSlopeFactors);
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
        // The downhill credit shortens an arc's effective distance, raising
        // its effective speed by up to the inverse of the smallest possible
        // sub-segment multiplier — the A* heuristic's bound must cover it.
        let attainable = attainable / self.min_slope_multiplier();
        if !self.max_speed.is_finite() || self.max_speed < attainable {
            return Err(ProfileError::MaxSpeedTooLow);
        }
        Ok(())
    }

    /// The smallest sub-segment multiplier this definition can produce: the
    /// steepest clamped descent's credit, floored — or 1.0 with slope off.
    fn min_slope_multiplier(&self) -> f64 {
        if self.slope_uphill == 0.0 && self.slope_downhill == 0.0 {
            return 1.0;
        }
        (1.0 - self.slope_downhill * MAX_SLOPE).max(MIN_SLOPE_MULTIPLIER)
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
    /// Private with a read-only accessor: the cached speed bound below is
    /// correct only for the costs it was measured over.
    arc_millis: Vec<u32>,
    /// The car delay model's per-slot cost split, present only when a delay
    /// model compiled: the distance-proportional base and the two junction
    /// endpoint delays, with `base + head + tail == arc_millis` per permitted
    /// slot. Partial (snapped) traversals prorate the base alone and add only
    /// the endpoint delay of a junction actually crossed; full relaxations
    /// keep reading `arc_millis`.
    car_partials: Option<CarPartials>,
    /// The greatest *chord* speed any permitted arc attains on this network:
    /// straight-line metres between the arc's endpoint coordinates (one
    /// shared vertex-coordinate table) per second of compiled cost. The
    /// goal-directed search divides by it, and because every distance in
    /// that bound derives from the same table, the heuristic stays
    /// admissible whatever the stored edge lengths claim — and tighter than
    /// a length-based bound wherever edges bend. Computed lazily on the
    /// first goal-directed query, so matrix-only workloads never pay for it
    /// (or for the table); infinite when a permitted zero-cost arc spans a
    /// nonzero chord, which disables the distance term entirely.
    max_effective_speed: std::sync::OnceLock<f64>,
}

/// The car cost split behind [`CompiledStreetProfile::car_partials`]: per
/// slot, the distance-proportional base cost and the junction delay charged
/// at each end of the traversal — `head` at the slot's target vertex, `tail`
/// at its source.
#[derive(Debug, Clone)]
struct CarPartials {
    base: Vec<u32>,
    head: Vec<u32>,
    tail: Vec<u32>,
}

/// The on-network millisecond cost of traversing `fraction` of an arc whose
/// full cost is `arc`.
pub(super) fn partial_millis(arc: u32, fraction: f64) -> u64 {
    (f64::from(arc) * fraction).ceil() as u64
}

impl CompiledStreetProfile {
    /// The per-arc millisecond costs, `u32::MAX` for a forbidden arc.
    pub fn arc_millis(&self) -> &[u32] {
        &self.arc_millis
    }

    /// The cost of leaving a mid-edge snap toward the slot's head over the
    /// given fraction: the head junction is crossed, the one behind the snap
    /// is not — unless the fraction is the whole edge, where the snap sits
    /// exactly on the tail vertex and that junction is crossed too (the full
    /// arc cost, agreeing with the graph relaxation for the same trip).
    /// Without a cost split this is the plain proration.
    pub(super) fn departing_partial(&self, slot: usize, fraction: f64) -> u64 {
        match &self.car_partials {
            Some(partials) => partial_millis(partials.base[slot], fraction)
                .saturating_add(u64::from(partials.head[slot]))
                .saturating_add(if fraction >= 1.0 {
                    u64::from(partials.tail[slot])
                } else {
                    0
                }),
            None => partial_millis(self.arc_millis[slot], fraction),
        }
    }

    /// The cost of entering the slot at its tail and stopping at a mid-edge
    /// snap over the given fraction: the tail junction is crossed, the head
    /// is never reached — unless the fraction is the whole edge, where the
    /// snap sits exactly on the head vertex and that junction is crossed too.
    pub(super) fn arriving_partial(&self, slot: usize, fraction: f64) -> u64 {
        match &self.car_partials {
            Some(partials) => partial_millis(partials.base[slot], fraction)
                .saturating_add(u64::from(partials.tail[slot]))
                .saturating_add(if fraction >= 1.0 {
                    u64::from(partials.head[slot])
                } else {
                    0
                }),
            None => partial_millis(self.arc_millis[slot], fraction),
        }
    }

    /// The cost of a same-edge trip over `fraction` of the slot: two interior
    /// snaps cross no junction, but a trip that starts exactly at the
    /// traversal's tail vertex enters the element through that junction, and
    /// one that ends exactly at its head vertex crosses that one — matching
    /// what the seed/egress path charges for the identical trip, so the
    /// same-edge fast path can never undercut the graph route.
    pub(super) fn same_edge_partial(
        &self,
        slot: usize,
        fraction: f64,
        enters_at_tail: bool,
        exits_at_head: bool,
    ) -> u64 {
        match &self.car_partials {
            Some(partials) => partial_millis(partials.base[slot], fraction)
                .saturating_add(if enters_at_tail {
                    u64::from(partials.tail[slot])
                } else {
                    0
                })
                .saturating_add(if exits_at_head {
                    u64::from(partials.head[slot])
                } else {
                    0
                }),
            None => partial_millis(self.arc_millis[slot], fraction),
        }
    }

    /// The lazily cached chord-speed bound (see the field's documentation).
    pub(super) fn effective_speed_cache(&self) -> &std::sync::OnceLock<f64> {
        &self.max_effective_speed
    }
}

impl StreetNetwork {
    /// Compiles `definition` into the per-arc millisecond costs for this
    /// network's graph, after validating it. An arc is forbidden (`u32::MAX`)
    /// when the profile's mode bit is absent from its `adj_access`, else it
    /// costs `ceil(length / effective_speed * 1000)` where the effective speed
    /// scales the base speed by the arc edge's class multipliers, falling to
    /// the dismount speed on a `FLAG_DISMOUNT` arc.
    ///
    /// With elevations installed and a slope-aware definition, `length` is
    /// the arc's effective distance: the physical length scaled by the
    /// direction's mean slope multiplier (see [`Self::slope_means`]).
    /// Dismount arcs keep the multiplier — the same person pushes the same
    /// bicycle up the same hill.
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
        if definition.mode == StreetMode::Car {
            return self.compile_car(definition);
        }
        let mode = definition.mode.bit();
        let meters = self.arrays().adj_meters();
        let edges = self.arrays().adj_edges();
        let attributes = self.street_attributes();
        // A non-walk profile has no permissions to route by without the
        // attributes — surface that rather than returning an all-forbidden set.
        if attributes.is_none() && definition.mode != StreetMode::Walk {
            return Err(ProfileError::MissingAttributes);
        }
        // With elevations installed and a nonzero slope factor, every edge
        // gets a distance-weighted mean multiplier per direction; the arc
        // loop below scales its length by the one for its direction.
        let slope = self.slope_means(definition);
        let slot_reverse = slope.as_ref().map(|_| self.slot_directions());
        let mut arc_millis = vec![u32::MAX; meters.len()];
        for slot in 0..meters.len() {
            let permitted = match attributes {
                Some(attributes) => attributes.adj_access[slot] & mode != 0,
                None => true, // attribute-free walking graph: every arc walkable
            };
            if !permitted {
                continue;
            }
            let mut length = meters[slot];
            if let (Some(means), Some(reverse)) = (&slope, &slot_reverse) {
                let (forward_mean, reverse_mean) = means[edges[slot] as usize];
                length *= if reverse[slot] {
                    reverse_mean
                } else {
                    forward_mean
                };
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
            let millis = (length / speed * 1000.0).ceil();
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
            car_partials: None,
            max_effective_speed: std::sync::OnceLock::new(),
        })
    }

    /// Compiles a car profile: each permitted arc at its own persisted
    /// driving speed, plus — with a delay model — the element's
    /// attribute-derived delay under the Jaakkola calibration, exactly
    /// edge-separable. Without a model (`car: None`, the default regime)
    /// every arc is its free-flow seconds.
    ///
    /// With a model, an element's cost is exclusive between multipliers and
    /// penalties: a roundabout interior charges `b/4` in place of endpoint
    /// shares; otherwise a junction-affected element — one with a
    /// topological, signalized, or ramp-junction endpoint, a property of the
    /// classes, never of the numeric share values — sums each junction
    /// endpoint's share: `½·b` at a topological or signalized endpoint, at a
    /// ramp-junction endpoint `¼·b` for a non-ramp element and the period's
    /// ramp share of `b` for a ramp element (the low-speed branch below
    /// [`CAR_HIGH_SPEED_KMH`]); only a junction-free element scales by a
    /// multiplier instead: the ramp multiplier on a fast ramp, the
    /// congestion multiplier on a fast non-ramp element, free-flow below
    /// the threshold. `b` always comes from the element's own group.
    ///
    /// The compiled profile also carries the per-slot cost split (base and
    /// endpoint delays) the partial-traversal costs read, with
    /// `base + head + tail == arc_millis` on every permitted slot.
    fn compile_car(
        &self,
        definition: &StreetProfileDefinition,
    ) -> Result<CompiledStreetProfile, ProfileError> {
        let meters = self.arrays().adj_meters();
        let edges = self.arrays().adj_edges();
        let (Some(attributes), Some(car)) = (self.street_attributes(), self.car_attributes())
        else {
            return Err(ProfileError::MissingAttributes);
        };
        // Each slot stores the class at its own head; its tail class is the
        // sibling slot's head. Pairing the two slots of every edge makes both
        // available per traversal direction.
        let mut tail_classes = vec![0u8; meters.len()];
        let mut first_slot = vec![usize::MAX; self.edge_count() as usize];
        for (slot, &edge) in edges.iter().enumerate() {
            let edge = edge as usize;
            if first_slot[edge] == usize::MAX {
                first_slot[edge] = slot;
            } else {
                let sibling = first_slot[edge];
                tail_classes[slot] = car.adj_junction[sibling];
                tail_classes[sibling] = car.adj_junction[slot];
            }
        }
        let mut arc_millis = vec![u32::MAX; meters.len()];
        let mut partials = definition.car.as_ref().map(|_| CarPartials {
            base: vec![0; meters.len()],
            head: vec![0; meters.len()],
            tail: vec![0; meters.len()],
        });
        for slot in 0..meters.len() {
            if attributes.adj_access[slot] & MODE_CAR == 0 {
                continue;
            }
            let speed_kmh = f64::from(car.adj_car_speed[slot]);
            let mut base = meters[slot] / (speed_kmh / 3.6);
            let (mut head, mut tail) = (0.0f64, 0.0f64);
            if let Some(model) = &definition.car {
                let edge = edges[slot] as usize;
                let code = attributes.edge_highway[edge] as usize;
                let b = model.group_seconds[model.groups[code] as usize];
                let is_ramp = RAMP_HIGHWAY[code];
                let fast = speed_kmh >= CAR_HIGH_SPEED_KMH;
                let junction = |class: u8| {
                    matches!(
                        class,
                        JUNCTION_TOPOLOGICAL | JUNCTION_SIGNALS | JUNCTION_RAMP
                    )
                };
                let share = |class: u8| match class {
                    JUNCTION_TOPOLOGICAL | JUNCTION_SIGNALS => 0.5,
                    JUNCTION_RAMP if is_ramp => {
                        if fast {
                            model.ramp_share_high
                        } else {
                            model.ramp_share_low
                        }
                    }
                    JUNCTION_RAMP => 0.25,
                    _ => 0.0,
                };
                let (head_class, tail_class) = (car.adj_junction[slot], tail_classes[slot]);
                if attributes.edge_flags[edge] & FLAG_ROUNDABOUT != 0 {
                    // The interior charge is circulating delay, spread over
                    // the element like distance rather than pinned to an end.
                    base += b / 4.0;
                } else if junction(head_class) || junction(tail_class) {
                    head = b * share(head_class);
                    tail = b * share(tail_class);
                } else if fast {
                    base *= if is_ramp {
                        model.ramp_multiplier
                    } else {
                        model.congestion_multiplier
                    };
                }
            }
            let millis = ((base + head + tail) * 1000.0).ceil();
            if millis >= u32::MAX as f64 {
                return Err(ProfileError::ArcCostOverflow);
            }
            let total = if meters[slot] > 0.0 {
                (millis as u32).max(1)
            } else {
                millis as u32
            };
            arc_millis[slot] = total;
            if let Some(partials) = &mut partials {
                // The endpoint components round to the nearest millisecond;
                // the base absorbs the ceil so the three sum to the total.
                let head = ((head * 1000.0).round() as u32).min(total);
                let tail = ((tail * 1000.0).round() as u32).min(total - head);
                partials.head[slot] = head;
                partials.tail[slot] = tail;
                partials.base[slot] = total - head - tail;
            }
        }
        Ok(CompiledStreetProfile {
            definition: definition.clone(),
            arc_millis,
            car_partials: partials,
            max_effective_speed: std::sync::OnceLock::new(),
        })
    }

    /// The distance-weighted mean slope multiplier per edge and direction
    /// (`(forward, reverse)`), integrating `1 + f(s)` over the stored
    /// sub-segments — `None` when the network carries no elevations or the
    /// definition's slope is off, which compiles flat.
    ///
    /// A sub-segment with unavailable elevation (NaN) or no length counts as
    /// flat in both directions: unavailable data never invents a penalty or a
    /// credit, so an elevation-less profile means multiplier 1.0 exactly and
    /// the compiled costs are bit-identical to a flat compilation.
    fn slope_means(&self, definition: &StreetProfileDefinition) -> Option<Vec<(f64, f64)>> {
        if definition.slope_uphill == 0.0 && definition.slope_downhill == 0.0 {
            return None;
        }
        let elevations = self.elevations()?;
        let offsets = self.arrays().coordinate_offsets();
        let cumulative = self.arrays().cumulative();
        let multiplier = |slope: f64| -> f64 {
            let f = if slope > 0.0 {
                definition.slope_uphill * slope
            } else {
                definition.slope_downhill * slope
            };
            (1.0 + f).max(MIN_SLOPE_MULTIPLIER)
        };
        let mut means = Vec::with_capacity(offsets.len() - 1);
        for edge in 0..offsets.len() - 1 {
            let start = offsets[edge] as usize;
            let end = offsets[edge + 1] as usize;
            let (mut total, mut forward, mut reverse) = (0.0f64, 0.0f64, 0.0f64);
            for point in start..end.saturating_sub(1) {
                let length = f64::from(cumulative[point + 1]) - f64::from(cumulative[point]);
                if length.is_nan() || length <= 0.0 {
                    continue;
                }
                let rise = f64::from(elevations[point + 1]) - f64::from(elevations[point]);
                let (up, down) = if rise.is_nan() {
                    (1.0, 1.0)
                } else {
                    let slope = (rise / length).clamp(-MAX_SLOPE, MAX_SLOPE);
                    (multiplier(slope), multiplier(-slope))
                };
                total += length;
                forward += length * up;
                reverse += length * down;
            }
            means.push(if total > 0.0 {
                (forward / total, reverse / total)
            } else {
                (1.0, 1.0)
            });
        }
        Some(means)
    }

    /// Whether each adjacency slot traverses its edge against the stored
    /// geometry direction, from the CSR: a slot under vertex `v` over edge
    /// `e` is forward iff `v` is the edge's stored `from` endpoint. A
    /// self-loop reads as forward both ways — its two arcs traverse the same
    /// profile.
    fn slot_directions(&self) -> Vec<bool> {
        let adjacency = self.arrays().adjacency_offsets();
        let edges = self.arrays().adj_edges();
        let endpoints = self.arrays().endpoints();
        let mut reverse = vec![false; edges.len()];
        for vertex in 0..adjacency.len() - 1 {
            for slot in adjacency[vertex] as usize..adjacency[vertex + 1] as usize {
                reverse[slot] = endpoints[2 * edges[slot] as usize] != vertex as u32;
            }
        }
        reverse
    }
}

/// A class multiplier by code. Every attribute path validates codes against
/// the table sizes (`check_street_attributes`) and every profile validates its
/// table lengths, so the code is always in range — indexed directly rather
/// than silently defaulting an out-of-range (ABI-drifted) code to neutral.
fn multiplier(multipliers: &[f64], code: u8) -> f64 {
    multipliers[code as usize]
}
