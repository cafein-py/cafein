//! Journey pricing over reconstructed candidate legs.
//!
//! Fares are journey-level: discounts, transfer windows, zone extents,
//! and caps make the price a function of the whole leg sequence and its
//! timing. Like emissions, they never enter the routing loop — a
//! candidate is priced at reconstruction time from the legs its label
//! chain yields.
//!
//! The two models mirror `cafein.fares`: the r5r-style rule-based
//! calculator and GTFS zone-set products. Python resolves identifiers,
//! types, and zones into the flat arrays here; pricing is pure
//! arithmetic. NaN marks a journey the model cannot price (a route
//! without a fare row, a stop without a zone).

/// A route without a fare row, a stop without a zone.
pub const NO_FARE: u32 = u32::MAX;

/// One transit leg of a candidate journey, in ride order.
#[derive(Debug, Clone, Copy)]
pub struct FareLeg {
    /// The ridden pattern's route index.
    pub route: u32,
    /// The boarding stop's index.
    pub board_stop: u32,
    /// The alighting stop's index.
    pub alight_stop: u32,
    /// The boarding time on the queried day's clock, in seconds.
    pub board_time: u32,
}

/// A flattened fare model.
pub enum FareTables {
    RuleBased(RuleFares),
    Zone(ZoneFares),
}

impl FareTables {
    /// The price of a journey riding `legs`, in order; an empty slice
    /// (a walking-only journey) is free.
    pub fn price(&self, legs: &[FareLeg]) -> f64 {
        match self {
            FareTables::RuleBased(tables) => tables.price(legs),
            FareTables::Zone(tables) => tables.price(legs),
        }
    }
}

/// The r5r-style rule-based fare model (`FareStructure` in Python),
/// with per-route full fares resolved ahead of time.
pub struct RuleFares {
    /// Per route: index into the type arrays; `NO_FARE` marks a route
    /// without a fare row.
    pub route_type: Vec<u32>,
    /// Per route: the resolved full fare (the route or type fare).
    pub route_fare: Vec<f64>,
    /// Per type: rides of the same type are free after the first.
    pub unlimited_transfers: Vec<bool>,
    /// Per type: a discounted transfer may return to the same route.
    pub allow_same_route: Vec<bool>,
    /// `type_count²` ordered pair totals, first type major; NaN marks a
    /// pair without an integration fare.
    pub pair_fare: Vec<f64>,
    /// How many transfers may price as integrations.
    pub max_discounted_transfers: u32,
    /// Seconds between boardings within which an integration applies.
    pub transfer_allowance: f64,
    /// Ceiling on the journey total (infinite: uncapped).
    pub fare_cap: f64,
}

impl RuleFares {
    /// Mirrors `FareStructure.price`: the first ride pays its full
    /// fare; each further ride pays in full unless its type allows
    /// unlimited transfers (same type: free) or an in-time discounted
    /// transfer applies, in which case the pair total replaces the two
    /// full fares.
    pub fn price(&self, legs: &[FareLeg]) -> f64 {
        let Some((first, rest)) = legs.split_first() else {
            return 0.0;
        };
        let count = self.unlimited_transfers.len();
        let mut previous_type = self.route_type[first.route as usize];
        if previous_type == NO_FARE {
            return f64::NAN;
        }
        let mut previous_fare = self.route_fare[first.route as usize];
        let mut total = previous_fare;
        let mut previous_route = first.route;
        let mut previous_board = first.board_time;
        let mut discounts = 0;
        for ride in rest {
            let kind = self.route_type[ride.route as usize];
            if kind == NO_FARE {
                return f64::NAN;
            }
            let fare = self.route_fare[ride.route as usize];
            // Rides within an unlimited-transfers type are free and
            // spend neither a discount nor the transfer clock; a later
            // integration prices off this ride's route.
            if kind == previous_type && self.unlimited_transfers[kind as usize] {
                previous_route = ride.route;
                previous_fare = fare;
                continue;
            }
            let pair = self.pair_fare[previous_type as usize * count + kind as usize];
            let allowed = kind != previous_type
                || self.allow_same_route[kind as usize]
                || ride.route != previous_route;
            let in_time = ride.board_time as f64 - previous_board as f64 <= self.transfer_allowance;
            if discounts < self.max_discounted_transfers && !pair.is_nan() && allowed && in_time {
                // The pair price is the total of both legs; the first
                // leg's full fare is already counted.
                total += pair - previous_fare;
                discounts += 1;
            } else {
                total += fare;
            }
            previous_fare = fare;
            previous_type = kind;
            previous_route = ride.route;
            previous_board = ride.board_time;
        }
        // `min` would coerce a NaN total (a fare row without a price)
        // into the cap; unpriceable stays unpriceable.
        if total.is_nan() {
            return f64::NAN;
        }
        total.min(self.fare_cap)
    }
}

/// The continuation state of a partially priced journey under the
/// rule-based model: everything the calculator's future increments
/// read, plus the running (uncapped) total. The frontier engine's
/// labels carry one of these; `fare_cap` applies at result time,
/// never mid-search.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FareState {
    /// The running total, uncapped; NaN once any boarding was
    /// unpriceable by amount (a fare row without a price).
    pub total: f64,
    pub previous_type: u32,
    pub previous_route: u32,
    pub previous_fare: f64,
    pub previous_board: u32,
    pub discounts: u32,
}

impl RuleFares {
    /// The state after a journey's first boarding; `None` when the
    /// route has no fare row.
    pub fn board_first(&self, route: u32, board_time: u32) -> Option<FareState> {
        let kind = self.route_type[route as usize];
        if kind == NO_FARE {
            return None;
        }
        let fare = self.route_fare[route as usize];
        Some(FareState {
            total: fare,
            previous_type: kind,
            previous_route: route,
            previous_fare: fare,
            previous_board: board_time,
            discounts: 0,
        })
    }

    /// The state after one further boarding — the incremental form of
    /// [`RuleFares::price`], branch for branch; `None` when the route
    /// has no fare row.
    pub fn board_next(&self, state: &FareState, route: u32, board_time: u32) -> Option<FareState> {
        let kind = self.route_type[route as usize];
        if kind == NO_FARE {
            return None;
        }
        let fare = self.route_fare[route as usize];
        let mut next = *state;
        // Rides within an unlimited-transfers type are free and spend
        // neither a discount nor the transfer clock; a later
        // integration prices off this ride's route.
        if kind == state.previous_type && self.unlimited_transfers[kind as usize] {
            next.previous_route = route;
            next.previous_fare = fare;
            return Some(next);
        }
        let count = self.unlimited_transfers.len();
        let pair = self.pair_fare[state.previous_type as usize * count + kind as usize];
        let allowed = kind != state.previous_type
            || self.allow_same_route[kind as usize]
            || route != state.previous_route;
        let in_time = board_time as f64 - state.previous_board as f64 <= self.transfer_allowance;
        if state.discounts < self.max_discounted_transfers && !pair.is_nan() && allowed && in_time {
            next.total += pair - state.previous_fare;
            next.discounts += 1;
        } else {
            next.total += fare;
        }
        next.previous_fare = fare;
        next.previous_type = kind;
        next.previous_route = route;
        next.previous_board = board_time;
        Some(next)
    }

    /// The journey total a state reports: the cap applied, NaN kept.
    pub fn capped_total(&self, state: &FareState) -> f64 {
        if state.total.is_nan() {
            return f64::NAN;
        }
        state.total.min(self.fare_cap)
    }

    /// The largest single integration saving the tables allow — the
    /// sound per-remaining-discount pruning margin: a label may still
    /// save at most `margin × remaining discounts` off its running
    /// total.
    pub fn max_discount_margin(&self) -> f64 {
        let count = self.unlimited_transfers.len();
        let mut margin: f64 = 0.0;
        for first in 0..count {
            for second in 0..count {
                let pair = self.pair_fare[first * count + second];
                if pair.is_nan() {
                    continue;
                }
                // The increment `pair − previous_fare` replaces the
                // full fare `fare`; the saving against paying in full
                // is `fare + previous_fare − pair`, maximised over the
                // full fares the types can carry.
                let best_first = self.best_full_fare(first as u32);
                let best_second = self.best_full_fare(second as u32);
                margin = margin.max((best_first + best_second - pair).max(0.0));
            }
        }
        margin
    }

    fn best_full_fare(&self, kind: u32) -> f64 {
        let mut best: f64 = 0.0;
        for (route, &route_kind) in self.route_type.iter().enumerate() {
            if route_kind == kind && !self.route_fare[route].is_nan() {
                best = best.max(self.route_fare[route]);
            }
        }
        best
    }

    fn cheapest_full_fare(&self, kind: u32) -> f64 {
        let mut cheapest = f64::INFINITY;
        for (route, &route_kind) in self.route_type.iter().enumerate() {
            if route_kind == kind && !self.route_fare[route].is_nan() {
                cheapest = cheapest.min(self.route_fare[route]);
            }
        }
        cheapest
    }

    /// Whether spending a discount can never cost more than paying in
    /// full — the calculator *forces* the integration branch, so a
    /// pair total above the cheapest full fares it replaces makes
    /// unspent discount capacity a liability, and `discounts ≤` stops
    /// being a safe dominance axis. Computed once per query.
    pub fn discounts_are_monotone(&self) -> bool {
        let count = self.unlimited_transfers.len();
        for first in 0..count {
            for second in 0..count {
                let pair = self.pair_fare[first * count + second];
                if pair.is_nan() {
                    continue;
                }
                let floor =
                    self.cheapest_full_fare(first as u32) + self.cheapest_full_fare(second as u32);
                if pair > floor {
                    return false;
                }
            }
        }
        true
    }
}

/// The fare-state half of the frontier dominance relation: `a` prices
/// every continuation at most as high as `b`. Continuation cost reads
/// only the fields compared here — equal previous type and route keep
/// the branch structure identical, a lower total and fewer spent
/// discounts never price worse, a later previous boarding keeps a
/// fresher window, and a higher previous full fare only shrinks the
/// integration increment `pair − previous_fare`. The caller conjoins
/// arrival ≤. Because the calculator *forces* the integration branch,
/// two axes need table-dependent gates:
/// `discounts_monotone` is [`RuleFares::discounts_are_monotone`] —
/// when the tables guarantee an integration never costs more than
/// paying full, fewer spent discounts is a safe ≤ axis; otherwise
/// only **equal** spent discounts compare. `freshness_monotone` may
/// only be true when `discounts_monotone` holds **and** the discount
/// budget covers every boarding the query can make
/// (`max_discounted_transfers ≥` the ride bound) — then a fresher
/// window's forced integrations are pointwise no dearer; with a
/// scarce budget a fresher window can squander the last discount on a
/// weak integration a staler label saves for a larger one, so only
/// **equal** previous boarding times compare. NaN totals neither
/// dominate nor are dominated — a NaN running total never recovers,
/// so the engine drops such labels at creation rather than relying on
/// dominance.
pub fn state_dominates(
    a: &FareState,
    b: &FareState,
    discounts_monotone: bool,
    freshness_monotone: bool,
) -> bool {
    let discounts_ok = if discounts_monotone {
        a.discounts <= b.discounts
    } else {
        a.discounts == b.discounts
    };
    let freshness_ok = if freshness_monotone {
        a.previous_board >= b.previous_board
    } else {
        a.previous_board == b.previous_board
    };
    a.previous_type == b.previous_type
        && a.previous_route == b.previous_route
        && a.total <= b.total
        && discounts_ok
        && freshness_ok
        && a.previous_fare >= b.previous_fare
}

/// A zone ticket: a price valid for the zones in the bitmask, for
/// `transfers` further boardings within `duration` seconds of the
/// first.
#[derive(Debug, Clone, Copy)]
pub struct ZoneProduct {
    pub price: f64,
    /// Bitmask over the model's zone indexes.
    pub zones: u128,
    /// Seconds of validity from the first covered boarding; infinite
    /// when the feed sets no window.
    pub duration: f64,
    /// Boardings after the first; `NO_FARE` when unlimited.
    pub transfers: u32,
}

/// The GTFS zone-set fare model (`ZoneFareStructure` in Python).
pub struct ZoneFares {
    /// Per stop: the stop's zone index; `NO_FARE` marks a stop without
    /// a zone.
    pub stop_zone: Vec<u32>,
    pub products: Vec<ZoneProduct>,
}

impl ZoneFares {
    /// Mirrors `ZoneFareStructure.price`: the cheapest chain of
    /// tickets in which each ticket covers the zones of every leg it
    /// spans (a leg contributes its boarding and alighting stops'
    /// zones) within its window and transfer count.
    pub fn price(&self, legs: &[FareLeg]) -> f64 {
        if legs.is_empty() {
            return 0.0;
        }
        let mut needs = Vec::with_capacity(legs.len());
        for leg in legs {
            let board = self.stop_zone[leg.board_stop as usize];
            let alight = self.stop_zone[leg.alight_stop as usize];
            if board == NO_FARE || alight == NO_FARE {
                return f64::NAN;
            }
            needs.push(((1u128 << board) | (1u128 << alight), leg.board_time));
        }
        // cost[at] = the cheapest chain covering legs at.. — a ticket
        // covers a forward stretch, so the table fills back to front.
        let count = needs.len();
        let mut cost = vec![f64::NAN; count + 1];
        cost[count] = 0.0;
        for at in (0..count).rev() {
            let mut cheapest = f64::NAN;
            for product in &self.products {
                if needs[at].0 & !product.zones != 0 {
                    continue;
                }
                // The ticket covers boardings within its window (and
                // its transfer count), as far as the zones allow.
                let mut end = at;
                while end + 1 < count
                    && needs[end + 1].0 & !product.zones == 0
                    && (needs[end + 1].1 - needs[at].1) as f64 <= product.duration
                    && (end + 1 - at) as u32 <= product.transfers
                {
                    end += 1;
                }
                for split in at..=end {
                    let candidate = product.price + cost[split + 1];
                    if cheapest.is_nan() || candidate < cheapest {
                        cheapest = candidate;
                    }
                }
            }
            cost[at] = cheapest;
        }
        cost[0]
    }
}

#[cfg(test)]
#[path = "fares_tests.rs"]
mod tests;
