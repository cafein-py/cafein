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
