use super::*;

fn leg(route: u32, board_time: u32) -> FareLeg {
    FareLeg {
        route,
        board_stop: 0,
        alight_stop: 0,
        board_time,
    }
}

#[track_caller]
fn assert_price(price: f64, expected: f64) {
    assert!(
        (price - expected).abs() < 1e-9,
        "price {price} != {expected}"
    );
}

/// Routes 0–1 are buses at 4.80, route 2 rail at 4.50 with
/// unlimited transfers, route 3 has no fare row, route 4 a fare
/// row without a price — the r5r vignette's shape.
fn vignette() -> RuleFares {
    RuleFares {
        route_type: vec![0, 0, 1, NO_FARE, 0],
        route_fare: vec![4.8, 4.8, 4.5, f64::NAN, f64::NAN],
        unlimited_transfers: vec![false, true],
        allow_same_route: vec![false, false],
        pair_fare: vec![7.2, 8.37, 8.37, f64::NAN],
        max_discounted_transfers: 1,
        transfer_allowance: 3600.0,
        fare_cap: f64::INFINITY,
    }
}

#[test]
fn rule_fares_follow_the_vignette() {
    let fares = vignette();
    assert_price(fares.price(&[]), 0.0);
    assert_price(fares.price(&[leg(0, 0)]), 4.8);
    assert_price(fares.price(&[leg(0, 0), leg(1, 1800)]), 7.2);
    assert_price(fares.price(&[leg(0, 0), leg(1, 3601)]), 9.6);
    assert_price(fares.price(&[leg(0, 0), leg(2, 1800)]), 8.37);
    // Rail rides after rail are free and re-anchor the integration.
    assert_price(fares.price(&[leg(2, 0), leg(2, 1800)]), 4.5);
    assert_price(fares.price(&[leg(2, 0), leg(2, 1800), leg(0, 3000)]), 8.37);
    // One discount only; reboarding the same bus route pays full.
    assert_price(fares.price(&[leg(0, 0), leg(1, 1200), leg(0, 2400)]), 12.0);
    assert_price(fares.price(&[leg(0, 0), leg(0, 1800)]), 9.6);
    assert!(fares.price(&[leg(3, 0)]).is_nan());
    assert!(fares.price(&[leg(4, 0)]).is_nan());
}

#[test]
fn rule_fares_cap_the_total() {
    let fares = RuleFares {
        fare_cap: 8.0,
        ..vignette()
    };
    assert_price(fares.price(&[leg(0, 0), leg(1, 1200), leg(0, 2400)]), 8.0);
    // The cap never turns an unpriceable journey into a price.
    assert!(fares.price(&[leg(4, 0)]).is_nan());
}

#[test]
fn zone_fares_chain_the_cheapest_tickets() {
    // Zones A=0, B=1; stops 0–1 in A, stop 2 in B, stop 3 zoneless.
    let fares = ZoneFares {
        stop_zone: vec![0, 0, 1, NO_FARE],
        products: vec![
            ZoneProduct {
                price: 2.8,
                zones: 0b11,
                duration: 4800.0,
                transfers: NO_FARE,
            },
            ZoneProduct {
                price: 2.0,
                zones: 0b10,
                duration: 4800.0,
                transfers: NO_FARE,
            },
        ],
    };
    let ride = |board: u32, alight: u32, at: u32| FareLeg {
        route: 0,
        board_stop: board,
        alight_stop: alight,
        board_time: at,
    };
    assert_price(fares.price(&[]), 0.0);
    assert_price(fares.price(&[ride(0, 1, 0)]), 2.8);
    assert_price(fares.price(&[ride(2, 2, 0)]), 2.0);
    assert_price(fares.price(&[ride(0, 2, 0)]), 2.8);
    // Two boardings inside the window ride one ticket; a boarding
    // beyond it buys a second one.
    assert_price(fares.price(&[ride(0, 1, 0), ride(1, 0, 1800)]), 2.8);
    assert_price(fares.price(&[ride(0, 1, 0), ride(1, 0, 7200)]), 5.6);
    // The cheaper single-zone ticket wins where it suffices.
    assert_price(fares.price(&[ride(0, 1, 0), ride(2, 2, 7200)]), 4.8);
    assert!(fares.price(&[ride(0, 3, 0)]).is_nan());
}

#[test]
fn zone_fares_respect_transfer_counts() {
    let fares = ZoneFares {
        stop_zone: vec![0],
        products: vec![ZoneProduct {
            price: 1.0,
            zones: 0b1,
            duration: f64::INFINITY,
            transfers: 1,
        }],
    };
    let ride = |at: u32| FareLeg {
        route: 0,
        board_stop: 0,
        alight_stop: 0,
        board_time: at,
    };
    assert_price(fares.price(&[ride(0), ride(1)]), 1.0);
    assert_price(fares.price(&[ride(0), ride(1), ride(2)]), 2.0);
}

/// Folding the incremental steps equals the journey pricer, branch
/// for branch, over randomized boarding sequences (a deterministic
/// LCG; both NaN counts as equal).
#[test]
fn stepping_the_state_machine_equals_the_price() {
    let fares = vignette();
    let capped = RuleFares {
        fare_cap: 8.0,
        ..vignette()
    };
    let mut seed: u64 = 0x5eed;
    let mut next = move || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (seed >> 33) as u32
    };
    for tables in [&fares, &capped] {
        for _ in 0..500 {
            let length = 1 + (next() % 5) as usize;
            let mut legs = Vec::with_capacity(length);
            let mut time = 0;
            for _ in 0..length {
                time += next() % 3000;
                legs.push(leg(next() % 5, time));
            }
            let expected = tables.price(&legs);
            let mut state = tables.board_first(legs[0].route, legs[0].board_time);
            for ride in &legs[1..] {
                state =
                    state.and_then(|state| tables.board_next(&state, ride.route, ride.board_time));
            }
            let folded = state.map_or(f64::NAN, |state| tables.capped_total(&state));
            assert!(
                (folded.is_nan() && expected.is_nan()) || (folded - expected).abs() < 1e-9,
                "fold {folded} != price {expected} for {legs:?}"
            );
        }
    }
}

/// The plan's dominance counterexamples: a higher previous full fare
/// only helps (the integration increment shrinks), a fresher window
/// only helps, and states differing on route or type never compare.
#[test]
fn state_dominance_orders_the_continuations() {
    let fares = vignette();
    // Two labels equal but for the previous full fare: the higher one
    // dominates, and after one more boarding it is indeed cheaper.
    let high = FareState {
        total: 6.0,
        previous_type: 0,
        previous_route: 0,
        previous_fare: 4.8,
        previous_board: 0,
        discounts: 0,
    };
    let low = FareState {
        previous_fare: 2.0,
        ..high
    };
    assert!(state_dominates(&high, &low, true, true));
    assert!(!state_dominates(&low, &high, true, true));
    let high_next = fares.board_next(&high, 1, 1800).unwrap();
    let low_next = fares.board_next(&low, 1, 1800).unwrap();
    assert!(high_next.total < low_next.total);
    // A fresher window dominates an equal-but-staler one, and only
    // the fresher label still integrates after the stale window ends.
    let fresh = FareState {
        previous_board: 3000,
        ..high
    };
    assert!(state_dominates(&fresh, &high, true, true));
    // Freshness needs the ample-budget gate; without it, equality.
    assert!(!state_dominates(&fresh, &high, true, false));
    let fresh_next = fares.board_next(&fresh, 1, 3601).unwrap();
    let stale_next = fares.board_next(&high, 1, 3601).unwrap();
    assert!(fresh_next.total < stale_next.total);
    // Different previous route or type: no comparison either way.
    let other_route = FareState {
        previous_route: 1,
        ..high
    };
    assert!(!state_dominates(&high, &other_route, true, true));
    assert!(!state_dominates(&other_route, &high, true, true));
    // A NaN total neither dominates nor is dominated — once NaN,
    // always NaN, so the engine drops such labels at creation.
    let unpriced = FareState {
        total: f64::NAN,
        ..high
    };
    assert!(!state_dominates(&unpriced, &high, true, true));
    assert!(!state_dominates(&high, &unpriced, true, true));
}

/// The pruning margin bounds every achievable single-integration
/// saving: on the vignette, integrating two buses saves
/// 4.8 + 4.8 − 7.2 = 2.4, and no sequence saves more per discount.
#[test]
fn the_discount_margin_bounds_the_savings() {
    let fares = vignette();
    let margin = fares.max_discount_margin();
    assert!((margin - 2.4).abs() < 1e-9, "margin {margin}");
    // A full-fare two-bus journey minus its integrated price never
    // exceeds the margin.
    let full = fares.price(&[leg(0, 0), leg(1, 3601)]);
    let integrated = fares.price(&[leg(0, 0), leg(1, 1800)]);
    assert!(full - integrated <= margin + 1e-9);
}

/// A pair total above the full fares it replaces makes unspent
/// discount capacity a liability: the tables report non-monotone
/// discounts, `discounts ≤` stops comparing, and the forced expensive
/// integration really does invert the ordering.
#[test]
fn harmful_pairs_disable_the_discount_axis() {
    let fares = vignette();
    assert!(fares.discounts_are_monotone());
    let harmful = RuleFares {
        pair_fare: vec![12.0, 8.37, 8.37, f64::NAN],
        ..vignette()
    };
    assert!(!harmful.discounts_are_monotone());
    let fresh = FareState {
        total: 4.8,
        previous_type: 0,
        previous_route: 0,
        previous_fare: 4.8,
        previous_board: 0,
        discounts: 0,
    };
    let spent = FareState {
        discounts: 1,
        ..fresh
    };
    // Under monotone tables the fresh label would dominate; under the
    // harmful tables only equal spent discounts compare.
    assert!(state_dominates(&fresh, &spent, true, true));
    assert!(!state_dominates(&fresh, &spent, false, false));
    // And the inversion is real: the fresh label is forced into the
    // 12.0 pair while the exhausted one pays 4.8 in full.
    let fresh_next = harmful.board_next(&fresh, 1, 1800).unwrap();
    let spent_next = harmful.board_next(&spent, 1, 1800).unwrap();
    assert!(fresh_next.total > spent_next.total);
}

/// With a scarce discount budget even monotone tables make window
/// freshness unsafe: the fresher label is forced to spend its one
/// discount on a weak integration a staler label saves for the big
/// one two boardings later.
#[test]
fn scarce_discounts_disable_the_freshness_axis() {
    // Two types: buses (4.8, routes 0-1) and rail (4.5, route 2).
    // Bus→bus integrates at 9.5 (saving 0.1), bus→rail at 5.0
    // (saving 4.3); both pairs are monotone.
    let fares = RuleFares {
        route_type: vec![0, 0, 1],
        route_fare: vec![4.8, 4.8, 4.5],
        unlimited_transfers: vec![false, false],
        allow_same_route: vec![false, false],
        pair_fare: vec![9.5, 5.0, f64::NAN, f64::NAN],
        max_discounted_transfers: 1,
        transfer_allowance: 600.0,
        fare_cap: f64::INFINITY,
    };
    assert!(fares.discounts_are_monotone());
    let fresh = FareState {
        total: 4.8,
        previous_type: 0,
        previous_route: 0,
        previous_fare: 4.8,
        previous_board: 1000,
        discounts: 0,
    };
    let stale = FareState {
        previous_board: 0,
        ..fresh
    };
    // Boarding a bus at 1400 (inside only the fresh window), then
    // rail at 1800: the stale label finishes cheaper.
    let fresh_end = fares
        .board_next(&fares.board_next(&fresh, 1, 1400).unwrap(), 2, 1800)
        .unwrap();
    let stale_end = fares
        .board_next(&fares.board_next(&stale, 1, 1400).unwrap(), 2, 1800)
        .unwrap();
    assert!(stale_end.total < fresh_end.total);
    // So freshness compares only under an ample budget; with the
    // scarce one, equal boarding times are required.
    assert!(!state_dominates(&fresh, &stale, true, false));
    assert!(state_dominates(&fresh, &stale, true, true));
    // And with the budget covering every boarding, freshness really
    // is safe: the same sequences keep the fresh label no dearer.
    let ample = RuleFares {
        max_discounted_transfers: 99,
        ..fares
    };
    let fresh_end = ample
        .board_next(&ample.board_next(&fresh, 1, 1400).unwrap(), 2, 1800)
        .unwrap();
    let stale_end = ample
        .board_next(&ample.board_next(&stale, 1, 1400).unwrap(), 2, 1800)
        .unwrap();
    assert!(fresh_end.total <= stale_end.total);
}
