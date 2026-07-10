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
