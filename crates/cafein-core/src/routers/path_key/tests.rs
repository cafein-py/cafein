use super::{challenger_wins, PathToken};

#[test]
fn later_root_departures_dominate_topology() {
    let ride = [PathToken::Ride {
        trip: 9,
        day_offset: 0,
        board: 0,
        alight: 1,
    }];
    let walk = [PathToken::Walk {
        from: 0,
        to: 1,
        duration: 60,
    }];
    assert!(challenger_wins(100, &walk, 50, &ride));
    assert!(!challenger_wins(50, &ride, 100, &walk));
    // Equal roots: a ride beats a walk, then fields decide.
    assert!(challenger_wins(50, &ride, 50, &walk));
    let earlier_trip = [PathToken::Ride {
        trip: 3,
        day_offset: 0,
        board: 0,
        alight: 1,
    }];
    assert!(challenger_wins(50, &earlier_trip, 50, &ride));
}
