use std::collections::HashMap;

use super::*;

fn walking(stop_count: u32, edges: &[(u32, u32, u32, f64)]) -> Transfers {
    let edges: Vec<(StopIdx, StopIdx, u32, f64)> = edges
        .iter()
        .map(|&(from, to, seconds, meters)| (StopIdx(from), StopIdx(to), seconds, meters))
        .collect();
    Transfers::from_edges(stop_count, &edges).unwrap()
}

fn rentals(stop_count: u32, edges: &[(u32, u32, u32, f64)]) -> Vec<Vec<RentalEdge>> {
    let mut rows = vec![Vec::new(); stop_count as usize];
    for &(from, to, seconds, meters) in edges {
        rows[from as usize].push(RentalEdge {
            to,
            seconds,
            network_meters: meters,
            total_meters: meters + 10.0,
        });
    }
    rows
}

fn merged_map(edges: &[MergedEdge]) -> HashMap<(u32, u32), (u32, f64)> {
    edges
        .iter()
        .map(|&(from, to, seconds, meters)| ((from.0, to.0), (seconds, meters)))
        .collect()
}

#[test]
fn a_faster_rental_wins_the_pair_and_keeps_its_token() {
    // Walking 0→1 takes 600 s; the scooter rides it in 200 s.
    let walking = walking(2, &[(0, 1, 600, 500.0)]);
    let rentals = rentals(2, &[(0, 1, 200, 900.0)]);
    let (edges, tokens) = merge_mode_transfers(&walking, &rentals, 2, 3600);
    let merged = merged_map(&edges);
    assert_eq!(merged[&(0, 1)].0, 200);
    let token = tokens[&(0, 1)];
    assert_eq!(token.pickup, StopIdx(0));
    assert_eq!(token.drop, StopIdx(1));
    assert_eq!(token.ride_seconds, 200);
    assert_eq!(token.pre_seconds, 0);
    assert_eq!(token.post_seconds, 0);
}

#[test]
fn an_equal_time_rental_loses_to_walking() {
    let walking = walking(2, &[(0, 1, 300, 250.0)]);
    let rentals = rentals(2, &[(0, 1, 300, 800.0)]);
    let (edges, tokens) = merge_mode_transfers(&walking, &rentals, 2, 3600);
    let merged = merged_map(&edges);
    assert_eq!(merged[&(0, 1)], (300, 250.0));
    assert!(tokens.is_empty());
}

#[test]
fn the_closure_walks_around_one_rental_at_most() {
    // 0 -walk- 1 -ride- 2 -walk- 3: the composed edge 0→3 exists with
    // the walking split recorded; a second ride 3→4 must not chain into
    // 0→4 through the first rental.
    let walking = walking(5, &[(0, 1, 100, 80.0), (2, 3, 120, 100.0)]);
    let rentals = rentals(5, &[(1, 2, 200, 700.0), (3, 4, 150, 500.0)]);
    let (edges, tokens) = merge_mode_transfers(&walking, &rentals, 5, 3600);
    let merged = merged_map(&edges);
    assert_eq!(merged[&(0, 3)].0, 100 + 200 + 120);
    let token = tokens[&(0, 3)];
    assert_eq!(token.pickup, StopIdx(1));
    assert_eq!(token.drop, StopIdx(2));
    assert_eq!(token.pre_seconds, 100);
    assert_eq!(token.post_seconds, 120);
    // One rental per transfer: 0→4 would need two.
    assert!(!merged.contains_key(&(0, 4)));
    // 3→4 stands on its own.
    assert_eq!(merged[&(3, 4)].0, 150);
}

#[test]
fn the_cutoff_bounds_rental_transfers_only() {
    let walking = walking(3, &[(0, 1, 500, 400.0)]);
    let rentals = rentals(3, &[(1, 2, 600, 2000.0)]);
    let (edges, _) = merge_mode_transfers(&walking, &rentals, 3, 1000);
    let merged = merged_map(&edges);
    assert!(merged.contains_key(&(0, 1)));
    // 500 + 600 exceeds the 1000 s budget for the rental-bearing
    // movement.
    assert!(!merged.contains_key(&(0, 2)));
    // The walking closure keeps its own budget: a walking row far past
    // the rental cutoff survives verbatim, or the merged set would be
    // weaker than the set it extends.
    let (edges, tokens) = merge_mode_transfers(&walking, &rentals, 3, 100);
    let merged = merged_map(&edges);
    assert_eq!(merged[&(0, 1)], (500, 400.0));
    assert!(tokens.is_empty());
}

#[test]
fn walking_edges_survive_untouched_where_no_rental_reaches() {
    let walking = walking(
        3,
        &[(0, 1, 300, 250.0), (1, 2, 200, 160.0), (0, 2, 500, 410.0)],
    );
    let rentals = vec![Vec::new(); 3];
    let (edges, tokens) = merge_mode_transfers(&walking, &rentals, 3, 3600);
    let merged = merged_map(&edges);
    assert_eq!(merged[&(0, 1)], (300, 250.0));
    assert_eq!(merged[&(0, 2)], (500, 410.0));
    assert!(tokens.is_empty());
}

/// The pre-walk is one closure row, never a chain: rows 0→1 and 1→2
/// exist but 0→2 does not (a real closure lacking it means the walk
/// exceeds its cutoff), so no merged edge composes 0-walk-walk-ride-3 —
/// while the covered single-row movement from 1 still merges.
#[test]
fn walks_never_chain_past_the_closure() {
    let walking = Transfers::from_edges(
        4,
        &[
            (StopIdx(0), StopIdx(1), 100, 100.0),
            (StopIdx(1), StopIdx(2), 100, 100.0),
        ],
    )
    .unwrap();
    let mut rentals = vec![Vec::new(); 4];
    rentals[2].push(RentalEdge {
        to: 3,
        seconds: 50,
        network_meters: 40.0,
        total_meters: 45.0,
    });
    let (edges, tokens) = merge_mode_transfers(&walking, &rentals, 4, 600);
    assert!(!edges
        .iter()
        .any(|&(from, to, _, _)| (from, to) == (StopIdx(0), StopIdx(3))));
    let ride = edges
        .iter()
        .find(|&&(from, to, _, _)| (from, to) == (StopIdx(1), StopIdx(3)))
        .unwrap();
    assert_eq!(ride.2, 150);
    assert_eq!(tokens[&(1, 3)].pre_seconds, 100);
}

/// The carriage merge: a strictly faster ride replaces the walking row
/// and is flagged as a winner; an equal-time ride loses to walking; a
/// ride-only pair is added. Rows never mix modes.
#[test]
fn carriage_rides_win_strictly_and_tie_to_walking() {
    let walking = Transfers::from_edges(
        4,
        &[
            (StopIdx(0), StopIdx(1), 100, 100.0),
            (StopIdx(0), StopIdx(2), 200, 200.0),
        ],
    )
    .unwrap();
    let mut rides = vec![Vec::new(); 4];
    rides[0].push(RentalEdge {
        to: 1,
        seconds: 100,
        network_meters: 300.0,
        total_meters: 320.0,
    });
    rides[0].push(RentalEdge {
        to: 2,
        seconds: 150,
        network_meters: 500.0,
        total_meters: 520.0,
    });
    rides[0].push(RentalEdge {
        to: 3,
        seconds: 400,
        network_meters: 1500.0,
        total_meters: 1520.0,
    });
    let (edges, winners) = merge_carriage_transfers(&walking, &rides, 4);
    let mut edges = edges;
    edges.sort_unstable_by_key(|&(from, to, _, _)| (from, to));
    assert_eq!(
        edges,
        vec![
            (StopIdx(0), StopIdx(1), 100, 100.0),
            (StopIdx(0), StopIdx(2), 150, 520.0),
            (StopIdx(0), StopIdx(3), 400, 1520.0),
        ]
    );
    assert_eq!(winners.len(), 2);
    assert_eq!(winners[&(0, 2)], 500.0);
    assert_eq!(winners[&(0, 3)], 1500.0);
}
