//! The merged mode-transfer set: shared-vehicle stop-to-stop transfers
//! beside the walking closure, projected into the [`Transfers`] shape the
//! engines already relax — the time-only `ModeTransfers` of the design.
//!
//! A merged transfer uses at most one rental — a transfer is one
//! pickup–travel–drop-off — and its pre- and post-walk are each **one
//! walking-closure row**: within the walking cutoff a chain of rows is
//! already covered by its direct row, and a longer chain would compose
//! a walk the closure itself refuses. The per-stop bounded search is
//! thus walk-row? ∘ rental ∘ walk-row? under the transfer budget. Ties
//! fall to walking, then the earlier-seeded rental; the walking hot
//! path never changes — the merged set is a second [`Transfers`] the
//! policy queries relax.

use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::collections::HashMap;

use rayon::prelude::*;

use crate::timetable::StopIdx;
use crate::transfers::Transfers;

/// One shared-vehicle candidate edge between two stops' street links.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RentalEdge {
    pub to: u32,
    pub seconds: u32,
    /// Meters ridden on the network, connectors excluded.
    pub network_meters: f64,
    /// Network meters plus both link connectors.
    pub total_meters: f64,
}

/// The rental leg behind a merged transfer edge that used one: where the
/// vehicle was picked up and dropped, and what the ride itself cost —
/// the reconstruction token. The walking legs on either side derive
/// from the closure (`pre_seconds` walking before the pickup,
/// `post_seconds` after the drop).
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct RentalToken {
    pub pickup: StopIdx,
    pub drop: StopIdx,
    pub ride_seconds: u32,
    pub ride_network_meters: f64,
    pub ride_total_meters: f64,
    pub pre_seconds: u32,
    pub post_seconds: u32,
}

/// A merged transfer edge as [`Transfers::from_edges`] consumes it.
pub type MergedEdge = (StopIdx, StopIdx, u32, f64);

/// One closed row from a source: target, seconds, meters, rental use.
type ClosedRow = (u32, u32, f64, Option<RentalUse>);

/// The state a Dijkstra label carries beside its cost: whether (and
/// which) rental the path used, and the walking split around it.
#[derive(Debug, Clone, Copy)]
struct RentalUse {
    pickup: u32,
    drop: u32,
    ride_seconds: u32,
    ride_network_meters: f64,
    ride_total_meters: f64,
    pre_seconds: u32,
}

/// Builds the merged, closed transfer set from the walking closure and
/// per-stop rental candidate edges, with one reconstruction token per
/// merged edge that used a rental.
///
/// The walking closure passes through verbatim — its rows keep their
/// own budget, and losing a walking transfer would make the merged set
/// weaker than the set it extends. `cutoff` bounds a rental-bearing
/// transfer's total duration (pre-walk, ride, and post-walk): the
/// policy's transfer budget covers that whole movement, so a rental
/// within budget cannot smuggle unbounded walking around itself. Per
/// pair the faster of the two wins; ties fall to walking.
pub fn merge_mode_transfers(
    walking: &Transfers,
    rentals: &[Vec<RentalEdge>],
    stop_count: u32,
    cutoff: u32,
) -> (Vec<MergedEdge>, HashMap<(u32, u32), RentalToken>) {
    let stops = stop_count as usize;
    assert_eq!(rentals.len(), stops, "rental rows must cover every stop");
    let per_source: Vec<(u32, Vec<ClosedRow>)> = (0..stop_count)
        .into_par_iter()
        .filter_map(|source| {
            let reached = close_from(walking, rentals, stops, cutoff, source);
            if reached.is_empty() && walking.from_stop(StopIdx(source)).is_empty() {
                None
            } else {
                Some((source, reached))
            }
        })
        .collect();
    let mut edges = Vec::new();
    let mut tokens = HashMap::new();
    for (source, reached) in per_source {
        let mut rental_best: HashMap<u32, ClosedRow> =
            reached.into_iter().map(|row| (row.0, row)).collect();
        for edge in walking.from_stop(StopIdx(source)) {
            // A walking row survives untouched unless a rental path is
            // strictly faster for the pair.
            match rental_best.get(&edge.to.0) {
                Some(&(_, seconds, _, _)) if seconds < edge.duration => {}
                _ => {
                    rental_best.remove(&edge.to.0);
                    edges.push((StopIdx(source), edge.to, edge.duration, edge.meters));
                }
            }
        }
        for (target, seconds, meters, rental) in rental_best.into_values() {
            edges.push((StopIdx(source), StopIdx(target), seconds, meters));
            let rental = rental.expect("only rental-bearing rows remain");
            tokens.insert(
                (source, target),
                RentalToken {
                    pickup: StopIdx(rental.pickup),
                    drop: StopIdx(rental.drop),
                    ride_seconds: rental.ride_seconds,
                    ride_network_meters: rental.ride_network_meters,
                    ride_total_meters: rental.ride_total_meters,
                    pre_seconds: rental.pre_seconds,
                    post_seconds: seconds - rental.pre_seconds - rental.ride_seconds,
                },
            );
        }
    }
    (edges, tokens)
}

/// The bounded search from one source stop, returning only the
/// **rental-bearing** rows. The pre- and post-walk are each **one
/// closure row**: chains within the walking cutoff are already covered
/// by their direct row (that is what a closure is), and a chain past
/// the cutoff would compose a walk the closure refuses — and one whose
/// edge the reconstruction could not resolve. States: 0 = at the
/// source, 1 = walked one closure row (rental only from here), 2 =
/// dropped off (one walking row still available), 3 = walked after the
/// drop (terminal). Pure walking rows are the caller's — the walking
/// closure already holds them, at its own budget.
fn close_from(
    walking: &Transfers,
    rentals: &[Vec<RentalEdge>],
    stops: usize,
    cutoff: u32,
    source: u32,
) -> Vec<ClosedRow> {
    const UNREACHED: u32 = u32::MAX;
    const STATES: u32 = 4;
    // Per (stop, state): best seconds, walked+ridden meters, rental use.
    let mut best = vec![(UNREACHED, 0.0_f64, None::<RentalUse>); stops * STATES as usize];
    let mut heap: BinaryHeap<Reverse<(u32, u32)>> = BinaryHeap::new();
    let slot = |stop: u32, state: u32| (stop * STATES + state) as usize;
    best[slot(source, 0)] = (0, 0.0, None);
    heap.push(Reverse((0, source * STATES)));
    while let Some(Reverse((seconds, key))) = heap.pop() {
        let (stop, state) = (key / STATES, key % STATES);
        if seconds > best[slot(stop, state)].0 {
            continue;
        }
        let (_, meters, rental) = best[slot(stop, state)];
        if state == 0 || state == 2 {
            for edge in walking.from_stop(StopIdx(stop)) {
                let Some(next) = seconds.checked_add(edge.duration) else {
                    continue;
                };
                if next > cutoff {
                    continue;
                }
                let target = slot(edge.to.0, state + 1);
                if next < best[target].0 {
                    best[target] = (next, meters + edge.meters, rental);
                    heap.push(Reverse((next, edge.to.0 * STATES + state + 1)));
                }
            }
        }
        if state <= 1 {
            for edge in &rentals[stop as usize] {
                let Some(next) = seconds.checked_add(edge.seconds) else {
                    continue;
                };
                if next > cutoff {
                    continue;
                }
                let target = slot(edge.to, 2);
                if next < best[target].0 {
                    best[target] = (
                        next,
                        meters + edge.total_meters,
                        Some(RentalUse {
                            pickup: stop,
                            drop: edge.to,
                            ride_seconds: edge.seconds,
                            ride_network_meters: edge.network_meters,
                            ride_total_meters: edge.total_meters,
                            pre_seconds: seconds,
                        }),
                    );
                    heap.push(Reverse((next, edge.to * STATES + 2)));
                }
            }
        }
    }
    let mut reached = Vec::new();
    for target in 0..stops as u32 {
        if target == source {
            continue;
        }
        let dropped = best[slot(target, 2)];
        let walked = best[slot(target, 3)];
        let (seconds, meters, rental) = if dropped.0 <= walked.0 {
            dropped
        } else {
            walked
        };
        if seconds == UNREACHED {
            continue;
        }
        reached.push((target, seconds, meters, rental));
    }
    reached
}

#[cfg(test)]
mod tests;
