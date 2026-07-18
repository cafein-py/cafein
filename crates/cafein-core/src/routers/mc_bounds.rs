//! The resolved-trip time bounds behind `max_slower`: a plain
//! time-only range sweep over the trips with a resolved emission
//! factor, shared by the multicriteria engines so both restrict
//! against one definition.

use crate::routers::router::Request;
use crate::tbtr::{earliest_boardable, DayView, ViewTrip};
use crate::timetable::{StopIdx, Timetable};
use crate::transfers::Transfers;

/// The plain time-only bounds behind `max_slower`: per departure pass
/// (descending, range-RAPTOR shared state and round-capped like the
/// multicriteria search) the earliest per-stop arrival over the trips
/// with a **resolved emission factor** — the same trip set the
/// multicriteria search can board, so its per-pass fastest journey
/// always achieves the destination bound and the cutoff floor provably
/// keeps that journey alive. Returns one per-stop snapshot per
/// departure, in the departures' (descending) order; unreachable stops
/// hold `u32::MAX`.
pub(crate) fn resolved_bounds(
    view: &DayView,
    timetable: &Timetable,
    footpaths: &Transfers,
    factors: &[f64],
    request: &Request,
    departures: &[u32],
) -> Vec<Vec<u32>> {
    let stop_count = timetable.stop_count() as usize;
    let mut best = vec![u32::MAX; stop_count];
    let mut queue: Vec<Vec<(u16, u32)>> = vec![Vec::new(); view.line_count() as usize];
    let mut touched: Vec<u32> = Vec::new();
    let mut snapshots = Vec::with_capacity(departures.len());
    for &departure in departures {
        let mut fresh: Vec<StopIdx> = Vec::new();
        for &(stop, seconds) in &request.access {
            let arrival = departure.saturating_add(seconds);
            if arrival < best[stop.0 as usize] {
                best[stop.0 as usize] = arrival;
                fresh.push(stop);
            }
        }
        // The same saturation as `Search::pass`: the bound must not come
        // from a round the multicriteria search cannot reach.
        for _ in 1..=request.max_transfers.min(254) as u32 + 1 {
            if fresh.is_empty() {
                break;
            }
            // Boarding times are captured at queueing so a round's own
            // alights cannot fuel same-round boardings (round separation,
            // matching the multicriteria search's label queue).
            for &stop in &fresh {
                let at = best[stop.0 as usize];
                for served in timetable.patterns_at_stop(stop) {
                    let positions = timetable.pattern_stops(served.pattern).len();
                    if served.position as usize + 1 >= positions {
                        continue;
                    }
                    for line in view.lines_of_pattern(served.pattern).into_iter().flatten() {
                        if queue[line as usize].is_empty() {
                            touched.push(line);
                        }
                        queue[line as usize].push((served.position, at));
                    }
                }
            }
            let mut rode: Vec<StopIdx> = Vec::new();
            let lines = std::mem::take(&mut touched);
            for &line in &lines {
                let mut entries = std::mem::take(&mut queue[line as usize]);
                entries.sort_unstable_by_key(|&(position, _)| position);
                let pattern = view.line_pattern(line);
                let stops = timetable.pattern_stops(pattern);
                let offset = view.line_day_offset(line);
                let mut current: Option<ViewTrip> = None;
                let mut queued = 0usize;
                for (position, &stop) in stops.iter().enumerate().skip(entries[0].0 as usize) {
                    if let Some(trip) = current {
                        let arrival = view.stored_times(timetable, trip)[position].arrival - offset;
                        if arrival < best[stop.0 as usize] {
                            best[stop.0 as usize] = arrival;
                            rode.push(stop);
                        }
                    }
                    while queued < entries.len() && entries[queued].0 as usize == position {
                        let (_, at) = entries[queued];
                        queued += 1;
                        // The earliest boardable trip whose factor resolves.
                        let Some(first) =
                            earliest_boardable(view, timetable, line, position as u16, at)
                        else {
                            continue;
                        };
                        for rank in first.0..view.line_trips(line).end {
                            let trip = ViewTrip(rank);
                            if !factors[view.backing(trip).0 as usize].is_finite() {
                                continue;
                            }
                            let departs =
                                view.stored_times(timetable, trip)[position].departure - offset;
                            let held_departs = current.map(|held| {
                                view.stored_times(timetable, held)[position].departure - offset
                            });
                            if held_departs.is_none_or(|held| departs < held) {
                                current = Some(trip);
                            }
                            break;
                        }
                    }
                }
                entries.clear();
                queue[line as usize] = entries;
            }
            let mut next = rode.clone();
            for &stop in &rode {
                let at = best[stop.0 as usize];
                for footpath in footpaths.from_stop(stop) {
                    let arrival = at.saturating_add(footpath.duration);
                    if arrival < best[footpath.to.0 as usize] {
                        best[footpath.to.0 as usize] = arrival;
                        next.push(footpath.to);
                    }
                }
            }
            fresh = next;
        }
        snapshots.push(best.clone());
    }
    snapshots
}
