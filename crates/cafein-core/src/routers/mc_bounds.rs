//! The resolved-trip time bounds behind `max_slower`: a plain
//! time-only range sweep over the trips with a resolved emission
//! factor, shared by the multicriteria engines so both restrict
//! against one definition.

use crate::routers::router::Request;
use crate::tbtr::{earliest_boardable, DayView, ViewTrip};
use crate::timetable::{StopIdx, Timetable};
use crate::transfers::Transfers;

/// The plain time-only bounds behind `max_slower`: per departure pass
/// (descending, range-shared state and round-capped like the
/// multicriteria search) the earliest per-stop arrival over the trips
/// with a **resolved emission factor** — the same trip set the
/// multicriteria search can board, so its per-pass fastest journey
/// always achieves the destination bound and the cutoff floor provably
/// keeps that journey alive. The sweep is ride-aware: arrivals are
/// tracked per round, so a faster arrival that exhausted the transfer
/// cap cannot suppress a slower fewer-rides one that still has
/// capacity to continue. Returns one per-stop snapshot per departure
/// (the ride-cap level), in the departures' (descending) order;
/// unreachable stops hold `u32::MAX`.
pub(crate) fn resolved_bounds(
    view: &DayView,
    timetable: &Timetable,
    footpaths: &Transfers,
    factors: &[f64],
    request: &Request,
    departures: &[u32],
) -> Vec<Vec<u32>> {
    let stop_count = timetable.stop_count() as usize;
    // The bound must count exactly the supply the restricted search can
    // use, so the exclusion masks apply here identically.
    let exclusions = request.exclusions.as_deref();
    // The same saturation as `Search::pass`: the bound must not come
    // from a round the multicriteria search cannot reach.
    let rounds = request.max_transfers.min(254) as usize + 1;
    // `best[r][stop]`: the earliest arrival using at most `r` rides,
    // shared across the descending passes; a write at level `r`
    // propagates to every higher level, so the snapshot is the cap's.
    let mut best: Vec<Vec<u32>> = vec![vec![u32::MAX; stop_count]; rounds + 1];
    let improve = |best: &mut Vec<Vec<u32>>, level: usize, stop: StopIdx, at: u32| -> bool {
        if at >= best[level][stop.0 as usize] {
            return false;
        }
        for row in &mut best[level..] {
            if at < row[stop.0 as usize] {
                row[stop.0 as usize] = at;
            }
        }
        true
    };
    let mut queue: Vec<Vec<(u16, u32)>> = vec![Vec::new(); view.line_count() as usize];
    let mut touched: Vec<u32> = Vec::new();
    let mut snapshots = Vec::with_capacity(departures.len());
    for &departure in departures {
        let mut fresh: Vec<StopIdx> = Vec::new();
        for &(stop, seconds) in &request.access {
            if exclusions.is_some_and(|excluded| excluded.excludes_stop(stop)) {
                continue;
            }
            let arrival = departure.saturating_add(seconds);
            if improve(&mut best, 0, stop, arrival) {
                fresh.push(stop);
            }
        }
        for round in 1..=rounds {
            if fresh.is_empty() {
                break;
            }
            // Boarding times are captured at queueing so a round's own
            // alights cannot fuel same-round boardings (round separation,
            // matching the multicriteria search's label queue); a round
            // boards from the previous level's arrivals.
            for &stop in &fresh {
                let at = best[round - 1][stop.0 as usize];
                for served in timetable.patterns_at_stop(stop) {
                    if exclusions.is_some_and(|excluded| {
                        excluded.excludes_route(timetable.pattern_route(served.pattern))
                    }) {
                        continue;
                    }
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
                        if !exclusions.is_some_and(|excluded| excluded.excludes_stop(stop))
                            && improve(&mut best, round, stop, arrival)
                        {
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
                            if !factors[view.backing(trip).0 as usize].is_finite()
                                || exclusions.is_some_and(|excluded| {
                                    excluded.excludes_trip(view.backing(trip))
                                })
                            {
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
                let at = best[round][stop.0 as usize];
                for footpath in footpaths.from_stop(stop) {
                    let arrival = at.saturating_add(footpath.duration);
                    if exclusions.is_some_and(|excluded| excluded.excludes_stop(footpath.to)) {
                        continue;
                    }
                    if improve(&mut best, round, footpath.to, arrival) {
                        next.push(footpath.to);
                    }
                }
            }
            fresh = next;
        }
        snapshots.push(best[rounds].clone());
    }
    snapshots
}

#[cfg(test)]
mod tests {
    use super::resolved_bounds;
    use crate::routers::router::Request;
    use crate::tbtr::DayView;
    use crate::timetable::{StopIdx, StopTime, TimetableBuilder};
    use crate::transfers::Transfers;

    fn time(at: u32) -> StopTime {
        StopTime {
            arrival: at,
            departure: at,
        }
    }

    #[test]
    fn bounds_track_rides_across_passes() {
        // The late pass reaches M faster but exhausts the two-ride cap;
        // the early pass needs its own slower one-ride arrival at M to
        // continue to D. A ride-blind sweep suppresses that label and
        // loses D's bound.
        let mut builder = TimetableBuilder::new(4);
        let a1 = builder.add_pattern(&[StopIdx(0), StopIdx(1)], 0).unwrap();
        let a2 = builder.add_pattern(&[StopIdx(1), StopIdx(2)], 1).unwrap();
        let b = builder.add_pattern(&[StopIdx(0), StopIdx(2)], 2).unwrap();
        let cd = builder.add_pattern(&[StopIdx(2), StopIdx(3)], 3).unwrap();
        builder
            .add_trip(a1, vec![time(300), time(320)], 0, 0)
            .unwrap();
        builder
            .add_trip(a2, vec![time(340), time(360)], 1, 0)
            .unwrap();
        builder
            .add_trip(b, vec![time(10), time(380)], 2, 0)
            .unwrap();
        builder
            .add_trip(cd, vec![time(400), time(450)], 3, 0)
            .unwrap();
        let timetable = builder.finish();
        let view = DayView::universal(&timetable);
        let footpaths = Transfers::empty(4);
        let request = Request {
            departure: 0,
            access: vec![(StopIdx(0), 0)],
            egress: vec![(StopIdx(3), 0)],
            active_services: vec![true],
            active_services_previous: vec![false],
            max_transfers: 1,
            exclusions: None,
        };
        let bounds = resolved_bounds(
            &view,
            &timetable,
            &footpaths,
            &[10.0; 4],
            &request,
            &[300, 0],
        );
        // The late pass cannot reach D within two rides; the early one
        // reaches it at 450 through its own one-ride label at M.
        assert_eq!(bounds[0][3], u32::MAX);
        assert_eq!(bounds[1][3], 450);
        assert_eq!(bounds[1][2], 360);
    }
}
