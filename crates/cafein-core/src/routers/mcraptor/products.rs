//! The query products: one-pair routes, range profiles, and the
//! least-emissions and frontier matrices.

use super::*;

/// The multicriteria journeys for a single departure: the Pareto set
/// over (arrival, emissions bucket), as full journeys. A positive `slack`
/// (seconds) widens the set to the suboptimal journeys arriving within the
/// band; `max_options`, when set, caps the returned count.
#[allow(clippy::too_many_arguments)]
pub fn route(
    view: &DayView,
    timetable: &Timetable,
    footpaths: &Transfers,
    geometry: &TripGeometry,
    factors: &[f64],
    request: &Request,
    bucket: f64,
    slack: u32,
    max_options: Option<usize>,
    route_penalties: &[u32],
    max_slower: Option<u32>,
) -> Vec<Journey> {
    profile(
        view,
        timetable,
        footpaths,
        geometry,
        factors,
        request,
        &[request.departure],
        bucket,
        slack,
        max_options,
        route_penalties,
        max_slower,
    )
}

/// The multicriteria departure-window profile: the Pareto set over
/// (departure, arrival, emissions bucket), each journey's departure
/// being the latest time the origin can be left to catch it.
#[allow(clippy::too_many_arguments)]
pub fn route_range(
    view: &DayView,
    timetable: &Timetable,
    footpaths: &Transfers,
    geometry: &TripGeometry,
    factors: &[f64],
    request: &Request,
    window: u32,
    bucket: f64,
    slack: u32,
    max_options: Option<usize>,
    route_penalties: &[u32],
    max_slower: Option<u32>,
) -> Vec<Journey> {
    let departures = departure_candidates(timetable, request, window);
    profile(
        view,
        timetable,
        footpaths,
        geometry,
        factors,
        request,
        &departures,
        bucket,
        slack,
        max_options,
        route_penalties,
        max_slower,
    )
}

/// The least-emissions cost matrix over McRAPTOR's candidate set: per
/// origin–destination cell, the cleanest journey (ties toward the
/// shorter travel time) among the (departure, arrival, emissions
/// bucket) Pareto candidates of the departure window — the same
/// widened set `journey_frontier`'s pareto candidates draw from, so a
/// cell can be strictly cleaner than the interim objective's, which
/// only sees time-optimal journeys. Candidates fold per pass at label
/// creation, so a `budget` (travel-time cap in seconds) is applied
/// against each label's own departure. Requests fan out over origins
/// with rayon; the access seeds are the zero-ride floor of the
/// origin's own cell.
#[allow(clippy::too_many_arguments)]
pub fn least_emissions_matrix(
    view: &DayView,
    timetable: &Timetable,
    footpaths: &Transfers,
    inputs: &CostInputs<'_>,
    requests: &[Request],
    destinations: &[StopIdx],
    egress: &[Vec<(u32, u32, f64)>],
    access_meters: &[Vec<(StopIdx, f64)>],
    egress_active: bool,
    window: u32,
    budget: Option<u32>,
    bucket: f64,
) -> Vec<Vec<CostRow>> {
    assert!(
        bucket.is_finite() && bucket > 0.0,
        "the emissions bucket must be positive"
    );
    assert_eq!(
        egress.len(),
        timetable.stop_count() as usize,
        "the egress map must be per stop"
    );
    assert_eq!(
        access_meters.len(),
        requests.len(),
        "the access-meter map must be per request"
    );
    // Door-to-door mode is set by the caller, not inferred from the egress map:
    // an all-empty map (every located destination unsnappable or beyond the cap)
    // must still keep the zero-walk direct credit off, leaving those
    // destinations unreachable as the stop-as-coordinate route would.
    // A stop holds one slot, so repeated destination stops share the
    // first occurrence's slot and their rows are re-expanded to the
    // requested order after folding, as `frontier_matrix` does.
    let mut slots = vec![0u32; timetable.stop_count() as usize];
    let mut cell_of: Vec<usize> = Vec::with_capacity(destinations.len());
    let mut unique = 0usize;
    for &stop in destinations {
        let slot = slots[stop.0 as usize];
        if slot == 0 {
            unique += 1;
            slots[stop.0 as usize] = unique as u32;
            cell_of.push(unique - 1);
        } else {
            cell_of.push(slot as usize - 1);
        }
    }
    let cell_count = if egress_active {
        destinations.len()
    } else {
        unique
    };
    let runs: Vec<(Vec<CostRow>, Box<McRaptorStats>)> = requests
        .par_iter()
        .zip(access_meters.par_iter())
        .map(|(request, access_meters)| {
            let mut search = Search::start(
                view,
                timetable,
                footpaths,
                inputs.geometry,
                inputs.factors,
                bucket,
                0,
                &[],
                request.exclusions.as_deref(),
            );
            let departures = departure_candidates(timetable, request, window);
            let mut best: Vec<Option<(f64, u32, u32, f64)>> = vec![None; cell_count];
            for &departure in &departures {
                let mut fold = Some(MatrixFold {
                    slots: &slots,
                    egress,
                    egress_active,
                    budget,
                    best: &mut best,
                });
                search.pass(request, departure, &mut fold, &mut None);
            }
            let rows = if egress_active {
                best.into_iter()
                    .enumerate()
                    .filter_map(|(slot, winner)| {
                        winner.map(|winner| {
                            search.cost_row(inputs, winner, destinations[slot].0, access_meters)
                        })
                    })
                    .collect()
            } else {
                destinations
                    .iter()
                    .zip(&cell_of)
                    .filter_map(|(&stop, &cell)| {
                        best[cell]
                            .map(|winner| search.cost_row(inputs, winner, stop.0, access_meters))
                    })
                    .collect()
            };
            (rows, search.stats)
        })
        .collect();
    if std::env::var_os("CAFEIN_MCRAPTOR_PROF").is_some() {
        let mut reduced = McRaptorStats::default();
        for (_, stats) in &runs {
            reduced.absorb(stats);
        }
        reduced.report("least_emissions_matrix");
    }
    runs.into_iter().map(|run| run.0).collect()
}

/// The batched Pareto frontiers: per request × destination slot, the
/// (departure, arrival, emissions bucket) Pareto journeys of the
/// departure window — each cell exactly the single-pair `route_range`
/// set (strict frontier: no slack, no cap, no penalties). Stop mode
/// takes the destination stops; door-to-door mode (`egress_active`)
/// takes a per-stop final-egress map over `slot_count` destination
/// points, the walking-only journey being the caller's overlay as in
/// the one-pair coordinate route. One window profile per request
/// serves every slot; requests fan out with rayon.
#[allow(clippy::too_many_arguments)]
pub fn frontier_matrix(
    view: &DayView,
    timetable: &Timetable,
    footpaths: &Transfers,
    geometry: &TripGeometry,
    factors: &[f64],
    requests: &[Request],
    destinations: &[StopIdx],
    egress: &[Vec<(u32, u32, f64)>],
    egress_active: bool,
    slot_count: usize,
    window: u32,
    bucket: f64,
    max_slower: Option<u32>,
) -> Vec<Vec<Vec<Journey>>> {
    assert!(
        bucket.is_finite() && bucket > 0.0,
        "the emissions bucket must be positive"
    );
    if egress_active {
        assert_eq!(
            egress.len(),
            timetable.stop_count() as usize,
            "the egress map must be per stop"
        );
    } else {
        assert_eq!(
            destinations.len(),
            slot_count,
            "stop mode takes one slot per destination stop"
        );
    }
    // A stop holds one slot, so repeated destination stops share the
    // first occurrence's slot and their cells are re-expanded to the
    // requested order after assembly.
    let mut slots = vec![0u32; timetable.stop_count() as usize];
    let mut cell_of: Vec<usize> = Vec::with_capacity(destinations.len());
    let mut unique = 0usize;
    for &stop in destinations {
        let slot = slots[stop.0 as usize];
        if slot == 0 {
            unique += 1;
            slots[stop.0 as usize] = unique as u32;
            cell_of.push(unique - 1);
        } else {
            cell_of.push(slot as usize - 1);
        }
    }
    let bag_count = if egress_active { slot_count } else { unique };
    // Every (stop, slot, final-walk seconds) pair, for the per-slot
    // destination bounds of the `max_slower` restriction.
    let slot_egress: Vec<(StopIdx, usize, u32)> = if max_slower.is_none() {
        Vec::new()
    } else if egress_active {
        egress
            .iter()
            .enumerate()
            .flat_map(|(stop, entries)| {
                entries
                    .iter()
                    .map(move |&(slot, seconds, _)| (StopIdx(stop as u32), slot as usize, seconds))
            })
            .collect()
    } else {
        destinations
            .iter()
            .zip(&cell_of)
            .map(|(&stop, &slot)| (stop, slot, 0))
            .collect()
    };
    let runs: Vec<(Vec<Vec<Journey>>, Box<McRaptorStats>, u64)> = requests
        .par_iter()
        .map(|request| {
            let origin_started = std::time::Instant::now();
            // Per pass, the restriction's per-slot destination bounds; the
            // cutoff floor is the farthest reachable slot's bound, so a
            // label needed by any cell survives, and each cell's output
            // band anchors at its own slot's bound.
            let departures = departure_candidates(timetable, request, window);
            let restricted = max_slower.map(|band| {
                let bounds =
                    resolved_bounds(view, timetable, footpaths, factors, request, &departures);
                let floors: Vec<Vec<u32>> = bounds
                    .iter()
                    .map(|per_stop| {
                        let mut slot_floors = vec![u32::MAX; bag_count];
                        for &(stop, slot, seconds) in &slot_egress {
                            let bound = per_stop[stop.0 as usize].saturating_add(seconds);
                            slot_floors[slot] = slot_floors[slot].min(bound);
                        }
                        slot_floors
                    })
                    .collect();
                (band, bounds, floors)
            });
            let mut search = Search::start(
                view,
                timetable,
                footpaths,
                geometry,
                factors,
                bucket,
                0,
                &[],
                request.exclusions.as_deref(),
            );
            let mut bags: Vec<DestinationBag> = std::iter::repeat_with(DestinationBag::default)
                .take(bag_count)
                .collect();
            for (index, &departure) in departures.iter().enumerate() {
                if let Some((band, bounds, floors)) = &restricted {
                    let floor = floors[index]
                        .iter()
                        .copied()
                        .filter(|&floor| floor != u32::MAX)
                        .max()
                        .unwrap_or(u32::MAX);
                    search.set_cutoff(&bounds[index], floor, *band);
                }
                let mut frontier = Some(FrontierFold {
                    slots: &slots,
                    egress,
                    egress_active,
                    bags: &mut bags,
                });
                search.pass(request, departure, &mut None, &mut frontier);
            }
            let cells: Vec<Vec<Journey>> = bags
                .iter()
                .enumerate()
                .map(|(slot, bag)| {
                    let mut journeys: Vec<Journey> = bag
                        .entries
                        .iter()
                        .filter(|entry| match &restricted {
                            None => true,
                            Some((band, _, floors)) => {
                                let index = departures
                                    .binary_search_by(|probe| probe.cmp(&entry.departure).reverse())
                                    .expect("every entry comes from a pass departure");
                                entry.arrival <= floors[index][slot].saturating_add(*band)
                            }
                        })
                        .map(|arrived| search.assemble(arrived))
                        .collect();
                    journeys.sort_by_key(|journey| {
                        (journey.departure, journey.arrival, journey.rides())
                    });
                    journeys
                })
                .collect();
            let cells: Vec<Vec<Journey>> = if egress_active || unique == destinations.len() {
                cells
            } else {
                cell_of.iter().map(|&cell| cells[cell].clone()).collect()
            };
            (
                cells,
                search.stats,
                origin_started.elapsed().as_nanos() as u64,
            )
        })
        .collect();
    if std::env::var_os("CAFEIN_MCRAPTOR_PROF").is_some() && !runs.is_empty() {
        // The per-origin service-time distribution: each timer starts
        // inside the Rayon task, so queue waiting is excluded.
        let mut walls: Vec<u64> = runs.iter().map(|run| run.2).collect();
        walls.sort_unstable();
        let total_relaxations: u64 = runs
            .iter()
            .map(|run| run.1.footpath_label_edge_relaxations)
            .sum();
        let mut slowest_share = 0.0f64;
        for (index, (cells, stats, wall)) in runs.iter().enumerate() {
            let rows: usize = cells.iter().map(|cell| cell.len()).sum();
            eprintln!(
                "MCRAPTOR-ORIGIN index={index} wall_ms={} departure_passes={} \
                 rounds={} rode_labels={} footpath_relaxations={} \
                 bag_calls={} labels_created={} rows={rows}",
                wall / 1_000_000,
                stats.departure_passes,
                stats.rounds_entered,
                stats.rode_labels,
                stats.footpath_label_edge_relaxations,
                stats.route_bag_calls + stats.footpath_bag_calls,
                stats.labels_created,
            );
            if *wall == walls[walls.len() - 1] && total_relaxations > 0 {
                slowest_share =
                    stats.footpath_label_edge_relaxations as f64 / total_relaxations as f64;
            }
        }
        let percentile = |fraction: f64| walls[((walls.len() - 1) as f64 * fraction) as usize];
        let p50 = percentile(0.5);
        eprintln!(
            "MCRAPTOR-ORIGINS min_ms={} p50_ms={} p90_ms={} max_ms={} \
             max_over_p50={:.2} slowest_relaxation_share={:.3}",
            walls[0] / 1_000_000,
            p50 / 1_000_000,
            percentile(0.9) / 1_000_000,
            walls[walls.len() - 1] / 1_000_000,
            walls[walls.len() - 1] as f64 / p50.max(1) as f64,
            slowest_share,
        );
        let mut reduced = McRaptorStats::default();
        for (_, stats, _) in &runs {
            reduced.absorb(stats);
        }
        reduced.report("frontier_matrix");
    }
    runs.into_iter().map(|run| run.0).collect()
}

#[allow(clippy::too_many_arguments)]
pub(super) fn profile(
    view: &DayView,
    timetable: &Timetable,
    footpaths: &Transfers,
    geometry: &TripGeometry,
    factors: &[f64],
    request: &Request,
    departures: &[u32],
    bucket: f64,
    slack: u32,
    max_options: Option<usize>,
    route_penalties: &[u32],
    max_slower: Option<u32>,
) -> Vec<Journey> {
    assert!(
        bucket.is_finite() && bucket > 0.0,
        "the emissions bucket must be positive"
    );
    // The `max_slower` restriction: per pass, the resolved-trip bounds
    // anchor the per-stop band and the destination bound floors it (so
    // the pass's fastest journey survives the pruning); the same floor
    // drives the output band below.
    let restricted = max_slower.map(|band| {
        let bounds = resolved_bounds(view, timetable, footpaths, factors, request, departures);
        let floors: Vec<u32> = bounds
            .iter()
            .map(|per_stop| {
                request
                    .egress
                    .iter()
                    .map(|&(stop, seconds)| per_stop[stop.0 as usize].saturating_add(seconds))
                    .min()
                    .unwrap_or(u32::MAX)
            })
            .collect();
        (band, bounds, floors)
    });
    let mut search = Search::start(
        view,
        timetable,
        footpaths,
        geometry,
        factors,
        bucket,
        slack,
        route_penalties,
        request.exclusions.as_deref(),
    );
    for (index, &departure) in departures.iter().enumerate() {
        if let Some((band, bounds, floors)) = &restricted {
            search.set_cutoff(&bounds[index], floors[index], *band);
        }
        search.pass(request, departure, &mut None, &mut None);
    }
    // The output band: a journey stays within `max_slower` of its own
    // pass's plain destination bound (an unreachable bound keeps
    // everything — nothing anchors the band that pass).
    let banded: Vec<Arrived>;
    let entries: &[Arrived] = match &restricted {
        Some((band, _, floors)) => {
            banded = search
                .destination
                .entries
                .iter()
                .filter(|entry| {
                    let index = departures
                        .binary_search_by(|probe| probe.cmp(&entry.departure).reverse())
                        .expect("every entry comes from a pass departure");
                    entry.arrival <= floors[index].saturating_add(*band)
                })
                .copied()
                .collect();
            &banded
        }
        None => &search.destination.entries,
    };
    let kept = cap_entries(entries, max_options);
    let mut journeys: Vec<Journey> = kept
        .into_iter()
        .map(|arrived| search.assemble(arrived))
        .collect();
    journeys.sort_by_key(|journey| (journey.departure, journey.arrival, journey.rides()));
    search.stats.report("profile");
    journeys
}

/// Strict (departure↓, arrival, emissions bucket) domination between two
/// destination entries — the relation that ranks suboptimal arrivals under
/// `max_options`.
pub(super) fn strictly_dominates(a: &Arrived, b: &Arrived) -> bool {
    a.departure >= b.departure
        && a.effective() <= b.effective()
        && a.key <= b.key
        && (a.departure > b.departure || a.effective() < b.effective() || a.key < b.key)
}

/// The destination entries to assemble. Without a cap (or when the set
/// already fits) every entry is kept; otherwise the strict frontier — the
/// entries no other entry strictly dominates — is kept in full and the
/// suboptimal arrivals of smallest time-gap above it fill the remainder up
/// to `max_options`, ties toward the cleaner emissions. A suboptimal entry's
/// gap is the seconds by which its nearest strict-frontier dominator arrives
/// earlier. The cap never drops a frontier (optimal) journey, so the result
/// can exceed `max_options` when the frontier itself is larger.
pub(super) fn cap_entries(entries: &[Arrived], max_options: Option<usize>) -> Vec<&Arrived> {
    let cap = match max_options {
        Some(cap) if entries.len() > cap => cap,
        _ => return entries.iter().collect(),
    };
    let on_frontier: Vec<bool> = entries
        .iter()
        .map(|entry| !entries.iter().any(|other| strictly_dominates(other, entry)))
        .collect();
    let mut ranked: Vec<(&Arrived, bool, u32)> = entries
        .iter()
        .zip(&on_frontier)
        .map(|(entry, &frontier)| {
            let gap = if frontier {
                0
            } else {
                entries
                    .iter()
                    .zip(&on_frontier)
                    .filter(|(other, &f)| f && strictly_dominates(other, entry))
                    .map(|(other, _)| entry.effective().saturating_sub(other.effective()))
                    .min()
                    .unwrap_or(u32::MAX)
            };
            (entry, frontier, gap)
        })
        .collect();
    // Frontier entries first (always kept), then suboptimals by time-gap.
    ranked.sort_by(|(a, fa, ga), (b, fb, gb)| {
        fb.cmp(fa)
            .then(ga.cmp(gb))
            .then(a.key.cmp(&b.key))
            .then(a.grams.total_cmp(&b.grams))
    });
    let frontier = on_frontier.iter().filter(|&&f| f).count();
    let keep = cap.max(frontier);
    ranked
        .into_iter()
        .take(keep)
        .map(|(entry, _, _)| entry)
        .collect()
}
