//! McRAPTOR: multicriteria RAPTOR over (arrival, emissions).
//!
//! Round-based label bags per stop over (arrival, grams CO₂e). Arrivals
//! compare exactly; grams compare at a configurable bucket width, so
//! labels within one bucket count as equal and bag sizes — and the
//! search — stay bounded. Each bag insertion may substitute a
//! same-bucket representative arriving no later, so a reported
//! journey's emissions sit within one bucket of a true frontier value
//! per insertion its labels survived — a worst case of
//! `(2 × rides + 1) × bucket`, in practice well under one bucket. A
//! vanishing bucket (one microgram, matching the label quantization)
//! reproduces the exhaustive oracle's exact frontier.
//!
//! The emissions firewall holds: a label's grams update is one
//! cumulative-distance subtraction per alight, nothing per-leg beyond
//! that enters the search. Trips without a resolved emission factor are
//! skipped — journeys riding them can never sit on an emissions
//! frontier. Boarding considers, besides the earliest boardable trip,
//! the later trips of the line whose factor strictly improves on every
//! earlier boardable one: waiting for a cleaner vehicle can hold a true
//! Pareto point. On lines with uniform factors — the common case — this
//! collapses to the classic earliest-trip rule.

use rayon::prelude::*;

use crate::exhaustive::quantized;
use crate::fares::FareLeg;
use crate::geometry::{wkb_multi_line_string, TripGeometry};
use crate::journey::{Journey, Leg};
use crate::raptor::{departure_candidates, CostInputs, CostRow};
use crate::router::Request;
use crate::tbtr::{earliest_boardable, DayView, ViewTrip};
use crate::timetable::{StopIdx, Timetable, TripIdx};
use crate::transfers::Transfers;

/// How a label reached its stop; parents index the label arena.
#[derive(Debug, Clone, Copy)]
enum Origin {
    Access,
    Ride {
        parent: u32,
        trip: ViewTrip,
        board: u16,
        alight: u16,
    },
    Walk {
        parent: u32,
        duration: u32,
    },
}

#[derive(Debug, Clone, Copy)]
struct Label {
    arrival: u32,
    grams: f64,
    stop: StopIdx,
    /// The profile pass that grew this label's chain: label chains
    /// never cross passes, so travel time is `arrival - departure`.
    departure: u32,
    origin: Origin,
}

/// One bag entry; `key` is the grams bucket, `grams` the exact
/// (microgram-quantized) value behind it, `rides` the transit legs the
/// label used to reach the stop.
#[derive(Debug, Clone, Copy)]
struct Entry {
    arrival: u32,
    key: i64,
    grams: f64,
    rides: u8,
}

/// A per-stop label bag under bucketed dominance, cumulative across
/// rounds and profile passes. Shared with the trip-based engine, whose
/// stop bags follow the same contract.
#[derive(Debug, Clone, Default)]
pub(crate) struct Bag {
    entries: Vec<Entry>,
}

impl Bag {
    /// Inserts unless an entry arriving no later, in the same or a
    /// cleaner bucket, AND on no more rides already dominates it; evicts
    /// what the newcomer covers. The `rides` axis is what makes the
    /// cumulative-across-passes bag sound under the second criterion: a
    /// later-departure label may only suppress an earlier-departure one
    /// when it also used no more transit legs, so it keeps at least the
    /// onward-transfer budget to reproduce every continuation. Dropping
    /// it lets a later-but-more-transferred journey wrongly evict a
    /// cleaner earlier one that still had transfers to spend. An entry
    /// equal on arrival, bucket and rides but strictly dirtier in exact
    /// grams is refined (replaced), keeping the bucket's representative
    /// as clean as the search has seen. The trip-based engine passes
    /// `rides = 0` throughout (its rounds are ranked in the trip bags,
    /// not here), so its dominance stays exactly `(arrival, key)`.
    pub(crate) fn insert(&mut self, arrival: u32, grams: f64, key: i64, rides: u8) -> bool {
        self.insert_slack(arrival, grams, key, rides, 0)
    }

    /// `insert` under a time slack: an entry rejects the newcomer only
    /// when it is at least `slack` seconds earlier (and no dirtier / no
    /// more rides), so a journey dominated by a nearer one survives as a
    /// suboptimal alternative. Same-class (`arrival`, `key`, `rides`)
    /// duplicates still reduce to the cleanest representative, and eviction
    /// likewise needs the full `slack` margin. `slack = 0` is exactly
    /// strict `insert`; the trip-based and exhaustive engines only call the
    /// strict form, so they keep unchanged dominance.
    pub(crate) fn insert_slack(
        &mut self,
        arrival: u32,
        grams: f64,
        key: i64,
        rides: u8,
        slack: u32,
    ) -> bool {
        for entry in &self.entries {
            if entry.key <= key && entry.rides <= rides {
                if entry.arrival == arrival && entry.key == key && entry.rides == rides {
                    if grams >= entry.grams {
                        return false;
                    }
                } else if entry.arrival.saturating_add(slack) <= arrival {
                    return false;
                }
            }
        }
        self.entries.retain(|entry| {
            !((key <= entry.key
                && rides <= entry.rides
                && arrival.saturating_add(slack) <= entry.arrival)
                || (entry.arrival == arrival
                    && entry.key == key
                    && entry.rides == rides
                    && grams < entry.grams))
        });
        self.entries.push(Entry {
            arrival,
            key,
            grams,
            rides,
        });
        true
    }
}

/// A destination frontier entry; `departure` is the profile pass that
/// produced it.
#[derive(Debug, Clone, Copy)]
struct Arrived {
    departure: u32,
    arrival: u32,
    key: i64,
    grams: f64,
    label: u32,
}

/// The destination bag: Pareto over (departure descending, arrival,
/// grams bucket). Passes run at descending departures, so entries from
/// later departures are never evicted by earlier ones.
#[derive(Debug, Default)]
struct DestinationBag {
    entries: Vec<Arrived>,
}

impl DestinationBag {
    /// Inserts under the same time slack as the stop bags: a frontier entry
    /// rejects the candidate only when it is at least `slack` seconds
    /// earlier and no dirtier, so suboptimal arrivals within the band are
    /// retained. `slack = 0` is the strict (departure↓, arrival, bucket)
    /// Pareto frontier.
    fn insert(&mut self, candidate: Arrived, slack: u32) -> bool {
        for entry in &self.entries {
            if entry.departure >= candidate.departure && entry.key <= candidate.key {
                if entry.departure == candidate.departure
                    && entry.arrival == candidate.arrival
                    && entry.key == candidate.key
                {
                    if candidate.grams >= entry.grams {
                        return false;
                    }
                } else if entry.arrival.saturating_add(slack) <= candidate.arrival {
                    return false;
                }
            }
        }
        self.entries.retain(|entry| {
            !((candidate.departure >= entry.departure
                && candidate.key <= entry.key
                && candidate.arrival.saturating_add(slack) <= entry.arrival)
                || (entry.departure == candidate.departure
                    && entry.arrival == candidate.arrival
                    && entry.key == candidate.key
                    && candidate.grams < entry.grams))
        });
        self.entries.push(candidate);
        true
    }
}

/// A label riding a trip during one line scan. `kappa` folds the grams
/// at boarding and the boarding position's cumulative distance into one
/// value comparable between same-trip riders: at every future alight
/// both share the trip's arrival, so the lower `kappa` alights with
/// fewer grams everywhere.
#[derive(Debug, Clone, Copy)]
struct Rider {
    trip: ViewTrip,
    board: u16,
    kappa: f64,
    grams: f64,
    factor: f64,
    parent: u32,
    departure: u32,
}

/// Per-destination fold state for the emissions matrix: the cleanest
/// (then fastest) label seen per destination, folded at label creation
/// so cross-pass bag evictions cannot lose a budget-qualifying
/// candidate. Mirrors the interim fold's rules: lower grams win, ties
/// resolve toward the shorter travel time, a travel-time budget
/// disqualifies outright.
struct MatrixFold<'a> {
    /// Per stop: destination slot + 1, or 0 when not a destination.
    slots: &'a [u32],
    /// Per stop: the destinations within a final walk of it —
    /// ``(destination slot, walk seconds, walk meters)``. Empty unless a
    /// street egress is folded (the door-to-door emissions matrix), in which
    /// case a label alighting at the stop also credits each of those
    /// destinations with the final walk added.
    egress: &'a [Vec<(u32, u32, f64)>],
    /// Whether a street egress is folded (door-to-door mode). When set, every
    /// located destination carries its own egress self-entry — its connector to
    /// its coordinate — so the zero-walk direct credit below is left to that
    /// entry and skipped, matching the single-pair door-to-door route's arrival
    /// at the coordinate.
    egress_active: bool,
    budget: Option<u32>,
    /// Per slot: (grams, seconds, label, egress meters).
    best: &'a mut [Option<(f64, u32, u32, f64)>],
}

impl MatrixFold<'_> {
    fn fold(&mut self, label: &Label, index: u32) {
        let base = label.arrival - label.departure;
        let stop = label.stop.0 as usize;
        // In closure mode the label alighting at its own stop, when that stop is
        // a destination, is credited with no final walk. In door-to-door mode
        // that credit is left to the egress map's self-entry (the destination's
        // connector), so the arrival lands at the coordinate as the single-pair
        // route reports it.
        if !self.egress_active {
            let slot = self.slots[stop];
            if slot != 0 {
                self.credit(slot as usize - 1, label.grams, base, index, 0.0);
            }
            return;
        }
        // Door-to-door mode. An access seed has not ridden; it never folds a
        // cell here. Its walking-only journey to a destination is the explicit
        // coordinate-to-coordinate direct walk, overlaid in the matrix layer
        // (`merge_direct_walk_cells`) where the diagonal is a true zero — folding
        // the access label would instead credit the access-walk-to-the-stop cost.
        if matches!(label.origin, Origin::Access) {
            return;
        }
        // A label that has ridden (a ride's alight, or a transfer off one — every
        // transfer label is a footpath off a ride, so `Access` above is the only
        // zero-ride origin) takes a final egress walk, bounded by
        // `max_walking_time`, to every reachable destination, matching what the
        // single-pair route folds. Read by index so the immutable borrow of
        // `egress` does not overlap `credit`'s mutable borrow of `best`.
        for i in 0..self.egress[stop].len() {
            let (dest, walk_seconds, walk_meters) = self.egress[stop][i];
            self.credit(
                dest as usize,
                label.grams,
                base.saturating_add(walk_seconds),
                index,
                walk_meters,
            );
        }
    }

    /// Keeps the cleanest (then fastest) crediting of `slot`; a travel-time
    /// budget disqualifies outright. `egress_meters` is the final walk's
    /// distance, zero when the label alights at the destination itself.
    fn credit(&mut self, slot: usize, grams: f64, seconds: u32, index: u32, egress_meters: f64) {
        if self.budget.is_some_and(|budget| seconds > budget) {
            return;
        }
        let best = &mut self.best[slot];
        let better = match best {
            None => true,
            Some((at_grams, at, _, _)) => {
                grams < *at_grams || (grams == *at_grams && seconds < *at)
            }
        };
        if better {
            *best = Some((grams, seconds, index, egress_meters));
        }
    }
}

struct Search<'a> {
    view: &'a DayView,
    timetable: &'a Timetable,
    footpaths: &'a Transfers,
    geometry: &'a TripGeometry,
    factors: &'a [f64],
    bucket: f64,
    /// The time-slack band, in seconds; 0 is the strict frontier.
    slack: u32,
    arena: Vec<Label>,
    bags: Vec<Bag>,
    destination: DestinationBag,
    /// Per line: the (position, label) boardings queued this round.
    queue: Vec<Vec<(u16, u32)>>,
    touched: Vec<u32>,
}

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
    let mut slots = vec![0u32; timetable.stop_count() as usize];
    for (index, stop) in destinations.iter().enumerate() {
        slots[stop.0 as usize] = index as u32 + 1;
    }
    requests
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
            );
            let departures = departure_candidates(timetable, request, window);
            let mut best: Vec<Option<(f64, u32, u32, f64)>> = vec![None; destinations.len()];
            for &departure in &departures {
                let mut fold = Some(MatrixFold {
                    slots: &slots,
                    egress,
                    egress_active,
                    budget,
                    best: &mut best,
                });
                search.pass(request, departure, &mut fold);
            }
            best.into_iter()
                .enumerate()
                .filter_map(|(slot, winner)| {
                    winner.map(|winner| {
                        search.cost_row(inputs, winner, destinations[slot].0, access_meters)
                    })
                })
                .collect()
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn profile(
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
) -> Vec<Journey> {
    assert!(
        bucket.is_finite() && bucket > 0.0,
        "the emissions bucket must be positive"
    );
    let mut search = Search::start(view, timetable, footpaths, geometry, factors, bucket, slack);
    for &departure in departures {
        search.pass(request, departure, &mut None);
    }
    let kept = cap_entries(&search.destination.entries, max_options);
    let mut journeys: Vec<Journey> = kept
        .into_iter()
        .map(|arrived| search.assemble(arrived))
        .collect();
    journeys.sort_by_key(|journey| (journey.departure, journey.arrival, journey.rides()));
    journeys
}

/// Strict (departure↓, arrival, emissions bucket) domination between two
/// destination entries — the relation that ranks suboptimal arrivals under
/// `max_options`.
fn strictly_dominates(a: &Arrived, b: &Arrived) -> bool {
    a.departure >= b.departure
        && a.arrival <= b.arrival
        && a.key <= b.key
        && (a.departure > b.departure || a.arrival < b.arrival || a.key < b.key)
}

/// The destination entries to assemble. Without a cap (or when the set
/// already fits) every entry is kept; otherwise the strict frontier — the
/// entries no other entry strictly dominates — is kept in full and the
/// suboptimal arrivals of smallest time-gap above it fill the remainder up
/// to `max_options`, ties toward the cleaner emissions. A suboptimal entry's
/// gap is the seconds by which its nearest strict-frontier dominator arrives
/// earlier. The cap never drops a frontier (optimal) journey, so the result
/// can exceed `max_options` when the frontier itself is larger.
fn cap_entries(entries: &[Arrived], max_options: Option<usize>) -> Vec<&Arrived> {
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
                    .map(|(other, _)| entry.arrival.saturating_sub(other.arrival))
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

impl<'a> Search<'a> {
    fn start(
        view: &'a DayView,
        timetable: &'a Timetable,
        footpaths: &'a Transfers,
        geometry: &'a TripGeometry,
        factors: &'a [f64],
        bucket: f64,
        slack: u32,
    ) -> Search<'a> {
        Search {
            view,
            timetable,
            footpaths,
            geometry,
            factors,
            bucket,
            slack,
            arena: Vec::new(),
            bags: vec![Bag::default(); timetable.stop_count() as usize],
            destination: DestinationBag::default(),
            queue: vec![Vec::new(); view.line_count() as usize],
            touched: Vec::new(),
        }
    }

    fn key(&self, grams: f64) -> i64 {
        (grams / self.bucket).floor() as i64
    }

    /// One profile pass: seed the access labels at `departure`, then
    /// ride/walk/join for each round. Bags persist across passes —
    /// labels of later departures suppress what they dominate on
    /// (arrival, bucket, rides), the range-RAPTOR reuse made sound under
    /// the emissions criterion by ranking the rides used (see
    /// `Bag::insert`). A matrix fold, when given, sees every label the
    /// pass creates (the access seeds are the zero-ride floor of the
    /// origin's own cell).
    fn pass(&mut self, request: &Request, departure: u32, fold: &mut Option<MatrixFold<'_>>) {
        let mut fresh: Vec<u32> = Vec::new();
        for &(stop, seconds) in &request.access {
            let arrival = departure.saturating_add(seconds);
            let label = self.arena.len() as u32;
            let key = self.key(0.0);
            if self.bags[stop.0 as usize].insert_slack(arrival, 0.0, key, 0, self.slack) {
                self.arena.push(Label {
                    arrival,
                    grams: 0.0,
                    stop,
                    departure,
                    origin: Origin::Access,
                });
                if let Some(fold) = fold {
                    fold.fold(&self.arena[label as usize], label);
                }
                fresh.push(label);
            }
        }
        for round in 1..=request.max_transfers as u32 + 1 {
            if fresh.is_empty() {
                break;
            }
            let rides = round as u8;
            for &label in &fresh {
                let stop = self.arena[label as usize].stop;
                for served in self.timetable.patterns_at_stop(stop) {
                    let positions = self.timetable.pattern_stops(served.pattern).len();
                    if served.position as usize + 1 >= positions {
                        continue;
                    }
                    for line in self
                        .view
                        .lines_of_pattern(served.pattern)
                        .into_iter()
                        .flatten()
                    {
                        if self.queue[line as usize].is_empty() {
                            self.touched.push(line);
                        }
                        self.queue[line as usize].push((served.position, label));
                    }
                }
            }
            let mut rode: Vec<u32> = Vec::new();
            let touched = std::mem::take(&mut self.touched);
            for &line in &touched {
                self.scan_line(line, rides, &mut rode);
            }
            // One footpath hop from the improving ride labels; the
            // closed transfer contract makes chains redundant.
            let mut next = rode.clone();
            for &label in &rode {
                let from = self.arena[label as usize];
                let key = self.key(from.grams);
                for footpath in self.footpaths.from_stop(from.stop) {
                    let arrival = from.arrival.saturating_add(footpath.duration);
                    let walked = self.arena.len() as u32;
                    if self.bags[footpath.to.0 as usize]
                        .insert_slack(arrival, from.grams, key, rides, self.slack)
                    {
                        self.arena.push(Label {
                            arrival,
                            grams: from.grams,
                            stop: footpath.to,
                            departure: from.departure,
                            origin: Origin::Walk {
                                parent: label,
                                duration: footpath.duration,
                            },
                        });
                        next.push(walked);
                    }
                }
            }
            for &label in &next {
                let reached = self.arena[label as usize];
                if let Some(fold) = fold {
                    fold.fold(&reached, label);
                }
                for &(stop, seconds) in &request.egress {
                    if stop == reached.stop {
                        let arrival = reached.arrival.saturating_add(seconds);
                        self.destination.insert(
                            Arrived {
                                departure,
                                arrival,
                                key: self.key(reached.grams),
                                grams: reached.grams,
                                label,
                            },
                            self.slack,
                        );
                    }
                }
            }
            fresh = next;
        }
    }

    /// Scans one line: boarding labels enter the rider bag at their
    /// queued positions, riders alight at every later position. Besides
    /// the earliest boardable trip, later trips join while their factor
    /// strictly improves; same-trip riders reduce to the lowest
    /// `kappa`.
    fn scan_line(&mut self, line: u32, rides: u8, rode: &mut Vec<u32>) {
        let mut entries = std::mem::take(&mut self.queue[line as usize]);
        entries.sort_unstable_by_key(|&(position, _)| position);
        let pattern = self.view.line_pattern(line);
        let stops = self.timetable.pattern_stops(pattern);
        let offset = self.view.line_day_offset(line);
        let mut riders: Vec<Rider> = Vec::new();
        let mut queued = 0;
        for position in entries[0].0 as usize..stops.len() {
            for rider in &riders {
                if (rider.board as usize) >= position {
                    continue;
                }
                let times = self.view.stored_times(self.timetable, rider.trip);
                let arrival = times[position].arrival - offset;
                let meters = self.geometry.leg_distance(
                    self.view.backing(rider.trip),
                    rider.board,
                    position as u16,
                ) as f64;
                let grams = quantized(rider.grams + meters / 1000.0 * rider.factor);
                let label = self.arena.len() as u32;
                let key = self.key(grams);
                if self.bags[stops[position].0 as usize]
                    .insert_slack(arrival, grams, key, rides, self.slack)
                {
                    self.arena.push(Label {
                        arrival,
                        grams,
                        stop: stops[position],
                        departure: rider.departure,
                        origin: Origin::Ride {
                            parent: rider.parent,
                            trip: rider.trip,
                            board: rider.board,
                            alight: position as u16,
                        },
                    });
                    rode.push(label);
                }
            }
            while queued < entries.len() && entries[queued].0 as usize == position {
                let (_, label) = entries[queued];
                queued += 1;
                let boarding = self.arena[label as usize];
                let Some(first) = earliest_boardable(
                    self.view,
                    self.timetable,
                    line,
                    position as u16,
                    boarding.arrival,
                ) else {
                    continue;
                };
                let mut cleanest = f64::INFINITY;
                // The latest boarded trip whose factor set `cleanest` — the
                // nearest no-dirtier earlier departure a within-slack trip is
                // measured against (only tracked when relaxing).
                let mut last_clean_departure = if self.slack > 0 {
                    self.view.stored_times(self.timetable, first)[position].departure - offset
                } else {
                    0
                };
                for rank in first.0..self.view.line_trips(line).end {
                    let trip = ViewTrip(rank);
                    let factor = self.factors[self.view.backing(trip).0 as usize];
                    if !factor.is_finite() {
                        continue;
                    }
                    if factor < cleanest {
                        cleanest = factor;
                        if self.slack > 0 {
                            last_clean_departure =
                                self.view.stored_times(self.timetable, trip)[position].departure
                                    - offset;
                        }
                    } else if self.slack == 0 {
                        continue;
                    } else {
                        // Under a slack, board a later same-line trip that is
                        // not strictly cleaner when it departs within the band of
                        // the nearest no-dirtier boarded trip — the "next
                        // departure" alternative strict Pareto drops. The stop
                        // bags prune the ones that fall more than `slack` behind
                        // at their alights.
                        let departure = self.view.stored_times(self.timetable, trip)[position]
                            .departure
                            - offset;
                        if departure.saturating_sub(last_clean_departure) > self.slack {
                            continue;
                        }
                    }
                    let travelled =
                        self.geometry
                            .leg_distance(self.view.backing(trip), 0, position as u16)
                            as f64;
                    let kappa = boarding.grams - travelled / 1000.0 * factor;
                    if riders
                        .iter()
                        .any(|rider| rider.trip == trip && rider.kappa <= kappa)
                    {
                        continue;
                    }
                    riders.retain(|rider| !(rider.trip == trip && kappa < rider.kappa));
                    riders.push(Rider {
                        trip,
                        board: position as u16,
                        kappa,
                        grams: boarding.grams,
                        factor,
                        parent: label,
                        departure: boarding.departure,
                    });
                }
            }
        }
        entries.clear();
        self.queue[line as usize] = entries;
    }

    /// Walks a winning label's chain into a cost row, mirroring the
    /// interim reconstruction: transit and transfer meters summed leg
    /// by leg, geometry legs reversed into ride order, fare legs
    /// priced in order. Emissions come from the label itself — the
    /// same cumulative-distance sums, already microgram-quantized.
    fn cost_row(
        &self,
        inputs: &CostInputs<'_>,
        winner: (f64, u32, u32, f64),
        to: u32,
        access_meters: &[(StopIdx, f64)],
    ) -> CostRow {
        let (grams, seconds, mut at, egress_meters) = winner;
        let mut rides = 0u32;
        let mut transit_meters = 0.0;
        // The final walk to the destination (zero when the label alights there).
        let mut walk_meters = egress_meters;
        let mut legs: Vec<(TripIdx, u16, u16)> = Vec::new();
        let mut fare_legs: Vec<FareLeg> = Vec::new();
        loop {
            let label = &self.arena[at as usize];
            match label.origin {
                Origin::Access => break,
                Origin::Ride {
                    parent,
                    trip,
                    board,
                    alight,
                } => {
                    rides += 1;
                    let backing = self.view.backing(trip);
                    transit_meters += inputs.geometry.leg_distance(backing, board, alight) as f64;
                    if inputs.with_geometry {
                        legs.push((backing, board, alight));
                    }
                    if inputs.fares.is_some() {
                        let pattern = self.timetable.trip_pattern(backing);
                        let stops = self.timetable.pattern_stops(pattern);
                        fare_legs.push(FareLeg {
                            route: self.timetable.pattern_route(pattern),
                            board_stop: stops[board as usize].0,
                            alight_stop: stops[alight as usize].0,
                            board_time: self.timetable.trip_stop_times(backing)[board as usize]
                                .departure
                                .saturating_sub(self.view.day_offset(trip)),
                        });
                    }
                    at = parent;
                }
                Origin::Walk { parent, .. } => {
                    let from = &self.arena[parent as usize];
                    walk_meters += self
                        .footpaths
                        .from_stop(from.stop)
                        .iter()
                        .find(|transfer| transfer.to == label.stop)
                        .map(|transfer| transfer.meters)
                        .unwrap_or(0.0);
                    at = parent;
                }
            }
        }
        // The initial walk from the origin coordinate to the boarded stop, in
        // door-to-door mode (empty access-meter map otherwise).
        let access_stop = self.arena[at as usize].stop;
        walk_meters += access_meters
            .iter()
            .find(|(stop, _)| *stop == access_stop)
            .map(|(_, meters)| *meters)
            .unwrap_or(0.0);
        let geometry = match (inputs.with_geometry, inputs.leg_geometry) {
            (true, Some(shapes)) => {
                let parts: Vec<Vec<(f64, f64)>> = legs
                    .iter()
                    .rev()
                    .map(|&(trip, board, alight)| shapes.leg_coordinates(trip, board, alight))
                    .collect();
                Some(wkb_multi_line_string(&parts))
            }
            _ => None,
        };
        let fare = match inputs.fares {
            Some(tables) => {
                fare_legs.reverse();
                tables.price(&fare_legs)
            }
            None => f64::NAN,
        };
        CostRow {
            to,
            seconds,
            rides,
            transit_meters,
            walk_meters,
            emission_grams: grams,
            fare,
            geometry,
        }
    }

    /// Walks a destination entry's label chain back into the journey
    /// contract.
    fn assemble(&self, arrived: &Arrived) -> Journey {
        let last = &self.arena[arrived.label as usize];
        let mut legs = vec![Leg::Egress {
            from_stop: last.stop,
            departure: last.arrival,
            arrival: arrived.arrival,
        }];
        let mut at = arrived.label;
        loop {
            let label = &self.arena[at as usize];
            match label.origin {
                Origin::Access => {
                    legs.push(Leg::Access {
                        to_stop: label.stop,
                        departure: arrived.departure,
                        arrival: label.arrival,
                    });
                    break;
                }
                Origin::Ride {
                    parent,
                    trip,
                    board,
                    alight,
                } => {
                    let line = self.view.line_of(trip);
                    let offset = self.view.line_day_offset(line);
                    let stops = self.timetable.pattern_stops(self.view.line_pattern(line));
                    let times = self.view.stored_times(self.timetable, trip);
                    legs.push(Leg::Transit {
                        trip: self.view.backing(trip),
                        board_stop: stops[board as usize],
                        alight_stop: stops[alight as usize],
                        board_position: board,
                        alight_position: alight,
                        board_time: times[board as usize].departure - offset,
                        alight_time: times[alight as usize].arrival - offset,
                    });
                    at = parent;
                }
                Origin::Walk { parent, duration } => {
                    let from = &self.arena[parent as usize];
                    legs.push(Leg::Transfer {
                        from_stop: from.stop,
                        to_stop: label.stop,
                        departure: from.arrival,
                        arrival: from.arrival.saturating_add(duration),
                    });
                    at = parent;
                }
            }
        }
        legs.reverse();
        Journey {
            departure: arrived.departure,
            arrival: arrived.arrival,
            legs,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exhaustive::pareto_oracle;
    use crate::geometry::DistanceProvenance;
    use crate::timetable::{StopTime, TimetableBuilder, TripIdx};

    fn time(at: u32) -> StopTime {
        StopTime {
            arrival: at,
            departure: at,
        }
    }

    fn request(access: StopIdx, egress: StopIdx, max_transfers: u8) -> Request {
        Request {
            departure: 0,
            access: vec![(access, 0)],
            egress: vec![(egress, 0)],
            active_services: Vec::new(),
            active_services_previous: Vec::new(),
            max_transfers,
        }
    }

    fn grams_of(journey: &Journey, geometry: &TripGeometry, factors: &[f64]) -> f64 {
        quantized(
            journey
                .legs
                .iter()
                .map(|leg| match leg {
                    Leg::Transit {
                        trip,
                        board_position,
                        alight_position,
                        ..
                    } => {
                        geometry.leg_distance(*trip, *board_position, *alight_position) as f64
                            / 1000.0
                            * factors[trip.0 as usize]
                    }
                    _ => 0.0,
                })
                .sum(),
        )
    }

    fn triples(
        journeys: &[Journey],
        geometry: &TripGeometry,
        factors: &[f64],
    ) -> Vec<(u32, f64, u32)> {
        let mut triples: Vec<(u32, f64, u32)> = journeys
            .iter()
            .map(|journey| {
                (
                    journey.arrival,
                    grams_of(journey, geometry, factors),
                    journey.rides() as u32,
                )
            })
            .collect();
        triples.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.partial_cmp(&b.1).unwrap()));
        triples
    }

    fn oracle_triples(points: &[crate::exhaustive::ParetoPoint]) -> Vec<(u32, f64, u32)> {
        points
            .iter()
            .map(|point| (point.arrival, point.grams, point.rides))
            .collect()
    }

    /// The exhaustive oracle's frontier fixture: a fast dirty direct
    /// line, a slow clean one, and a cleaner-but-slower combination
    /// over a transfer.
    fn frontier_fixture() -> (Timetable, TripGeometry, [f64; 4]) {
        let mut builder = TimetableBuilder::new(4);
        let dirty = builder.add_pattern(&[StopIdx(0), StopIdx(3)], 0).unwrap();
        let clean = builder.add_pattern(&[StopIdx(0), StopIdx(3)], 1).unwrap();
        let combo_a = builder.add_pattern(&[StopIdx(0), StopIdx(1)], 2).unwrap();
        let combo_b = builder.add_pattern(&[StopIdx(1), StopIdx(3)], 3).unwrap();
        builder
            .add_trip(dirty, vec![time(100), time(500)], 0, 0)
            .unwrap();
        builder
            .add_trip(clean, vec![time(100), time(900)], 1, 0)
            .unwrap();
        builder
            .add_trip(combo_a, vec![time(100), time(300)], 2, 0)
            .unwrap();
        builder
            .add_trip(combo_b, vec![time(400), time(700)], 3, 0)
            .unwrap();
        let timetable = builder.finish();
        let geometry = TripGeometry::from_trips(
            &timetable,
            vec![
                (TripIdx(0), vec![0.0, 1000.0], DistanceProvenance::CrowFly),
                (TripIdx(1), vec![0.0, 1000.0], DistanceProvenance::CrowFly),
                (TripIdx(2), vec![0.0, 500.0], DistanceProvenance::CrowFly),
                (TripIdx(3), vec![0.0, 500.0], DistanceProvenance::CrowFly),
            ],
        )
        .unwrap();
        (timetable, geometry, [100.0, 10.0, 100.0, 10.0])
    }

    #[test]
    fn matches_the_oracle_with_a_vanishing_bucket() {
        let (timetable, geometry, factors) = frontier_fixture();
        let view = DayView::universal(&timetable);
        let footpaths = Transfers::empty(4);
        let request = request(StopIdx(0), StopIdx(3), 3);
        let journeys = route(
            &view, &timetable, &footpaths, &geometry, &factors, &request, 1e-6, 0, None,
        );
        let points = pareto_oracle(
            &view,
            &timetable,
            &footpaths,
            &geometry,
            &factors,
            0,
            &request.access,
            &request.egress,
            3,
        );
        assert_eq!(
            triples(&journeys, &geometry, &factors),
            oracle_triples(&points)
        );
        assert_eq!(
            oracle_triples(&points),
            vec![(500, 100.0, 1), (700, 55.0, 2), (900, 10.0, 1)]
        );
    }

    /// Two frontier journeys — a fast dirty line and a slower cleaner one
    /// — plus a middle line that strict Pareto drops as dominated but a
    /// time slack keeps as a suboptimal alternative.
    fn relaxed_fixture() -> (Timetable, TripGeometry, [f64; 3]) {
        let mut builder = TimetableBuilder::new(2);
        let fast = builder.add_pattern(&[StopIdx(0), StopIdx(1)], 0).unwrap();
        let clean = builder.add_pattern(&[StopIdx(0), StopIdx(1)], 1).unwrap();
        let middle = builder.add_pattern(&[StopIdx(0), StopIdx(1)], 2).unwrap();
        builder
            .add_trip(fast, vec![time(100), time(500)], 0, 0)
            .unwrap();
        builder
            .add_trip(clean, vec![time(100), time(700)], 1, 0)
            .unwrap();
        builder
            .add_trip(middle, vec![time(100), time(600)], 2, 0)
            .unwrap();
        let timetable = builder.finish();
        let geometry = TripGeometry::from_trips(
            &timetable,
            vec![
                (TripIdx(0), vec![0.0, 1000.0], DistanceProvenance::CrowFly),
                (TripIdx(1), vec![0.0, 1000.0], DistanceProvenance::CrowFly),
                (TripIdx(2), vec![0.0, 1000.0], DistanceProvenance::CrowFly),
            ],
        )
        .unwrap();
        (timetable, geometry, [100.0, 60.0, 100.0])
    }

    #[test]
    fn zero_slack_matches_the_strict_frontier() {
        let (timetable, geometry, factors) = relaxed_fixture();
        let view = DayView::universal(&timetable);
        let footpaths = Transfers::empty(2);
        let request = request(StopIdx(0), StopIdx(1), 3);
        let strict = route(
            &view, &timetable, &footpaths, &geometry, &factors, &request, 1e-6, 0, None,
        );
        // Only the fast dirty line and the slow clean one; the middle line
        // is strictly dominated and dropped.
        assert_eq!(
            triples(&strict, &geometry, &factors),
            vec![(500, 100.0, 1), (700, 60.0, 1)]
        );
    }

    #[test]
    fn a_time_slack_keeps_the_suboptimal_middle_line() {
        let (timetable, geometry, factors) = relaxed_fixture();
        let view = DayView::universal(&timetable);
        let footpaths = Transfers::empty(2);
        let request = request(StopIdx(0), StopIdx(1), 3);
        // 200 s exceeds the 100 s the fast line beats the middle line by,
        // so the middle line (600, dominated by 500) is retained.
        let relaxed = route(
            &view, &timetable, &footpaths, &geometry, &factors, &request, 1e-6, 200, None,
        );
        assert_eq!(
            triples(&relaxed, &geometry, &factors),
            vec![(500, 100.0, 1), (600, 100.0, 1), (700, 60.0, 1)]
        );
    }

    #[test]
    fn max_options_keeps_the_frontier_over_the_suboptimal() {
        let (timetable, geometry, factors) = relaxed_fixture();
        let view = DayView::universal(&timetable);
        let footpaths = Transfers::empty(2);
        let request = request(StopIdx(0), StopIdx(1), 3);
        // The relaxed set is three journeys; a cap of two keeps the strict
        // frontier and drops the suboptimal middle line.
        let capped = route(
            &view,
            &timetable,
            &footpaths,
            &geometry,
            &factors,
            &request,
            1e-6,
            200,
            Some(2),
        );
        assert_eq!(
            triples(&capped, &geometry, &factors),
            vec![(500, 100.0, 1), (700, 60.0, 1)]
        );
    }

    /// One line, two trips one headway apart at the same factor: the later
    /// trip is strictly dominated and never boarded by the strict line scan.
    fn same_line_fixture() -> (Timetable, TripGeometry, [f64; 2]) {
        let mut builder = TimetableBuilder::new(2);
        let line = builder.add_pattern(&[StopIdx(0), StopIdx(1)], 0).unwrap();
        builder
            .add_trip(line, vec![time(100), time(500)], 0, 0)
            .unwrap();
        builder
            .add_trip(line, vec![time(200), time(600)], 1, 0)
            .unwrap();
        let timetable = builder.finish();
        let geometry = TripGeometry::from_trips(
            &timetable,
            vec![
                (TripIdx(0), vec![0.0, 1000.0], DistanceProvenance::CrowFly),
                (TripIdx(1), vec![0.0, 1000.0], DistanceProvenance::CrowFly),
            ],
        )
        .unwrap();
        (timetable, geometry, [100.0, 100.0])
    }

    #[test]
    fn a_time_slack_boards_the_next_same_line_trip() {
        let (timetable, geometry, factors) = same_line_fixture();
        let view = DayView::universal(&timetable);
        let footpaths = Transfers::empty(2);
        let request = request(StopIdx(0), StopIdx(1), 3);
        // Strict Pareto boards only the earliest trip; the later same-factor
        // trip arrives later at no lower emissions, so it is dropped.
        let strict = route(
            &view, &timetable, &footpaths, &geometry, &factors, &request, 1e-6, 0, None,
        );
        assert_eq!(triples(&strict, &geometry, &factors), vec![(500, 100.0, 1)]);
        // A 200 s slack boards the next departure too — the "one trip later"
        // alternative the strict line scan never surfaces.
        let relaxed = route(
            &view, &timetable, &footpaths, &geometry, &factors, &request, 1e-6, 200, None,
        );
        assert_eq!(
            triples(&relaxed, &geometry, &factors),
            vec![(500, 100.0, 1), (600, 100.0, 1)]
        );
    }

    /// One line, three trips: a dirty first, a much-later clean one, and a
    /// middle-factor trip just after the clean one. The middle trip is within
    /// slack of the clean frontier trip but far beyond the first departure —
    /// only measuring against the nearest no-dirtier boarded trip admits it.
    fn later_frontier_fixture() -> (Timetable, TripGeometry, [f64; 3]) {
        let mut builder = TimetableBuilder::new(2);
        let line = builder.add_pattern(&[StopIdx(0), StopIdx(1)], 0).unwrap();
        builder
            .add_trip(line, vec![time(100), time(500)], 0, 0)
            .unwrap();
        builder
            .add_trip(line, vec![time(900), time(1300)], 1, 0)
            .unwrap();
        builder
            .add_trip(line, vec![time(1000), time(1400)], 2, 0)
            .unwrap();
        let timetable = builder.finish();
        let geometry = TripGeometry::from_trips(
            &timetable,
            vec![
                (TripIdx(0), vec![0.0, 1000.0], DistanceProvenance::CrowFly),
                (TripIdx(1), vec![0.0, 1000.0], DistanceProvenance::CrowFly),
                (TripIdx(2), vec![0.0, 1000.0], DistanceProvenance::CrowFly),
            ],
        )
        .unwrap();
        (timetable, geometry, [100.0, 10.0, 50.0])
    }

    #[test]
    fn a_time_slack_boards_the_next_trip_after_a_later_frontier() {
        let (timetable, geometry, factors) = later_frontier_fixture();
        let view = DayView::universal(&timetable);
        let footpaths = Transfers::empty(2);
        let request = request(StopIdx(0), StopIdx(1), 3);
        // Strict Pareto keeps the two frontier trips only.
        let strict = route(
            &view, &timetable, &footpaths, &geometry, &factors, &request, 1e-6, 0, None,
        );
        assert_eq!(
            triples(&strict, &geometry, &factors),
            vec![(500, 100.0, 1), (1300, 10.0, 1)]
        );
        // A 200 s slack admits the middle-factor trip 100 s after the clean
        // frontier trip — far beyond the first departure, so measuring only
        // against the first would wrongly drop it.
        let relaxed = route(
            &view, &timetable, &footpaths, &geometry, &factors, &request, 1e-6, 200, None,
        );
        assert_eq!(
            triples(&relaxed, &geometry, &factors),
            vec![(500, 100.0, 1), (1300, 10.0, 1), (1400, 50.0, 1)]
        );
    }

    #[test]
    fn a_wide_bucket_collapses_to_the_fastest_journey() {
        let (timetable, geometry, factors) = frontier_fixture();
        let view = DayView::universal(&timetable);
        let footpaths = Transfers::empty(4);
        let request = request(StopIdx(0), StopIdx(3), 3);
        let journeys = route(
            &view, &timetable, &footpaths, &geometry, &factors, &request, 1e9, 0, None,
        );
        assert_eq!(
            triples(&journeys, &geometry, &factors),
            vec![(500, 100.0, 1)]
        );
    }

    #[test]
    fn loop_backs_walk_nowhere() {
        // The oracle's regression shape: ride out and back, then walk
        // to the destination — cleaner and earlier than the direct
        // line, but suppressed by the access label's dominance.
        let mut builder = TimetableBuilder::new(3);
        let out = builder.add_pattern(&[StopIdx(0), StopIdx(1)], 0).unwrap();
        let back = builder.add_pattern(&[StopIdx(1), StopIdx(0)], 1).unwrap();
        let direct = builder.add_pattern(&[StopIdx(0), StopIdx(2)], 2).unwrap();
        builder
            .add_trip(out, vec![time(10), time(50)], 0, 0)
            .unwrap();
        builder
            .add_trip(back, vec![time(60), time(100)], 1, 0)
            .unwrap();
        builder
            .add_trip(direct, vec![time(20), time(200)], 2, 0)
            .unwrap();
        let timetable = builder.finish();
        let geometry = TripGeometry::from_trips(
            &timetable,
            vec![
                (TripIdx(0), vec![0.0, 400.0], DistanceProvenance::CrowFly),
                (TripIdx(1), vec![0.0, 400.0], DistanceProvenance::CrowFly),
                (TripIdx(2), vec![0.0, 800.0], DistanceProvenance::CrowFly),
            ],
        )
        .unwrap();
        let factors = [10.0, 10.0, 100.0];
        let footpaths = Transfers::from_edges(
            3,
            &[
                (StopIdx(0), StopIdx(2), 30, 30.0),
                (StopIdx(2), StopIdx(0), 30, 30.0),
            ],
        )
        .unwrap();
        let view = DayView::universal(&timetable);
        let request = request(StopIdx(0), StopIdx(2), 3);
        let journeys = route(
            &view, &timetable, &footpaths, &geometry, &factors, &request, 1e-6, 0, None,
        );
        assert_eq!(
            triples(&journeys, &geometry, &factors),
            vec![(200, 80.0, 1)]
        );
    }

    #[test]
    fn waits_for_the_cleaner_trip_on_a_mixed_factor_line() {
        // One line, a dirty early trip and a clean later one: the true
        // frontier holds both, so boarding must look past the earliest
        // boardable trip when a later factor strictly improves.
        let mut builder = TimetableBuilder::new(2);
        let line = builder.add_pattern(&[StopIdx(0), StopIdx(1)], 0).unwrap();
        builder
            .add_trip(line, vec![time(100), time(200)], 0, 0)
            .unwrap();
        builder
            .add_trip(line, vec![time(300), time(400)], 1, 0)
            .unwrap();
        let timetable = builder.finish();
        let geometry = TripGeometry::from_trips(
            &timetable,
            vec![
                (TripIdx(0), vec![0.0, 1000.0], DistanceProvenance::CrowFly),
                (TripIdx(1), vec![0.0, 1000.0], DistanceProvenance::CrowFly),
            ],
        )
        .unwrap();
        let factors = [100.0, 10.0];
        let view = DayView::universal(&timetable);
        let footpaths = Transfers::empty(2);
        let request = request(StopIdx(0), StopIdx(1), 1);
        let journeys = route(
            &view, &timetable, &footpaths, &geometry, &factors, &request, 1e-6, 0, None,
        );
        let points = pareto_oracle(
            &view,
            &timetable,
            &footpaths,
            &geometry,
            &factors,
            0,
            &request.access,
            &request.egress,
            1,
        );
        assert_eq!(
            triples(&journeys, &geometry, &factors),
            oracle_triples(&points)
        );
        assert_eq!(
            oracle_triples(&points),
            vec![(200, 100.0, 1), (400, 10.0, 1)]
        );
    }

    #[test]
    fn transfers_over_a_footpath_match_the_oracle() {
        // Ride, walk a footpath, ride again — the walked hop must
        // carry its grams unchanged and reconstruct as a transfer leg.
        let mut builder = TimetableBuilder::new(4);
        let first = builder.add_pattern(&[StopIdx(0), StopIdx(1)], 0).unwrap();
        let second = builder.add_pattern(&[StopIdx(2), StopIdx(3)], 1).unwrap();
        builder
            .add_trip(first, vec![time(0), time(100)], 0, 0)
            .unwrap();
        builder
            .add_trip(second, vec![time(200), time(300)], 1, 0)
            .unwrap();
        let timetable = builder.finish();
        let geometry = TripGeometry::from_trips(
            &timetable,
            vec![
                (TripIdx(0), vec![0.0, 1000.0], DistanceProvenance::CrowFly),
                (TripIdx(1), vec![0.0, 2000.0], DistanceProvenance::CrowFly),
            ],
        )
        .unwrap();
        let factors = [10.0, 20.0];
        let footpaths = Transfers::from_edges(
            4,
            &[
                (StopIdx(1), StopIdx(2), 50, 50.0),
                (StopIdx(2), StopIdx(1), 50, 50.0),
            ],
        )
        .unwrap();
        let view = DayView::universal(&timetable);
        let request = request(StopIdx(0), StopIdx(3), 3);
        let journeys = route(
            &view, &timetable, &footpaths, &geometry, &factors, &request, 1e-6, 0, None,
        );
        let points = pareto_oracle(
            &view,
            &timetable,
            &footpaths,
            &geometry,
            &factors,
            0,
            &request.access,
            &request.egress,
            3,
        );
        assert_eq!(
            triples(&journeys, &geometry, &factors),
            oracle_triples(&points)
        );
        assert_eq!(oracle_triples(&points), vec![(300, 50.0, 2)]);
        let legs: Vec<&str> = journeys[0]
            .legs
            .iter()
            .map(|leg| match leg {
                Leg::Access { .. } => "access",
                Leg::Transit { .. } => "transit",
                Leg::Transfer { .. } => "transfer",
                Leg::Egress { .. } => "egress",
            })
            .collect();
        assert_eq!(
            legs,
            vec!["access", "transit", "transfer", "transit", "egress"]
        );
    }

    #[test]
    fn skips_unresolved_factors() {
        let mut builder = TimetableBuilder::new(2);
        let line = builder.add_pattern(&[StopIdx(0), StopIdx(1)], 0).unwrap();
        builder
            .add_trip(line, vec![time(0), time(100)], 0, 0)
            .unwrap();
        let timetable = builder.finish();
        let geometry = TripGeometry::from_trips(
            &timetable,
            vec![(TripIdx(0), vec![0.0, 1000.0], DistanceProvenance::CrowFly)],
        )
        .unwrap();
        let view = DayView::universal(&timetable);
        let request = request(StopIdx(0), StopIdx(1), 1);
        let journeys = route(
            &view,
            &timetable,
            &Transfers::empty(2),
            &geometry,
            &[f64::NAN],
            &request,
            1e-6,
            0,
            None,
        );
        assert!(journeys.is_empty());
    }

    #[test]
    fn the_emissions_matrix_sees_past_the_time_candidates() {
        use crate::raptor::{Objective, Raptor};

        let (timetable, geometry, factors) = frontier_fixture();
        let view = DayView::universal(&timetable);
        let footpaths = Transfers::empty(4);
        let inputs = CostInputs {
            geometry: &geometry,
            factors: &factors,
            leg_geometry: None,
            with_geometry: false,
            fares: None,
        };
        let requests = vec![Request {
            departure: 0,
            access: vec![(StopIdx(0), 0)],
            egress: Vec::new(),
            active_services: vec![true],
            active_services_previous: vec![false],
            max_transfers: 3,
        }];
        let destinations = [StopIdx(0), StopIdx(3)];
        let rows = least_emissions_matrix(
            &view,
            &timetable,
            &footpaths,
            &inputs,
            &requests,
            &destinations,
            &vec![Vec::new(); timetable.stop_count() as usize],
            &vec![Vec::new(); requests.len()],
            false,
            600,
            None,
            1e-6,
        );
        // The cleanest journey is the slow clean line — invisible to
        // the interim objective, whose per-round RAPTOR arrivals only
        // ever hold the faster dirty alternatives.
        let cell = rows[0].iter().find(|row| row.to == 3).unwrap();
        assert_eq!((cell.seconds, cell.rides), (800, 1));
        assert!((cell.emission_grams - 10.0).abs() < 1e-9);
        assert_eq!(cell.transit_meters, 1000.0);
        let interim = Raptor.least_cost_matrix(
            &timetable,
            &footpaths,
            &inputs,
            &requests,
            &destinations,
            600,
            None,
            Objective::Emissions,
        );
        let interim_cell = interim[0].iter().find(|row| row.to == 3).unwrap();
        assert!((interim_cell.emission_grams - 100.0).abs() < 1e-9);
        assert!(cell.emission_grams < interim_cell.emission_grams);
        // The origin's own cell is the zero-ride floor.
        let floor = rows[0].iter().find(|row| row.to == 0).unwrap();
        assert_eq!((floor.seconds, floor.rides), (0, 0));
        assert_eq!(floor.emission_grams, 0.0);
    }

    #[test]
    fn a_budget_caps_the_matrix_travel_time() {
        let (timetable, geometry, factors) = frontier_fixture();
        let view = DayView::universal(&timetable);
        let footpaths = Transfers::empty(4);
        let inputs = CostInputs {
            geometry: &geometry,
            factors: &factors,
            leg_geometry: None,
            with_geometry: false,
            fares: None,
        };
        let requests = vec![Request {
            departure: 0,
            access: vec![(StopIdx(0), 0)],
            egress: Vec::new(),
            active_services: vec![true],
            active_services_previous: vec![false],
            max_transfers: 3,
        }];
        let cell = |budget: Option<u32>| {
            let rows = least_emissions_matrix(
                &view,
                &timetable,
                &footpaths,
                &inputs,
                &requests,
                &[StopIdx(3)],
                &vec![Vec::new(); timetable.stop_count() as usize],
                &vec![Vec::new(); requests.len()],
                false,
                600,
                budget,
                1e-6,
            );
            rows[0].iter().find(|row| row.to == 3).cloned()
        };
        // Tightening budgets walk the frontier: the clean line, the
        // transfer combination, the dirty direct, then nothing.
        let unbudgeted = cell(None).unwrap();
        assert_eq!((unbudgeted.seconds, unbudgeted.rides), (800, 1));
        let combo = cell(Some(600)).unwrap();
        assert_eq!((combo.seconds, combo.rides), (600, 2));
        assert!((combo.emission_grams - 55.0).abs() < 1e-9);
        let dirty = cell(Some(400)).unwrap();
        assert_eq!((dirty.seconds, dirty.rides), (400, 1));
        assert!((dirty.emission_grams - 100.0).abs() < 1e-9);
        assert!(cell(Some(100)).is_none());
    }

    #[test]
    fn matrix_rows_carry_their_transfer_walks() {
        let mut builder = TimetableBuilder::new(4);
        let first = builder.add_pattern(&[StopIdx(0), StopIdx(1)], 0).unwrap();
        let second = builder.add_pattern(&[StopIdx(2), StopIdx(3)], 1).unwrap();
        builder
            .add_trip(first, vec![time(0), time(100)], 0, 0)
            .unwrap();
        builder
            .add_trip(second, vec![time(200), time(300)], 1, 0)
            .unwrap();
        let timetable = builder.finish();
        let geometry = TripGeometry::from_trips(
            &timetable,
            vec![
                (TripIdx(0), vec![0.0, 1000.0], DistanceProvenance::CrowFly),
                (TripIdx(1), vec![0.0, 2000.0], DistanceProvenance::CrowFly),
            ],
        )
        .unwrap();
        let factors = [10.0, 20.0];
        let footpaths = Transfers::from_edges(
            4,
            &[
                (StopIdx(1), StopIdx(2), 50, 50.0),
                (StopIdx(2), StopIdx(1), 50, 50.0),
            ],
        )
        .unwrap();
        let view = DayView::universal(&timetable);
        let inputs = CostInputs {
            geometry: &geometry,
            factors: &factors,
            leg_geometry: None,
            with_geometry: false,
            fares: None,
        };
        let requests = vec![Request {
            departure: 0,
            access: vec![(StopIdx(0), 0)],
            egress: Vec::new(),
            active_services: vec![true],
            active_services_previous: vec![false],
            max_transfers: 3,
        }];
        let rows = least_emissions_matrix(
            &view,
            &timetable,
            &footpaths,
            &inputs,
            &requests,
            &[StopIdx(3)],
            &vec![Vec::new(); timetable.stop_count() as usize],
            &vec![Vec::new(); requests.len()],
            false,
            100,
            None,
            1e-6,
        );
        let cell = rows[0].iter().find(|row| row.to == 3).unwrap();
        assert_eq!((cell.seconds, cell.rides), (300, 2));
        assert_eq!(cell.transit_meters, 3000.0);
        assert_eq!(cell.walk_meters, 50.0);
        assert!((cell.emission_grams - 50.0).abs() < 1e-9);
        assert!(cell.fare.is_nan());
    }

    #[test]
    fn profiles_the_departure_window() {
        // Two departures of one line inside the window: the profile
        // keeps both journeys, each at the latest departure catching
        // it.
        let mut builder = TimetableBuilder::new(2);
        let line = builder.add_pattern(&[StopIdx(0), StopIdx(1)], 0).unwrap();
        builder
            .add_trip(line, vec![time(100), time(300)], 0, 0)
            .unwrap();
        builder
            .add_trip(line, vec![time(200), time(400)], 1, 0)
            .unwrap();
        let timetable = builder.finish();
        let geometry = TripGeometry::from_trips(
            &timetable,
            vec![
                (TripIdx(0), vec![0.0, 1000.0], DistanceProvenance::CrowFly),
                (TripIdx(1), vec![0.0, 1000.0], DistanceProvenance::CrowFly),
            ],
        )
        .unwrap();
        let factors = [50.0, 50.0];
        let view = DayView::universal(&timetable);
        let footpaths = Transfers::empty(2);
        let request = Request {
            departure: 50,
            access: vec![(StopIdx(0), 0)],
            egress: vec![(StopIdx(1), 0)],
            active_services: vec![true],
            active_services_previous: vec![false],
            max_transfers: 1,
        };
        let journeys = route_range(
            &view, &timetable, &footpaths, &geometry, &factors, &request, 200, 1e-6, 0, None,
        );
        let profile: Vec<(u32, u32)> = journeys
            .iter()
            .map(|journey| (journey.departure, journey.arrival))
            .collect();
        assert_eq!(profile, vec![(100, 300), (200, 400)]);
    }
}
