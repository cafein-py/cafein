//! The exact zone-ticket (time, fare) frontier search — the zone
//! tariffs' counterpart of [`fare_frontier`], sharing its profile
//! structure and departure disciplines.
//!
//! Labels carry the money spent plus the active ticket's remaining
//! resources (coverage, expiry, boardings). Each boarding either
//! extends the active ticket or branches into buying every covering
//! product — buying closes the previous ticket, so the branches
//! enumerate exactly the contiguous ticket partitions
//! [`ZoneFares::price`] folds over — and alighting discards branches
//! whose active ticket does not cover the alighting zone. Dominance
//! compares the routing axes and the remaining resources
//! (paid ≤, coverage ⊇, expiry ≥, boardings ≥); product identity is
//! irrelevant once paid. Seeds walk one bounded transfer before the
//! first boarding, fare free, and departure events are enumerated
//! over that walk-extended access set. A branch-and-bound money
//! ceiling — the caller's top cutoff, tightened to the best fare
//! folded so far — prunes exactly: paid totals only grow along a
//! chain, and equal spenders survive to improve travel time.

use crate::fares::{ZoneFares, NO_FARE};
use crate::routers::fare_frontier::{fold_candidate, Arrived, FrontierRow};
use crate::routers::raptor::earliest_active_trip;
use crate::routers::router::Request;
use crate::timetable::{StopIdx, Timetable, TripIdx};
use crate::transfers::Transfers;

const UNREACHED: u32 = u32::MAX;
pub const MONEY_EPSILON: f64 = 1e-9;
/// The arena handle of a caller-seeded row: not a real chain. A
/// returned row still carrying it means the search found nothing
/// better than the seed.
pub const SEED_NODE: u32 = u32::MAX;
const DAY_SECONDS: u32 = 86_400;

/// The network-and-tariff inputs one zone-frontier query runs on.
pub struct ZoneFrontierInputs<'a> {
    pub timetable: &'a Timetable,
    pub transfers: &'a Transfers,
    pub fares: &'a ZoneFares,
    /// Labels whose paid total exceeds this are dead: paid only grows.
    pub top_cutoff: f64,
    /// A bound on `arrival − departure`, seconds; `None` unbounded.
    pub max_duration: Option<u32>,
    /// As on the fare frontier: `Some(step)` rasterises the window,
    /// `None` runs one pass per departure event.
    pub departure_step: Option<u32>,
    /// Whether seeds relax one bounded transfer before the first
    /// boarding. On for stop access — an origin stop may only be
    /// boardable from a neighbour; off for point access, whose street
    /// rows arrive walk-complete, so seeding another walk would
    /// compose two walking segments past the bounded contract.
    pub seed_walks: bool,
}

/// The active ticket's remaining resources.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Ticket {
    product: u32,
    /// Absolute second the ticket stops covering boardings.
    expiry: u32,
    /// Boardings left on this ticket; `u32::MAX` when unlimited.
    remaining: u32,
}

/// A label's fare side: `None` before the first boarding (walking is
/// free), the paid total and active ticket after it. The kinds never
/// compare, exactly as on the rule-based frontier.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Side {
    Unboarded,
    Boarded { paid: f64, ticket: Ticket },
}

#[derive(Clone, Copy, Debug)]
struct Entry {
    arrival: u32,
    rides: u8,
    side: Side,
    /// This entry's node in the reconstruction arena; bags evict
    /// entries, the arena persists, so a folded winner's chain stays
    /// walkable after the search moves on.
    node: u32,
}

/// One step of a journey chain, parent-linked into the arena.
#[derive(Clone, Copy, Debug)]
pub enum Node {
    /// Left the origin at `leave` and reached `stop` (directly, or by
    /// the seed walk whose seconds are folded into `leave`).
    Seed { stop: StopIdx, leave: u32 },
    /// Rode `trip` from `board_position` to `alight_position`.
    Ride {
        parent: u32,
        trip: TripIdx,
        day_offset: u32,
        board_position: u16,
        alight_position: u16,
    },
    /// Walked one bounded transfer from `from`.
    Walk {
        parent: u32,
        from: StopIdx,
        to: StopIdx,
    },
}

/// One boarded run along a pattern; same-trip runs share every future
/// arrival, so only their states compare.
#[derive(Clone, Copy)]
struct Riding {
    trip: TripIdx,
    day_offset: u32,
    rides: u8,
    paid: f64,
    ticket: Ticket,
    /// The boarding entry's arena node and pattern position.
    parent: u32,
    board_position: u16,
}

/// The zone coverage of a product, as bits over the model's zones.
fn coverage(fares: &ZoneFares, product: u32) -> u128 {
    fares.products[product as usize].zones
}

/// A stop's zone bit; `None` for a zone-less stop (it can never be a
/// fare endpoint) and for zones past the 128 the model supports.
fn zone_bit(fares: &ZoneFares, stop: StopIdx) -> Option<u128> {
    match fares.stop_zone.get(stop.0 as usize).copied() {
        Some(zone) if zone != NO_FARE && zone < 128 => Some(1u128 << zone),
        _ => None,
    }
}

/// Whether `a` dominates `b`: routing axes plus remaining resources.
/// Boarded and Unboarded never compare — an unboarded label has spent
/// nothing but must still buy, a boarded one may ride free.
fn dominates(fares: &ZoneFares, a: &Entry, b: &Entry) -> bool {
    if a.arrival > b.arrival || a.rides > b.rides {
        return false;
    }
    match (&a.side, &b.side) {
        (Side::Unboarded, Side::Unboarded) => true,
        (
            Side::Boarded {
                paid: pa,
                ticket: ta,
            },
            Side::Boarded {
                paid: pb,
                ticket: tb,
            },
        ) => {
            *pa <= *pb + MONEY_EPSILON
                && coverage(fares, tb.product) & !coverage(fares, ta.product) == 0
                && ta.expiry >= tb.expiry
                && ta.remaining >= tb.remaining
        }
        _ => false,
    }
}

/// An unsorted nondominated bag, as on the rule-based frontier.
#[derive(Default, Clone)]
struct Bag {
    entries: Vec<Entry>,
}

impl Bag {
    /// Whether the bag would admit `candidate` — the dominance check
    /// alone, so callers can defer arena bookkeeping to admitted
    /// entries (dominance never reads the node).
    fn admits(&self, candidate: &Entry, fares: &ZoneFares) -> bool {
        !self
            .entries
            .iter()
            .any(|entry| dominates(fares, entry, candidate))
    }

    /// Inserts a candidate `admits` already accepted: evicts what it
    /// dominates and pushes, without rescanning admission.
    fn insert_admitted(&mut self, candidate: Entry, fares: &ZoneFares) {
        self.entries
            .retain(|entry| !dominates(fares, &candidate, entry));
        self.entries.push(candidate);
    }

    fn insert(&mut self, candidate: Entry, fares: &ZoneFares) -> bool {
        for entry in &self.entries {
            if dominates(fares, entry, &candidate) {
                return false;
            }
        }
        self.entries
            .retain(|entry| !dominates(fares, &candidate, entry));
        self.entries.push(candidate);
        true
    }
}

/// Per destination slot: the winner under the (fare, travel time,
/// rides) lexicographic order — the cheapest fare, at that fare's
/// fastest journey.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ZoneRow {
    pub fare: f64,
    pub travel_time: u32,
    pub rides: u8,
    /// The winning entry's arena node — the reconstruction handle.
    pub node: u32,
    /// The egress link's walk seconds to the destination slot (zero
    /// when the slot is the stop itself).
    pub egress_seconds: u32,
}

/// What the search folds admitted labels into.
pub enum Fold<'a> {
    /// Per destination slot, the (fare, travel time, rides)
    /// lexicographic winner — the cheapest fare, at that fare's
    /// fastest journey. The dynamic money bound and the arrival
    /// deadline apply: both only ever discard labels that cannot
    /// improve this order. `warm` carries one upper bound per slot —
    /// a fold fare where one exists, a ceiling otherwise — so a slot
    /// no journey serves cannot pin the bound at infinity: the
    /// effective money ceiling is the maximum over slots of
    /// `min(warm, best fare so far)`.
    Cheapest {
        rows: Vec<Option<ZoneRow>>,
        warm: Vec<f64>,
    },
    /// Per destination slot × ascending cutoff, the fastest journey
    /// whose fare fits — the rule-based frontier's product shape. The
    /// money ceiling stays the top cutoff: a pricier-but-faster
    /// journey must survive to win its cutoff, so the dynamic bound
    /// and deadline must not engage.
    Cutoffs {
        cutoffs: &'a [f64],
        rows: Vec<Vec<Option<FrontierRow>>>,
    },
}

pub struct ZoneFrontierSearch<'a> {
    inputs: &'a ZoneFrontierInputs<'a>,
    request: &'a Request,
    bags: Vec<Bag>,
    marked: Vec<StopIdx>,
    is_marked: Vec<bool>,
    destination_map: Vec<Vec<(u32, u32)>>,
    fold: Fold<'a>,
    /// Exact-phase sidecar for the bounded walking set, exactly as on
    /// the rule-based frontier.
    exact_walks: bool,
    arrivals: Vec<Bag>,
    arrivals_touched: Vec<StopIdx>,
    /// Branch-and-bound on money: no label may spend past the best
    /// fare already folded (an equal spend may still improve travel
    /// time). Starts at the caller's cutoff and only tightens — paid
    /// totals only grow along a chain, so the prune is exact.
    paid_bound: f64,
    /// The per-slot deadline index: open slots' money bounds
    /// descending, each carrying the slowest best travel time over
    /// its prefix. A spend that cannot beat any fare may still tie
    /// one — but only boarding no later than some tie-able slot's
    /// best arrival. The index is departure-independent, so it is
    /// rebuilt only after a row improved; until then it is
    /// conservatively stale.
    deadline_index: Vec<(f64, u32)>,
    /// Whether a row improved since the index was last rebuilt.
    deadline_dirty: bool,
    /// Per slot, `min(best fare, ceiling)` — the incremental backing
    /// of the money bound: values only fall, so the maximum is
    /// recomputed only when a maximizer falls.
    slot_value: Vec<f64>,
    /// The current pass's departure — the deadline index's anchor.
    pass_departure: u32,
    /// First marked position per pattern for the round being queued;
    /// `u16::MAX` when unqueued (restored as patterns drain).
    queue_position: Vec<u16>,
    /// The reconstruction arena; grows monotonically per query. Only
    /// the cheapest fold walks chains, so the cutoff fold skips the
    /// bookkeeping — a whole-day search's arena can reach gigabytes.
    pub arena: Vec<Node>,
    keep_chains: bool,
}

impl<'a> ZoneFrontierSearch<'a> {
    pub fn new(
        inputs: &'a ZoneFrontierInputs<'a>,
        request: &'a Request,
        destinations: &[Vec<(StopIdx, u32)>],
        fold: Fold<'a>,
    ) -> ZoneFrontierSearch<'a> {
        let stop_count = inputs.timetable.stop_count() as usize;
        let exact_walks = !inputs.transfers.closed();
        let warm_ceiling = match &fold {
            Fold::Cheapest { warm, .. } => warm.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
            Fold::Cutoffs { .. } => inputs.top_cutoff,
        };
        let keep_chains = matches!(&fold, Fold::Cheapest { .. });
        let slot_value: Vec<f64> = match &fold {
            Fold::Cheapest { rows, warm } => rows
                .iter()
                .enumerate()
                .map(|(slot, row)| {
                    let ceiling = warm.get(slot).copied().unwrap_or(inputs.top_cutoff);
                    row.as_ref().map_or(ceiling, |row| row.fare.min(ceiling))
                })
                .collect(),
            Fold::Cutoffs { .. } => Vec::new(),
        };
        ZoneFrontierSearch {
            inputs,
            request,
            bags: vec![Bag::default(); stop_count],
            marked: Vec::new(),
            is_marked: vec![false; stop_count],
            destination_map: {
                let mut map: Vec<Vec<(u32, u32)>> = vec![Vec::new(); stop_count];
                for (slot, egress) in destinations.iter().enumerate() {
                    for &(stop, seconds) in egress {
                        map[stop.0 as usize].push((slot as u32, seconds));
                    }
                }
                map
            },
            fold,
            exact_walks,
            arrivals: if exact_walks {
                vec![Bag::default(); stop_count]
            } else {
                Vec::new()
            },
            arrivals_touched: Vec::new(),
            paid_bound: warm_ceiling.min(inputs.top_cutoff),
            deadline_index: Vec::new(),
            deadline_dirty: true,
            slot_value,
            pass_departure: 0,
            queue_position: vec![u16::MAX; inputs.timetable.pattern_count() as usize],
            arena: Vec::new(),
            keep_chains,
        }
    }

    /// Whether a boarding at `board_time` with `paid` spent can still
    /// improve any destination row. A spend strictly under the money
    /// bound can still beat some slot's fare — a cheaper journey may
    /// arrive arbitrarily late, so no deadline applies. At or above
    /// the bound only fare ties remain, and a tie improves nothing
    /// past the tie-able slots' slowest best arrival; the strict
    /// comparison keeps equal-second boardings, which can still tie
    /// travel time and improve the rides tie-break.
    fn hopeless(&self, paid: f64, board_time: u32) -> bool {
        if paid + MONEY_EPSILON < self.paid_bound {
            return false;
        }
        let cut = paid - MONEY_EPSILON;
        let end = self
            .deadline_index
            .partition_point(|&(fare, _)| fare >= cut);
        match end.checked_sub(1) {
            None => true,
            Some(last) => {
                board_time
                    > self
                        .pass_departure
                        .saturating_add(self.deadline_index[last].1)
            }
        }
    }

    /// Records `node` in the arena and returns its handle — or
    /// [`SEED_NODE`] when no fold will ever walk a chain.
    fn note(&mut self, node: Node) -> u32 {
        if !self.keep_chains {
            return SEED_NODE;
        }
        self.arena.push(node);
        (self.arena.len() - 1) as u32
    }

    /// Rebuilds the deadline index from the fold's current rows. It
    /// runs at every row improvement — rare — so `hopeless` always
    /// reads fare/deadline pairs that are simultaneously true: a
    /// tightened bound must never pair with a superseded row's
    /// shorter travel time.
    fn rearm(&mut self) {
        self.deadline_dirty = false;
        self.deadline_index.clear();
        match &self.fold {
            Fold::Cheapest { rows, warm } => {
                let mut entries: Vec<(f64, u32)> = Vec::with_capacity(rows.len());
                for (slot, row) in rows.iter().enumerate() {
                    let ceiling = warm.get(slot).copied().unwrap_or(self.inputs.top_cutoff);
                    if ceiling < 0.0 {
                        continue;
                    }
                    match row {
                        Some(row) => entries.push((row.fare.min(ceiling), row.travel_time)),
                        None => entries.push((ceiling, UNREACHED)),
                    }
                }
                entries.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
                let mut slowest = 0u32;
                for (fare, travel_time) in entries {
                    slowest = slowest.max(travel_time);
                    self.deadline_index.push((fare, slowest));
                }
            }
            // The cutoff fold takes no deadline — a pricier-but-faster
            // journey must survive to win its cutoff — but spends past
            // the top cutoff stay dead.
            Fold::Cutoffs { .. } => self
                .deadline_index
                .push((self.inputs.top_cutoff, UNREACHED)),
        }
    }

    fn mark(&mut self, stop: StopIdx) {
        if !self.is_marked[stop.0 as usize] {
            self.is_marked[stop.0 as usize] = true;
            self.marked.push(stop);
        }
    }

    fn collect(&mut self, stop: StopIdx, entry: &Entry, departure: u32) {
        if self.destination_map[stop.0 as usize].is_empty() {
            return;
        }
        let fare = match &entry.side {
            // Walking is free — but only stop queries may fold an
            // unboarded label: point access arrives walk-complete, so
            // reaching an egress link without boarding would compose
            // two independently bounded street walks. Point walking
            // rides the direct street search instead.
            Side::Unboarded => {
                if !self.inputs.seed_walks {
                    return;
                }
                0.0
            }
            Side::Boarded { paid, .. } => *paid,
        };
        let mut improved = false;
        for index in 0..self.destination_map[stop.0 as usize].len() {
            let (slot, seconds) = self.destination_map[stop.0 as usize][index];
            let Some(arrival) = entry
                .arrival
                .checked_add(seconds)
                .filter(|&at| at != UNREACHED)
            else {
                continue;
            };
            if let Some(cap) = self.inputs.max_duration {
                if arrival.saturating_sub(departure) > cap {
                    continue;
                }
            }
            if !fare.is_finite() {
                continue;
            }
            match &mut self.fold {
                Fold::Cheapest { rows, warm } => {
                    let row = ZoneRow {
                        fare,
                        travel_time: arrival - departure,
                        rides: entry.rides,
                        node: entry.node,
                        egress_seconds: seconds,
                    };
                    let better = match &rows[slot as usize] {
                        None => true,
                        Some(current) => {
                            (row.fare, row.travel_time, row.rides)
                                < (current.fare, current.travel_time, current.rides)
                        }
                    };
                    if better {
                        rows[slot as usize] = Some(row);
                        let ceiling = warm
                            .get(slot as usize)
                            .copied()
                            .unwrap_or(self.inputs.top_cutoff);
                        let value = row.fare.min(ceiling);
                        let fallen = self.slot_value[slot as usize];
                        if value < fallen {
                            self.slot_value[slot as usize] = value;
                            // Values only fall; the bound moves only
                            // when a maximizer falls.
                            if fallen >= self.paid_bound - MONEY_EPSILON {
                                let bound = self
                                    .slot_value
                                    .iter()
                                    .cloned()
                                    .fold(f64::NEG_INFINITY, f64::max);
                                if bound < self.paid_bound {
                                    self.paid_bound = bound;
                                }
                            }
                        }
                        improved = true;
                    }
                }
                Fold::Cutoffs { cutoffs, rows } => {
                    fold_candidate(
                        &mut rows[slot as usize],
                        cutoffs,
                        Arrived {
                            departure,
                            arrival,
                            rides: entry.rides,
                            fare,
                        },
                    );
                }
            }
        }
        if improved {
            self.rearm();
        }
    }

    /// Runs one departure pass; passes come in strictly decreasing
    /// departure order, as on every profile engine.
    pub fn pass(&mut self, departure: u32) {
        // The deadline index is anchored on this pass's departure;
        // the money bound carries across passes. A negative ceiling
        // closes its slot — it accepts no journey, so it neither pins
        // the money bound nor holds a deadline open — while a zero
        // ceiling stays open for zero-fare tariffs.
        self.pass_departure = departure;
        if self.deadline_dirty {
            self.rearm();
        }
        let horizon = self
            .inputs
            .max_duration
            .and_then(|cap| departure.checked_add(cap))
            .unwrap_or(UNREACHED);
        let mut seeds: Vec<(StopIdx, Entry)> = Vec::new();
        for &(stop, seconds) in &self.request.access {
            let Some(arrival) = departure
                .checked_add(seconds)
                .filter(|&at| at != UNREACHED && at <= horizon)
            else {
                continue;
            };
            let node = self.note(Node::Seed {
                stop,
                leave: departure,
            });
            seeds.push((
                stop,
                Entry {
                    arrival,
                    rides: 0,
                    side: Side::Unboarded,
                    node,
                },
            ));
        }
        // Seeds walk one bounded transfer before any boarding, fare
        // free and still unboarded: an origin (a zone-less one
        // especially) may only be boardable from a neighbouring stop.
        let mut walk_nodes: Vec<(StopIdx, Entry, StopIdx)> = Vec::new();
        for &(stop, entry) in seeds.iter().filter(|_| self.inputs.seed_walks) {
            for edge in self.inputs.transfers.from_stop(stop) {
                let Some(arrival) = entry
                    .arrival
                    .checked_add(edge.duration)
                    .filter(|&at| at != UNREACHED && at <= horizon)
                else {
                    continue;
                };
                walk_nodes.push((
                    edge.to,
                    Entry {
                        arrival,
                        rides: 0,
                        side: Side::Unboarded,
                        node: u32::MAX, // assigned below, after the borrow ends
                    },
                    stop,
                ));
            }
        }
        let mut seed_walks: Vec<(StopIdx, Entry)> = Vec::new();
        for (to, mut entry, from) in walk_nodes {
            entry.node = self.note(Node::Seed {
                stop: to,
                leave: departure,
            });
            let _ = from;
            seed_walks.push((to, entry));
        }
        for (stop, entry) in seeds.into_iter().chain(seed_walks) {
            if self.bags[stop.0 as usize].insert(entry, self.inputs.fares) {
                self.mark(stop);
                self.collect(stop, &entry, departure);
            }
        }

        let rounds = (self.request.max_transfers as usize).min(254) + 1;
        for round in 1..=rounds {
            let marked = std::mem::take(&mut self.marked);
            for &stop in &marked {
                self.is_marked[stop.0 as usize] = false;
            }
            if marked.is_empty() {
                break;
            }
            let timetable = self.inputs.timetable;
            // Pooled queue: first marked position per pattern, reset
            // as patterns dequeue — the RAPTOR discipline, no per-round
            // allocation.
            let mut queued: Vec<(u32, u16)> = Vec::new();
            for &stop in &marked {
                for pattern_stop in timetable.patterns_at_stop(stop) {
                    let slot = &mut self.queue_position[pattern_stop.pattern.0 as usize];
                    if *slot == u16::MAX {
                        queued.push((pattern_stop.pattern.0, 0));
                    }
                    if pattern_stop.position < *slot {
                        *slot = pattern_stop.position;
                    }
                }
            }
            for entry in &mut queued {
                entry.1 = self.queue_position[entry.0 as usize];
                self.queue_position[entry.0 as usize] = u16::MAX;
            }
            queued.sort_unstable();

            let mut writes: Vec<(StopIdx, Entry, Node)> = Vec::new();
            for (pattern, start_position) in queued {
                self.scan_pattern(
                    crate::timetable::PatternIdx(pattern),
                    start_position as usize,
                    round,
                    horizon,
                    &mut writes,
                );
            }
            for (stop, mut entry, node) in writes {
                // The arena records admitted entries only: a candidate
                // dominated on arrival never parents a chain, and in a
                // saturated search those are the overwhelming majority.
                let into_arrivals = self.exact_walks
                    && self.arrivals[stop.0 as usize].admits(&entry, self.inputs.fares);
                let into_bag = self.bags[stop.0 as usize].admits(&entry, self.inputs.fares);
                if into_arrivals || into_bag {
                    entry.node = self.note(node);
                }
                if into_arrivals {
                    let first = self.arrivals[stop.0 as usize].entries.is_empty();
                    self.arrivals[stop.0 as usize].insert_admitted(entry, self.inputs.fares);
                    if first {
                        self.arrivals_touched.push(stop);
                    }
                }
                if into_bag {
                    self.bags[stop.0 as usize].insert_admitted(entry, self.inputs.fares);
                    self.mark(stop);
                    self.collect(stop, &entry, departure);
                }
            }

            // Walking phase, fare unchanged; the bounded set relaxes
            // the round's transit arrivals only (the exact phase).
            let sources: Vec<StopIdx> = if self.exact_walks {
                std::mem::take(&mut self.arrivals_touched)
            } else {
                self.marked.clone()
            };
            let mut walks: Vec<(StopIdx, Entry, StopIdx)> = Vec::new();
            for &source in &sources {
                let candidates: Vec<Entry> = if self.exact_walks {
                    self.arrivals[source.0 as usize].entries.clone()
                } else {
                    self.bags[source.0 as usize]
                        .entries
                        .iter()
                        .filter(|entry| entry.rides as usize == round)
                        .copied()
                        .collect()
                };
                for edge in self.inputs.transfers.from_stop(source) {
                    for entry in &candidates {
                        let Some(arrival) = entry
                            .arrival
                            .checked_add(edge.duration)
                            .filter(|&at| at != UNREACHED && at <= horizon)
                        else {
                            continue;
                        };
                        walks.push((
                            edge.to,
                            Entry {
                                arrival,
                                rides: entry.rides,
                                side: entry.side,
                                node: entry.node,
                            },
                            source,
                        ));
                    }
                }
            }
            for (stop, mut entry, from) in walks {
                if !self.bags[stop.0 as usize].admits(&entry, self.inputs.fares) {
                    continue;
                }
                entry.node = self.note(Node::Walk {
                    parent: entry.node,
                    from,
                    to: stop,
                });
                self.bags[stop.0 as usize].insert_admitted(entry, self.inputs.fares);
                self.mark(stop);
                // Stop queries fold walked arrivals (their slots ARE
                // the stops — this is the one bounded footpath); point
                // egress links carry their own complete street walk,
                // so folding a walked label there would chain two.
                // The label stays boardable either way.
                if self.inputs.seed_walks {
                    self.collect(stop, &entry, departure);
                }
            }
            if self.exact_walks {
                for &stop in &sources {
                    self.arrivals[stop.0 as usize].entries.clear();
                }
            }
        }
    }

    fn scan_pattern(
        &mut self,
        pattern: crate::timetable::PatternIdx,
        start_position: usize,
        round: usize,
        horizon: u32,
        writes: &mut Vec<(StopIdx, Entry, Node)>,
    ) {
        let timetable = self.inputs.timetable;
        let stops = timetable.pattern_stops(pattern);
        let mut riding: Vec<Riding> = Vec::new();
        for position in start_position..stops.len() {
            let stop = stops[position];
            // Alight everything riding here whose ticket covers the
            // alighting zone; the rest die (unpriceable alight).
            for run in &riding {
                let times = timetable.trip_stop_times(run.trip);
                let arrival = times[position].arrival.saturating_sub(run.day_offset);
                if arrival > horizon {
                    continue;
                }
                match zone_bit(self.inputs.fares, stop) {
                    Some(bit) if coverage(self.inputs.fares, run.ticket.product) & bit != 0 => {
                        let entry = Entry {
                            arrival,
                            rides: run.rides,
                            side: Side::Boarded {
                                paid: run.paid,
                                ticket: run.ticket,
                            },
                            node: u32::MAX,
                        };
                        // Stage-time prefilter against the resident
                        // bags (stable until the round barrier); the
                        // flush's per-structure admission is the one
                        // decision that also sees in-round evolution.
                        // A round-global pending-dominance sweep was
                        // measured quadratic and regressed the metro
                        // benchmark five-fold — dominated same-round
                        // duplicates are rare enough to leave to the
                        // flush.
                        let admitted = self.bags[stop.0 as usize].admits(&entry, self.inputs.fares)
                            || (self.exact_walks
                                && self.arrivals[stop.0 as usize]
                                    .admits(&entry, self.inputs.fares));
                        if admitted {
                            writes.push((
                                stop,
                                entry,
                                Node::Ride {
                                    parent: run.parent,
                                    trip: run.trip,
                                    day_offset: run.day_offset,
                                    board_position: run.board_position,
                                    alight_position: position as u16,
                                },
                            ));
                        }
                    }
                    _ => {}
                }
            }
            if position + 1 == stops.len() {
                continue;
            }
            let boardable: Vec<Entry> = self.bags[stop.0 as usize]
                .entries
                .iter()
                .filter(|entry| (entry.rides as usize) < round)
                .copied()
                .collect();
            if boardable.is_empty() {
                continue;
            }
            let Some(bit) = zone_bit(self.inputs.fares, stop) else {
                // A zone-less boarding stop can never be a fare
                // endpoint: nothing boards here.
                continue;
            };
            let trip_range = timetable.pattern_trip_range(pattern);
            for entry in boardable {
                for (stream, active) in [
                    &self.request.active_services,
                    &self.request.active_services_previous,
                ]
                .into_iter()
                .enumerate()
                {
                    if active.iter().all(|&on| !on) {
                        continue;
                    }
                    let day_offset = if stream == 0 { 0 } else { DAY_SECONDS };
                    let Some(threshold) = entry.arrival.checked_add(day_offset) else {
                        continue;
                    };
                    let Some(first) = earliest_active_trip(
                        timetable,
                        active,
                        self.request.exclusions.as_deref(),
                        pattern,
                        position,
                        threshold,
                    ) else {
                        continue;
                    };
                    for trip in first.0..trip_range.end {
                        let trip_idx = TripIdx(trip);
                        let board_time = timetable.trip_stop_times(trip_idx)[position]
                            .departure
                            .saturating_sub(day_offset);
                        if board_time > horizon {
                            break;
                        }
                        if !active
                            .get(timetable.trip_service(trip_idx) as usize)
                            .copied()
                            .unwrap_or(false)
                        {
                            continue;
                        }
                        if let Some(excluded) = self.request.exclusions.as_deref() {
                            if excluded.excludes_trip(trip_idx) {
                                continue;
                            }
                        }
                        // Extending is only ever useful on the first
                        // boardable trip: a later extend has a strictly
                        // later arrival and identical resources. Buys
                        // explore later trips — a later purchase
                        // carries a fresher expiry.
                        if trip == first.0 {
                            self.extend(
                                &entry,
                                bit,
                                trip_idx,
                                day_offset,
                                board_time,
                                position as u16,
                                &mut riding,
                            );
                        }
                        self.buy(
                            &entry,
                            bit,
                            trip_idx,
                            day_offset,
                            board_time,
                            position as u16,
                            &mut riding,
                        );
                    }
                }
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn extend(
        &mut self,
        entry: &Entry,
        bit: u128,
        trip: TripIdx,
        day_offset: u32,
        board_time: u32,
        board_position: u16,
        riding: &mut Vec<Riding>,
    ) {
        let Side::Boarded { paid, ticket } = entry.side else {
            return;
        };
        if coverage(self.inputs.fares, ticket.product) & bit == 0 {
            return;
        }
        if board_time > ticket.expiry {
            return;
        }
        if ticket.remaining == 0 {
            return;
        }
        if self.hopeless(paid, board_time) {
            return;
        }
        let run = Riding {
            trip,
            day_offset,
            rides: entry.rides + 1,
            paid,
            parent: entry.node,
            board_position,
            ticket: Ticket {
                remaining: if ticket.remaining == u32::MAX {
                    u32::MAX
                } else {
                    ticket.remaining - 1
                },
                ..ticket
            },
        };
        self.admit_run(run, riding);
    }

    #[allow(clippy::too_many_arguments)]
    fn buy(
        &mut self,
        entry: &Entry,
        bit: u128,
        trip: TripIdx,
        day_offset: u32,
        board_time: u32,
        board_position: u16,
        riding: &mut Vec<Riding>,
    ) {
        let paid_before = match entry.side {
            Side::Unboarded => 0.0,
            Side::Boarded { paid, .. } => paid,
        };
        for (product, spec) in self.inputs.fares.products.iter().enumerate() {
            if spec.zones & bit == 0 {
                continue;
            }
            if !spec.price.is_finite() {
                continue;
            }
            let paid = paid_before + spec.price;
            if paid > self.inputs.top_cutoff + MONEY_EPSILON {
                continue;
            }
            if self.hopeless(paid, board_time) {
                continue;
            }
            let expiry = if spec.duration.is_finite() {
                board_time.saturating_add(spec.duration as u32)
            } else {
                UNREACHED
            };
            let run = Riding {
                trip,
                day_offset,
                rides: entry.rides + 1,
                paid,
                parent: entry.node,
                board_position,
                ticket: Ticket {
                    product: product as u32,
                    expiry,
                    remaining: spec.transfers,
                },
            };
            self.admit_run(run, riding);
        }
    }

    /// Inserts a run unless a same-trip run resource-dominates it.
    fn admit_run(&mut self, run: Riding, riding: &mut Vec<Riding>) {
        let fares = self.inputs.fares;
        let stronger = |a: &Riding, b: &Riding| {
            a.rides <= b.rides
                && a.paid <= b.paid + MONEY_EPSILON
                && coverage(fares, b.ticket.product) & !coverage(fares, a.ticket.product) == 0
                && a.ticket.expiry >= b.ticket.expiry
                && a.ticket.remaining >= b.ticket.remaining
        };
        let mut admitted = true;
        riding.retain(|other| {
            if other.trip != run.trip || other.day_offset != run.day_offset {
                return true;
            }
            if stronger(other, &run) {
                admitted = false;
                return true;
            }
            !stronger(&run, other)
        });
        if admitted {
            riding.push(run);
        }
    }

    pub fn into_fold(self) -> Fold<'a> {
        self.fold
    }
}

/// One step of a reconstructed journey, in travel order.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ChainStep {
    Ride {
        trip: TripIdx,
        day_offset: u32,
        board_position: u16,
        alight_position: u16,
    },
    Walk {
        from: StopIdx,
        to: StopIdx,
    },
}

/// Walks a winner's chain out of the arena: the origin-leave time,
/// the seeded stop, and the steps in travel order.
pub fn chain(arena: &[Node], node: u32) -> (u32, StopIdx, Vec<ChainStep>) {
    let mut steps = Vec::new();
    let mut at = node;
    loop {
        match arena[at as usize] {
            Node::Seed { stop, leave } => {
                steps.reverse();
                return (leave, stop, steps);
            }
            Node::Ride {
                parent,
                trip,
                day_offset,
                board_position,
                alight_position,
            } => {
                steps.push(ChainStep::Ride {
                    trip,
                    day_offset,
                    board_position,
                    alight_position,
                });
                at = parent;
            }
            Node::Walk { parent, from, to } => {
                steps.push(ChainStep::Walk { from, to });
                at = parent;
            }
        }
    }
}

/// The profile driver: event passes or the rasterised grid, exactly
/// as the rule-based frontier chooses them.
pub fn zone_frontier(
    inputs: &ZoneFrontierInputs<'_>,
    request: &Request,
    destinations: &[Vec<(StopIdx, u32)>],
    window: u32,
) -> (Vec<Option<ZoneRow>>, Vec<Node>) {
    let warm = vec![inputs.top_cutoff; destinations.len()];
    zone_frontier_warm(inputs, request, destinations, window, warm, &[])
}

/// [`zone_frontier`] with per-slot warm upper bounds: fold fares
/// where they exist, ceilings otherwise. Any slot's answer is exact
/// whenever its warm bound was a true upper bound on that cell's
/// optimum (a fold fare always is); an unreachable slot merely
/// bounds, never pins, the search.
///
/// `seeds` optionally carries the warm journeys' travel times: a
/// seeded slot starts with a row at (its warm fare, that travel
/// time), arming the money bound and the deadline index from the
/// first pass. A returned row still carrying [`SEED_NODE`] means
/// nothing beat the seed — the caller keeps its own journey.
pub fn zone_frontier_warm(
    inputs: &ZoneFrontierInputs<'_>,
    request: &Request,
    destinations: &[Vec<(StopIdx, u32)>],
    window: u32,
    warm: Vec<f64>,
    seeds: &[Option<u32>],
) -> (Vec<Option<ZoneRow>>, Vec<Node>) {
    let rows = (0..destinations.len())
        .map(|slot| {
            seeds
                .get(slot)
                .copied()
                .flatten()
                .map(|travel_time| ZoneRow {
                    fare: warm[slot],
                    travel_time,
                    rides: u8::MAX,
                    node: SEED_NODE,
                    egress_seconds: 0,
                })
        })
        .collect();
    let fold = Fold::Cheapest { rows, warm };
    let (fold, arena) = drive(inputs, request, destinations, window, fold);
    let Fold::Cheapest { rows, .. } = fold else {
        unreachable!("the cheapest fold returns the cheapest fold")
    };
    (rows, arena)
}

/// The per-cutoff product: [`fare_frontier`]'s row shape, priced by
/// the zone-ticket state machine.
pub fn zone_frontier_cutoffs<'a>(
    inputs: &'a ZoneFrontierInputs<'a>,
    request: &'a Request,
    destinations: &[Vec<(StopIdx, u32)>],
    window: u32,
    cutoffs: &'a [f64],
) -> Vec<Vec<Option<FrontierRow>>> {
    let fold = Fold::Cutoffs {
        cutoffs,
        rows: vec![vec![None; cutoffs.len()]; destinations.len()],
    };
    let (fold, _) = drive(inputs, request, destinations, window, fold);
    let Fold::Cutoffs { rows, .. } = fold else {
        unreachable!("the cutoff fold returns the cutoff fold")
    };
    rows
}

fn drive<'a>(
    inputs: &'a ZoneFrontierInputs<'a>,
    request: &'a Request,
    destinations: &[Vec<(StopIdx, u32)>],
    window: u32,
    fold: Fold<'a>,
) -> (Fold<'a>, Vec<Node>) {
    let departures = match inputs.departure_step {
        Some(step) => {
            crate::routers::fare_frontier::sampled_departures(request.departure, window, step)
        }
        None => {
            // Seeds walk one transfer before boarding (stop queries),
            // so the event passes must include trip departures at
            // walk-reachable stops, shifted by the walk — otherwise a
            // journey whose only boarding follows the seed walk has no
            // pass at its catchable departure. Point queries arrive
            // walk-complete and skip the extension.
            let mut access = request.access.clone();
            if inputs.seed_walks {
                for &(stop, seconds) in &request.access {
                    for edge in inputs.transfers.from_stop(stop) {
                        access.push((edge.to, seconds.saturating_add(edge.duration)));
                    }
                }
            }
            let extended = Request {
                access,
                egress: request.egress.clone(),
                active_services: request.active_services.clone(),
                active_services_previous: request.active_services_previous.clone(),
                exclusions: request.exclusions.clone(),
                ..*request
            };
            crate::routers::raptor::departure_candidates(inputs.timetable, &extended, window)
        }
    };
    let mut search = ZoneFrontierSearch::new(inputs, request, destinations, fold);
    for &departure in &departures {
        search.pass(departure);
    }
    let arena = std::mem::take(&mut search.arena);
    (search.into_fold(), arena)
}

#[cfg(test)]
#[path = "zone_frontier/tests.rs"]
mod tests;
