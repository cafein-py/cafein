//! The scan structures: segments, trip bags, sinks, closure
//! batches, and the prune envelope.

use super::*;

/// How a segment came to board its trip.
#[derive(Debug, Clone, Copy)]
pub(super) enum SegOrigin {
    Access {
        stop: StopIdx,
        seconds: u32,
    },
    Transfer {
        parent: u32,
        alight: u16,
    },
    Walked {
        parent: u32,
        alight: u16,
        duration: u32,
    },
}

/// One boarded trip during the scan; `grams` are the journey's grams
/// at boarding.
#[derive(Debug, Clone, Copy)]
pub(super) struct Segment {
    pub(super) trip: ViewTrip,
    pub(super) board: u16,
    pub(super) grams: f64,
    pub(super) departure: u32,
    pub(super) origin: SegOrigin,
}

/// Queue lifecycle of an arena segment: enqueued, consumed by its
/// round's scan, or cancelled by a covering trip-bag admission before
/// scanning. Cancelled segments stay in the arena — indices are
/// reconstruction identities.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum SegmentState {
    Pending = 0,
    Scanned = 1,
    Cancelled = 2,
}

/// One trip-bag admission: the (board, κ) point plus the queued
/// segment to cancel if a covering admission evicts it before its
/// round scans it.
#[derive(Debug, Clone, Copy)]
pub(super) struct TripBagEntry {
    pub(super) kappa: f64,
    pub(super) key: i64,
    pub(super) pending_segment: u32,
    pub(super) board: u16,
}

/// The per-(trip, round) Pareto bag over (board, κ): boarding no later
/// along the pattern with a κ in the same or a cleaner bucket covers
/// every alight the newcomer could make. Equal slots refine toward the
/// exact-cleaner κ. Evicting an entry reports its segment through the
/// cancellation callback — the caller downgrades only still-pending
/// segments, so scanned work (destination leaves, transfer and
/// WalkBoard parents) is never revoked.
#[derive(Debug, Clone, Default)]
pub(super) struct TripBag {
    pub(super) entries: Vec<TripBagEntry>,
}

impl TripBag {
    pub(super) fn admits(
        &mut self,
        board: u16,
        kappa: f64,
        key: i64,
        pending_segment: u32,
        mut cancel: impl FnMut(u32),
    ) -> bool {
        for entry in &self.entries {
            if entry.board <= board
                && entry.key <= key
                && !(entry.board == board && entry.key == key && kappa < entry.kappa)
            {
                return false;
            }
        }
        self.entries.retain(|entry| {
            let covered = board <= entry.board && key <= entry.key;
            if covered && entry.pending_segment != NONE_U32 {
                cancel(entry.pending_segment);
            }
            !covered
        });
        self.entries.push(TripBagEntry {
            kappa,
            key,
            pending_segment,
            board,
        });
        true
    }
}

/// A destination frontier entry: the leaf segment, where it alighted,
/// and how the egress was joined.
#[derive(Debug, Clone, Copy)]
pub(super) struct Arrived {
    pub(super) departure: u32,
    pub(super) arrival: u32,
    pub(super) key: i64,
    pub(super) grams: f64,
    pub(super) leaf: u32,
    pub(super) alight: u16,
    /// A final footpath hop before the egress, when joined via one.
    pub(super) walk: Option<(StopIdx, u32)>,
}

/// The sentinel leaf of a zero-ride (access floor) matrix winner.
pub(super) const ACCESS_LEAF: u32 = u32::MAX;

/// One matrix winner: the cleanest (then fastest) point folded for a
/// destination, with the chain to rebuild its cost row.
#[derive(Debug, Clone, Copy)]
pub(super) struct Winner {
    pub(super) grams: f64,
    pub(super) seconds: u32,
    pub(super) leaf: u32,
    pub(super) alight: u16,
    /// The point was reached over a final footpath hop.
    pub(super) walked: bool,
}

/// Per-destination fold state for the emissions matrix, mirroring the
/// McRAPTOR matrix fold: candidates fold per pass at creation (an
/// end-of-search bag readout would lose budget-qualifying candidates
/// to cross-pass evictions), lower grams win, ties resolve toward the
/// shorter travel time, a travel-time budget disqualifies outright.
pub(super) struct MatrixSink<'a> {
    /// Per stop: destination slot + 1, or 0 when not a destination.
    pub(super) slots: &'a [u32],
    pub(super) budget: Option<u32>,
    pub(super) best: &'a mut [Option<Winner>],
}

impl MatrixSink<'_> {
    pub(super) fn fold(
        &mut self,
        stop: StopIdx,
        seconds: u32,
        grams: f64,
        leaf: u32,
        alight: u16,
        walked: bool,
    ) {
        let slot = self.slots[stop.0 as usize];
        if slot == 0 {
            return;
        }
        if self.budget.is_some_and(|budget| seconds > budget) {
            return;
        }
        let best = &mut self.best[slot as usize - 1];
        let better = match best {
            None => true,
            Some(winner) => {
                grams < winner.grams || (grams == winner.grams && seconds < winner.seconds)
            }
        };
        if better {
            *best = Some(Winner {
                grams,
                seconds,
                leaf,
                alight,
                walked,
            });
        }
    }
}

pub(super) const NONE_U32: u32 = u32::MAX;

/// One direct admission queued for the round's closure sweep, linked
/// per source stop in direct-scan order.
#[derive(Clone, Copy)]
pub(super) struct ClosurePoint {
    pub(super) grams: f64,
    pub(super) key: i64,
    pub(super) segment: u32,
    pub(super) arrival: u32,
    pub(super) next: u32,
    pub(super) alight: u16,
    pub(super) live: bool,
}

/// A live point compacted for relaxation, without the list link.
#[derive(Clone, Copy)]
pub(super) struct ActiveClosurePoint {
    pub(super) grams: f64,
    pub(super) key: i64,
    pub(super) segment: u32,
    pub(super) arrival: u32,
    pub(super) alight: u16,
}

/// The round's pending closure sources: per-stop linked buckets of
/// admitted direct points (stable indices, only touched stops reset)
/// swept edge-major after the direct scans. A batch never spans two
/// rounds, two departure passes, walked labels, or two requests.
pub(super) struct ClosureBatch {
    pub(super) heads: Box<[u32]>,
    pub(super) tails: Box<[u32]>,
    pub(super) touched: Vec<StopIdx>,
    pub(super) points: Vec<ClosurePoint>,
    pub(super) active: Vec<ActiveClosurePoint>,
    pub(super) evicted: u64,
}

impl ClosureBatch {
    pub(super) fn new(stop_count: usize) -> Self {
        ClosureBatch {
            heads: vec![NONE_U32; stop_count].into_boxed_slice(),
            tails: vec![NONE_U32; stop_count].into_boxed_slice(),
            touched: Vec::new(),
            points: Vec::new(),
            active: Vec::new(),
            evicted: 0,
        }
    }

    /// Queues a direct admission under the batch-local gate: the
    /// fixed-round, fixed-departure (arrival, key) relation with the
    /// stop bags' equal-slot exact-grams refinement. Rejection is rare
    /// (the stop bag admitted the point already) but preserves the
    /// invariant that no queued point covers another.
    pub(super) fn offer(
        &mut self,
        source: StopIdx,
        segment: u32,
        alight: u16,
        arrival: u32,
        grams: f64,
        key: i64,
    ) -> bool {
        let slot = source.0 as usize;
        let mut cursor = self.heads[slot];
        while cursor != NONE_U32 {
            let point = &mut self.points[cursor as usize];
            let next = point.next;
            if point.live {
                if point.arrival <= arrival
                    && point.key <= key
                    && !(point.arrival == arrival && point.key == key && grams < point.grams)
                {
                    return false;
                }
                if arrival <= point.arrival && key <= point.key {
                    point.live = false;
                    self.evicted += 1;
                }
            }
            cursor = next;
        }
        let index = self.points.len() as u32;
        self.points.push(ClosurePoint {
            grams,
            key,
            segment,
            arrival,
            next: NONE_U32,
            alight,
            live: true,
        });
        if self.heads[slot] == NONE_U32 {
            self.heads[slot] = index;
            self.touched.push(source);
        } else {
            self.points[self.tails[slot] as usize].next = index;
        }
        self.tails[slot] = index;
        true
    }

    pub(super) fn reset(&mut self) {
        for &stop in &self.touched {
            self.heads[stop.0 as usize] = NONE_U32;
            self.tails[stop.0 as usize] = NONE_U32;
        }
        self.touched.clear();
        self.points.clear();
        self.active.clear();
        self.evicted = 0;
    }
}

/// A next-round boarding discovered by a round's join sweep, executed
/// by its expansion sweep under the round's pruning envelope.
pub(super) struct WalkBoard {
    pub(super) segment: u32,
    pub(super) alight: u16,
    pub(super) to: StopIdx,
    pub(super) reached: u32,
    pub(super) grams: f64,
    pub(super) duration: u32,
}

/// The one-pair round's pruning envelope: per arrival, the cleanest
/// key the destination has reached at or before it. A candidate at or
/// above the envelope is dominated — its continuations only grow in
/// arrival and grams, and passes descend, so nothing it leads to can
/// join a new frontier entry (the same plain (arrival ≤, key ≤)
/// dominance the one-pair pruning has always used). The bound is
/// non-increasing in arrival while a segment's alights only grow in
/// both axes, so a pruned alight ends its segment's expansion
/// outright. The batched product runs unpruned: an all-slots envelope
/// is a max over every destination rebuilt from every accumulated
/// frontier each round — measured far costlier at scale than the
/// expansion it trims, and one unserved slot disables it entirely.
pub(super) struct PruneEnvelope {
    pub(super) arrivals: Vec<u32>,
    pub(super) bounds: Vec<i64>,
}

impl PruneEnvelope {
    /// Never prunes: the destination has no entry yet, or the search
    /// has no one-pair destination at all (the batched frontier and
    /// emissions-matrix fold modes).
    pub(super) fn none() -> PruneEnvelope {
        PruneEnvelope {
            arrivals: Vec::new(),
            bounds: Vec::new(),
        }
    }

    pub(super) fn build(entries: &[Arrived]) -> PruneEnvelope {
        // The destination's entries sorted by arrival under prefix-min
        // keys: from each arrival, the cleanest key already achieved.
        if entries.is_empty() {
            return PruneEnvelope::none();
        }
        let mut sorted: Vec<(u32, i64)> = entries
            .iter()
            .map(|entry| (entry.arrival, entry.key))
            .collect();
        sorted.sort_unstable();
        let mut prefix = i64::MAX;
        for slot in &mut sorted {
            prefix = prefix.min(slot.1);
            slot.1 = prefix;
        }
        sorted.dedup_by(|later, earlier| {
            if earlier.0 == later.0 {
                earlier.1 = later.1;
                true
            } else {
                false
            }
        });
        let (arrivals, bounds) = sorted.into_iter().unzip();
        PruneEnvelope { arrivals, bounds }
    }

    pub(super) fn prunes(&self, arrival: u32, key: i64) -> bool {
        match self
            .arrivals
            .partition_point(|&at| at <= arrival)
            .checked_sub(1)
        {
            None => false,
            Some(index) => self.bounds[index] <= key,
        }
    }
}

/// Per-destination-slot frontier state for the batched product: every
/// egress join the one-pair search would make feeds its slot's
/// destination frontier instead, under the same `join` rules, so a
/// batched cell's journeys equal the single-pair query's. Stop mode
/// joins a destination stop's own alights and footpath walks with no
/// final walk; door-to-door mode (`egress_active`) walks each join
/// through the per-stop final-egress map, the walking-only journey
/// being the caller's overlay.
pub(super) struct FrontierSink<'a> {
    /// Per stop: destination slot + 1, or 0 when not a destination.
    pub(super) slots: &'a [u32],
    /// Per stop: `(destination slot, walk seconds, walk meters)` final
    /// egress; consulted only in door-to-door mode.
    pub(super) egress: &'a [Vec<(u32, u32, f64)>],
    pub(super) egress_active: bool,
    /// Per slot: the destination frontier the cell assembles from.
    pub(super) bags: &'a mut [Vec<Arrived>],
}
