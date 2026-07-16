//! Per-stop multicriteria bags: labels, dominance, and the insert
//! probes.

use super::*;

/// How a label reached its stop; parents index the label arena.
#[derive(Debug, Clone, Copy)]
pub(super) enum Origin {
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
pub(super) struct Label {
    pub(super) arrival: u32,
    pub(super) grams: f64,
    pub(super) stop: StopIdx,
    /// The profile pass that grew this label's chain: label chains
    /// never cross passes, so travel time is `arrival - departure`.
    pub(super) departure: u32,
    /// Route-penalty seconds accumulated along the chain (soft-penalty
    /// diverse); the bags dominate on `arrival + penalty` while the
    /// reconstructed journey keeps the true `arrival`. Zero without
    /// penalties.
    pub(super) penalty: u32,
    pub(super) origin: Origin,
}

/// One bag entry; `arrival` is the true arrival, `penalty` the
/// accumulated route penalty (0 without one), `key` the grams bucket,
/// `grams` the exact (microgram-quantized) value behind it, `rides` the
/// transit legs the label used to reach the stop.
#[derive(Debug, Clone, Copy)]
pub(super) struct Entry {
    pub(super) arrival: u32,
    pub(super) penalty: u32,
    pub(super) key: i64,
    pub(super) grams: f64,
    pub(super) rides: u8,
}

/// A per-stop label bag under bucketed dominance, cumulative across
/// rounds and profile passes. Shared with the trip-based engine, whose
/// stop bags follow the same contract.
#[derive(Debug, Clone, Default)]
pub(crate) struct Bag {
    pub(super) entries: Vec<Entry>,
}

/// The strict (`penalty = 0`, `slack = 0`) rejection relation: does
/// `entry` reject the candidate? Equivalent to the rejection scan of
/// `insert_slack(arrival, 0, grams, key, rides, 0)`, including the
/// entry-penalty reads.
pub(super) fn rejects_strict(entry: &Entry, arrival: u32, grams: f64, key: i64, rides: u8) -> bool {
    if entry.key <= key && entry.rides <= rides && entry.arrival <= arrival {
        if entry.arrival == arrival
            && entry.penalty == 0
            && entry.key == key
            && entry.rides == rides
        {
            grams >= entry.grams
        } else {
            entry.arrival.saturating_add(entry.penalty) <= arrival
        }
    } else {
        false
    }
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
    /// as clean as the search has seen. The trip-based engine ranks the
    /// transit round in `rides` for its direct and closure arrivals.
    ///
    /// The strict path self-organises the entry vector: a rejection
    /// swaps its witness to slot 0 and an admission swaps the new entry
    /// there, so the workload's recent certificates reject in one
    /// probe. Entry order has no semantic consumer — rejection is an
    /// existential query, eviction is set-based, and a cleaner exact
    /// tie continues the scan rather than admitting early — so only the
    /// private vector permutation differs from a stable-order bag.
    pub(crate) fn insert(&mut self, arrival: u32, grams: f64, key: i64, rides: u8) -> bool {
        for index in 0..self.entries.len() {
            if rejects_strict(&self.entries[index], arrival, grams, key, rides) {
                if index != 0 {
                    self.entries.swap(0, index);
                }
                return false;
            }
        }
        self.entries.retain(|entry| {
            !((key <= entry.key
                && rides <= entry.rides
                && arrival <= entry.arrival
                && arrival <= entry.arrival.saturating_add(entry.penalty))
                || (entry.arrival == arrival
                    && entry.penalty == 0
                    && entry.key == key
                    && entry.rides == rides
                    && grams < entry.grams))
        });
        self.entries.push(Entry {
            arrival,
            penalty: 0,
            key,
            grams,
            rides,
        });
        let last = self.entries.len() - 1;
        if last != 0 {
            self.entries.swap(0, last);
        }
        true
    }

    /// `insert` under a route penalty and a time slack. Dominance runs on
    /// two time axes: an entry may reject the newcomer only when it reaches
    /// the stop no later in **true arrival** — so it catches every onward
    /// connection the newcomer could — and is at least `slack` seconds
    /// earlier on the **effective arrival** (`arrival + penalty`), no
    /// dirtier and on no more rides. A penalized label arriving physically
    /// earlier is therefore never suppressed by an unpenalized one arriving
    /// later, even though its effective arrival is worse. Same-class
    /// (`arrival`, `penalty`, `key`, `rides`) duplicates reduce to the
    /// cleanest representative, and eviction likewise needs the full `slack`
    /// margin. Without penalties effective equals true, so this is exactly
    /// the single-axis `(arrival, key, rides)` dominance; `slack = 0` is
    /// strict `insert`, the only form the trip-based and exhaustive engines
    /// call.
    pub(crate) fn insert_slack(
        &mut self,
        arrival: u32,
        penalty: u32,
        grams: f64,
        key: i64,
        rides: u8,
        slack: u32,
    ) -> bool {
        let effective = arrival.saturating_add(penalty);
        for entry in &self.entries {
            if entry.key <= key && entry.rides <= rides && entry.arrival <= arrival {
                let entry_effective = entry.arrival.saturating_add(entry.penalty);
                if entry.arrival == arrival
                    && entry.penalty == penalty
                    && entry.key == key
                    && entry.rides == rides
                {
                    if grams >= entry.grams {
                        return false;
                    }
                } else if entry_effective.saturating_add(slack) <= effective {
                    return false;
                }
            }
        }
        self.entries.retain(|entry| {
            let entry_effective = entry.arrival.saturating_add(entry.penalty);
            !((key <= entry.key
                && rides <= entry.rides
                && arrival <= entry.arrival
                && effective.saturating_add(slack) <= entry_effective)
                || (entry.arrival == arrival
                    && entry.penalty == penalty
                    && entry.key == key
                    && entry.rides == rides
                    && grams < entry.grams))
        });
        self.entries.push(Entry {
            arrival,
            penalty,
            key,
            grams,
            rides,
        });
        true
    }

    /// Strict `insert` with probe accounting for the trip-based
    /// engine's closure diagnostics: identical decisions and identical
    /// self-organising swaps, recording the bag length before the
    /// call, the one-based rejecting depth (or the complete pre-call
    /// length on admission), and the eviction-walk depth.
    pub(crate) fn insert_probed(
        &mut self,
        arrival: u32,
        grams: f64,
        key: i64,
        rides: u8,
        probes: &mut InsertProbes,
    ) -> bool {
        probes.length = self.entries.len() as u32;
        probes.examined = 0;
        probes.retained = 0;
        for index in 0..self.entries.len() {
            probes.examined += 1;
            if rejects_strict(&self.entries[index], arrival, grams, key, rides) {
                if index != 0 {
                    self.entries.swap(0, index);
                }
                return false;
            }
        }
        let retained = &mut probes.retained;
        self.entries.retain(|entry| {
            *retained += 1;
            !((key <= entry.key
                && rides <= entry.rides
                && arrival <= entry.arrival
                && arrival <= entry.arrival.saturating_add(entry.penalty))
                || (entry.arrival == arrival
                    && entry.penalty == 0
                    && entry.key == key
                    && entry.rides == rides
                    && grams < entry.grams))
        });
        self.entries.push(Entry {
            arrival,
            penalty: 0,
            key,
            grams,
            rides,
        });
        let last = self.entries.len() - 1;
        if last != 0 {
            self.entries.swap(0, last);
        }
        true
    }

    /// A bag with a prescribed entry order, for order-sensitivity
    /// tests (reachable strict bags are antichains; tests may build
    /// unreachable orders deliberately).
    #[cfg(test)]
    pub(crate) fn from_entries(entries: Vec<(u32, u32, i64, f64, u8)>) -> Bag {
        Bag {
            entries: entries
                .into_iter()
                .map(|(arrival, penalty, key, grams, rides)| Entry {
                    arrival,
                    penalty,
                    key,
                    grams,
                    rides,
                })
                .collect(),
        }
    }

    /// Stable-order `insert_slack` with probe accounting for the
    /// R0 attribution runs: identical decisions and identical entry
    /// order to `insert_slack`, recording the bag length before the
    /// call, the one-based depth of the rejection scan (or the
    /// complete pre-call length on admission), and the eviction-walk
    /// depth.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn insert_slack_probed(
        &mut self,
        arrival: u32,
        penalty: u32,
        grams: f64,
        key: i64,
        rides: u8,
        slack: u32,
        probes: &mut InsertProbes,
    ) -> bool {
        probes.length = self.entries.len() as u32;
        probes.examined = 0;
        probes.retained = 0;
        let effective = arrival.saturating_add(penalty);
        for entry in &self.entries {
            probes.examined += 1;
            if entry.key <= key && entry.rides <= rides && entry.arrival <= arrival {
                let entry_effective = entry.arrival.saturating_add(entry.penalty);
                if entry.arrival == arrival
                    && entry.penalty == penalty
                    && entry.key == key
                    && entry.rides == rides
                {
                    if grams >= entry.grams {
                        return false;
                    }
                } else if entry_effective.saturating_add(slack) <= effective {
                    return false;
                }
            }
        }
        let retained = &mut probes.retained;
        self.entries.retain(|entry| {
            *retained += 1;
            let entry_effective = entry.arrival.saturating_add(entry.penalty);
            !((key <= entry.key
                && rides <= entry.rides
                && arrival <= entry.arrival
                && effective.saturating_add(slack) <= entry_effective)
                || (entry.arrival == arrival
                    && entry.penalty == penalty
                    && entry.key == key
                    && entry.rides == rides
                    && grams < entry.grams))
        });
        self.entries.push(Entry {
            arrival,
            penalty,
            key,
            grams,
            rides,
        });
        true
    }

    /// The R1 dispatch: a strict search (zero slack, no route
    /// penalties) inserts through the self-organising strict path —
    /// identical decisions, recent witnesses swapped to the front —
    /// while slack and penalty searches keep the stable-order general
    /// path byte-for-byte. Set evolution is order-independent in both
    /// relations, so which path ran is invisible in results.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn insert_label(
        &mut self,
        strict: bool,
        arrival: u32,
        penalty: u32,
        grams: f64,
        key: i64,
        rides: u8,
        slack: u32,
    ) -> bool {
        if strict {
            debug_assert!(penalty == 0, "a strict search carries no penalty");
            self.insert(arrival, grams, key, rides)
        } else {
            self.insert_slack(arrival, penalty, grams, key, rides, slack)
        }
    }

    /// The probed twin of `insert_label`: identical decisions and
    /// identical permutations per mode, with probe accounting.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn insert_label_probed(
        &mut self,
        strict: bool,
        arrival: u32,
        penalty: u32,
        grams: f64,
        key: i64,
        rides: u8,
        slack: u32,
        probes: &mut InsertProbes,
    ) -> bool {
        if strict {
            debug_assert!(penalty == 0, "a strict search carries no penalty");
            self.insert_probed(arrival, grams, key, rides, probes)
        } else {
            self.insert_slack_probed(arrival, penalty, grams, key, rides, slack, probes)
        }
    }

    /// The number of retained entries, for diagnostic bag censuses.
    pub(super) fn entry_count(&self) -> usize {
        self.entries.len()
    }

    /// Whether the exact label tuple is still retained — the stale-
    /// `rode` audit's membership test.
    pub(super) fn contains_exact(
        &self,
        arrival: u32,
        penalty: u32,
        key: i64,
        grams: f64,
        rides: u8,
    ) -> bool {
        self.entries.iter().any(|entry| {
            entry.arrival == arrival
                && entry.penalty == penalty
                && entry.key == key
                && entry.grams == grams
                && entry.rides == rides
        })
    }

    /// The bag's entries as comparable tuples, for differential tests.
    #[cfg(test)]
    pub(crate) fn snapshot(&self) -> Vec<(u32, u32, i64, u64, u8)> {
        self.entries
            .iter()
            .map(|entry| {
                (
                    entry.arrival,
                    entry.penalty,
                    entry.key,
                    entry.grams.to_bits(),
                    entry.rides,
                )
            })
            .collect()
    }
}

/// Per-call probe depths of one strict `insert_probed`.
#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct InsertProbes {
    /// Entries walked by the rejection scan before returning.
    pub examined: u32,
    /// Entries the eviction walk examined (admissions only).
    pub retained: u32,
    /// Bag length before the call.
    pub length: u32,
}

/// Histogram bucket of a pre-call bag length.
pub(super) fn length_bucket(length: u32) -> usize {
    match length {
        0 => 0,
        1 => 1,
        2 => 2,
        3..=4 => 3,
        5..=8 => 4,
        9..=16 => 5,
        17..=32 => 6,
        33..=128 => 7,
        _ => 8,
    }
}

/// Histogram bucket of a one-based rejecting depth.
pub(super) fn depth_bucket(depth: u32) -> usize {
    match depth {
        0..=1 => 0,
        2 => 1,
        3..=4 => 2,
        5..=8 => 3,
        9..=16 => 4,
        17..=32 => 5,
        33..=64 => 6,
        65..=128 => 7,
        _ => 8,
    }
}
