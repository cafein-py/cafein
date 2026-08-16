//! Precomputed stop-to-stop transfers (footpaths), CSR by origin stop.

use crate::timetable::{StopIdx, TimetableError};

/// A walkable connection to another stop.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Transfer {
    pub to: StopIdx,
    /// Walking time in seconds.
    pub duration: u32,
    /// Walked distance in meters.
    pub meters: f64,
}

/// All stop-to-stop transfers of a network.
///
/// Transfers are single bounded walks between consecutive rides: the
/// engines relax them with the exact transfer phase (walks extend
/// transit arrivals, never other walks), so no chain of transfers can
/// exceed the walking cutoff. A set may declare itself `closed`
/// (transitively complete), which lets the engines use the cheaper
/// label-improving relaxation — the empty set trivially qualifies.
#[derive(Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Transfers {
    /// CSR offsets into `edges`, one entry per stop plus a tail.
    offsets: Vec<u32>,
    edges: Vec<Transfer>,
    /// Whether the set is transitively complete (the cheaper
    /// label-improving relaxation is sound). Not persisted; loaded
    /// sets take the exact phase.
    #[serde(skip, default = "closed_default")]
    closed: bool,
}

/// Deserialized sets take the exact transfer phase: correct for
/// bounded sets by definition, and for legacy closures too — their
/// chains are single edges, so walk-after-ride relaxation loses
/// nothing.
fn closed_default() -> bool {
    false
}

/// The incoming-edge view of a transfer set: for each stop, the
/// `(from, duration)` pairs of every forward edge that ends there. A
/// derived index — built from the forward CSR on demand and never
/// persisted — so reverse relaxation walks genuine incoming edges
/// rather than assuming the set is symmetric.
#[derive(Debug)]
pub struct ReversedTransfers {
    offsets: Vec<u32>,
    edges: Vec<(StopIdx, u32)>,
}

impl ReversedTransfers {
    /// Builds the incoming-edge CSR of `transfers`.
    pub fn build(transfers: &Transfers) -> ReversedTransfers {
        let stops = transfers.offsets.len() - 1;
        let mut counts = vec![0u32; stops + 1];
        for edge in &transfers.edges {
            counts[edge.to.0 as usize + 1] += 1;
        }
        for index in 1..counts.len() {
            counts[index] += counts[index - 1];
        }
        let offsets = counts.clone();
        let mut cursor = offsets.clone();
        let mut edges = vec![(StopIdx(0), 0u32); transfers.edges.len()];
        for (from, range) in transfers
            .offsets
            .windows(2)
            .enumerate()
            .map(|(stop, window)| (stop as u32, window[0] as usize..window[1] as usize))
        {
            for edge in &transfers.edges[range] {
                let slot = cursor[edge.to.0 as usize] as usize;
                edges[slot] = (StopIdx(from), edge.duration);
                cursor[edge.to.0 as usize] += 1;
            }
        }
        ReversedTransfers { offsets, edges }
    }

    /// The `(from, duration)` incoming edges of `stop`.
    pub fn into_stop(&self, stop: StopIdx) -> &[(StopIdx, u32)] {
        let start = self.offsets[stop.0 as usize] as usize;
        let end = self.offsets[stop.0 as usize + 1] as usize;
        &self.edges[start..end]
    }
}

impl Transfers {
    /// A network with no transfers.
    pub fn empty(stop_count: u32) -> Transfers {
        Transfers {
            offsets: vec![0; stop_count as usize + 1],
            edges: Vec::new(),
            closed: true,
        }
    }

    /// Builds the CSR structure from `(from, to, duration, meters)` edges.
    ///
    /// Duplicate `(from, to)` pairs keep one edge — the fastest, and the
    /// shortest on equal durations — so the meters reported for a
    /// relaxed transfer always belong to the edge routing used.
    pub fn from_edges(
        stop_count: u32,
        edges: &[(StopIdx, StopIdx, u32, f64)],
    ) -> Result<Transfers, TimetableError> {
        for &(from, to, _, _) in edges {
            for stop in [from, to] {
                if stop.0 >= stop_count {
                    return Err(TimetableError::StopOutOfRange {
                        stop: stop.0,
                        stop_count,
                    });
                }
            }
        }
        let mut edges = edges.to_vec();
        edges.sort_by(|a, b| {
            (a.0, a.1, a.2)
                .cmp(&(b.0, b.1, b.2))
                .then(a.3.total_cmp(&b.3))
        });
        edges.dedup_by_key(|&mut (from, to, _, _)| (from, to));
        let edges = &edges[..];
        let mut offsets = vec![0u32; stop_count as usize + 1];
        for (from, _, _, _) in edges {
            offsets[from.0 as usize + 1] += 1;
        }
        for stop in 0..stop_count as usize {
            offsets[stop + 1] += offsets[stop];
        }
        let mut sorted = vec![
            Transfer {
                to: StopIdx(0),
                duration: 0,
                meters: 0.0,
            };
            edges.len()
        ];
        let mut cursor = offsets.clone();
        for &(from, to, duration, meters) in edges {
            let slot = cursor[from.0 as usize] as usize;
            sorted[slot] = Transfer {
                to,
                duration,
                meters,
            };
            cursor[from.0 as usize] += 1;
        }
        Ok(Transfers {
            offsets,
            edges: sorted,
            closed: true,
        })
    }

    /// Declares that this set does not satisfy the transitive-closure
    /// contract, switching RAPTOR to its exact transfer phase.
    pub fn mark_unclosed(&mut self) {
        self.closed = false;
    }

    /// Whether the set satisfies the transitive-closure contract.
    pub fn closed(&self) -> bool {
        self.closed
    }

    /// Number of transfer edges.
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// The transfers leaving a stop.
    pub fn from_stop(&self, stop: StopIdx) -> &[Transfer] {
        let start = self.offsets[stop.0 as usize] as usize;
        let end = self.offsets[stop.0 as usize + 1] as usize;
        &self.edges[start..end]
    }

    /// The `from_stop` slice's index range into the edge-major order,
    /// for arrays aligned with the CSR edges (the merged set's per-edge
    /// rental meters).
    pub fn edge_range(&self, stop: StopIdx) -> std::ops::Range<usize> {
        self.offsets[stop.0 as usize] as usize..self.offsets[stop.0 as usize + 1] as usize
    }
}

#[cfg(test)]
#[path = "transfers_tests.rs"]
mod tests;
