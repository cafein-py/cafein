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
/// The input edge list must already be transitively closed (the footpath
/// precompute's responsibility): routing relaxes one transfer hop per
/// round and does not chain transfers.
#[derive(Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Transfers {
    /// CSR offsets into `edges`, one entry per stop plus a tail.
    offsets: Vec<u32>,
    edges: Vec<Transfer>,
}

impl Transfers {
    /// A network with no transfers.
    pub fn empty(stop_count: u32) -> Transfers {
        Transfers {
            offsets: vec![0; stop_count as usize + 1],
            edges: Vec::new(),
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
        })
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn groups_edges_by_origin_stop() {
        let transfers = Transfers::from_edges(
            3,
            &[
                (StopIdx(2), StopIdx(0), 30, 30.0),
                (StopIdx(0), StopIdx(1), 60, 58.5),
                (StopIdx(0), StopIdx(2), 90, 90.0),
            ],
        )
        .unwrap();
        assert_eq!(
            transfers.from_stop(StopIdx(0)),
            &[
                Transfer {
                    to: StopIdx(1),
                    duration: 60,
                    meters: 58.5,
                },
                Transfer {
                    to: StopIdx(2),
                    duration: 90,
                    meters: 90.0,
                },
            ]
        );
        assert_eq!(transfers.from_stop(StopIdx(1)), &[]);
        assert_eq!(transfers.from_stop(StopIdx(2)).len(), 1);
    }

    #[test]
    fn keeps_one_edge_per_stop_pair() {
        // The fastest duplicate wins; on equal durations, the shortest.
        let transfers = Transfers::from_edges(
            2,
            &[
                (StopIdx(0), StopIdx(1), 60, 59.0),
                (StopIdx(0), StopIdx(1), 45, 44.0),
                (StopIdx(0), StopIdx(1), 45, 43.5),
            ],
        )
        .unwrap();
        assert_eq!(
            transfers.from_stop(StopIdx(0)),
            &[Transfer {
                to: StopIdx(1),
                duration: 45,
                meters: 43.5,
            }]
        );
        assert_eq!(transfers.edge_count(), 1);
    }

    #[test]
    fn rejects_out_of_range_stops() {
        assert_eq!(
            Transfers::from_edges(1, &[(StopIdx(0), StopIdx(1), 30, 30.0)]),
            Err(TimetableError::StopOutOfRange {
                stop: 1,
                stop_count: 1
            })
        );
    }
}
