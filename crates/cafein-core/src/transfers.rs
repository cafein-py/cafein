//! Precomputed stop-to-stop transfers (footpaths), CSR by origin stop.

use crate::timetable::{StopIdx, TimetableError};

/// A walkable connection to another stop.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Transfer {
    pub to: StopIdx,
    /// Walking time in seconds.
    pub duration: u32,
}

/// All stop-to-stop transfers of a network.
///
/// The input edge list must already be transitively closed (the footpath
/// precompute's responsibility): routing relaxes one transfer hop per
/// round and does not chain transfers.
#[derive(Debug, PartialEq, Eq)]
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

    /// Builds the CSR structure from `(from, to, duration)` edges.
    pub fn from_edges(
        stop_count: u32,
        edges: &[(StopIdx, StopIdx, u32)],
    ) -> Result<Transfers, TimetableError> {
        for &(from, to, _) in edges {
            for stop in [from, to] {
                if stop.0 >= stop_count {
                    return Err(TimetableError::StopOutOfRange {
                        stop: stop.0,
                        stop_count,
                    });
                }
            }
        }
        let mut offsets = vec![0u32; stop_count as usize + 1];
        for (from, _, _) in edges {
            offsets[from.0 as usize + 1] += 1;
        }
        for stop in 0..stop_count as usize {
            offsets[stop + 1] += offsets[stop];
        }
        let mut sorted = vec![
            Transfer {
                to: StopIdx(0),
                duration: 0
            };
            edges.len()
        ];
        let mut cursor = offsets.clone();
        for &(from, to, duration) in edges {
            let slot = cursor[from.0 as usize] as usize;
            sorted[slot] = Transfer { to, duration };
            cursor[from.0 as usize] += 1;
        }
        Ok(Transfers {
            offsets,
            edges: sorted,
        })
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
                (StopIdx(2), StopIdx(0), 30),
                (StopIdx(0), StopIdx(1), 60),
                (StopIdx(0), StopIdx(2), 90),
            ],
        )
        .unwrap();
        assert_eq!(
            transfers.from_stop(StopIdx(0)),
            &[
                Transfer {
                    to: StopIdx(1),
                    duration: 60
                },
                Transfer {
                    to: StopIdx(2),
                    duration: 90
                },
            ]
        );
        assert_eq!(transfers.from_stop(StopIdx(1)), &[]);
        assert_eq!(transfers.from_stop(StopIdx(2)).len(), 1);
    }

    #[test]
    fn rejects_out_of_range_stops() {
        assert_eq!(
            Transfers::from_edges(1, &[(StopIdx(0), StopIdx(1), 30)]),
            Err(TimetableError::StopOutOfRange {
                stop: 1,
                stop_count: 1
            })
        );
    }
}
