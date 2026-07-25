//! The canonical path key: a topology-only total order over
//! equal-arrival journeys, shared by RAPTOR and TBTR so both engines
//! elect the same representative whatever chain their scan order meets
//! first. Nothing here reads distances, factors, fares, or geometry —
//! the emissions firewall stays intact; the key orders journeys by the
//! rides and walks they are made of.

/// One step of a journey chain, listed destination → origin. The
/// variant order is the tie-break rank: at an equal arrival a direct
/// ride beats a walked arrival. `Access` terminates every sequence, so
/// two distinct chains always differ at some earlier token.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum PathToken {
    Ride {
        trip: u32,
        day_offset: u32,
        board: u16,
        alight: u16,
    },
    Walk {
        from: u32,
        to: u32,
        duration: u32,
    },
    Access {
        stop: u32,
        duration: u32,
    },
}

/// Whether a challenger chain is canonically preferred over the
/// incumbent at an exact arrival tie. The later root departure wins
/// outright — a chain retained from a later departure of a range scan
/// dominates an equal-arrival chain leaving earlier — then the
/// lexicographically smaller destination-to-origin token sequence.
pub(crate) fn challenger_wins(
    challenger_root: u32,
    challenger: &[PathToken],
    incumbent_root: u32,
    incumbent: &[PathToken],
) -> bool {
    match challenger_root.cmp(&incumbent_root) {
        std::cmp::Ordering::Greater => true,
        std::cmp::Ordering::Less => false,
        std::cmp::Ordering::Equal => challenger < incumbent,
    }
}

#[cfg(test)]
mod tests;
