//! The algorithm-agnostic journey output contract.
//!
//! Every router returns journeys in this shape; router internals must not
//! leak into it. Transit legs reference the trip and the board/alight
//! positions within its pattern, which is all later per-leg annotation
//! (distances, geometry, emissions) needs.

use crate::timetable::{StopIdx, TripIdx};

/// One journey from origin to destination.
///
/// All times are seconds past the service day's start.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Journey {
    pub legs: Vec<Leg>,
    /// The requested departure time.
    pub departure: u32,
    /// Arrival at the destination, including the egress leg.
    pub arrival: u32,
}

impl Journey {
    /// The number of transit legs (rides) in the journey.
    pub fn rides(&self) -> usize {
        self.legs
            .iter()
            .filter(|leg| matches!(leg, Leg::Transit { .. }))
            .count()
    }
}

/// One leg of a journey.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Leg {
    /// From the origin to the first stop.
    Access {
        to_stop: StopIdx,
        departure: u32,
        arrival: u32,
    },
    /// Riding one trip between two positions of its pattern.
    Transit {
        trip: TripIdx,
        board_stop: StopIdx,
        alight_stop: StopIdx,
        board_position: u16,
        alight_position: u16,
        board_time: u32,
        alight_time: u32,
    },
    /// A stop-to-stop transfer (footpath).
    Transfer {
        from_stop: StopIdx,
        to_stop: StopIdx,
        departure: u32,
        arrival: u32,
    },
    /// From the last stop to the destination.
    Egress {
        from_stop: StopIdx,
        departure: u32,
        arrival: u32,
    },
}
