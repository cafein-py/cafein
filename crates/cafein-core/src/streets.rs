//! The query-time street network for walking access and egress.
//!
//! The Python-side build hands the walking graph over as flat arrays:
//! vertices are implicit indices, edges carry their cost length in meters
//! and their geometry. A query snaps a coordinate to its nearest edge
//! through a packed static index over the edge segments — a virtual node at
//! the snap point, with the split edge's cost pro-rated by the fraction each
//! side covers — and runs a cutoff-bounded Dijkstra to collect every transit
//! stop reachable on foot, entering stops through the same snap links the
//! footpath precompute used. Walking is undirected, so one search serves
//! access and egress alike.
//!
//! Edges and vertices are renumbered along a Hilbert curve at build time, so
//! spatially-nearby streets are nearby in every edge-indexed array — the ids
//! are internal, and only exactly-equal snap or cost ties can resolve
//! differently than under the input order.
//!
//! Geometry is stored in geographic coordinates (longitude/latitude); there
//! is no global projection. Distances use a local `cos(latitude)` evaluated
//! at the point's own latitude, so they stay accurate over country-scale
//! latitude ranges (following R5/OpenTripPlanner). Segments are densified to
//! a maximum length at build time, so the local-scale model is exact: the
//! snap foot, the connector distance, and the geometry lengths are all
//! computed in a frame local to the relevant short segment. The packed index
//! stores segment boxes in lon/lat and is used only for envelope queries —
//! never metric nearest-neighbour, whose degree-Euclidean distance would be
//! wrong.
//!
//! Coordinates are assumed to be a contiguous extract within a continuous
//! longitude range — the regional OSM extracts cafein consumes never cross the
//! antimeridian. Distance measurement still uses the shortest signed longitude
//! delta defensively, but the snap envelope is a single longitude interval, so
//! snapping is not supported across ±180°.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};

use rayon::prelude::*;

use crate::timetable::StopIdx;

/// How a stop enters the street graph: snapped onto an edge at a fraction
/// of its cost length, over a straight connector to the snap point.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct StopLink {
    pub stop: StopIdx,
    /// Index of the edge the stop snapped onto.
    pub edge: u32,
    /// Snap position as a fraction of the edge, 0 at its `from` vertex.
    pub fraction: f64,
    /// Straight-line distance from the stop to the snap point, in meters.
    pub connector: f64,
}

/// A stop link as the network stores it: the input [`StopLink`] with its
/// edge's endpoint vertices denormalised in, so a load can rebuild the
/// vertex→link index from the links alone, without the street arrays.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct StoredLink {
    pub stop: StopIdx,
    /// Index of the edge the stop snapped onto (internal numbering).
    pub edge: u32,
    /// Snap position as a fraction of the edge, 0 at its `from` vertex.
    pub fraction: f64,
    /// Straight-line distance from the stop to the snap point, in meters.
    pub connector: f64,
    /// The edge's endpoint vertices.
    pub from: u32,
    pub to: u32,
}

/// A stop reached by the walking search.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WalkedStop {
    pub stop: StopIdx,
    /// Walking time in whole seconds, rounded up.
    pub seconds: u32,
    /// The exact walked street-path length in meters, connectors
    /// included.
    pub meters: f64,
}

/// A coordinate snapped onto the street network.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Snap {
    pub edge: u32,
    /// Snap position as a fraction of the edge, 0 at its `from` vertex.
    pub fraction: f64,
    /// Straight-line distance from the coordinate to the snap point, in
    /// meters.
    pub connector: f64,
}

/// Errors raised while assembling a [`StreetNetwork`].
#[derive(Debug, PartialEq)]
pub enum StreetError {
    /// The coordinate offsets do not describe the coordinate arrays.
    InvalidOffsets,
    /// An edge has fewer than two geometry coordinates.
    ShortGeometry { edge: u32 },
    /// An edge has a non-finite coordinate.
    InvalidCoordinates { edge: u32 },
    /// An edge endpoint is not below the vertex count.
    VertexOutOfRange { edge: u32, vertex_count: u32 },
    /// An edge's cost length is negative or not finite.
    InvalidLength { edge: u32 },
    /// A stop link references an edge that does not exist.
    LinkEdgeOutOfRange { link: usize, edge_count: u32 },
    /// A stop link's stop index is not below the stop count.
    StopOutOfRange { stop: u32, stop_count: u32 },
    /// A stop link's fraction is outside [0, 1] or its connector is
    /// negative or not finite.
    InvalidLink { link: usize },
    /// A mapped array range is out of bounds, misaligned, or shaped
    /// inconsistently with the index payload.
    InvalidMapping,
}

impl std::fmt::Display for StreetError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StreetError::InvalidOffsets => {
                write!(f, "coordinate offsets do not match the coordinate arrays")
            }
            StreetError::ShortGeometry { edge } => {
                write!(f, "edge {edge} has fewer than two geometry coordinates")
            }
            StreetError::InvalidCoordinates { edge } => {
                write!(f, "edge {edge} has a non-finite coordinate")
            }
            StreetError::VertexOutOfRange { edge, vertex_count } => {
                write!(
                    f,
                    "edge {edge} has an endpoint out of range ({vertex_count} vertices)"
                )
            }
            StreetError::InvalidLength { edge } => {
                write!(f, "edge {edge} has a negative or non-finite length")
            }
            StreetError::LinkEdgeOutOfRange { link, edge_count } => {
                write!(
                    f,
                    "stop link {link} references an edge out of range ({edge_count} edges)"
                )
            }
            StreetError::StopOutOfRange { stop, stop_count } => {
                write!(f, "stop index {stop} is out of range ({stop_count} stops)")
            }
            StreetError::InvalidLink { link } => {
                write!(
                    f,
                    "stop link {link} has a fraction outside [0, 1] or an invalid connector"
                )
            }
            StreetError::InvalidMapping => {
                write!(f, "a mapped street array is out of bounds or misaligned")
            }
        }
    }
}

impl std::error::Error for StreetError {}

mod geo;
mod graph;
mod index;
mod paths;
mod search;
mod snap;

use geo::*;
pub use graph::{Backing, MappedStreets, StreetNetwork, StreetNetworkParts};
use index::*;
use search::*;

#[cfg(test)]
mod tests;
