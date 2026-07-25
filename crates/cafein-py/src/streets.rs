//! The standalone street network exposed to Python.
//!
//! A `TransportNetwork` carries a street network for walking access and
//! egress; this is the street graph on its own — built from the union OSM
//! extraction, routable by any compiled profile. Journeys never enter it, so
//! it holds no timetable, transfers, or stop links.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use cafein_core::streets::{
    CompiledStreetProfile, EdgeAttributes, Snap, StreetNetwork as CoreStreetNetwork,
    StreetProfileDefinition,
};

/// The street-mode names accepted by the public API, with the shipped profile
/// definition each resolves to.
fn profile_definition(mode: &str) -> PyResult<StreetProfileDefinition> {
    match mode {
        "walk" => Ok(StreetProfileDefinition::walk()),
        "bicycle" => Ok(StreetProfileDefinition::bicycle()),
        "e_bike" => Ok(StreetProfileDefinition::e_bike()),
        "e_scooter" => Ok(StreetProfileDefinition::e_scooter()),
        other => Err(PyValueError::new_err(format!(
            "unknown street mode '{other}'; expected one of \
             'walk', 'bicycle', 'e_bike', 'e_scooter'"
        ))),
    }
}

/// A routable street network built from an OpenStreetMap extract.
#[pyclass]
pub struct StreetNetwork {
    inner: CoreStreetNetwork,
    /// Compiled profiles kept by the exact definition that produced them, so
    /// repeated queries reuse the per-arc millisecond array instead of
    /// rescanning every arc. Keyed by full equality, never by name: a changed
    /// speed or multiplier compiles afresh rather than reusing stale costs.
    profiles: Vec<(StreetProfileDefinition, CompiledStreetProfile)>,
}

#[pymethods]
impl StreetNetwork {
    /// Builds the network from the union extraction's flat arrays.
    ///
    /// Every attribute array carries one entry per physical edge, in the same
    /// order as `edges`.
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(
        vertex_count: u32,
        edges: Vec<(u32, u32, f64)>,
        coordinate_offsets: Vec<u32>,
        longitudes: Vec<f64>,
        latitudes: Vec<f64>,
        edge_highway: Vec<u8>,
        edge_surface: Vec<u8>,
        edge_smoothness: Vec<u8>,
        edge_flags: Vec<u16>,
        access_forward: Vec<u8>,
        access_reverse: Vec<u8>,
        facility_forward: Vec<u8>,
        facility_reverse: Vec<u8>,
    ) -> PyResult<StreetNetwork> {
        let inner = CoreStreetNetwork::new_multimodal(
            vertex_count,
            0,
            &edges,
            &coordinate_offsets,
            &longitudes,
            &latitudes,
            Vec::new(),
            EdgeAttributes {
                highway: &edge_highway,
                surface: &edge_surface,
                smoothness: &edge_smoothness,
                flags: &edge_flags,
                access_forward: &access_forward,
                access_reverse: &access_reverse,
                facility_forward: &facility_forward,
                facility_reverse: &facility_reverse,
            },
        )
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
        Ok(StreetNetwork {
            inner,
            profiles: Vec::new(),
        })
    }

    #[getter]
    fn vertex_count(&self) -> u32 {
        self.inner.vertex_count()
    }

    #[getter]
    fn edge_count(&self) -> u32 {
        self.inner.edge_count()
    }

    /// The travel time in whole seconds from `origin` to `destination` under
    /// `mode`, or `None` when the destination is not reachable within
    /// `max_seconds`.
    ///
    /// Coordinates are `(latitude, longitude)` in EPSG:4326. A coordinate that
    /// does not snap to the network raises `ValueError` — unreachable and
    /// unsnappable are different answers.
    fn travel_time(
        &mut self,
        origin: (f64, f64),
        destination: (f64, f64),
        mode: &str,
        max_seconds: f64,
        max_snap_distance: f64,
    ) -> PyResult<Option<u32>> {
        // Compile first: snapping is profile-aware, so that a bicycle query
        // does not land on a footway it may not enter.
        let index = self.compiled(mode)?;
        let (_, profile) = &self.profiles[index];
        let from = self.snap_endpoint(origin, max_snap_distance, "origin", profile)?;
        let to = self.snap_endpoint(destination, max_snap_distance, "destination", profile)?;
        // A coordinate routes to itself in no time. The connector is the walk
        // between the coordinate and its street, so routing through the network
        // would charge it twice — leaving a point a positive distance from
        // itself. Snapping still runs first, so a coordinate off the network is
        // an error rather than a silent zero.
        if origin == destination {
            return Ok((max_seconds.is_finite() && max_seconds >= 0.0).then_some(0));
        }
        Ok(self
            .inner
            .directed_travel_time(&from, &to, profile, max_seconds))
    }
}

impl StreetNetwork {
    /// Snaps one endpoint, naming it in the error so a failure says which
    /// coordinate was off the network.
    fn snap_endpoint(
        &self,
        (latitude, longitude): (f64, f64),
        max_snap_distance: f64,
        role: &str,
        profile: &CompiledStreetProfile,
    ) -> PyResult<Snap> {
        self.inner
            .snap_for_profile(latitude, longitude, max_snap_distance, profile)
            .ok_or_else(|| {
                PyValueError::new_err(format!(
                    "the {role} ({latitude}, {longitude}) is farther than \
                     {max_snap_distance} m from any street '{}' can use",
                    profile.definition.name
                ))
            })
    }

    /// The position of `mode`'s compiled profile, compiling and caching it on
    /// first use. Returns an index rather than a reference so the cache can be
    /// extended while the network stays borrowed.
    fn compiled(&mut self, mode: &str) -> PyResult<usize> {
        let definition = profile_definition(mode)?;
        if let Some(index) = self
            .profiles
            .iter()
            .position(|(cached, _)| *cached == definition)
        {
            return Ok(index);
        }
        let compiled = self
            .inner
            .compile_profile(&definition)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        self.profiles.push((definition, compiled));
        Ok(self.profiles.len() - 1)
    }
}
