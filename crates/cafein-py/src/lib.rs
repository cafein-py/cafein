//! Python bindings for cafein.

use std::collections::HashMap;

use chrono::NaiveDate;
use numpy::{IntoPyArray, PyArray2, PyArray3, PyArrayMethods, PyReadonlyArray1};
use pyo3::exceptions::{PyKeyError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList};

use cafein_core::ch::ContractionHierarchy;
use cafein_core::exhaustive;
use cafein_core::fares::{FareTables, RuleFares, ZoneFares, ZoneProduct, NO_FARE};
use cafein_core::geometry::{wkb_multi_line_string, DistanceProvenance, LegGeometry, TripGeometry};
use cafein_core::journey::{Journey, Leg};
use cafein_core::mcraptor;
use cafein_core::mctbtr::McTbtrEngine;
use cafein_core::mcultra::compute_mcultra_shortcuts;
use cafein_core::raptor::{CostInputs, CostRow, Objective, Raptor};
use cafein_core::router::{factor_fingerprint, same_factors, Exclusions, Request, TransitRouter};
use cafein_core::streets::{
    Backing, MappedStreets, Snap, StopLink, StoredLink, StreetAttributes, StreetNetwork,
    StreetNetworkParts, WalkedStop,
};
use cafein_core::tbtr::{DayView, TbtrEngine};
use cafein_core::timetable::{StopIdx, Timetable, TripIdx};
use cafein_core::transfers::Transfers;
use cafein_core::ultra::{compute_shortcuts, Shortcut};
use cafein_gtfs::{build_timetable, Feed, RouteType, ServiceCalendar, TimetableBuild};

/// A routable public-transport network built from GTFS data.
#[pyclass]
struct TransportNetwork {
    feed: Feed,
    build: TimetableBuild,
    transfers: Transfers,
    /// The ULTRA shortcut set, when computed (`compute_ultra_shortcuts`):
    /// the intermediate transfers as a `Transfers`, derived from the
    /// installed street network. The time engines relax it in place of the
    /// closure `transfers` when it is present; the emissions/fare engines
    /// always use the closure. Persisted with the artifact and restored on
    /// load, scoped by `ultra_window`.
    ultra_transfers: Option<Transfers>,
    /// The source-departure window the ULTRA set was computed for, in
    /// seconds since midnight. A time query outside it falls back to the
    /// closure, so a partial-window set never silently drops transfers.
    ultra_window: Option<(u32, u32)>,
    /// The McULTRA (emissions-aware) shortcut set, when computed
    /// (`compute_mcultra_shortcuts`): the coordinate emissions engines relax it
    /// in place of the closure when a whole-day set is present and the query's
    /// factor vector matches the one it was built for (`mcultra_factors`).
    /// Persisted with the artifact and restored on load, with its window and
    /// factor vector.
    mcultra_transfers: Option<Transfers>,
    mcultra_window: Option<(u32, u32)>,
    /// The per-trip emission-factor vector the McULTRA set was built with,
    /// compared exactly (`same_factors`); a query using different factors
    /// falls back to the closure.
    mcultra_factors: Option<Vec<f64>>,
    /// The cached time-only TBTR transfer set, when computed
    /// (`compute_tbtr_transfers`), keyed by the date string it was built for.
    /// A `router="tbtr"` stop time matrix on the same date — single-departure
    /// or windowed — reuses it (build once, query many); other queries rebuild
    /// ad-hoc. Persisted with the artifact and restored on load, keyed by its
    /// date.
    tbtr_time_transfers: Option<(String, cafein_core::tbtr::TransferSet)>,
    /// The cached multicriteria TBTR transfer set with the date and the
    /// factor vector it was built for (`compute_mctbtr_transfers`).
    mctbtr_transfers: Option<(String, Vec<f64>, cafein_core::tbtr::TransferSet)>,
    geometry: Option<TripGeometry>,
    leg_geometry: Option<LegGeometry>,
    streets: Option<StreetNetwork>,
    stops_by_id: HashMap<String, StopLookup>,
    stops_by_qualified_id: HashMap<String, StopIdx>,
    trips_by_public_id: HashMap<String, TripIdx>,
    /// STREETS-section bytes the load explicitly read — 0 for a lazy
    /// mapped load; the laziness tests assert on it.
    streets_bytes_read: u64,
}

/// Resolution of a raw GTFS stop_id, which merged feeds can duplicate.
#[derive(Clone, Copy)]
enum StopLookup {
    Unique(StopIdx),
    Ambiguous,
}

/// A coordinate query's exact walk lengths in meters, keyed by the stop
/// each access or egress link enters the network through.
struct WalkMaps {
    access: HashMap<StopIdx, f64>,
    egress: HashMap<StopIdx, f64>,
}

impl WalkMaps {
    fn new(access: &[WalkedStop], egress: &[WalkedStop]) -> WalkMaps {
        let meters =
            |walks: &[WalkedStop]| walks.iter().map(|walk| (walk.stop, walk.meters)).collect();
        WalkMaps {
            access: meters(access),
            egress: meters(egress),
        }
    }
}

impl Backing for MappedArtifact {
    fn bytes(&self) -> &[u8] {
        &self.0
    }
}

mod artifact;
mod cost_matrices;
mod frontiers;
mod network;
mod options;
mod points;
mod routes;
mod time_matrices;

use artifact::*;
use cost_matrices::*;
use options::*;
use points::*;

#[pymodule]
fn _cafein(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_class::<TransportNetwork>()?;
    Ok(())
}
