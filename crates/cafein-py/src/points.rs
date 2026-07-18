//! Street-linking machinery for the coordinate surfaces.

use super::*;

impl TransportNetwork {
    /// A stop's `(latitude, longitude)`, or `None` when the feed omits it.
    pub(super) fn stop_coordinate(&self, stop: StopIdx) -> Option<(f64, f64)> {
        let stop = &self.feed.stops[stop.0 as usize];
        Some((stop.latitude?, stop.longitude?))
    }

    /// The per-destination final-walk egress for the one-to-all time queries:
    /// a bounded (`max_walking_time`) `link_many` from every stop's coordinate,
    /// giving `egress[t]` the stops within a final walk of `t` and their walk
    /// seconds. The same bounded construction the coordinate query and the cost
    /// matrix use — walking is undirected, so a search from `t` yields the
    /// `s -> t` egress. Built once per query and reused across a matrix's
    /// origins. `None` means the stop has no coordinate (it cannot be located,
    /// so it keeps its bare transit arrival); `Some(list)` treats the stop's
    /// coordinate as the destination — an empty list is a located-but-unreachable
    /// coordinate (its connector exceeds the cap), which gets no arrival, exactly
    /// as `route_between_coordinates` would refuse it.
    pub(super) fn final_egress(
        &self,
        streets: &StreetNetwork,
        speed: f64,
        max_walking_time: f64,
        max_snap_distance: f64,
    ) -> Vec<Option<Vec<(StopIdx, u32)>>> {
        let stop_count = self.build.timetable.stop_count() as usize;
        let mut coordinates: Vec<(f64, f64)> = Vec::with_capacity(stop_count);
        let mut slots: Vec<Option<usize>> = Vec::with_capacity(stop_count);
        for index in 0..stop_count {
            match self.stop_coordinate(StopIdx(index as u32)) {
                Some(coordinate) => {
                    slots.push(Some(coordinates.len()));
                    coordinates.push(coordinate);
                }
                None => slots.push(None),
            }
        }
        let links = streets.link_many(&coordinates, speed, max_walking_time, max_snap_distance);
        (0..stop_count)
            .map(|index| {
                slots[index].map(|slot| {
                    let mut sources = Vec::new();
                    if let Some(reached) = &links[slot] {
                        sources.extend(reached.iter().map(|walk| (walk.stop, walk.seconds)));
                    }
                    sources
                })
            })
            .collect()
    }

    /// The street egress for the emissions cost matrix, keyed by source stop:
    /// `map[s]` lists `(destination slot, walk seconds, walk meters)` for every
    /// matrix destination reachable by a final walk off stop `s` — the reverse
    /// of `final_egress`, carrying metres for the reported walk distance. A
    /// destination without a coordinate (or unreachable within
    /// `max_walking_time`) contributes no sources, so it is reached only by
    /// alighting there directly.
    pub(super) fn matrix_street_egress(
        &self,
        streets: &StreetNetwork,
        destinations: &[StopIdx],
        speed: f64,
        max_walking_time: f64,
        max_snap_distance: f64,
    ) -> Vec<Vec<(u32, u32, f64)>> {
        let stop_count = self.build.timetable.stop_count() as usize;
        let mut coordinates: Vec<(f64, f64)> = Vec::with_capacity(destinations.len());
        let mut slot_of: Vec<u32> = Vec::with_capacity(destinations.len());
        for (slot, &destination) in destinations.iter().enumerate() {
            if let Some(coordinate) = self.stop_coordinate(destination) {
                slot_of.push(slot as u32);
                coordinates.push(coordinate);
            }
        }
        let links = streets.link_many(&coordinates, speed, max_walking_time, max_snap_distance);
        let mut map = vec![Vec::new(); stop_count];
        for (index, reached) in links.iter().enumerate() {
            if let Some(reached) = reached {
                let slot = slot_of[index];
                for walk in reached {
                    map[walk.stop.0 as usize].push((slot, walk.seconds, walk.meters));
                }
            }
        }
        // A destination with no coordinate cannot be a door-to-door coordinate,
        // so it is reachable only by a direct alight — give it a bare zero-walk
        // self-entry. A located destination is left to its `link_many` connector:
        // if its coordinate does not snap or lies beyond the cap it carries no
        // entry and is simply unreachable, exactly as the single-pair coordinate
        // route would refuse it (rather than crediting it as a free alight).
        for (slot, &destination) in destinations.iter().enumerate() {
            if self.stop_coordinate(destination).is_none() {
                map[destination.0 as usize].push((slot as u32, 0, 0.0));
            }
        }
        map
    }

    /// The location-based access for the emissions cost matrix, one entry per
    /// origin: the stops within an initial walk of the origin's coordinate — its
    /// own connector included — as `(stop, walk seconds, walk meters)`, plus a
    /// `located` flag. A coordinate that **snaps** is `located` (`true`) even
    /// when no stop is reachable within the cap — its access is then empty (no
    /// transit boarding), but it stays on the door-to-door path so its
    /// direct-walk overlay still applies. Only a missing coordinate or a failed
    /// snap gives the board-at-origin fallback `[(origin, 0, 0)]` with `false`,
    /// routing that origin over the closure rather than the intermediate-only
    /// set. The initial-walk analogue of `matrix_street_egress`; the metres are
    /// threaded into the reported walk distance.
    #[allow(clippy::type_complexity)]
    pub(super) fn matrix_location_access(
        &self,
        streets: &StreetNetwork,
        origins: &[StopIdx],
        speed: f64,
        max_walking_time: f64,
        max_snap_distance: f64,
    ) -> (Vec<Vec<(StopIdx, u32, f64)>>, Vec<bool>) {
        let mut coordinates: Vec<(f64, f64)> = Vec::with_capacity(origins.len());
        let mut coordinate_of: Vec<Option<usize>> = Vec::with_capacity(origins.len());
        for &origin in origins {
            match self.stop_coordinate(origin) {
                Some(coordinate) => {
                    coordinate_of.push(Some(coordinates.len()));
                    coordinates.push(coordinate);
                }
                None => coordinate_of.push(None),
            }
        }
        let links = streets.link_many(&coordinates, speed, max_walking_time, max_snap_distance);
        origins
            .iter()
            .zip(coordinate_of)
            .map(
                |(&origin, slot)| match slot.and_then(|slot| links[slot].as_ref()) {
                    // Snapped — located even when no stop is reachable within the
                    // cap: it still takes the direct-walk overlay, an empty access
                    // just means no transit boarding. Only a missing coordinate or
                    // a failed snap falls back to closure board-at-origin routing.
                    Some(reached) => (
                        reached
                            .iter()
                            .map(|walk| (walk.stop, walk.seconds, walk.meters))
                            .collect(),
                        true,
                    ),
                    None => (vec![(origin, 0, 0.0)], false),
                },
            )
            .unzip()
    }

    /// The explicit coordinate-to-coordinate direct street walks for the
    /// emissions cost matrix, per origin: `(destination slot, walk seconds, walk
    /// metres)` for every destination the origin's coordinate reaches on foot
    /// within the cap. Built from `walk_matrix`, which snaps both coordinates,
    /// zeroes the same-coordinate diagonal, and returns nothing for a coordinate
    /// that does not snap — so a cell matches the single-pair route's direct
    /// walk rather than inferring one from stop connectors. A stop with no
    /// coordinate contributes and receives no walk.
    pub(super) fn matrix_direct_walks(
        &self,
        streets: &StreetNetwork,
        origins: &[StopIdx],
        destinations: &[StopIdx],
        speed: f64,
        max_walking_time: f64,
        max_snap_distance: f64,
    ) -> Vec<Vec<(u32, u32, f64)>> {
        let mut origin_coords: Vec<(f64, f64)> = Vec::new();
        let mut origin_row: Vec<Option<usize>> = Vec::with_capacity(origins.len());
        for &origin in origins {
            match self.stop_coordinate(origin) {
                Some(coordinate) => {
                    origin_row.push(Some(origin_coords.len()));
                    origin_coords.push(coordinate);
                }
                None => origin_row.push(None),
            }
        }
        let mut dest_coords: Vec<(f64, f64)> = Vec::new();
        let mut dest_col: Vec<Option<usize>> = Vec::with_capacity(destinations.len());
        for &destination in destinations {
            match self.stop_coordinate(destination) {
                Some(coordinate) => {
                    dest_col.push(Some(dest_coords.len()));
                    dest_coords.push(coordinate);
                }
                None => dest_col.push(None),
            }
        }
        let walk = streets.walk_matrix(
            &origin_coords,
            &dest_coords,
            speed,
            max_walking_time,
            max_snap_distance,
        );
        origin_row
            .iter()
            .map(|&row| match row {
                None => Vec::new(),
                Some(row) => dest_col
                    .iter()
                    .enumerate()
                    .filter_map(|(slot, &col)| {
                        let (seconds, meters) = walk[row][col?]?;
                        Some((slot as u32, seconds, meters))
                    })
                    .collect(),
            })
            .collect()
    }

    /// Folds one **bounded** final walk into a one-to-all arrival array,
    /// location-based: each *located* target stop's arrival becomes the earliest
    /// `arrival[source] + walk(source -> target)` over the sources in
    /// `egress[target]` (within `max_walking_time`). The egress is
    /// `link_many(target.coordinate)`, which includes the target itself via its
    /// connector, so a transit-reached target keeps its own arrival plus that
    /// connector — the arrival *at the stop's coordinate*, matching
    /// `route_between_coordinates`. `egress[target] == None` (no coordinate)
    /// keeps the bare RAPTOR arrival; `Some(empty)` (coordinate unreachable
    /// within `max_walking_time`) yields no arrival, as the coordinate query
    /// would. A single hop over a snapshot of the RAPTOR arrivals, so a final
    /// walk never chains. Used by the one-to-all time queries under a whole-day
    /// ULTRA set.
    pub(super) fn fold_final_transfers(
        &self,
        arrivals: &mut [Option<u32>],
        egress: &[Option<Vec<(StopIdx, u32)>>],
    ) {
        let reached: Vec<Option<u32>> = arrivals.to_vec();
        for (target, entry) in egress.iter().enumerate() {
            let Some(sources) = entry else {
                continue; // no coordinate: keep the bare transit arrival
            };
            let mut best = None;
            for &(source, walk) in sources {
                let Some(at) = reached[source.0 as usize] else {
                    continue;
                };
                if let Some(candidate) = at.checked_add(walk).filter(|&at| at != u32::MAX) {
                    best = Some(best.map_or(candidate, |current: u32| current.min(candidate)));
                }
            }
            arrivals[target] = best;
        }
    }
}

/// The `(stop, seconds)` request offsets of a walking-search result.
pub(super) fn request_offsets(walks: &[WalkedStop]) -> Vec<(StopIdx, u32)> {
    walks.iter().map(|walk| (walk.stop, walk.seconds)).collect()
}

/// A coordinate query's endpoints, for drawing its walk legs.
pub(super) struct CoordinateEnds {
    pub(super) origin: (f64, f64),
    pub(super) origin_snap: Snap,
    pub(super) destination: (f64, f64),
    pub(super) destination_snap: Snap,
}

/// The indices of points the linking could not snap.
pub(super) fn unsnapped(links: &[Option<Vec<WalkedStop>>]) -> Vec<u32> {
    links
        .iter()
        .enumerate()
        .filter_map(|(index, links)| links.is_none().then_some(index as u32))
        .collect()
}

/// Each destination point's `(stop, seconds, meters)` egress table;
/// unsnapped points get an empty table and stay unreachable.
pub(super) fn egress_tables(links: &[Option<Vec<WalkedStop>>]) -> Vec<Vec<(StopIdx, u32, f64)>> {
    links
        .iter()
        .map(|links| {
            links
                .as_deref()
                .unwrap_or(&[])
                .iter()
                .map(|walk| (walk.stop, walk.seconds, walk.meters))
                .collect()
        })
        .collect()
}

/// The walking speed in m/s of validated street-query parameters, or a
/// `ValueError` naming the parameter that is out of range.
pub(super) fn validated_walking_speed(
    walking_speed_kmph: f64,
    max_walking_time: f64,
    max_snap_distance: f64,
) -> PyResult<f64> {
    if !walking_speed_kmph.is_finite() || walking_speed_kmph <= 0.0 {
        return Err(PyValueError::new_err(
            "walking_speed_kmph must be a positive, finite number",
        ));
    }
    if !max_walking_time.is_finite() || max_walking_time < 0.0 {
        return Err(PyValueError::new_err(
            "max_walking_time must be a non-negative, finite number",
        ));
    }
    if !max_snap_distance.is_finite() || max_snap_distance < 0.0 {
        return Err(PyValueError::new_err(
            "max_snap_distance must be a non-negative, finite number",
        ));
    }
    Ok(walking_speed_kmph / 3.6)
}

/// The stops walkable from a coordinate, or a `ValueError` when the
/// coordinate is invalid or off the network; `side` prefixes the message
/// (e.g. `"origin "`) to name the endpoint.
pub(super) fn coordinate_links(
    streets: &StreetNetwork,
    coordinate: (f64, f64),
    walking_speed: f64,
    max_walking_time: f64,
    max_snap_distance: f64,
    side: &str,
) -> PyResult<Vec<WalkedStop>> {
    let (lat, lon) = coordinate;
    if !lat.is_finite() || !lon.is_finite() {
        return Err(PyValueError::new_err(format!(
            "{side}lat and lon must be finite"
        )));
    }
    streets
        .access_stops(lat, lon, walking_speed, max_walking_time, max_snap_distance)
        .ok_or_else(|| {
            PyValueError::new_err(format!(
                "{side}({lat}, {lon}) is farther than {max_snap_distance} m \
                 from the walking network"
            ))
        })
}

/// Encodes coordinates as a little-endian WKB LineString (XY).
pub(super) fn wkb_line_string<'py>(
    py: Python<'py>,
    coordinates: &[(f64, f64)],
) -> Bound<'py, PyBytes> {
    PyBytes::new(py, &cafein_core::geometry::wkb_line_string(coordinates))
}
