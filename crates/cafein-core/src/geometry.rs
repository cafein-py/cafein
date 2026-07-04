//! Per-trip cumulative travel distances, CSR by trip.
//!
//! The distances come from the Python-side fallback ladder (GTFS shapes,
//! linear referencing, crow-fly estimation); each trip carries the
//! provenance tier of its estimate so uncertainty stays visible per leg.

use crate::timetable::{Timetable, TripIdx};

/// How a trip's distances were estimated (the fallback-ladder tier).
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum DistanceProvenance {
    /// Valid `shape_dist_traveled` values taken directly from the feed.
    ShapeDist,
    /// Stops linear-referenced onto the feed's shape geometry.
    ShapeLinRef,
    /// Distances cut from a matched OSM route relation.
    OsmRelation,
    /// Distances map-matched onto the mode's network graph.
    MapMatched,
    /// Great-circle distances scaled by a mode detour coefficient.
    CrowFly,
}

/// Cumulative distances in meters at every stop position of every trip.
#[derive(Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct TripGeometry {
    /// CSR offsets into `distances`, one entry per trip plus a tail.
    offsets: Vec<u32>,
    distances: Vec<f32>,
    provenance: Vec<DistanceProvenance>,
}

/// Errors raised while assembling a [`TripGeometry`].
#[derive(Debug, PartialEq, Eq)]
pub enum GeometryError {
    /// A trip index is not below the timetable's trip count.
    TripOutOfRange { trip: u32, trip_count: u32 },
    /// A trip was given twice.
    DuplicateTrip { trip: u32 },
    /// Not every trip of the timetable was given distances.
    MissingTrips { missing: u32 },
    /// A trip's distance count differs from its pattern's stop count.
    LengthMismatch {
        trip: u32,
        distances: usize,
        stops: usize,
    },
    /// A trip's cumulative distances decrease, or are not finite.
    InvalidDistances { trip: u32, position: usize },
}

impl std::fmt::Display for GeometryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GeometryError::TripOutOfRange { trip, trip_count } => {
                write!(f, "trip index {trip} is out of range ({trip_count} trips)")
            }
            GeometryError::DuplicateTrip { trip } => {
                write!(f, "trip index {trip} was given distances twice")
            }
            GeometryError::MissingTrips { missing } => {
                write!(f, "{missing} trip(s) were given no distances")
            }
            GeometryError::LengthMismatch {
                trip,
                distances,
                stops,
            } => write!(
                f,
                "trip index {trip} has {distances} distances but its pattern has {stops} stops"
            ),
            GeometryError::InvalidDistances { trip, position } => write!(
                f,
                "trip index {trip} has a decreasing or non-finite distance at position {position}"
            ),
        }
    }
}

impl std::error::Error for GeometryError {}

impl TripGeometry {
    /// Builds the CSR structure from per-trip cumulative distances.
    ///
    /// Every trip of the timetable must appear exactly once, with one
    /// non-decreasing, finite cumulative distance per pattern stop.
    pub fn from_trips(
        timetable: &Timetable,
        trips: Vec<(TripIdx, Vec<f32>, DistanceProvenance)>,
    ) -> Result<TripGeometry, GeometryError> {
        let trip_count = timetable.trip_count();
        let mut per_trip: Vec<Option<(Vec<f32>, DistanceProvenance)>> =
            (0..trip_count).map(|_| None).collect();
        for (trip, distances, provenance) in trips {
            if trip.0 >= trip_count {
                return Err(GeometryError::TripOutOfRange {
                    trip: trip.0,
                    trip_count,
                });
            }
            let stops = timetable.pattern_stops(timetable.trip_pattern(trip)).len();
            if distances.len() != stops {
                return Err(GeometryError::LengthMismatch {
                    trip: trip.0,
                    distances: distances.len(),
                    stops,
                });
            }
            for (position, value) in distances.iter().enumerate() {
                if !value.is_finite() {
                    return Err(GeometryError::InvalidDistances {
                        trip: trip.0,
                        position,
                    });
                }
            }
            for (position, pair) in distances.windows(2).enumerate() {
                if pair[1] < pair[0] {
                    return Err(GeometryError::InvalidDistances {
                        trip: trip.0,
                        position: position + 1,
                    });
                }
            }
            let slot = &mut per_trip[trip.0 as usize];
            if slot.is_some() {
                return Err(GeometryError::DuplicateTrip { trip: trip.0 });
            }
            *slot = Some((distances, provenance));
        }
        let missing = per_trip.iter().filter(|slot| slot.is_none()).count() as u32;
        if missing > 0 {
            return Err(GeometryError::MissingTrips { missing });
        }

        let mut offsets = Vec::with_capacity(trip_count as usize + 1);
        offsets.push(0u32);
        let mut flat = Vec::new();
        let mut provenance = Vec::with_capacity(trip_count as usize);
        for slot in per_trip {
            let (distances, tier) = slot.unwrap();
            flat.extend_from_slice(&distances);
            offsets.push(flat.len() as u32);
            provenance.push(tier);
        }
        Ok(TripGeometry {
            offsets,
            distances: flat,
            provenance,
        })
    }

    /// The distance in meters travelled on `trip` between two positions.
    pub fn leg_distance(&self, trip: TripIdx, board_position: u16, alight_position: u16) -> f32 {
        let start = self.offsets[trip.0 as usize] as usize;
        self.distances[start + alight_position as usize]
            - self.distances[start + board_position as usize]
    }

    /// The provenance tier of a trip's distances.
    pub fn provenance(&self, trip: TripIdx) -> DistanceProvenance {
        self.provenance[trip.0 as usize]
    }
}

/// Errors raised while assembling a [`LegGeometry`].
#[derive(Debug, PartialEq)]
pub enum LegGeometryError {
    /// A polyline's coordinate and measure arrays disagree, are shorter
    /// than two points, or carry non-finite or decreasing values.
    InvalidPolyline { polyline: u32 },
    /// A trip references a polyline that does not exist.
    PolylineOutOfRange { trip: u32, polyline_count: u32 },
    /// A trip index is not below the timetable's trip count.
    TripOutOfRange { trip: u32, trip_count: u32 },
    /// A trip was given twice.
    DuplicateTrip { trip: u32 },
    /// Not every trip of the timetable was given positions.
    MissingTrips { missing: u32 },
    /// A trip's position count differs from its pattern's stop count,
    /// or its positions are non-finite, decreasing, or outside its
    /// polyline's measure range.
    InvalidPositions { trip: u32 },
}

impl std::fmt::Display for LegGeometryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LegGeometryError::InvalidPolyline { polyline } => {
                write!(f, "polyline {polyline} is malformed")
            }
            LegGeometryError::PolylineOutOfRange {
                trip,
                polyline_count,
            } => write!(
                f,
                "trip index {trip} references a polyline out of range ({polyline_count} polylines)"
            ),
            LegGeometryError::TripOutOfRange { trip, trip_count } => {
                write!(f, "trip index {trip} is out of range ({trip_count} trips)")
            }
            LegGeometryError::DuplicateTrip { trip } => {
                write!(f, "trip index {trip} was given positions twice")
            }
            LegGeometryError::MissingTrips { missing } => {
                write!(f, "{missing} trip(s) were given no positions")
            }
            LegGeometryError::InvalidPositions { trip } => {
                write!(f, "trip index {trip} has invalid stop positions")
            }
        }
    }
}

impl std::error::Error for LegGeometryError {}

/// Leg geometries: per-trip polylines with the stops located along them.
///
/// Polylines are deduplicated (trips of one shape share it) and carry a
/// monotone measure at every vertex; each trip stores its stops'
/// positions in its polyline's measure. A transit leg's geometry is the
/// polyline's slice between the board and alight positions, found by
/// binary search and endpoint interpolation.
#[derive(Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct LegGeometry {
    /// CSR offsets into the coordinate arrays, one per polyline plus a
    /// tail.
    coordinate_offsets: Vec<u32>,
    /// Polyline coordinates, in EPSG:4326.
    xs: Vec<f64>,
    ys: Vec<f64>,
    /// The monotone measure at every polyline vertex.
    measures: Vec<f64>,
    /// Each trip's polyline.
    polyline_of: Vec<u32>,
    /// CSR offsets into `positions`, one per trip plus a tail.
    position_offsets: Vec<u32>,
    /// Stop positions in the polyline's measure, one per pattern stop.
    positions: Vec<f64>,
}

impl LegGeometry {
    /// Builds the structure from deduplicated polylines and per-trip
    /// stop positions. Every trip of the timetable must appear exactly
    /// once, with one position per pattern stop, non-decreasing and
    /// within its polyline's measure range.
    pub fn new(
        timetable: &Timetable,
        polylines: &[(Vec<f64>, Vec<f64>, Vec<f64>)],
        trips: Vec<(TripIdx, u32, Vec<f64>)>,
    ) -> Result<LegGeometry, LegGeometryError> {
        let mut coordinate_offsets = Vec::with_capacity(polylines.len() + 1);
        coordinate_offsets.push(0u32);
        let mut xs = Vec::new();
        let mut ys = Vec::new();
        let mut measures = Vec::new();
        for (index, (lons, lats, line_measures)) in polylines.iter().enumerate() {
            let polyline = index as u32;
            if lons.len() != lats.len()
                || lons.len() != line_measures.len()
                || lons.len() < 2
                || lons.iter().any(|value| !value.is_finite())
                || lats.iter().any(|value| !value.is_finite())
                || line_measures.iter().any(|value| !value.is_finite())
                || line_measures.windows(2).any(|pair| pair[1] < pair[0])
            {
                return Err(LegGeometryError::InvalidPolyline { polyline });
            }
            xs.extend_from_slice(lons);
            ys.extend_from_slice(lats);
            measures.extend_from_slice(line_measures);
            coordinate_offsets.push(xs.len() as u32);
        }

        let trip_count = timetable.trip_count();
        let mut per_trip: Vec<Option<(u32, Vec<f64>)>> = (0..trip_count).map(|_| None).collect();
        for (trip, polyline, positions) in trips {
            if trip.0 >= trip_count {
                return Err(LegGeometryError::TripOutOfRange {
                    trip: trip.0,
                    trip_count,
                });
            }
            if polyline as usize >= polylines.len() {
                return Err(LegGeometryError::PolylineOutOfRange {
                    trip: trip.0,
                    polyline_count: polylines.len() as u32,
                });
            }
            let stops = timetable.pattern_stops(timetable.trip_pattern(trip)).len();
            let line_measures = &polylines[polyline as usize].2;
            let first = *line_measures.first().expect("validated above");
            let last = *line_measures.last().expect("validated above");
            if positions.len() != stops
                || positions.iter().any(|value| !value.is_finite())
                || positions.windows(2).any(|pair| pair[1] < pair[0])
                || positions
                    .iter()
                    .any(|&value| value < first - 1e-6 || value > last + 1e-6)
            {
                return Err(LegGeometryError::InvalidPositions { trip: trip.0 });
            }
            let slot = &mut per_trip[trip.0 as usize];
            if slot.is_some() {
                return Err(LegGeometryError::DuplicateTrip { trip: trip.0 });
            }
            *slot = Some((polyline, positions));
        }
        let missing = per_trip.iter().filter(|slot| slot.is_none()).count() as u32;
        if missing > 0 {
            return Err(LegGeometryError::MissingTrips { missing });
        }

        let mut polyline_of = Vec::with_capacity(trip_count as usize);
        let mut position_offsets = Vec::with_capacity(trip_count as usize + 1);
        position_offsets.push(0u32);
        let mut positions = Vec::new();
        for slot in per_trip {
            let (polyline, trip_positions) = slot.expect("checked for missing trips");
            polyline_of.push(polyline);
            positions.extend_from_slice(&trip_positions);
            position_offsets.push(positions.len() as u32);
        }
        Ok(LegGeometry {
            coordinate_offsets,
            xs,
            ys,
            measures,
            polyline_of,
            position_offsets,
            positions,
        })
    }

    /// The coordinates travelled on `trip` between two stop positions:
    /// the polyline slice between the stops' measures, endpoints
    /// interpolated. Always at least two points; a zero-length slice
    /// repeats the point.
    pub fn leg_coordinates(
        &self,
        trip: TripIdx,
        board_position: u16,
        alight_position: u16,
    ) -> Vec<(f64, f64)> {
        let base = self.position_offsets[trip.0 as usize] as usize;
        let from = self.positions[base + board_position as usize];
        let to = self.positions[base + alight_position as usize];
        let polyline = self.polyline_of[trip.0 as usize] as usize;
        let start = self.coordinate_offsets[polyline] as usize;
        let end = self.coordinate_offsets[polyline + 1] as usize;
        let measures = &self.measures[start..end];

        let mut coordinates = Vec::new();
        coordinates.push(self.interpolate(start, measures, from));
        let after = measures.partition_point(|&measure| measure <= from);
        let until = measures.partition_point(|&measure| measure < to);
        for vertex in after..until {
            coordinates.push((self.xs[start + vertex], self.ys[start + vertex]));
        }
        coordinates.push(self.interpolate(start, measures, to));
        coordinates
    }

    /// The point at `measure` along a polyline, by linear interpolation
    /// between the surrounding vertices.
    fn interpolate(&self, start: usize, measures: &[f64], measure: f64) -> (f64, f64) {
        let upper = measures
            .partition_point(|&at| at < measure)
            .clamp(1, measures.len() - 1);
        let lower = upper - 1;
        let span = measures[upper] - measures[lower];
        let along = if span > 0.0 {
            ((measure - measures[lower]) / span).clamp(0.0, 1.0)
        } else {
            0.0
        };
        (
            self.xs[start + lower] + along * (self.xs[start + upper] - self.xs[start + lower]),
            self.ys[start + lower] + along * (self.ys[start + upper] - self.ys[start + lower]),
        )
    }
}

/// Encodes coordinates as a little-endian WKB LineString (XY).
pub fn wkb_line_string(coordinates: &[(f64, f64)]) -> Vec<u8> {
    let mut wkb = Vec::with_capacity(9 + coordinates.len() * 16);
    wkb.push(1u8);
    wkb.extend_from_slice(&2u32.to_le_bytes());
    wkb.extend_from_slice(&(coordinates.len() as u32).to_le_bytes());
    for &(x, y) in coordinates {
        wkb.extend_from_slice(&x.to_le_bytes());
        wkb.extend_from_slice(&y.to_le_bytes());
    }
    wkb
}

/// Encodes line parts as a little-endian WKB MultiLineString (XY).
pub fn wkb_multi_line_string(parts: &[Vec<(f64, f64)>]) -> Vec<u8> {
    let coordinates: usize = parts.iter().map(Vec::len).sum();
    let mut wkb = Vec::with_capacity(9 + parts.len() * 9 + coordinates * 16);
    wkb.push(1u8);
    wkb.extend_from_slice(&5u32.to_le_bytes());
    wkb.extend_from_slice(&(parts.len() as u32).to_le_bytes());
    for part in parts {
        wkb.extend_from_slice(&wkb_line_string(part));
    }
    wkb
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::timetable::{StopIdx, StopTime, TimetableBuilder};

    fn timetable() -> Timetable {
        let mut builder = TimetableBuilder::new(3);
        let pattern = builder
            .add_pattern(&[StopIdx(0), StopIdx(1), StopIdx(2)], 0)
            .unwrap();
        for departure in [0, 100] {
            builder
                .add_trip(
                    pattern,
                    vec![
                        StopTime {
                            arrival: departure,
                            departure,
                        },
                        StopTime {
                            arrival: departure + 60,
                            departure: departure + 60,
                        },
                        StopTime {
                            arrival: departure + 120,
                            departure: departure + 120,
                        },
                    ],
                    0,
                    0,
                )
                .unwrap();
        }
        builder.finish()
    }

    #[test]
    fn stores_leg_distances_and_provenance() {
        let timetable = timetable();
        let geometry = TripGeometry::from_trips(
            &timetable,
            vec![
                (
                    TripIdx(1),
                    vec![100.0, 700.0, 1600.0],
                    DistanceProvenance::CrowFly,
                ),
                (
                    TripIdx(0),
                    vec![0.0, 500.0, 1200.0],
                    DistanceProvenance::ShapeDist,
                ),
            ],
        )
        .unwrap();
        assert_eq!(geometry.leg_distance(TripIdx(0), 0, 2), 1200.0);
        assert_eq!(geometry.leg_distance(TripIdx(1), 1, 2), 900.0);
        assert_eq!(
            geometry.provenance(TripIdx(0)),
            DistanceProvenance::ShapeDist
        );
        assert_eq!(geometry.provenance(TripIdx(1)), DistanceProvenance::CrowFly);
    }

    #[test]
    fn slices_leg_geometry_between_stops() {
        let timetable = timetable();
        // One straight polyline shared by both trips; stops at measures
        // 0, 50, and 200 — the middle stop between vertices.
        let polylines = vec![(
            vec![24.0, 24.1, 24.2],
            vec![60.0, 60.0, 60.0],
            vec![0.0, 100.0, 200.0],
        )];
        let geometry = LegGeometry::new(
            &timetable,
            &polylines,
            vec![
                (TripIdx(0), 0, vec![0.0, 50.0, 200.0]),
                (TripIdx(1), 0, vec![0.0, 50.0, 200.0]),
            ],
        )
        .unwrap();
        assert_eq!(
            geometry.leg_coordinates(TripIdx(0), 0, 1),
            vec![(24.0, 60.0), (24.05, 60.0)]
        );
        assert_eq!(
            geometry.leg_coordinates(TripIdx(0), 1, 2),
            vec![(24.05, 60.0), (24.1, 60.0), (24.2, 60.0)]
        );
        assert_eq!(
            geometry.leg_coordinates(TripIdx(1), 0, 2),
            vec![(24.0, 60.0), (24.1, 60.0), (24.2, 60.0)]
        );
        // A zero-length slice still yields a drawable two-point line.
        assert_eq!(
            geometry.leg_coordinates(TripIdx(0), 0, 0),
            vec![(24.0, 60.0), (24.0, 60.0)]
        );
    }

    #[test]
    fn rejects_inconsistent_leg_geometry() {
        let timetable = timetable();
        let line = (vec![24.0, 24.2], vec![60.0, 60.0], vec![0.0, 200.0]);
        let full = |trip| (trip, 0u32, vec![0.0, 50.0, 200.0]);
        assert_eq!(
            LegGeometry::new(
                &timetable,
                &[(vec![24.0], vec![60.0], vec![0.0])],
                vec![full(TripIdx(0)), full(TripIdx(1))],
            ),
            Err(LegGeometryError::InvalidPolyline { polyline: 0 })
        );
        assert_eq!(
            LegGeometry::new(
                &timetable,
                &[(vec![24.0, 24.2], vec![60.0, 60.0], vec![200.0, 0.0])],
                vec![full(TripIdx(0)), full(TripIdx(1))],
            ),
            Err(LegGeometryError::InvalidPolyline { polyline: 0 })
        );
        assert_eq!(
            LegGeometry::new(
                &timetable,
                std::slice::from_ref(&line),
                vec![full(TripIdx(0))]
            ),
            Err(LegGeometryError::MissingTrips { missing: 1 })
        );
        assert_eq!(
            LegGeometry::new(
                &timetable,
                std::slice::from_ref(&line),
                vec![full(TripIdx(0)), full(TripIdx(0))],
            ),
            Err(LegGeometryError::DuplicateTrip { trip: 0 })
        );
        assert_eq!(
            LegGeometry::new(
                &timetable,
                std::slice::from_ref(&line),
                vec![full(TripIdx(0)), (TripIdx(1), 1, vec![0.0, 50.0, 200.0])],
            ),
            Err(LegGeometryError::PolylineOutOfRange {
                trip: 1,
                polyline_count: 1
            })
        );
        assert_eq!(
            LegGeometry::new(
                &timetable,
                std::slice::from_ref(&line),
                vec![full(TripIdx(0)), (TripIdx(1), 0, vec![0.0, 200.0])],
            ),
            Err(LegGeometryError::InvalidPositions { trip: 1 })
        );
        assert_eq!(
            LegGeometry::new(
                &timetable,
                std::slice::from_ref(&line),
                vec![full(TripIdx(0)), (TripIdx(1), 0, vec![0.0, 50.0, 500.0])],
            ),
            Err(LegGeometryError::InvalidPositions { trip: 1 })
        );
        assert_eq!(
            LegGeometry::new(
                &timetable,
                std::slice::from_ref(&line),
                vec![full(TripIdx(0)), (TripIdx(1), 0, vec![50.0, 0.0, 200.0])],
            ),
            Err(LegGeometryError::InvalidPositions { trip: 1 })
        );
    }

    #[test]
    fn rejects_inconsistent_input() {
        let timetable = timetable();
        let full = |trip| (trip, vec![0.0, 1.0, 2.0], DistanceProvenance::CrowFly);
        assert_eq!(
            TripGeometry::from_trips(&timetable, vec![full(TripIdx(0))]),
            Err(GeometryError::MissingTrips { missing: 1 })
        );
        assert_eq!(
            TripGeometry::from_trips(&timetable, vec![full(TripIdx(0)), full(TripIdx(0))]),
            Err(GeometryError::DuplicateTrip { trip: 0 })
        );
        assert_eq!(
            TripGeometry::from_trips(&timetable, vec![full(TripIdx(2))]),
            Err(GeometryError::TripOutOfRange {
                trip: 2,
                trip_count: 2
            })
        );
        assert_eq!(
            TripGeometry::from_trips(
                &timetable,
                vec![
                    full(TripIdx(0)),
                    (TripIdx(1), vec![0.0, 1.0], DistanceProvenance::CrowFly)
                ]
            ),
            Err(GeometryError::LengthMismatch {
                trip: 1,
                distances: 2,
                stops: 3
            })
        );
        assert_eq!(
            TripGeometry::from_trips(
                &timetable,
                vec![
                    full(TripIdx(0)),
                    (TripIdx(1), vec![0.0, 2.0, 1.0], DistanceProvenance::CrowFly)
                ]
            ),
            Err(GeometryError::InvalidDistances {
                trip: 1,
                position: 2
            })
        );
        assert_eq!(
            TripGeometry::from_trips(
                &timetable,
                vec![
                    full(TripIdx(0)),
                    (
                        TripIdx(1),
                        vec![f32::NAN, 1.0, 2.0],
                        DistanceProvenance::CrowFly
                    )
                ]
            ),
            Err(GeometryError::InvalidDistances {
                trip: 1,
                position: 0
            })
        );
    }
}
