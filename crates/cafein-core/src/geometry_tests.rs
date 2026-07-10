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
