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
