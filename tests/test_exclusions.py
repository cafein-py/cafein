"""Query-time exclusion sets: disruption and accessibility filters."""

import pytest

from test_frontier import build_two_line_gtfs

DATE, DEPARTURE = "2022-02-22", "08:00:00"


@pytest.fixture()
def two_line_network(tmp_path):
    from cafein import TransportNetwork

    feed = build_two_line_gtfs(tmp_path / "two_line_gtfs.zip")
    return TransportNetwork.from_gtfs([str(feed)])


def used(journeys, key):
    return {
        leg[key]
        for journey in journeys
        for leg in journey["legs"]
        if leg["type"] == "transit"
    }


def test_excluding_used_supply_reroutes(two_line_network):
    baseline = two_line_network.route_between_stops("A", "B", DATE, DEPARTURE)
    assert baseline
    routes = used(baseline, "route_id")
    assert routes
    # Excluding every ridden route removes those journeys; whatever is
    # left (possibly nothing) rides none of them.
    rerouted = two_line_network.route_between_stops(
        "A", "B", DATE, DEPARTURE, exclude_routes=sorted(routes)
    )
    assert not (used(rerouted, "route_id") & routes)
    # Excluding the ridden trips is finer: those departures vanish,
    # while later trips of the same routes still run.
    trips = used(baseline, "trip_id")
    by_trip = two_line_network.route_between_stops(
        "A", "B", DATE, DEPARTURE, exclude_trips=sorted(trips)
    )
    assert by_trip
    assert not (used(by_trip, "trip_id") & trips)
    assert used(by_trip, "route_id") & routes


def test_excluded_origin_or_destination_is_unreachable(two_line_network):
    assert (
        two_line_network.route_between_stops(
            "A", "B", DATE, DEPARTURE, exclude_stops=["A"]
        )
        == []
    )
    assert (
        two_line_network.route_between_stops(
            "A", "B", DATE, DEPARTURE, exclude_stops=["B"]
        )
        == []
    )


def test_unknown_route_and_trip_ids_are_ignored(two_line_network):
    baseline = two_line_network.route_between_stops("A", "B", DATE, DEPARTURE)
    same = two_line_network.route_between_stops(
        "A",
        "B",
        DATE,
        DEPARTURE,
        exclude_routes=["no-such-route"],
        exclude_trips=["no-such-trip"],
    )
    assert same == baseline
    # Stop ids must resolve, as everywhere else.
    with pytest.raises(KeyError):
        two_line_network.route_between_stops(
            "A", "B", DATE, DEPARTURE, exclude_stops=["no-such-stop"]
        )


def test_frontier_exclusions_force_the_raptor_family(
    two_line_network, capfd, monkeypatch
):
    from cafein import journey_frontier

    args = (two_line_network, "A", "B", DATE, DEPARTURE, 1800)
    kwargs = dict(candidates="pareto")
    baseline = journey_frontier(*args, **kwargs)
    assert len(baseline) > 0
    legs = two_line_network.route_between_stops("A", "B", DATE, DEPARTURE)
    routes = sorted(used(legs, "route_id"))
    assert routes
    with pytest.raises(ValueError, match="exclusions require router='raptor'"):
        journey_frontier(*args, router="tbtr", exclude_routes=routes, **kwargs)
    # Auto falls back to McRAPTOR even with a matching cached set: the
    # stats flag is the dispatch observable.
    two_line_network.compute_mctbtr_transfers(DATE)
    monkeypatch.setenv("CAFEIN_MCTBTR_PROF", "1")
    capfd.readouterr()
    auto = journey_frontier(*args, exclude_routes=routes, **kwargs)
    assert "MCTBTR-STATS" not in capfd.readouterr().err
    raptor = journey_frontier(*args, router="raptor", exclude_routes=routes, **kwargs)
    assert auto.equals(raptor)


def test_exclusions_compose_with_diverse_alternatives(two_line_network):
    from cafein import DetailedItineraries

    baseline = DetailedItineraries(
        two_line_network,
        ["A"],
        ["B"],
        DATE,
        DEPARTURE,
        candidates="diverse",
        max_options=2,
        geometries=False,
    )
    ridden = set(baseline["route_id"].dropna())
    assert len(ridden) > 1
    excluded = sorted(ridden)[0]
    # A hard exclusion on top of the diverse penalization loop: every
    # round honours it, and the surviving alternatives ride the rest.
    composed = DetailedItineraries(
        two_line_network,
        ["A"],
        ["B"],
        DATE,
        DEPARTURE,
        candidates="diverse",
        max_options=2,
        geometries=False,
        exclude_routes=[excluded],
    )
    assert len(composed) > 0
    assert excluded not in set(composed["route_id"].dropna())


def filtered_feed(source, target, drop_route):
    """The two-line feed with one route's supply genuinely removed."""
    import csv
    import io
    import zipfile

    with zipfile.ZipFile(source) as archive:
        tables = {name: archive.read(name).decode() for name in archive.namelist()}

    def rows(name):
        return list(csv.DictReader(io.StringIO(tables[name])))

    dropped_trips = {
        row["trip_id"] for row in rows("trips.txt") if row["route_id"] == drop_route
    }

    def write(name, keep):
        kept = [row for row in rows(name) if keep(row)]
        out = io.StringIO()
        writer = csv.DictWriter(out, fieldnames=kept[0].keys())
        writer.writeheader()
        writer.writerows(kept)
        tables[name] = out.getvalue()

    write("routes.txt", lambda row: row["route_id"] != drop_route)
    write("trips.txt", lambda row: row["route_id"] != drop_route)
    write("stop_times.txt", lambda row: row["trip_id"] not in dropped_trips)
    with zipfile.ZipFile(target, "w") as archive:
        for name, text in tables.items():
            archive.writestr(name, text)
    return target


def test_exclusions_match_a_rebuilt_feed(two_line_network, tmp_path):
    from cafein import TransportNetwork

    baseline = two_line_network.route_between_stops("A", "B", DATE, DEPARTURE)
    excluded = sorted(used(baseline, "route_id"))[0]

    def normalized(journeys):
        return [
            (
                journey["arrival"],
                [leg["trip_id"] for leg in journey["legs"] if leg["type"] == "transit"],
            )
            for journey in journeys
        ]

    with_exclusions = two_line_network.route_between_stops(
        "A", "B", DATE, DEPARTURE, exclude_routes=[excluded]
    )
    source = build_two_line_gtfs(tmp_path / "full.zip")
    rebuilt = TransportNetwork.from_gtfs(
        [str(filtered_feed(source, tmp_path / "without.zip", excluded))]
    )
    oracle = rebuilt.route_between_stops("A", "B", DATE, DEPARTURE)
    assert normalized(with_exclusions) == normalized(oracle)
    assert with_exclusions


def test_frontier_exclusions_match_a_rebuilt_feed(two_line_network, tmp_path):
    from cafein import TransportNetwork, journey_frontier

    columns = ["departure", "arrival", "travel_time", "rides", "frontier"]
    baseline = two_line_network.route_between_stops("A", "B", DATE, DEPARTURE)
    excluded = sorted(used(baseline, "route_id"))[0]
    source = build_two_line_gtfs(tmp_path / "full.zip")
    rebuilt = TransportNetwork.from_gtfs(
        [str(filtered_feed(source, tmp_path / "without.zip", excluded))]
    )
    for candidates in ("time", "pareto", "relaxed"):
        with_exclusions = journey_frontier(
            two_line_network,
            "A",
            "B",
            DATE,
            DEPARTURE,
            1800,
            candidates=candidates,
            exclude_routes=[excluded],
        )
        oracle = journey_frontier(
            rebuilt, "A", "B", DATE, DEPARTURE, 1800, candidates=candidates
        )
        assert len(with_exclusions) > 0, candidates
        assert with_exclusions[columns].equals(oracle[columns]), candidates


def test_unknown_only_ids_leave_the_router_untouched(two_line_network):
    from cafein import journey_frontier

    # Unknown-only lists resolve to no exclusions: explicit TBTR stays
    # accepted, so a disruption list naming absent supply cannot flip
    # the engine.
    two_line_network.compute_mctbtr_transfers(DATE)
    frame = journey_frontier(
        two_line_network,
        "A",
        "B",
        DATE,
        DEPARTURE,
        1800,
        candidates="pareto",
        router="tbtr",
        exclude_routes=["no-such-route"],
        exclude_trips=["no-such-trip"],
    )
    assert len(frame) > 0


def test_itineraries_take_exclusions(network_with_footpaths):
    from cafein import DetailedItineraries

    baseline = DetailedItineraries(
        network_with_footpaths,
        ["1100602"],
        ["1040280"],
        "2022-02-22",
        "08:30:00",
        geometries=False,
    )
    ridden = set(baseline["route_id"].dropna())
    assert ridden
    rerouted = DetailedItineraries(
        network_with_footpaths,
        ["1100602"],
        ["1040280"],
        "2022-02-22",
        "08:30:00",
        geometries=False,
        exclude_routes=sorted(ridden),
    )
    assert not (set(rerouted["route_id"].dropna()) & ridden)


def test_matrix_exclusions_match_a_rebuilt_feed(two_line_network, tmp_path):
    from cafein import TransportNetwork, TravelTimeMatrix

    legs = two_line_network.route_between_stops("A", "B", DATE, DEPARTURE)
    excluded = sorted(used(legs, "route_id"))[0]
    source = build_two_line_gtfs(tmp_path / "full.zip")
    rebuilt = TransportNetwork.from_gtfs(
        [str(filtered_feed(source, tmp_path / "without.zip", excluded))]
    )
    kwargs = dict(date=DATE, departure=DEPARTURE)
    with_exclusions = TravelTimeMatrix(
        two_line_network, exclude_routes=[excluded], **kwargs
    )
    oracle = TravelTimeMatrix(rebuilt, **kwargs)
    assert len(with_exclusions) > 0
    assert with_exclusions.equals(oracle)
    # The one-to-all query agrees with the rebuilt feed too.
    assert two_line_network.travel_times_from_stop(
        "A", DATE, DEPARTURE, exclude_routes=[excluded]
    ) == rebuilt.travel_times_from_stop("A", DATE, DEPARTURE)


def test_matrix_exclusions_router_contract(two_line_network):
    from cafein import TravelTimeMatrix

    two_line_network.compute_tbtr_transfers(DATE)
    kwargs = dict(date=DATE, departure=DEPARTURE)
    legs = two_line_network.route_between_stops("A", "B", DATE, DEPARTURE)
    excluded = sorted(used(legs, "route_id"))
    with pytest.raises(ValueError, match="exclusions require router='raptor'"):
        TravelTimeMatrix(
            two_line_network, router="tbtr", exclude_routes=excluded, **kwargs
        )
    # Auto falls back to RAPTOR with unchanged rows even over a matching
    # cached set; unknown-only ids leave explicit TBTR accepted.
    auto = TravelTimeMatrix(two_line_network, exclude_routes=excluded, **kwargs)
    raptor = TravelTimeMatrix(
        two_line_network, router="raptor", exclude_routes=excluded, **kwargs
    )
    assert auto.equals(raptor)
    unknown = TravelTimeMatrix(
        two_line_network, router="tbtr", exclude_routes=["no-such-route"], **kwargs
    )
    assert len(unknown) > 0
    # An excluded origin has no rows at all.
    empty = TravelTimeMatrix(two_line_network, ["A"], exclude_stops=["A"], **kwargs)
    assert empty.empty
