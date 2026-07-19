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


def test_batched_frontiers_match_a_rebuilt_feed(two_line_network, tmp_path):
    from cafein import TransportNetwork, frontier_table, journey_frontiers

    legs = two_line_network.route_between_stops("A", "B", DATE, DEPARTURE)
    excluded = sorted(used(legs, "route_id"))[0]
    source = build_two_line_gtfs(tmp_path / "full.zip")
    rebuilt = TransportNetwork.from_gtfs(
        [str(filtered_feed(source, tmp_path / "without.zip", excluded))]
    )
    args = (["A"], ["B"], DATE, DEPARTURE, 1800)
    frame = journey_frontiers(two_line_network, *args, exclude_routes=[excluded])
    oracle = journey_frontiers(rebuilt, *args)
    assert len(frame) > 0
    assert frame.equals(oracle)
    flat = frontier_table(two_line_network, *args, exclude_routes=[excluded])
    flat_oracle = frontier_table(rebuilt, *args)
    assert flat.equals(flat_oracle)
    # The band composes: its bounds count only the surviving supply —
    # on the journey and flat-table forms alike.
    banded = journey_frontiers(
        two_line_network, *args, exclude_routes=[excluded], max_slower=0
    )
    assert banded.equals(journey_frontiers(rebuilt, *args, max_slower=0))
    banded_flat = frontier_table(
        two_line_network, *args, exclude_routes=[excluded], max_slower=0
    )
    assert banded_flat.equals(frontier_table(rebuilt, *args, max_slower=0))


def test_batched_frontier_router_contract(two_line_network, capfd, monkeypatch):
    from cafein import journey_frontiers

    legs = two_line_network.route_between_stops("A", "B", DATE, DEPARTURE)
    excluded = sorted(used(legs, "route_id"))
    args = (two_line_network, ["A"], ["B"], DATE, DEPARTURE, 1800)
    with pytest.raises(ValueError, match="exclusions require router='raptor'"):
        journey_frontiers(*args, router="tbtr", exclude_routes=excluded)
    # Auto falls back even over a matching cached set: the batched path
    # emits its stats flag only when McTBTR actually runs.
    two_line_network.compute_mctbtr_transfers(DATE)
    monkeypatch.setenv("CAFEIN_MCTBTR_PROF", "1")
    capfd.readouterr()
    auto = journey_frontiers(*args, exclude_routes=excluded)
    assert "MCTBTR-STATS" not in capfd.readouterr().err
    raptor = journey_frontiers(*args, router="raptor", exclude_routes=excluded)
    assert auto.equals(raptor)
    # Unknown-only ids stay on the cached engine.
    capfd.readouterr()
    unknown = journey_frontiers(*args, exclude_routes=["no-such-route"])
    assert "MCTBTR-STATS" in capfd.readouterr().err
    assert len(unknown) > 0


def test_point_frontiers_take_exclusions(network_with_footpaths):
    import geopandas as gpd
    from shapely.geometry import Point

    from cafein import frontier_table, journey_frontiers

    origins = gpd.GeoDataFrame(
        {"id": ["origin"]}, geometry=[Point(24.9330, 60.1689)], crs="EPSG:4326"
    )
    destinations = gpd.GeoDataFrame(
        {"id": ["destination"]}, geometry=[Point(24.9505, 60.1690)], crs="EPSG:4326"
    )
    legs = network_with_footpaths.route_between_coordinates(
        (60.1689, 24.9330), (60.1690, 24.9505), "2022-02-22", "08:30:00"
    )
    ridden = sorted(used(legs, "route_id"))
    assert ridden
    args = (
        network_with_footpaths,
        origins,
        destinations,
        "2022-02-22",
        "08:30:00",
        1800,
    )
    for wrapper in (journey_frontiers, frontier_table):
        baseline = wrapper(*args)
        excluded = wrapper(*args, exclude_routes=ridden)
        raptor = wrapper(*args, router="raptor", exclude_routes=ridden)
        # The point form honours the exclusions (auto equals raptor under
        # them) and genuinely loses the excluded corridors.
        assert excluded.equals(raptor)
        assert not excluded.equals(baseline)
        with pytest.raises(ValueError, match="exclusions require router='raptor'"):
            wrapper(*args, router="tbtr", exclude_routes=ridden)


def test_cost_matrices_match_a_rebuilt_feed(two_line_network, tmp_path):
    import pytest as _pytest

    from cafein import TransportNetwork, TravelCostMatrix, travel_cost_table

    legs = two_line_network.route_between_stops("A", "B", DATE, DEPARTURE)
    excluded = sorted(used(legs, "route_id"))[0]
    source = build_two_line_gtfs(tmp_path / "full.zip")
    rebuilt = TransportNetwork.from_gtfs(
        [str(filtered_feed(source, tmp_path / "without.zip", excluded))]
    )
    args = (["A"], None, DATE, DEPARTURE)
    fastest = TravelCostMatrix(two_line_network, *args, exclude_routes=[excluded])
    assert len(fastest) > 0
    assert fastest.equals(TravelCostMatrix(rebuilt, *args))
    windowed = TravelCostMatrix(
        two_line_network,
        *args,
        optimize="emissions",
        window=1800,
        exclude_routes=[excluded],
    )
    assert windowed.equals(
        TravelCostMatrix(rebuilt, *args, optimize="emissions", window=1800)
    )
    _pytest.importorskip("pyarrow")
    flat = travel_cost_table(two_line_network, *args, exclude_routes=[excluded])
    assert flat.equals(travel_cost_table(rebuilt, *args))


def test_cost_matrix_router_contract(two_line_network):
    from cafein import TravelCostMatrix

    two_line_network.compute_tbtr_transfers(DATE)
    two_line_network.compute_mctbtr_transfers(DATE)
    legs = two_line_network.route_between_stops("A", "B", DATE, DEPARTURE)
    excluded = sorted(used(legs, "route_id"))
    args = (two_line_network, ["A"], ["B"], DATE, DEPARTURE)
    for kwargs in (
        {},
        {"optimize": "emissions", "window": 1800},
        {"optimize": "emissions", "window": 1800, "candidates": "pareto"},
    ):
        with pytest.raises(ValueError, match="exclusions require router='raptor'"):
            TravelCostMatrix(*args, router="tbtr", exclude_routes=excluded, **kwargs)
        auto = TravelCostMatrix(*args, exclude_routes=excluded, **kwargs)
        raptor = TravelCostMatrix(
            *args, router="raptor", exclude_routes=excluded, **kwargs
        )
        assert auto.equals(raptor)
    # Unknown-only ids leave the cached engines accepted; an excluded
    # origin has no rows.
    unknown = TravelCostMatrix(*args, router="tbtr", exclude_routes=["no-such-route"])

    assert len(unknown) > 0
    empty = TravelCostMatrix(*args, exclude_stops=["A"])
    assert empty.empty


def test_cost_matrix_unknown_ids_keep_the_cache(two_line_network, capfd, monkeypatch):
    from cafein import TravelCostMatrix

    two_line_network.compute_mctbtr_transfers(DATE)
    monkeypatch.setenv("CAFEIN_MCTBTR_PROF", "1")
    legs = two_line_network.route_between_stops("A", "B", DATE, DEPARTURE)
    excluded = sorted(used(legs, "route_id"))
    kwargs = dict(optimize="emissions", window=1800, candidates="pareto")
    args = (two_line_network, ["A"], ["B"], DATE, DEPARTURE)
    # The cost path's dispatch observable is inverted: the McRAPTOR
    # stats flag prints only when the fallback engine actually runs, so
    # unknown-only ids (which keep the cached McTBTR engine) stay
    # silent and real exclusions report.
    monkeypatch.setenv("CAFEIN_MCRAPTOR_PROF", "1")
    capfd.readouterr()
    TravelCostMatrix(*args, exclude_routes=["no-such-route"], **kwargs)
    assert "MCRAPTOR-STATS" not in capfd.readouterr().err
    capfd.readouterr()
    TravelCostMatrix(*args, exclude_routes=excluded, **kwargs)
    assert "MCRAPTOR-STATS" in capfd.readouterr().err


def test_point_cost_matrices_take_exclusions(network_with_footpaths):
    import geopandas as gpd
    from shapely.geometry import Point

    from cafein import TravelCostMatrix

    # A long pair (Korso-ish to the centre) where transit wins, so the
    # walking alternative cannot mask the exclusions.
    coords = {
        stop: (lat, lon)
        for stop, lat, lon in network_with_footpaths.stops
        if lat is not None
    }
    from_lat, from_lon = coords["1100602"]
    to_lat, to_lon = coords["1040280"]
    origins = gpd.GeoDataFrame(
        {"id": ["origin"]}, geometry=[Point(from_lon, from_lat)], crs="EPSG:4326"
    )
    destinations = gpd.GeoDataFrame(
        {"id": ["destination"]}, geometry=[Point(to_lon, to_lat)], crs="EPSG:4326"
    )
    legs = network_with_footpaths.route_between_coordinates(
        (from_lat, from_lon), (to_lat, to_lon), "2022-02-22", "08:30:00"
    )
    ridden = sorted(used(legs, "route_id"))
    assert ridden
    args = (network_with_footpaths, origins, destinations, "2022-02-22", "08:30:00")
    for kwargs in ({}, {"optimize": "emissions", "window": 1800}):
        with pytest.warns(UserWarning, match="route_type"):
            baseline = TravelCostMatrix(*args, **kwargs)
        with pytest.warns(UserWarning, match="route_type"):
            excluded = TravelCostMatrix(*args, exclude_routes=ridden, **kwargs)
        with pytest.warns(UserWarning, match="route_type"):
            raptor = TravelCostMatrix(
                *args, router="raptor", exclude_routes=ridden, **kwargs
            )
        # The point forms honour the exclusions: auto equals raptor
        # under them, and the fastest form loses its excluded corridor
        # (the emissions pick may already avoid it).
        assert excluded.equals(raptor)
        if not kwargs:
            assert not excluded.equals(baseline)
        with pytest.raises(ValueError, match="exclusions require router='raptor'"):
            TravelCostMatrix(*args, router="tbtr", exclude_routes=ridden, **kwargs)
