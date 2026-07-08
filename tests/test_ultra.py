"""ULTRA shortcut computation and its use by the time engines."""

import pytest

from cafein import TransportNetwork

QUERY_DATE, QUERY_TIME = "2022-02-22", "08:30:00"
# The compute/inspect tests use the full feed with a bounded window (fast, no
# routing). The routing tests use a centrally-cropped feed, because routing
# only relaxes a whole-day set and a whole-day compute over the region-wide
# feed's 195k trips takes minutes; cropped to the centre it takes seconds.
CUTOFF = 300.0
WINDOW = {"min_departure": 28800, "max_departure": 29400}  # 08:00–08:10
CENTRAL_BBOX = (60.14, 24.88, 60.20, 25.00)  # (min_lat, min_lon, max_lat, max_lon)
# A short access/egress radius makes journeys rely on transit and the
# intermediate transfers ULTRA widens, rather than walking the ends.
ACCESS = 200.0


@pytest.fixture(scope="module")
def ultra_network(helsinki_gtfs, kantakaupunki_pbf):
    """A Helsinki network with a bounded-window ULTRA set computed."""
    pytest.importorskip("cafein._cafein")
    with pytest.warns(UserWarning):
        network = TransportNetwork.from_gtfs(
            [str(helsinki_gtfs)], osm_pbf=str(kantakaupunki_pbf)
        )
    network.compute_ultra_shortcuts(max_transfer_time=CUTOFF, **WINDOW)
    return network


@pytest.fixture(scope="module")
def central_gtfs(helsinki_gtfs, tmp_path_factory):
    """`helsinki_gtfs` cropped to central Helsinki — only the stops in the
    bbox and the trips lying entirely within them — so a whole-day ULTRA
    compute (the only kind routing relaxes) is fast."""
    import zipfile

    import pandas as pd

    with zipfile.ZipFile(helsinki_gtfs) as archive:
        tables = {
            name[:-4]: pd.read_csv(archive.open(name), dtype=str)
            for name in archive.namelist()
            if name.endswith(".txt")
        }
    stops = tables["stops"]
    lat, lon = stops["stop_lat"].astype(float), stops["stop_lon"].astype(float)
    lo_lat, lo_lon, hi_lat, hi_lon = CENTRAL_BBOX
    inside = (lat >= lo_lat) & (lat <= hi_lat) & (lon >= lo_lon) & (lon <= hi_lon)
    kept = set(stops.loc[inside, "stop_id"])
    times = tables["stop_times"]
    trip_stops = times.groupby("trip_id")["stop_id"].apply(set)
    trips = set(trip_stops[trip_stops.apply(lambda s: s <= kept)].index)
    tables["stop_times"] = times[times["trip_id"].isin(trips)]
    tables["trips"] = tables["trips"][tables["trips"]["trip_id"].isin(trips)]
    on_trips = set(tables["stop_times"]["stop_id"])
    tables["stops"] = stops[stops["stop_id"].isin(on_trips)].copy()
    if "parent_station" in tables["stops"].columns:
        tables["stops"]["parent_station"] = ""  # drop now-dangling references
    tables["routes"] = tables["routes"][
        tables["routes"]["route_id"].isin(set(tables["trips"]["route_id"]))
    ]
    services = set(tables["trips"]["service_id"])
    for calendar in ("calendar", "calendar_dates"):
        if calendar in tables:
            tables[calendar] = tables[calendar][
                tables[calendar]["service_id"].isin(services)
            ]
    if "transfers" in tables:
        transfers = tables["transfers"]
        tables["transfers"] = transfers[
            transfers["from_stop_id"].isin(on_trips)
            & transfers["to_stop_id"].isin(on_trips)
        ]
    path = tmp_path_factory.mktemp("central") / "central.zip"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, frame in tables.items():
            archive.writestr(name + ".txt", frame.to_csv(index=False))
    return path


def _pareto(journeys):
    return sorted({(j["arrival"], j["rides"]) for j in journeys})


def _dominated(point, pareto):
    arrival, rides = point
    return any(a <= arrival and r <= rides for a, r in pareto)


def _door_to_door(net, coords, endpoints, time):
    """Door-to-door Pareto sets for the endpoint pairs that route."""
    out = {}
    for origin in endpoints:
        for destination in endpoints:
            if origin == destination:
                continue
            try:
                out[(origin, destination)] = _pareto(
                    net.route_between_coordinates(
                        coords[origin],
                        coords[destination],
                        QUERY_DATE,
                        time,
                        max_walking_time=ACCESS,
                    )
                )
            except ValueError:
                pass
    return out


def test_compute_returns_and_stores_a_shortcut_count(ultra_network):
    assert ultra_network.ultra_shortcut_count is not None
    assert ultra_network.ultra_shortcut_count > 0
    shortcuts = ultra_network.ultra_shortcuts
    assert len(shortcuts) == ultra_network.ultra_shortcut_count
    # Shortcuts are (origin, destination, seconds, meters) between distinct
    # stops, carrying the walked distance.
    origin, destination, seconds, meters = shortcuts[0]
    assert origin != destination
    assert isinstance(seconds, int)
    assert meters >= 0.0


def test_point_destination_routing(central_gtfs, kantakaupunki_pbf):
    # On the cropped feed a whole-day compute is fast. With a closure radius
    # (60 s) below the ULTRA walk cutoff (600 s), a whole-day set gives
    # door-to-door queries every closure journey (superset) plus journeys the
    # radius misses (unrestricted walking); an ULTRA-routed transfer leg
    # reports its distance. A partial-window set and stop-to-stop queries keep
    # the closure.
    with pytest.warns(UserWarning):
        net = TransportNetwork.from_gtfs(
            [str(central_gtfs)], osm_pbf=str(kantakaupunki_pbf), max_walking_time=60
        )
    coords = {s: (lat, lon) for s, lat, lon in net.stops if lat is not None}
    endpoints = list(coords)[:15]
    closure = _door_to_door(net, coords, endpoints, QUERY_TIME)
    stop_before = net.travel_times_from_stop(endpoints[0], QUERY_DATE, QUERY_TIME)
    assert any(pareto for pareto in closure.values())

    # A partial-window set is stored but not relaxed by routing.
    net.compute_ultra_shortcuts(
        max_transfer_time=600.0, min_departure=29700, max_departure=31500
    )
    assert _door_to_door(net, coords, endpoints, QUERY_TIME) == closure

    # A whole-day set is: superset, at least one gained journey, and transfer
    # legs report a distance looked up in the ULTRA set (which carries metres).
    net.compute_ultra_shortcuts(max_transfer_time=600.0)  # whole day
    ultra = _door_to_door(net, coords, endpoints, QUERY_TIME)
    improved = 0
    for pair, before in closure.items():
        for point in before:
            assert _dominated(point, ultra[pair])
        if ultra[pair] != before:
            improved += 1
    assert improved >= 1

    assert (
        net.travel_times_from_stop(endpoints[0], QUERY_DATE, QUERY_TIME) == stop_before
    )

    saw_transfer = False
    for origin, destination in closure:
        for journey in net.route_between_coordinates(
            coords[origin],
            coords[destination],
            QUERY_DATE,
            QUERY_TIME,
            max_walking_time=ACCESS,
        ):
            for leg in journey["legs"]:
                if leg["type"] == "transfer":
                    saw_transfer = True
                    assert leg["distance"] is not None
    assert saw_transfer


def test_point_matrix_supersets_and_emissions_ignore(central_gtfs, kantakaupunki_pbf):
    # The point-set travel-time matrix is a point-destination query, so a
    # whole-day ULTRA set never makes a cell slower; the emissions frontier
    # (McRAPTOR) keeps the closure, so computing ULTRA leaves it unchanged.
    import geopandas as gpd
    from shapely.geometry import Point

    from cafein import TravelTimeMatrix
    from cafein.frontier import journey_frontier

    with pytest.warns(UserWarning):
        net = TransportNetwork.from_gtfs(
            [str(central_gtfs)], osm_pbf=str(kantakaupunki_pbf), max_walking_time=60
        )
    coords = {s: (lat, lon) for s, lat, lon in net.stops if lat is not None}
    endpoints = list(coords)[:12]
    points = gpd.GeoDataFrame(
        {"id": endpoints},
        geometry=[Point(coords[s][1], coords[s][0]) for s in endpoints],
        crs="EPSG:4326",
    )
    # A stop pair the emissions frontier can route.
    frontier_pair = next(
        (
            (o, d)
            for o in endpoints
            for d in endpoints
            if o != d and net.route_between_stops(o, d, QUERY_DATE, QUERY_TIME)
        ),
        None,
    )
    assert frontier_pair is not None

    def matrix():
        frame = TravelTimeMatrix(
            net, points, points, date=QUERY_DATE, departure=QUERY_TIME
        )
        return {
            (row["from_id"], row["to_id"]): row["travel_time"]
            for _, row in frame.iterrows()
            if row["travel_time"] == row["travel_time"]  # drop NaN (unreachable)
        }

    def frontier():
        frame = journey_frontier(
            net, *frontier_pair, QUERY_DATE, QUERY_TIME, 1800, candidates="pareto"
        )
        rows = frame[["travel_time", "emissions", "rides", "frontier"]]
        return sorted(rows.round(3).itertuples(index=False, name=None))

    closure_matrix = matrix()
    frontier_before = frontier()
    assert closure_matrix and frontier_before

    net.compute_ultra_shortcuts(max_transfer_time=CUTOFF)  # whole day

    ultra_matrix = matrix()
    for cell, seconds in closure_matrix.items():
        assert cell in ultra_matrix and ultra_matrix[cell] <= seconds
    assert frontier() == frontier_before


def test_compute_is_deterministic(helsinki_gtfs, kantakaupunki_pbf, ultra_network):
    with pytest.warns(UserWarning):
        again = TransportNetwork.from_gtfs(
            [str(helsinki_gtfs)], osm_pbf=str(kantakaupunki_pbf)
        )
    again.compute_ultra_shortcuts(max_transfer_time=CUTOFF, **WINDOW)
    # The whole set — not merely its size — is reproduced, as the plan's
    # thread-independent determinism requires.
    assert again.ultra_shortcuts == ultra_network.ultra_shortcuts


def test_a_new_street_network_clears_the_shortcuts(helsinki_gtfs, kantakaupunki_pbf):
    # Rebuilding the street network invalidates shortcuts derived from the
    # old one, so recomputing is required rather than silently reusing them.
    from cafein import streets

    with pytest.warns(UserWarning):
        net = TransportNetwork.from_gtfs(
            [str(helsinki_gtfs)], osm_pbf=str(kantakaupunki_pbf)
        )
    net.compute_ultra_shortcuts(max_transfer_time=CUTOFF, **WINDOW)
    assert net.ultra_shortcut_count is not None

    _, street_network = streets.walking_streets(str(kantakaupunki_pbf), net.stops)
    net._core.set_street_network(*street_network)
    assert net.ultra_shortcut_count is None
    assert net.ultra_shortcuts is None


def test_count_is_none_before_computation(network):
    assert network.ultra_shortcut_count is None
    assert network.ultra_shortcuts is None


def test_compute_without_a_street_network_errors(network):
    with pytest.raises(ValueError, match="street network"):
        network.compute_ultra_shortcuts()
