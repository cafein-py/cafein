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

    # A partial-window set is stored but not relaxed by routing — door-to-door
    # and the one-to-all stop query alike keep the closure.
    net.compute_ultra_shortcuts(
        max_transfer_time=600.0, min_departure=29700, max_departure=31500
    )
    assert _door_to_door(net, coords, endpoints, QUERY_TIME) == closure
    assert (
        net.travel_times_from_stop(endpoints[0], QUERY_DATE, QUERY_TIME) == stop_before
    )

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


def test_save_load_round_trips_the_ultra_set(central_gtfs, kantakaupunki_pbf, tmp_path):
    # The shortcut set and its compute window survive save/load: a whole-day
    # set stays routed by point-destination queries, and a partial-window set
    # is restored but stays unused (its scope is not mistaken for whole-day).
    with pytest.warns(UserWarning):
        net = TransportNetwork.from_gtfs(
            [str(central_gtfs)], osm_pbf=str(kantakaupunki_pbf), max_walking_time=60
        )
    coords = {s: (lat, lon) for s, lat, lon in net.stops if lat is not None}
    endpoints = list(coords)[:15]
    closure = _door_to_door(net, coords, endpoints, QUERY_TIME)

    # A whole-day set: shortcuts and routing behaviour reproduce after a
    # round-trip, and the routed result genuinely differs from the closure so
    # the equality below is not trivially satisfied.
    net.compute_ultra_shortcuts(max_transfer_time=600.0)
    shortcuts = net.ultra_shortcuts
    routed = _door_to_door(net, coords, endpoints, QUERY_TIME)
    assert routed != closure
    stop_before = net.travel_times_from_stop(endpoints[0], QUERY_DATE, QUERY_TIME)
    whole = tmp_path / "whole.cafein"
    net.save(whole)
    loaded = TransportNetwork.load(whole)
    assert loaded.ultra_shortcuts == shortcuts
    assert _door_to_door(loaded, coords, endpoints, QUERY_TIME) == routed
    assert (
        loaded.travel_times_from_stop(endpoints[0], QUERY_DATE, QUERY_TIME)
        == stop_before
    )

    # A partial-window set: restored and inspectable, but routing keeps the
    # closure — the persisted window is a partial one, not whole-day.
    net.compute_ultra_shortcuts(
        max_transfer_time=600.0, min_departure=29700, max_departure=31500
    )
    partial_shortcuts = net.ultra_shortcuts
    partial = tmp_path / "partial.cafein"
    net.save(partial)
    loaded_partial = TransportNetwork.load(partial)
    assert loaded_partial.ultra_shortcut_count == len(partial_shortcuts)
    assert loaded_partial.ultra_shortcuts == partial_shortcuts
    assert _door_to_door(loaded_partial, coords, endpoints, QUERY_TIME) == closure


def test_save_load_without_an_ultra_set(network, tmp_path):
    # A network that never computed shortcuts round-trips to no set.
    path = tmp_path / "no_ultra.cafein"
    network.save(path)
    loaded = TransportNetwork.load(path)
    assert loaded.ultra_shortcut_count is None
    assert loaded.ultra_shortcuts is None


def test_from_gtfs_ultra_default_off(network_with_footpaths):
    # The flag is opt-in: a plain OSM build computes no shortcuts.
    assert network_with_footpaths.ultra_shortcut_count is None


def test_from_gtfs_ultra_requires_an_osm_extract(helsinki_gtfs):
    # ultra=True has no meaning without a street network to walk.
    with pytest.raises(ValueError, match="OSM extract"):
        TransportNetwork.from_gtfs([str(helsinki_gtfs)], ultra=True)


def test_from_gtfs_ultra_computes_the_whole_day_set(central_gtfs, kantakaupunki_pbf):
    # ultra=True computes the whole-day set at build time, identically to
    # building without the flag and calling compute_ultra_shortcuts().
    with pytest.warns(UserWarning):
        built = TransportNetwork.from_gtfs(
            [str(central_gtfs)], osm_pbf=str(kantakaupunki_pbf), ultra=True
        )
    with pytest.warns(UserWarning):
        manual = TransportNetwork.from_gtfs(
            [str(central_gtfs)], osm_pbf=str(kantakaupunki_pbf)
        )
    assert manual.ultra_shortcut_count is None
    manual.compute_ultra_shortcuts()
    assert built.ultra_shortcut_count and built.ultra_shortcut_count > 0
    assert built.ultra_shortcuts == manual.ultra_shortcuts


def test_route_between_stops_door_to_door_under_ultra(central_gtfs, kantakaupunki_pbf):
    # Under a whole-day ULTRA set, route_between_stops routes door-to-door
    # between the stops' coordinates (unrestricted initial/intermediate/final
    # walking) — equal to route_between_coordinates on those coordinates — and
    # finds journeys the closure board-at-origin path misses. Without the set
    # it keeps the closure and ignores the walking arguments.
    with pytest.warns(UserWarning):
        net = TransportNetwork.from_gtfs(
            [str(central_gtfs)], osm_pbf=str(kantakaupunki_pbf), max_walking_time=60
        )
    coords = {s: (lat, lon) for s, lat, lon in net.stops if lat is not None}
    endpoints = list(coords)[:15]

    def stops_pareto(o, d, **kw):
        return _pareto(net.route_between_stops(o, d, QUERY_DATE, QUERY_TIME, **kw))

    a, b = endpoints[0], endpoints[1]
    assert stops_pareto(a, b, max_walking_time=ACCESS) == stops_pareto(a, b)
    closure = {
        (o, d): stops_pareto(o, d) for o in endpoints for d in endpoints if o != d
    }

    net.compute_ultra_shortcuts(max_transfer_time=600.0)  # whole day

    improved = 0
    for (o, d), before in closure.items():
        via_stops = stops_pareto(o, d, max_walking_time=ACCESS)
        via_coords = _pareto(
            net.route_between_coordinates(
                coords[o], coords[d], QUERY_DATE, QUERY_TIME, max_walking_time=ACCESS
            )
        )
        assert via_stops == via_coords
        if via_stops != before:
            improved += 1
    assert improved >= 1

    # The windowed (range) query is door-to-door too.
    ranged = _pareto(
        net.route_between_stops(
            a, b, QUERY_DATE, QUERY_TIME, window=1800, max_walking_time=ACCESS
        )
    )
    assert ranged == _pareto(
        net.route_between_coordinates(
            coords[a],
            coords[b],
            QUERY_DATE,
            QUERY_TIME,
            window=1800,
            max_walking_time=ACCESS,
        )
    )


def test_travel_times_from_stop_door_to_door_under_ultra(
    central_gtfs, kantakaupunki_pbf
):
    # Under a whole-day ULTRA set, travel_times_from_stop routes door-to-door
    # from the origin stop's coordinate: it equals travel_times_from_coordinate
    # there (unrestricted initial/intermediate/final walking), reaches a
    # superset of the closure's stops, and reaches strictly more. Without the
    # set it keeps the closure board-at-origin path and ignores the walking
    # arguments.
    with pytest.warns(UserWarning):
        net = TransportNetwork.from_gtfs(
            [str(central_gtfs)], osm_pbf=str(kantakaupunki_pbf), max_walking_time=60
        )
    coords = {s: (lat, lon) for s, lat, lon in net.stops if lat is not None}
    origin = next(iter(coords))

    closure = net.travel_times_from_stop(origin, QUERY_DATE, QUERY_TIME)
    # The walking arguments are ignored without a whole-day set.
    assert (
        net.travel_times_from_stop(
            origin, QUERY_DATE, QUERY_TIME, max_walking_time=ACCESS
        )
        == closure
    )

    net.compute_ultra_shortcuts(max_transfer_time=600.0)  # whole day

    ultra = net.travel_times_from_stop(
        origin, QUERY_DATE, QUERY_TIME, max_walking_time=ACCESS
    )
    # Door-to-door: identical to routing from the origin stop's coordinate.
    assert ultra == net.travel_times_from_coordinate(
        coords[origin], QUERY_DATE, QUERY_TIME, max_walking_time=ACCESS
    )
    # Reachability superset of the closure, and strictly more stops reached
    # (the origin itself now costs its stop-to-platform connector walk, so this
    # is a reachability superset, not a strict per-stop time superset).
    assert set(closure) <= set(ultra)
    assert len(ultra) > len(closure)


def test_travel_time_matrix_mixed_origins_under_ultra(
    central_gtfs, kantakaupunki_pbf, tmp_path
):
    # A stop-origin travel_time_matrix under a whole-day set partitions its
    # origins: snappable origins route door-to-door, an off-network origin
    # falls back to the closure board-at-origin search, and rows follow input
    # order. Each row equals the matching per-origin travel_times_from_stop,
    # which gates identically.
    import zipfile

    import numpy as np
    import pandas as pd

    with zipfile.ZipFile(central_gtfs) as archive:
        tables = {
            name[:-4]: pd.read_csv(archive.open(name), dtype=str)
            for name in archive.namelist()
            if name.endswith(".txt")
        }
    offnet = tables["stops"]["stop_id"].iloc[0]
    tables["stops"].loc[
        tables["stops"]["stop_id"] == offnet, ["stop_lat", "stop_lon"]
    ] = [
        "0.0",
        "0.0",
    ]  # off the walking network — forces the closure fallback
    feed = tmp_path / "offnet.zip"
    with zipfile.ZipFile(feed, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, frame in tables.items():
            archive.writestr(name + ".txt", frame.to_csv(index=False))

    with pytest.warns(UserWarning):
        net = TransportNetwork.from_gtfs(
            [str(feed)], osm_pbf=str(kantakaupunki_pbf), max_walking_time=60
        )
    net.compute_ultra_shortcuts(max_transfer_time=600.0)  # whole day
    stops = [s for s, _, _ in net.stops]
    usable = [s for s, lat, _ in net.stops if lat is not None and abs(lat) > 1][:3]
    origins = [offnet, *usable]  # mixed, fallback origin first

    matrix = net.travel_time_matrix(
        origins, QUERY_DATE, QUERY_TIME, max_walking_time=ACCESS
    )
    unreached = np.iinfo(np.uint32).max
    for i, origin in enumerate(origins):
        row = {
            stops[j]: int(matrix[i, j])
            for j in range(len(stops))
            if matrix[i, j] != unreached
        }
        assert row == net.travel_times_from_stop(
            origin, QUERY_DATE, QUERY_TIME, max_walking_time=ACCESS
        )
    # The off-network origin fell back to closure board-at-origin (maps to 0);
    # a usable origin routes door-to-door (its own cell costs the connector).
    assert matrix[0, stops.index(offnet)] == 0
    assert matrix[1, stops.index(usable[0])] > 0


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
