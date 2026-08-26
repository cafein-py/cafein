"""Regression tests guarding specific fixed bugs.

One test per fixed defect; add new ones here rather than in a new file.
"""

import pytest

from cafein import exhaustive_frontier, journey_frontier, journey_frontiers
from cafein.matrices import travel_cost_table


def test_mcraptor_window_profile_keeps_cleaner_earlier_journeys(network_with_footpaths):
    """McRAPTOR's departure-window emissions profile must not drop an
    undominated cleaner-but-earlier-departing journey.

    The per-stop label bag is cumulative across the descending profile
    passes; before the fix its dominance ignored the rides used to reach
    a stop, so a later-departure journey that reached an intermediate
    stop with more transfers could suppress an earlier-departure journey
    that reached it with fewer — and thus still had the transfer budget
    for a cleaner continuation. On this pair that dropped the cleanest
    journey entirely: the window frontier collapsed to the single
    latest-departing (dirtiest) point.

    The exhaustive oracle at 08:30 (inside the window) pins the true
    minimum, so McRAPTOR over the window must reach a journey no dirtier
    than it, and must return more than the one dirtiest point.
    """
    origin, destination = "1010419", "4240227"
    oracle = exhaustive_frontier(
        network_with_footpaths,
        origin,
        destination,
        "2022-02-22 08:30:00",
        max_rides=5,
    )
    cleanest = oracle["emissions"].min()

    frontier = journey_frontier(
        network_with_footpaths,
        origin,
        destination,
        "2022-02-22 08:30:00",
        departure_time_window=15,
        max_rides=5,
        candidates="pareto",
        bucket=1e-6,
        router="raptor",
    )
    on_frontier = frontier[frontier["frontier"]]
    transit = on_frontier[on_frontier["rides"] >= 1]

    assert transit["emissions"].min() == pytest.approx(cleanest, abs=1e-3)
    assert len(transit) > 1


def test_mctbtr_window_profile_keeps_cleaner_earlier_journeys(network_with_footpaths):
    """The McTBTR profile has the same cross-pass hazard as McRAPTOR, in
    the stop bags that gate its query-time footpath relaxation.

    Those bags dominate on (arrival, emissions bucket) and persist across
    the descending profile passes; before the fix they ignored the rides
    used to reach a stop, so a later-departure arrival on more rides could
    suppress a cleaner fewer-rides arrival, and the walk that would reach
    a cleaner onward leg was never relaxed. On this pair McTBTR kept a
    journey a few grams dirtier than the oracle's cleanest; ranking the
    rides in the stop bags restores completeness.
    """
    origin, destination = "1010108", "3170218"
    oracle = exhaustive_frontier(
        network_with_footpaths,
        origin,
        destination,
        "2022-02-22 08:30:00",
        max_rides=5,
    )
    cleanest = oracle["emissions"].min()

    frontier = journey_frontier(
        network_with_footpaths,
        origin,
        destination,
        "2022-02-22 08:30:00",
        departure_time_window=15,
        max_rides=5,
        candidates="pareto",
        bucket=1e-6,
        router="tbtr",
    )
    transit = frontier[frontier["frontier"] & (frontier["rides"] >= 1)]

    assert transit["emissions"].min() == pytest.approx(cleanest, abs=1e-3)


def test_factor_loaders_reject_infinities():
    # Both factor loaders accepted +inf component values, which priced
    # infinite emissions instead of failing loudly; NA stays the marker
    # for an unresolved component.
    import pandas as pd
    import pytest

    from cafein import emissions

    street = pd.DataFrame(
        [
            {
                "street_mode": "car",
                "vehicle_class": "ICE",
                "service_model": "private",
                "vehicle": float("inf"),
                "fuel": 0.0,
                "infrastructure": 0.0,
                "operations": 0.0,
            }
        ]
    )
    with pytest.raises(ValueError, match="non-finite"):
        emissions.load_street_factors(street)
    transit = pd.DataFrame([{"route_type": 3, "vehicle": float("inf")}])
    with pytest.raises(ValueError, match="non-finite"):
        emissions.load_factors(transit)


def _install_two_edge_multimodal_graph(network):
    """A walk+e-scooter edge south of the first stop and a bicycle-only
    edge north of it, installed directly — the synthetic multimodal graph
    the mode-separation test uses, with every edge index and mask known."""
    stop, lat, lon = next((s, la, lo) for s, la, lo in network.stops if la is not None)
    south, north = lat - 0.0005, lat + 0.0011
    zeros = [0, 0]
    network._core.set_multimodal_streets(
        ["walk", "bicycle", "e_scooter"],
        4,
        [(0, 1, 200.0), (2, 3, 200.0)],
        [0, 2, 4],
        [lon - 0.001, lon + 0.001, lon - 0.001, lon + 0.001],
        [south, south, north, north],
        zeros,
        zeros,
        zeros,
        zeros,
        [1 | 4, 2],  # south: walk + e_scooter; north: bicycle only
        [1 | 4, 2],
        zeros,
        zeros,
    )
    return stop, lat, lon, south


def test_multimodal_leg_validates_the_street_choice_token(fresh_footpaths_network):
    """The leg rebuild must reject a malformed ``StreetChoice`` token at
    the boundary instead of feeding it to the core.

    Before the fix the caller-supplied ``(edge, fraction, connector)``
    triple became a core ``Snap`` unchecked: an out-of-range edge index
    panicked through unchecked indexing rather than raising ``ValueError``,
    and a non-finite or out-of-range fraction or a negative connector
    produced invalid or understated costs silently. An edge the resolved
    profile cannot traverse was accepted too, seeding the search from an
    arc the mode may not use.
    """
    network = fresh_footpaths_network
    stop, lat, lon, south = _install_two_edge_multimodal_graph(network)
    core = network._core
    rows = {r[0]: r for r in core._street_access_seconds(south, lon, "bicycle", 600.0)}
    _, _, edge, fraction, connector = rows[stop]
    good = core._multimodal_leg(
        south, lon, "bicycle", stop, edge, fraction, connector, False, 600.0, False
    )
    assert good is not None
    malformed = [
        (999, fraction, connector, "out of range"),
        (edge, float("nan"), connector, "fraction"),
        (edge, 1.5, connector, "fraction"),
        (edge, fraction, -1.0, "connector"),
        (edge, fraction, float("inf"), "connector"),
    ]
    for bad_edge, bad_fraction, bad_connector, message in malformed:
        with pytest.raises(ValueError, match=message):
            core._multimodal_leg(
                south,
                lon,
                "bicycle",
                stop,
                bad_edge,
                bad_fraction,
                bad_connector,
                False,
                600.0,
                False,
            )
    # The walk+e-scooter edge is a valid index the bicycle profile may not use.
    walk_rows = {
        r[0]: r for r in core._street_access_seconds(south, lon, "walk", 600.0)
    }
    walk_edge = walk_rows[stop][2]
    assert walk_edge != edge
    with pytest.raises(ValueError, match="not traversable"):
        core._multimodal_leg(
            south, lon, "bicycle", stop, walk_edge, 0.5, 1.0, False, 600.0, False
        )


def test_multimodal_zero_shortcut_validates_snap_and_cutoff(fresh_footpaths_network):
    """The equal-coordinate shortcut on the multimodal access surface must
    snap and check the cutoff before reporting a zero leg.

    Before the fix equal origin and destination coordinates returned a
    zero-duration leg before either endpoint was snapped or ``max_seconds``
    validated, so an equal pair arbitrarily far from the network — or a
    query with a negative or NaN cutoff — was reported reachable, unlike
    the direct matrix and standalone travel-time surfaces.
    """
    network = fresh_footpaths_network
    stop, lat, lon, south = _install_two_edge_multimodal_graph(network)
    core = network._core
    # An equal pair far from every edge is unsnappable, not a zero leg.
    far = (lat + 1.0, lon + 1.0)
    assert core._multimodal_direct_leg(far, far, "walk", 600.0, False) is None
    # A snappable equal pair is a zero leg only when the cutoff admits one.
    origin = (south, lon)
    assert core._multimodal_direct_leg(origin, origin, "walk", -1.0, False) is None
    nan = float("nan")
    assert core._multimodal_direct_leg(origin, origin, "walk", nan, False) is None
    seconds, network_m, connector_m, _ = core._multimodal_direct_leg(
        origin, origin, "walk", 0.0, False
    )
    assert (seconds, network_m, connector_m) == (0, 0.0, 0.0)
    # The stop-coincident shortcut in the leg rebuild obeys the same gate.
    rows = {r[0]: r for r in core._street_access_seconds(lat, lon, "walk", 600.0)}
    _, zero_seconds, edge, fraction, connector = rows[stop]
    assert zero_seconds == 0
    assert (
        core._multimodal_leg(
            lat, lon, "walk", stop, edge, fraction, connector, False, -1.0, False
        )
        is None
    )
    parts = core._multimodal_leg(
        lat, lon, "walk", stop, edge, fraction, connector, False, 600.0, False
    )
    assert parts is not None and parts[0] == 0


def test_street_modes_validate_before_any_file_is_read():
    # A bare string used to survive to tuple("walk") == ('w','a','l','k')
    # and only fail after the whole GTFS build (issue #237). The paths
    # here do not exist: reaching a file error would mean lazy validation.
    pytest.importorskip("cafein._cafein")
    from cafein import StreetNetwork, TransportNetwork

    with pytest.raises(TypeError, match="iterable of mode names"):
        TransportNetwork.from_gtfs(
            ["no-such-feed.zip"], osm_pbf="no-such.osm.pbf", street_modes="walk"
        )
    with pytest.raises(ValueError, match="unknown street mode"):
        TransportNetwork.from_gtfs(
            ["no-such-feed.zip"], osm_pbf="no-such.osm.pbf", street_modes=("flying",)
        )
    with pytest.raises(ValueError, match="duplicate street mode"):
        TransportNetwork.from_gtfs(
            ["no-such-feed.zip"],
            osm_pbf="no-such.osm.pbf",
            street_modes=("walk", "walk"),
        )
    with pytest.raises(TypeError, match="iterable of mode names"):
        StreetNetwork.from_osm("no-such.osm.pbf", modes="bicycle")


def test_id_collections_refuse_bare_strings(network):
    # A bare string used to dissolve into one-character ids: exclusions
    # matched nothing (silently wrong results), stop sets resolved to
    # confusing per-character KeyErrors (issue #237).
    from cafein import DetailedItineraries, TravelTimeMatrix

    with pytest.raises(TypeError, match="exclude_routes"):
        network.route_between_stops(
            "1040280", "1100602", "2022-02-22 08:30:00", exclude_routes="1001"
        )
    with pytest.raises(TypeError, match="exclude_trips"):
        network.travel_times_from_stop(
            "1040280", "2022-02-22 08:30:00", exclude_trips="t-1"
        )
    with pytest.raises(TypeError, match="origins"):
        network.travel_time_matrix("1040280", "2022-02-22 08:30:00")
    with pytest.raises(TypeError, match="origins"):
        TravelTimeMatrix(network, "1040280", None, "2022-02-22 08:30:00")
    with pytest.raises(TypeError, match="origins"):
        DetailedItineraries(network, origins="1040280", departure="2022-02-22 08:30:00")
    with pytest.raises(TypeError, match="exclude_stops"):
        journey_frontier(
            network,
            "1040280",
            "1100602",
            "2022-02-22 08:30:00",
            10,
            exclude_stops="1040280",
        )
    with pytest.raises(TypeError, match="components"):
        network.annotate_emissions([], components="fuel")
    with pytest.raises(TypeError, match="origins"):
        journey_frontiers(network, "1040280", ["1100602"], "2022-02-22 08:30:00", 10)
    with pytest.raises(TypeError, match="origins"):
        travel_cost_table(network, "1040280", None, "2022-02-22 08:30:00")


def test_component_selections_accept_one_shot_iterables(network):
    # `set(components)` used to run twice, so a generator was exhausted
    # by validation and the selection came out empty.
    journeys = network.route_between_stops("4810551", "1250551", "2022-02-22 08:30:00")
    annotated = network.annotate_emissions(journeys, components=iter(["fuel"]))
    legs = [leg for j in annotated for leg in j["legs"] if leg["type"] == "transit"]
    assert any(leg.get("emissions") is not None for leg in legs)
    # The frontier path resolves factors, then annotates: the selection
    # must be frozen once at entry, not consumed twice.
    frontier = journey_frontier(
        network,
        "4810551",
        "1250551",
        "2022-02-22 08:30:00",
        10,
        candidates="pareto",
        components=iter(["fuel"]),
    )
    assert len(frontier) > 0


def test_id_collections_refuse_bytes(network):
    # Bytes iterate as integers: b"1001" would become ("49", "48", ...).
    with pytest.raises(TypeError, match="exclude_routes"):
        network.route_between_stops(
            "1040280", "1100602", "2022-02-22 08:30:00", exclude_routes=b"1001"
        )
    with pytest.raises(TypeError, match="components"):
        network.annotate_emissions([], components=b"fuel")


def _chained_walk_feed(path):
    """Four stops in a line: a trip into B, a trip out of D, and nothing
    joining them but footpaths B→C and C→D."""
    import zipfile

    tables = {
        "agency.txt": [
            "agency_id,agency_name,agency_url,agency_timezone",
            "A,Test Agency,http://example.com,Europe/Helsinki",
        ],
        "stops.txt": [
            "stop_id,stop_name,stop_lat,stop_lon",
            "A,A,60.000,24.000",
            "B,B,60.001,24.000",
            "C,C,60.002,24.000",
            "D,D,60.003,24.000",
            "E,E,60.004,24.000",
        ],
        "routes.txt": ["route_id,route_short_name,route_type", "R1,1,3", "R2,2,3"],
        "trips.txt": ["route_id,service_id,trip_id", "R1,SV,T_IN", "R2,SV,T_OUT"],
        "stop_times.txt": [
            "trip_id,arrival_time,departure_time,stop_id,stop_sequence",
            "T_IN,08:00:00,08:00:00,A,1",
            "T_IN,08:10:00,08:10:00,B,2",
            "T_OUT,08:40:00,08:40:00,D,1",
            "T_OUT,08:50:00,08:50:00,E,2",
        ],
        "calendar.txt": [
            "service_id,monday,tuesday,wednesday,thursday,friday,saturday,"
            "sunday,start_date,end_date",
            "SV,1,1,1,1,1,1,1,20220101,20221231",
        ],
    }
    with zipfile.ZipFile(path, "w") as archive:
        for name, lines in tables.items():
            archive.writestr(name, "\n".join(lines) + "\n")
    return path


@pytest.mark.parametrize("router", ["raptor", "tbtr"])
def test_transfers_never_chain_two_bounded_walks(tmp_path, router):
    # Issue #249: transfers are single bounded street walks. B→C and
    # C→D are installed at 600 s each; their 1200 s composition is not,
    # so no engine may walk B→C→D between the two rides — riding out of
    # D is simply unreachable, while the single hop to C stands.
    import numpy as np

    from cafein import TransportNetwork
    from cafein.streets import Footpaths

    feed = _chained_walk_feed(tmp_path / "chained.zip")
    network = TransportNetwork.from_gtfs([str(feed)])
    network.set_transfers(
        Footpaths(
            ["A", "B", "C", "D", "E"],
            [1, 2, 2, 3],
            [2, 1, 3, 2],
            [600, 600, 600, 600],
            [600.0, 600.0, 600.0, 600.0],
        )
    )
    matrix = network.travel_time_matrix(["A"], "2022-02-22 07:55:00", router=router)
    unreachable = np.iinfo(np.uint32).max
    # Ride A→B (08:10), then one 600 s walk to C: 08:20.
    assert matrix[0][2] == 25 * 60
    # Chaining B→C→D would reach D at 08:30, in time for the 08:40
    # departure to E; a single bounded walk never does.
    assert matrix[0][3] == unreachable
    assert matrix[0][4] == unreachable


def test_multimodal_payload_tolerates_a_zero_length_edge(
    kantakaupunki_pbf, monkeypatch
):
    """A zero-length way must not break the DEM densification.

    OSM legally carries ways whose consecutive nodes sit at one
    position; their edge geometry is a two-point LineString of
    identical coordinates, which GEOS's segmentize refuses ("point
    array must contain 0 or >1 elements"). One such cycleway in the
    2026-08-20 Finland extract failed every DEM-enabled street build.
    """
    import geopandas
    import numpy
    import pandas
    from shapely.geometry import LineString

    from cafein import _osm, street_network

    real_union = _osm.union_network

    def union_with_degenerate_edge(osm_pbf, bounding_box=None, car=False):
        nodes, edges = real_union(osm_pbf, bounding_box=bounding_box, car=car)
        clone = edges.iloc[[0]].copy()
        point = clone.geometry.iloc[0].coords[0]
        clone.geometry = geopandas.GeoSeries(
            [LineString([point, point])], crs=edges.crs
        ).values
        clone["length"] = 0.0
        return nodes, pandas.concat([edges, clone], ignore_index=True)

    monkeypatch.setattr(_osm, "union_network", union_with_degenerate_edge)

    # Any dem enables the densification that used to crash; a flat
    # elevation callable keeps the test off the optional raster stack.
    payload = street_network.multimodal_payload(
        str(kantakaupunki_pbf),
        bounding_box=[24.93, 60.16, 24.96, 60.18],
        dem=lambda longitudes, latitudes: numpy.zeros(len(longitudes)),
    )
    assert payload is not None


def test_walking_knobs_validate_before_any_file_is_read():
    # The walking triple used to survive to the street build and only
    # fail after the whole GTFS ingest and PBF parse (issue #237). The
    # paths here do not exist: reaching a file error would mean lazy
    # validation.
    pytest.importorskip("cafein._cafein")
    from cafein import TransportNetwork
    from cafein.streets import walking_footpaths, walking_streets

    with pytest.raises(ValueError, match="walking_speed_kmph must be a positive"):
        TransportNetwork.from_gtfs(
            ["no-such-feed.zip"], osm_pbf="no-such.osm.pbf", walking_speed_kmph=0
        )
    with pytest.raises(ValueError, match="snap_distance must be a non-negative"):
        TransportNetwork.from_gtfs(
            ["no-such-feed.zip"], osm_pbf="no-such.osm.pbf", snap_distance=-1
        )
    with pytest.raises(TypeError, match="walking_speed_kmph must be a number"):
        TransportNetwork.from_gtfs(
            ["no-such-feed.zip"], osm_pbf="no-such.osm.pbf", walking_speed_kmph="fast"
        )
    with pytest.raises(ValueError, match="four finite numbers"):
        TransportNetwork.from_gtfs(
            ["no-such-feed.zip"],
            osm_pbf="no-such.osm.pbf",
            bounding_box=(24.9, 60.1, 25.0),
        )
    with pytest.raises(ValueError, match="each minimum below its maximum"):
        TransportNetwork.from_gtfs(
            ["no-such-feed.zip"],
            osm_pbf="no-such.osm.pbf",
            bounding_box=(25.0, 60.1, 24.9, 60.2),
        )
    for build in (walking_footpaths, walking_streets):
        with pytest.raises(ValueError, match="walking_speed_kmph must be a positive"):
            build("no-such.osm.pbf", [], walking_speed_kmph=-3.6)
        with pytest.raises(ValueError, match="snap_distance must be a non-negative"):
            build("no-such.osm.pbf", [], snap_distance=float("nan"))
        with pytest.raises(TypeError, match="bounding_box must be four numbers"):
            build("no-such.osm.pbf", [], bounding_box="helsinki")
    # Numeric-looking strings are the wrong KIND, never quietly parsed.
    with pytest.raises(TypeError, match="walking_speed_kmph must be a number"):
        walking_footpaths("no-such.osm.pbf", [], walking_speed_kmph="3.6")
    # The extractor validates its bounding box before the PBF read too.
    from cafein.streets import park_and_ride_facilities

    with pytest.raises(ValueError, match="each minimum below its maximum"):
        park_and_ride_facilities(
            "no-such.osm.pbf", bounding_box=(25.0, 60.1, 24.9, 60.2)
        )
    # A one-shot iterable bounding box validates AND survives: the
    # returned snapshot replaces the exhausted iterator.
    from cafein._validate import validated_bounding_box

    assert validated_bounding_box(iter((24.9, 60.1, 25.0, 60.2))) == [
        24.9,
        60.1,
        25.0,
        60.2,
    ]
    # An endless iterable refuses by length rather than hanging, and
    # an integer beyond float range refuses as non-finite everywhere —
    # never a raw OverflowError.
    import itertools

    with pytest.raises(ValueError, match="four finite numbers"):
        validated_bounding_box(itertools.count())
    # An empty geometry's NaN bounds refuse instead of reaching the
    # extract reader.
    from shapely.geometry import Polygon

    with pytest.raises(ValueError, match="geometry must have four finite bounds"):
        validated_bounding_box(Polygon())
    with pytest.raises(ValueError, match="walking_speed_kmph must be a positive"):
        walking_footpaths("no-such.osm.pbf", [], walking_speed_kmph=10**400)
    with pytest.raises(ValueError, match="non-negative, finite duration"):
        walking_footpaths("no-such.osm.pbf", [], max_walking_time=10**400)


def test_query_walking_knobs_fail_by_their_public_names(network):
    # The query gate names the public parameter (snap_distance), not
    # the internal Rust name it used to leak (issue #237).
    with pytest.raises(ValueError, match="snap_distance must be a non-negative"):
        network.route_between_coordinates(
            (60.1690, 24.9320),
            (60.1795, 24.9520),
            "2022-02-22 08:30:00",
            snap_distance=-5,
        )
    with pytest.raises(ValueError, match="walking_speed_kmph must be a positive"):
        network.travel_times_from_coordinate(
            (60.1690, 24.9320), "2022-02-22 08:30:00", walking_speed_kmph=0
        )
    # The stop-matrix branches that ignore the knobs still refuse
    # garbage — the reverse stop axis and the windowed form included.
    from cafein import TravelTimeMatrix

    with pytest.raises(ValueError, match="snap_distance must be a non-negative"):
        TravelTimeMatrix(
            network,
            ["1000202"],
            None,
            arrival="2022-02-22 09:30:00",
            snap_distance=-1,
        )
    with pytest.raises(ValueError, match="walking_speed_kmph must be a positive"):
        TravelTimeMatrix(
            network,
            ["1000202"],
            None,
            "2022-02-22 08:30:00",
            departure_time_window=10,
            percentiles=(50,),
            walking_speed_kmph=-1,
        )


def test_multimodal_options_validate_before_any_file_is_read():
    # speed_limits, urban_areas, dem, and dem_interval used to fail
    # only after the full extraction — and on from_gtfs, after the
    # whole GTFS ingest too (issue #237). The paths here do not exist:
    # reaching a file error would mean lazy validation.
    pytest.importorskip("cafein._cafein")
    geopandas = pytest.importorskip("geopandas")
    from shapely.geometry import Polygon

    from cafein import StreetNetwork, TransportNetwork

    def gtfs(**options):
        return TransportNetwork.from_gtfs(
            ["no-such-feed.zip"],
            osm_pbf="no-such.osm.pbf",
            street_modes=("walk", "car"),
            **options,
        )

    def osm(**options):
        return StreetNetwork.from_osm(
            "no-such.osm.pbf", modes=("walk", "car"), **options
        )

    for build in (gtfs, osm):
        with pytest.raises(ValueError, match="unknown speed_limits classes"):
            build(speed_limits={"warp_drive": 300.0})
        with pytest.raises(ValueError, match="must be a positive km/h number"):
            build(speed_limits={"motorway_inside": -80.0})
        with pytest.raises(ValueError, match="dem_interval must be a positive"):
            build(dem="no-such-dem.tif", dem_interval=0)
        with pytest.raises(ValueError, match="does not exist"):
            build(dem="no-such-dem.tif")
        bare = geopandas.GeoDataFrame(
            geometry=[Polygon([(24.9, 60.1), (24.9, 60.2), (25.0, 60.2)])]
        )
        with pytest.raises(ValueError, match="urban_areas must carry a CRS"):
            build(urban_areas=bare)
        with pytest.raises(ValueError, match="ISO 3166"):
            build(country="finland")
        with pytest.raises(ValueError, match="positive km/h number"):
            build(speed_limits={"motorway_inside": 10**400})
        with pytest.raises(ValueError, match="one-dimensional boolean mask"):
            build(urban_areas=True)
        with pytest.raises(ValueError, match="one-dimensional boolean mask"):
            build(urban_areas=[[True, False], [False, True]])
        with pytest.raises(TypeError, match="two positional arguments"):
            build(dem=lambda: None)
    # The validated snapshots are the build's inputs: mutating the
    # caller's objects afterwards changes nothing.
    import numpy as np

    from cafein.street_network import validate_street_options

    mask = np.array([True, True, False])
    limits = {"motorway_inside": 100.0}
    _modes, _dem, _country, held_mask, held_limits = validate_street_options(
        ("walk", "car"), urban_areas=mask, speed_limits=limits
    )
    mask[:] = False
    limits["motorway_inside"] = -1.0
    assert held_mask.sum() == 2
    assert held_limits["motorway_inside"] == 100.0
    # A frame whose active geometry has another name normalizes to the
    # canonical column, so the build's spatial join reads the shapes
    # the caller meant.
    from shapely.geometry import Polygon as _Polygon

    renamed = geopandas.GeoDataFrame(
        {"footprint": [_Polygon([(24.9, 60.1), (24.9, 60.2), (25.0, 60.2)])]},
        geometry="footprint",
        crs="EPSG:4326",
    )
    _m, _d, _c, held_areas, _l = validate_street_options(
        ("walk", "car"), urban_areas=renamed
    )
    assert list(held_areas.columns) == ["geometry"]
    assert held_areas.crs is not None
    # The coherence refusal keeps its priority: options without the
    # car mode name the missing mode, not their own contents.
    with pytest.raises(ValueError, match="configure car speeds"):
        TransportNetwork.from_gtfs(
            ["no-such-feed.zip"],
            osm_pbf="no-such.osm.pbf",
            street_modes=("walk",),
            speed_limits={"warp_drive": 300.0},
        )


def test_exposure_validates_before_the_street_frame_builds():
    # rasterize, thresholds, layer names, and every layer's spec used
    # to validate only after the lazy street-frame materialization —
    # and a malformed later layer failed only after the earlier layers
    # had fully ingested (issue #237). The network here explodes on
    # frame access: reaching it would prove lazy validation.
    from cafein.exposure import Exposure

    class _Untouchable:
        @property
        def streets_gdf(self):
            raise AssertionError("the street frame was built before validation")

    with pytest.raises(ValueError, match="rasterize must be a positive"):
        Exposure(_Untouchable(), noise=("no-such.tif", "band"), rasterize=-1.0)
    with pytest.raises(ValueError, match="must be a .source, value. pair"):
        Exposure(_Untouchable(), noise="no-such.tif")
    with pytest.raises(ValueError, match="thresholds name unknown layer"):
        Exposure(_Untouchable(), noise=("no-such.tif", "band"), thresholds={"other": 5})
    with pytest.raises(ValueError, match="collides with the cost"):
        Exposure(_Untouchable(), cost_noise=("no-such.tif", "band"))
    # A malformed LATER layer refuses before any earlier layer opens:
    # the first layer's raster does not exist, so touching it would
    # raise its own error instead of the spec refusal.
    with pytest.raises(ValueError, match="must be a .source, value. pair"):
        Exposure(
            _Untouchable(),
            aa=("no-such-raster.tif", "band"),
            zz="not-a-pair",
        )


def test_street_query_knobs_refuse_instead_of_emptying(network):
    # A negative snap_distance on the street matrices used to unroute
    # every point Rust-side and return an EMPTY matrix with only an
    # off-the-streets warning — a silent wrong answer (issue #237).
    geopandas = pytest.importorskip("geopandas")
    from shapely.geometry import Point

    from cafein import StreetNetwork, TravelTimeMatrix

    street = StreetNetwork.from_osm("tests/data/kantakaupunki.osm.pbf", modes=("walk",))
    origins = geopandas.GeoDataFrame(
        {"id": ["a"]},
        geometry=[Point(24.9320, 60.1690)],
        crs="EPSG:4326",
    )
    with pytest.raises(ValueError, match="snap_distance must be a non-negative"):
        TravelTimeMatrix(
            street, origins, origins, transport_mode="walk", snap_distance=-5
        )
    with pytest.raises(ValueError, match="max_street_time must be a non-negative"):
        TravelTimeMatrix(
            street,
            origins,
            origins,
            transport_mode="walk",
            max_street_time=float("nan"),
        )
    with pytest.raises(ValueError, match="snap_distance must be a non-negative"):
        street.travel_time(
            (60.1690, 24.9320), (60.1795, 24.9520), mode="walk", snap_distance=-5
        )
    # Percentile ranks refuse by name in Python, before any snapping.
    with pytest.raises(ValueError, match="within \\[0, 100\\]"):
        network.travel_time_matrix(
            ["1000202"],
            "2022-02-22 08:30:00",
            departure_time_window=10,
            percentiles=(150,),
        )
    with pytest.raises(ValueError, match="within \\[0, 100\\]"):
        network.travel_time_matrix(
            ["1000202"],
            "2022-02-22 08:30:00",
            departure_time_window=10,
            percentiles=(10**400,),
        )
    # A bare string would dissolve into digits: "50" is not [5.0, 0.0].
    with pytest.raises(TypeError, match="percentiles must be a collection"):
        network.travel_time_matrix(
            ["1000202"],
            "2022-02-22 08:30:00",
            departure_time_window=10,
            percentiles="50",
        )
    # Street-policy budgets refuse oversized integers by name too.
    from cafein.policy import StreetLegPolicy

    with pytest.raises(ValueError, match="positive, finite time budget"):
        StreetLegPolicy(access={"walk": 10**400})
    # The wrong KIND on the car options is a TypeError now, matching
    # the house discipline; the value refusals are unchanged.
    from cafein.emissions import _car_query_options

    with pytest.raises(TypeError, match="occupancy must be a number"):
        _car_query_options("car", "two", None)
    with pytest.raises(ValueError, match="at least 1"):
        _car_query_options("car", 0.5, None)
    with pytest.raises(ValueError, match="finite number of at least 1"):
        _car_query_options("car", 10**400, None)
    from cafein.policy import CarParkPolicy as _Policy

    with pytest.raises(ValueError, match="at least 1"):
        _Policy(
            facilities=geopandas.GeoDataFrame(
                {"id": ["f"]}, geometry=[Point(24.9330, 60.1990)], crs="EPSG:4326"
            ),
            occupancy=10**400,
        )


@pytest.mark.parametrize("streamed", [False, True])
def test_decay_parameters_keep_their_fractional_seconds(network, tmp_path, streamed):
    """Time-axis decay parameters must reach the core as exact seconds
    on both execution paths — the constructor and the streamed
    ``to_parquet``, which converts them separately.

    Before the fix `decay_params` went through the whole-second rounding
    of the router-clock inputs, so a half-life of ``ln2/0.1`` minutes
    (415.888 s) ran as 416 s and every exponential weight drifted by
    ``exp((x − x')·t)`` — 2.4 % on the farthest destinations.
    """
    import math

    import numpy as np
    import pandas as pd

    from cafein.accessibility import Accessibility, NearestDestinations

    departure = "2022-02-22 08:30"
    stops = [stop for stop, _, _ in network.stops]
    rng = np.random.default_rng(3)
    origins = list(rng.choice(stops, 8, replace=False))
    table = pd.DataFrame({"id": rng.choice(stops, 40, replace=False), "jobs": 1.0})
    costs = NearestDestinations(
        network,
        origins,
        list(table["id"]),
        departure,
        k=len(table),
        output_time_units="seconds",
    )
    half_life = math.log(2) / 0.1  # minutes: 415.888 s, not a whole number
    call = dict(
        opportunities="jobs",
        budgets=(float(costs["cost"].max()) / 60 + 1,),
        decay="exponential",
        decay_params={"half_life": half_life},
    )
    if streamed:
        parquet = pytest.importorskip("pyarrow.parquet")
        Accessibility.to_parquet(
            network,
            origins,
            table,
            departure,
            output=tmp_path / "decay.parquet",
            batch_size=3,
            **call,
        )
        result = parquet.read_table(tmp_path / "decay.parquet").to_pandas()
        result = result.astype({"from_id": str}).set_index("from_id")["accessibility"]
    else:
        result = Accessibility(network, origins, table, departure, **call)
        result = result.set_index("from_id")["accessibility"]
    weights = np.exp(-math.log(2) * costs["cost"] / (half_life * 60))
    expected = weights.groupby(costs["from_id"]).sum().reindex(origins, fill_value=0)
    assert np.allclose(result.reindex(origins), expected, rtol=1e-12)
