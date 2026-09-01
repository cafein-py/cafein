"""CarParkPolicy and the park-and-ride facilities intake."""

import datetime

import geopandas
import pandas as pd
import pytest
from shapely.geometry import Point, Polygon

from cafein.policy import CarParkPolicy
from cafein.streets import park_and_ride_facilities

KAMPPI = Point(24.9320, 60.1690)
HAKANIEMI = Point(24.9520, 60.1795)


def _facilities(**columns):
    frame = geopandas.GeoDataFrame(
        dict(columns), geometry=[KAMPPI, HAKANIEMI], crs="EPSG:4326"
    )
    return frame


def test_the_policy_fills_the_documented_defaults():
    policy = CarParkPolicy(facilities=_facilities())
    snapshot = policy.facilities
    assert snapshot["id"].tolist() == [0, 1]
    assert snapshot["search_seconds"].tolist() == [300.0, 300.0]
    assert snapshot["fee"].tolist() == [0.0, 0.0]
    assert policy.max_car_seconds == 30 * 60
    assert policy.max_facility_walk_seconds == 10 * 60
    assert policy.occupancy == 1.0
    assert policy.intersection_delays is True


def test_columns_override_and_nan_fills():
    policy = CarParkPolicy(
        facilities=_facilities(
            id=["p1", "p2"], search_seconds=[120.0, None], fee=[None, 4.0]
        ),
        max_car_time=datetime.timedelta(minutes=45),
        max_facility_walk_time=5,
        occupancy=2.0,
        vehicle_class="BEV",
        intersection_delays=False,
    )
    snapshot = policy.facilities
    assert snapshot["id"].tolist() == ["p1", "p2"]
    assert snapshot["search_seconds"].tolist() == [120.0, 300.0]
    assert snapshot["fee"].tolist() == [0.0, 4.0]
    assert policy.max_car_seconds == 45 * 60
    assert policy.max_facility_walk_seconds == 5 * 60
    assert policy.vehicle_class == "BEV"
    assert policy.intersection_delays is False


def test_the_snapshot_is_isolated_and_reprojected():
    metric = _facilities().to_crs(epsg=3067)
    policy = CarParkPolicy(facilities=metric)
    assert str(policy.facilities.crs).upper() == "EPSG:4326"
    assert policy.facilities.geometry.iloc[0].distance(KAMPPI) < 1e-6
    # Mutating the input afterwards never reaches the snapshot.
    metric["fee"] = 99.0
    assert policy.facilities["fee"].tolist() == [0.0, 0.0]


def test_policy_validation_refusals():
    bare = _facilities()
    bare.crs = None
    square = geopandas.GeoDataFrame(
        geometry=[Polygon([(24.9, 60.1), (24.9, 60.2), (25.0, 60.2)])],
        crs="EPSG:4326",
    )
    facilities = _facilities()
    for kwargs, exc, match in (
        # The facilities frame refusals...
        (
            dict(facilities=pd.DataFrame({"id": [1]})),
            ValueError,
            "must be a GeoDataFrame",
        ),
        (
            dict(facilities=_facilities().iloc[:0]),
            ValueError,
            "names no park-and-ride",
        ),
        (dict(facilities=bare), ValueError, "must carry a CRS"),
        (dict(facilities=square), ValueError, "representative_point"),
        (dict(facilities=_facilities(id=["p", "p"])), ValueError, "duplicates"),
        (dict(facilities=_facilities(id=["p", None])), ValueError, "missing values"),
        (
            dict(facilities=_facilities(search_seconds=[-1.0, 10.0])),
            ValueError,
            "finite and non-negative",
        ),
        (dict(facilities=_facilities(fee=[-2.0, 0.0])), ValueError, "EUR2017"),
        (
            dict(facilities=_facilities(search_seconds=["fast", 10.0])),
            ValueError,
            "non-numeric values",
        ),
        (
            dict(facilities=_facilities(fee=["free", 0.0])),
            ValueError,
            "non-numeric values",
        ),
        # ...and the policy parameter refusals.
        (
            dict(facilities=facilities, max_car_time=0),
            ValueError,
            "positive time budget",
        ),
        (
            dict(facilities=facilities, max_facility_walk_time=-5),
            ValueError,
            "non-negative, finite duration",
        ),
        (dict(facilities=facilities, max_car_time="30"), TypeError, "minutes"),
        (dict(facilities=facilities, occupancy=0.5), ValueError, "at least 1"),
        (
            dict(facilities=facilities, vehicle_class=3),
            TypeError,
            "emission-factor row",
        ),
    ):
        with pytest.raises(exc, match=match):
            CarParkPolicy(**kwargs)


def test_the_matrix_and_accessibility_refusals(network, tmp_path):
    from cafein import Accessibility, TravelCostMatrix, TravelTimeMatrix
    from cafein.policy import StreetLegPolicy

    policy = CarParkPolicy(facilities=_facilities())
    origins = geopandas.GeoDataFrame({"id": ["a"]}, geometry=[KAMPPI], crs="EPSG:4326")
    destinations = geopandas.GeoDataFrame(
        {"id": ["b"]}, geometry=[HAKANIEMI], crs="EPSG:4326"
    )
    # The surfaces demand the car side by name on a network without it.
    with pytest.raises(ValueError, match="multimodal car side"):
        TravelTimeMatrix(
            network,
            origins,
            destinations,
            "2022-02-22 08:30:00",
            street_policy=policy,
        )
    with pytest.raises(ValueError, match="multimodal car side"):
        network.route_between_coordinates(
            (60.1690, 24.9320),
            (60.1795, 24.9520),
            "2022-02-22 08:30:00",
            street_policy=policy,
        )
    stop_ids = list(network.stops_gdf["id"].iloc[:1])
    # The combination contract, each by name.
    with pytest.raises(ValueError, match="multimodal car side"):
        TravelTimeMatrix(
            network,
            origins,
            destinations,
            arrival="2022-02-22 09:30:00",
            street_policy=policy,
        )
    with pytest.raises(ValueError, match="departure_time_window"):
        TravelTimeMatrix(
            network,
            origins,
            destinations,
            "2022-02-22 08:30:00",
            departure_time_window=10,
            street_policy=policy,
        )
    with pytest.raises(ValueError, match="stop exclusions"):
        TravelTimeMatrix(
            network,
            origins,
            destinations,
            "2022-02-22 08:30:00",
            exclude_stops=stop_ids,
            street_policy=policy,
        )
    with pytest.raises(ValueError, match="fares"):
        TravelCostMatrix(
            network,
            origins,
            destinations,
            "2022-02-22 08:30:00",
            fares="hsl",
            street_policy=policy,
        )
    with pytest.raises(ValueError, match="router"):
        TravelCostMatrix(
            network,
            origins,
            destinations,
            "2022-02-22 08:30:00",
            router="tbtr",
            street_policy=policy,
        )
    with pytest.raises(NotImplementedError, match="do not stream"):
        TravelTimeMatrix.to_parquet(
            network,
            origins,
            destinations,
            "2022-02-22 08:30:00",
            street_policy=policy,
            output=str(tmp_path / "refused.parquet"),
        )
    # Accessibility's gate, each by name.
    with pytest.raises(ValueError, match="serves street_policy=CarParkPolicy"):
        Accessibility(
            network,
            origins,
            destinations,
            "2022-02-22 08:30:00",
            street_policy=StreetLegPolicy(access={"walk": 600}),
        )
    with pytest.raises(ValueError, match="cost='time'"):
        Accessibility(
            network,
            origins,
            destinations,
            "2022-02-22 08:30:00",
            cost="emissions",
            departure_time_window=10,
            street_policy=policy,
        )
    with pytest.raises(ValueError, match="multimodal car side"):
        Accessibility(
            network,
            origins,
            destinations,
            arrival="2022-02-22 09:30:00",
            street_policy=policy,
        )
    # The construction contract outranks the input shape: on a network
    # without the car side, stop-id inputs name the missing car side.
    with pytest.raises(ValueError, match="multimodal car side"):
        Accessibility(
            network,
            stop_ids,
            stop_ids,
            "2022-02-22 08:30:00",
            street_policy=policy,
        )
    with pytest.raises(NotImplementedError, match="does not stream"):
        Accessibility.to_parquet(
            network,
            origins,
            destinations,
            "2022-02-22 08:30:00",
            street_policy=policy,
            output=str(tmp_path / "refused-scores.parquet"),
        )


def test_the_extractor_contract_on_the_helsinki_fixture(kantakaupunki_pbf):
    frame = park_and_ride_facilities(str(kantakaupunki_pbf))
    assert list(frame.columns) == ["id", "park_ride", "geometry"]
    assert str(frame.crs).upper() == "EPSG:4326"
    assert len(frame) >= 1
    assert (frame.geometry.geom_type == "Point").all()
    assert (frame["park_ride"].str.lower() != "no").all()
    # The frame feeds the policy directly.
    policy = CarParkPolicy(facilities=frame)
    assert len(policy.facilities) == len(frame)


def test_the_extractor_semantics_on_controlled_content(kantakaupunki_pbf, tmp_path):
    import pyrosm

    osm = pyrosm.OSM(str(kantakaupunki_pbf))
    tagged = osm.get_data_by_custom_criteria(
        custom_filter={"park_ride": True}, filter_type="keep"
    )
    # Way-based elements round-trip pyrosm's subset writer
    # deterministically; the filtering semantics under test are
    # element-type independent. One real element per pinned case:
    # a plain "yes", a mode-specific value, and a "no".
    ways = tagged[tagged.geometry.geom_type != "Point"].sort_values("id")
    values = ways["park_ride"].astype(str).str.lower()
    yes_way = ways[values == "yes"].head(1)
    modal_way = ways[values.isin(["train", "metro", "bus", "tram"])].head(1)
    no_way = ways[values == "no"].head(1)
    assert len(yes_way) and len(modal_way) and len(no_way)
    plain = osm.get_data_by_custom_criteria(
        custom_filter={"amenity": ["parking"]}, filter_type="keep"
    )
    if "park_ride" in plain.columns:
        plain = plain[plain["park_ride"].isna()]
    # The tag columns differ per fetch, and Helsinki's park-and-ride
    # lots are also amenity=parking — exclude every tagged id so
    # "plain" really is plain.
    plain = plain[~plain["id"].isin(set(tagged["id"]))]
    plain = plain[plain.geometry.geom_type != "Point"].head(2)
    assert len(plain)
    subset = pd.concat([yes_way, modal_way, no_way, plain], ignore_index=False)
    staged = tmp_path / "mini.osm.pbf"
    osm.write_pbf(subset, str(staged), subset_only=True)
    frame = park_and_ride_facilities(str(staged))

    def qualified(rows):
        return set(rows["osm_type"].astype(str) + "/" + rows["id"].astype(str))

    # Exactly the qualifying elements survive: yes + the modal value;
    # the "no" facility and plain parking are absent.
    assert set(frame["id"]) == qualified(yes_way) | qualified(modal_way)
    assert not set(frame["id"]) & (qualified(no_way) | qualified(plain))
    assert (frame.geometry.geom_type == "Point").all()
    # An area facility's point lies inside its source ring.
    source = yes_way.iloc[0]
    key = f"{source['osm_type']}/{source['id']}"
    collapsed = frame[frame["id"] == key].geometry.iloc[0]
    assert source.geometry.contains(collapsed)


@pytest.fixture(scope="module")
def car_park_network(helsinki_gtfs, kantakaupunki_pbf):
    import warnings

    from cafein import TransportNetwork

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return TransportNetwork.from_gtfs(
            [str(helsinki_gtfs)],
            osm_pbf=str(kantakaupunki_pbf),
            street_modes=("walk", "car"),
            country="FI",
        )


PASILA = (60.1990, 24.9330)
# Zero stops within an 8-minute walk, but drivable.
FAR_ORIGIN = (60.1980, 24.9130)
DEPARTURE_TIME = "2022-02-22 08:30:00"


def _pasila_policy(**kwargs):
    frame = geopandas.GeoDataFrame(
        {"id": ["pasila"], "search_seconds": [240.0], "fee": [3.0]},
        geometry=[Point(PASILA[1], PASILA[0])],
        crs="EPSG:4326",
    )
    return CarParkPolicy(facilities=frame, **kwargs)


def test_the_composed_table_is_the_hand_minimum(car_park_network):
    from cafein.network import _car_park_offsets, _walk_options
    from cafein.street_network import _resolved_delays

    core = car_park_network._core
    second = (60.2010, 24.9420)
    frame = geopandas.GeoDataFrame(
        {"id": ["pasila", "haka"], "search_seconds": [240.0, 60.0]},
        geometry=[Point(PASILA[1], PASILA[0]), Point(second[1], second[0])],
        crs="EPSG:4326",
    )
    policy = CarParkPolicy(facilities=frame)
    best = _car_park_offsets(
        core, FAR_ORIGIN, policy, _walk_options(None, None, None), False
    )
    assert best
    model = _resolved_delays("car", True, None, None)
    drives = core._car_park_drive_seconds(
        FAR_ORIGIN[0], FAR_ORIGIN[1], [PASILA, second], model, 1800.0, 500.0, False
    )
    parks = [240, 60]
    walks = [
        core.access_stops(lat, lon, 3.6, 600.0, 1600.0) for lat, lon in (PASILA, second)
    ]
    winners = 0
    for stop, (total, token) in best.items():
        # The winner is the true per-stop minimum over BOTH facilities.
        candidates = {
            position: int(drives[position][0]) + parks[position] + int(w[stop])
            for position, w in enumerate(walks)
            if drives[position] is not None and stop in w
        }
        assert total == min(candidates.values())
        assert candidates[token[0]] == total
        if token[0] == 1:
            winners += 1
    # Each facility wins somewhere: the minimum is real, not a copy
    # of one facility's table.
    assert winners and winners < len(best)
    # End to end: the public route's chain carries the hand-selected
    # winner for the boarding stop it actually uses.
    journeys = car_park_network.route_between_coordinates(
        FAR_ORIGIN,
        (60.1795, 24.9520),
        DEPARTURE_TIME,
        street_policy=policy,
        max_walking_time=8,
    )
    ridden = next(j for j in journeys if j["rides"] >= 1)
    chain = ridden["legs"]
    assert chain[0]["mode"] == "car_park"
    boarding = chain[2]["to_stop"]
    expected_position = best[boarding][1][0]
    assert chain[0]["facility_id"] == frame["id"].iloc[expected_position]


def test_a_facility_exactly_at_a_stop_prices_like_every_walk(car_park_network):
    # A facility placed on a stop's own coordinate walks the stop's
    # snap connectors, exactly as the ordinary walking plane prices
    # that position — search and rebuild agree, nothing refuses.
    core = car_park_network._core
    stops = car_park_network.access_stops(*PASILA, max_walking_time=8)
    sid = min(stops, key=stops.get)
    row = car_park_network.stops_gdf.set_index("id").loc[sid]
    lat, lon = row.geometry.y, row.geometry.x
    shape = core._car_park_walk_leg(lat, lon, sid, 1600.0, False)
    assert shape is not None
    network_m, connector_m, _ = shape
    assert network_m == 0.0
    seconds = core.access_stops(lat, lon, 3.6, 600.0, 1600.0)[sid]
    # 3.6 km/h is one metre per second: the rebuilt metres price to
    # the searched seconds within rounding.
    assert abs((network_m + connector_m) - float(seconds)) <= 2.0
    frame = geopandas.GeoDataFrame(
        {"id": ["at-stop"]}, geometry=[Point(lon, lat)], crs="EPSG:4326"
    )
    journeys = car_park_network.route_between_coordinates(
        FAR_ORIGIN,
        (60.1795, 24.9520),
        DEPARTURE_TIME,
        street_policy=CarParkPolicy(facilities=frame),
        max_walking_time=8,
    )
    ridden = next(j for j in journeys if j["rides"] >= 1)
    chain = ridden["legs"]
    assert chain[0]["mode"] == "car_park"
    assert chain[1]["type"] == "park"
    walk = chain[2]
    assert walk["mode"] == "walk"
    if walk["to_stop"] == sid:
        assert walk["network_distance_m"] == 0.0
        assert walk["distance_m"] == connector_m


def test_the_car_chain_carries_when_walking_cannot(car_park_network):
    policy = _pasila_policy()
    journeys = car_park_network.route_between_coordinates(
        FAR_ORIGIN,
        (60.1795, 24.9520),
        DEPARTURE_TIME,
        street_policy=policy,
        max_walking_time=8,
    )
    assert journeys
    ridden = [j for j in journeys if j["rides"] >= 1]
    assert ridden
    legs = ridden[0]["legs"]
    kinds = [(leg["type"], leg.get("mode")) for leg in legs[:3]]
    assert kinds == [
        ("access", "car_park"),
        ("park", None),
        ("access", "walk"),
    ]
    assert legs[0]["facility_id"] == "pasila"
    assert legs[1]["facility_id"] == "pasila"
    # The access chain is contiguous; the wait sits at the stop, as
    # on every transit journey.
    assert legs[0]["arrival_s"] == legs[1]["departure_s"]
    assert legs[1]["arrival_s"] == legs[2]["departure_s"]
    assert legs[1]["arrival_s"] - legs[1]["departure_s"] == 240
    transit = next(leg for leg in legs if leg["type"] == "transit")
    assert legs[2]["arrival_s"] <= transit["departure_s"]
    assert legs[0]["geometry"] is not None
    assert legs[0]["network_distance_m"] > 1000


def test_the_walk_still_wins_nearby(car_park_network):
    policy = _pasila_policy()
    journeys = car_park_network.route_between_coordinates(
        (60.1690, 24.9320),
        (60.1795, 24.9520),
        DEPARTURE_TIME,
        street_policy=policy,
    )
    assert journeys
    for journey in journeys:
        assert all(leg.get("mode") != "car_park" for leg in journey["legs"])


def test_travel_times_seed_from_the_same_table(car_park_network):
    from cafein.network import _car_park_table, _walk_options

    policy = _pasila_policy()
    times = car_park_network.travel_times_from_coordinate(
        FAR_ORIGIN, DEPARTURE_TIME, street_policy=policy, max_walking_time=8
    )
    assert times
    offsets, tokens, _walking = _car_park_table(
        car_park_network._core,
        FAR_ORIGIN,
        policy,
        _walk_options(None, 480.0, None),
        False,
    )
    assert tokens
    seeded = dict(offsets)
    departed = 8 * 3600 + 30 * 60
    proven = [stop for stop in tokens if stop in times]
    assert proven, "no car-seeded stop appears in the travel times"
    for stop in proven:
        # Seeding bounds the arrival: the engine can only improve on
        # walking in at the composed seconds.
        assert times[stop] <= departed + seeded[stop]


def test_pricing_goldens(car_park_network):
    from cafein import DetailedItineraries, costs

    policy = _pasila_policy(vehicle_class="ICE", occupancy=2.0)
    origins = geopandas.GeoDataFrame(
        {"id": ["o"]}, geometry=[Point(FAR_ORIGIN[1], FAR_ORIGIN[0])], crs="EPSG:4326"
    )
    destinations = geopandas.GeoDataFrame(
        {"id": ["d"]}, geometry=[Point(24.9520, 60.1795)], crs="EPSG:4326"
    )
    frame = DetailedItineraries(
        car_park_network,
        origins,
        destinations,
        DEPARTURE_TIME,
        street_policy=policy,
        max_walking_time=8,
        perspectives=("private",),
    )
    drives = frame[frame["mode"] == "car_park"]
    assert len(drives)
    kilometres = drives["network_distance_m"].iloc[0] / 1000.0
    # ICE per-vehicle grams (24 + 126 + 12), divided by the occupancy.
    assert drives["emissions"].iloc[0] == pytest.approx(
        kilometres * 162.0 / 2.0, rel=1e-9
    )
    totals, _breakdown, label = costs.resolve_query(
        "car", ("private",), None, None, None
    )
    assert drives["cost_private"].iloc[0] == pytest.approx(
        kilometres * totals["private"], rel=1e-9
    )
    assert (drives["currency"] == label).all()
    parks = frame[frame["leg_type"] == "park"]
    assert parks["cost_private"].iloc[0] == pytest.approx(3.0)
    assert parks["facility_id"].iloc[0] == "pasila"
    assert "bike_aboard" not in frame.columns
    # Every shipped GEMMAT class prices its own per-vehicle factor.
    class_totals = {
        "ICE": 162.0,
        "HEV": 133.0,
        "PHEV": 88.0,
        "BEV": 70.0,
        "FCEV": 134.0,
    }
    for vehicle_class, grams_per_km in class_totals.items():
        classed = DetailedItineraries(
            car_park_network,
            origins,
            destinations,
            DEPARTURE_TIME,
            street_policy=_pasila_policy(vehicle_class=vehicle_class, occupancy=2.0),
            max_walking_time=8,
        )
        rows = classed[classed["mode"] == "car_park"]
        km = rows["network_distance_m"].iloc[0] / 1000.0
        assert rows["emissions"].iloc[0] == pytest.approx(
            km * grams_per_km / 2.0, rel=1e-9
        ), vehicle_class
    # facility_id keeps object dtype even with default integer ids.
    plain_frame = geopandas.GeoDataFrame(
        {"search_seconds": [240.0]},
        geometry=[Point(PASILA[1], PASILA[0])],
        crs="EPSG:4326",
    )
    numbered = DetailedItineraries(
        car_park_network,
        origins,
        destinations,
        DEPARTURE_TIME,
        street_policy=CarParkPolicy(facilities=plain_frame),
        max_walking_time=8,
    )
    assert numbered["facility_id"].dtype == object
    assert 0 in set(numbered["facility_id"].dropna())
    # Occupancy scales emissions only.
    solo = DetailedItineraries(
        car_park_network,
        origins,
        destinations,
        DEPARTURE_TIME,
        street_policy=_pasila_policy(vehicle_class="ICE", occupancy=1.0),
        max_walking_time=8,
        perspectives=("private",),
    )
    solo_drives = solo[solo["mode"] == "car_park"]
    assert solo_drives["emissions"].iloc[0] == pytest.approx(
        2.0 * drives["emissions"].iloc[0], rel=1e-9
    )
    assert solo_drives["cost_private"].iloc[0] == pytest.approx(
        drives["cost_private"].iloc[0], rel=1e-9
    )


def test_facility_semantics_and_refusals(car_park_network):
    # A facility in open water never snaps; the drivable one serves.
    mixed = geopandas.GeoDataFrame(
        {"id": ["sea", "pasila"]},
        geometry=[Point(24.80, 60.10), Point(PASILA[1], PASILA[0])],
        crs="EPSG:4326",
    )
    journeys = car_park_network.route_between_coordinates(
        FAR_ORIGIN,
        (60.1795, 24.9520),
        DEPARTURE_TIME,
        street_policy=CarParkPolicy(facilities=mixed),
        max_walking_time=8,
    )
    carried = [
        leg
        for journey in journeys
        for leg in journey["legs"]
        if leg.get("mode") == "car_park"
    ]
    assert carried and all(leg["facility_id"] == "pasila" for leg in carried)
    # Every facility unreachable refuses rather than silently walking.
    at_sea = geopandas.GeoDataFrame(
        {"id": ["sea"]}, geometry=[Point(24.80, 60.10)], crs="EPSG:4326"
    )
    with pytest.raises(ValueError, match="no facility is reachable by car"):
        car_park_network.route_between_coordinates(
            FAR_ORIGIN,
            (60.1795, 24.9520),
            DEPARTURE_TIME,
            street_policy=CarParkPolicy(facilities=at_sea),
        )
    policy = _pasila_policy()
    with pytest.raises(ValueError, match="departure or arrival window"):
        car_park_network.route_between_coordinates(
            FAR_ORIGIN,
            (60.1795, 24.9520),
            DEPARTURE_TIME,
            street_policy=policy,
            departure_time_window=30,
        )
    with pytest.raises(ValueError, match="stop exclusions"):
        car_park_network.route_between_coordinates(
            FAR_ORIGIN,
            (60.1795, 24.9520),
            DEPARTURE_TIME,
            street_policy=policy,
            exclude_stops=["1000202"],
        )
    with pytest.raises(ValueError, match="departure or arrival window"):
        car_park_network.route_between_coordinates(
            FAR_ORIGIN,
            (60.1795, 24.9520),
            arrival="2022-02-22 09:30:00",
            arrival_time_window=30,
            street_policy=policy,
        )
    # The arrive-by travel-times form has stop origins, so the
    # access-only car plane has no origin to drive from.
    with pytest.raises(ValueError, match="origins are the stops"):
        car_park_network.travel_times_from_coordinate(
            FAR_ORIGIN,
            arrival="2022-02-22 09:30:00",
            street_policy=policy,
        )
    from cafein import DetailedItineraries

    origins = geopandas.GeoDataFrame(
        {"id": ["o"]}, geometry=[Point(FAR_ORIGIN[1], FAR_ORIGIN[0])], crs="EPSG:4326"
    )
    destinations = geopandas.GeoDataFrame(
        {"id": ["d"]}, geometry=[Point(24.9520, 60.1795)], crs="EPSG:4326"
    )
    with pytest.raises(ValueError, match="time-optimal arm"):
        DetailedItineraries(
            car_park_network,
            origins,
            destinations,
            DEPARTURE_TIME,
            street_policy=policy,
            candidates="pareto",
        )
    with pytest.raises(ValueError, match="stays 'auto'"):
        DetailedItineraries(
            car_park_network,
            origins,
            destinations,
            DEPARTURE_TIME,
            street_policy=policy,
            router="raptor",
        )


@pytest.mark.parametrize("axis", ["departure", "arrival"])
def test_the_time_matrix_cells_reconcile_with_the_route(car_park_network, axis):
    from cafein import TravelTimeMatrix

    policy = _pasila_policy()
    origins = geopandas.GeoDataFrame(
        {"id": ["far", "near"]},
        geometry=[Point(FAR_ORIGIN[1], FAR_ORIGIN[0]), Point(24.9320, 60.1690)],
        crs="EPSG:4326",
    )
    destinations = geopandas.GeoDataFrame(
        {"id": ["d1", "d2"]},
        geometry=[Point(24.9520, 60.1795), Point(24.9500, 60.1841)],
        crs="EPSG:4326",
    )
    when = (
        {"departure": DEPARTURE_TIME}
        if axis == "departure"
        else {"arrival": "2022-02-22 09:30:00"}
    )
    frame = TravelTimeMatrix(
        car_park_network,
        origins,
        destinations,
        street_policy=policy,
        max_walking_time=8,
        output_time_units="seconds",
        **when,
    )
    cells = {(row.from_id, row.to_id): row.travel_time for row in frame.itertuples()}
    coordinates = {"far": FAR_ORIGIN, "near": (60.1690, 24.9320)}
    targets = {"d1": (60.1795, 24.9520), "d2": (60.1841, 24.9500)}
    for from_id, origin in coordinates.items():
        for to_id, destination in targets.items():
            journeys = car_park_network.route_between_coordinates(
                origin,
                destination,
                street_policy=policy,
                max_walking_time=8,
                **when,
            )
            if axis == "departure":
                best = min(j["arrival_s"] - j["departure_s"] for j in journeys)
            else:
                # Arrive-by elects the latest departure: the leading
                # journey is the cell's answer.
                best = journeys[0]["arrival_s"] - journeys[0]["departure_s"]
            cell = cells[(from_id, to_id)]
            if from_id == "far":
                # Walking cannot serve within the budget: purely the
                # engine run (the reverse election under a deadline) on
                # both sides, so the cell matches exactly.
                assert cell == best
            else:
                # A walking-won cell may differ by the two folds'
                # rounding conventions.
                assert abs(cell - best) <= 1


def test_the_cost_matrix_carries_the_car_chain(car_park_network):
    from cafein import DetailedItineraries, TravelCostMatrix

    policy = _pasila_policy(vehicle_class="ICE", occupancy=2.0)
    origins = geopandas.GeoDataFrame(
        {"id": ["o"]}, geometry=[Point(FAR_ORIGIN[1], FAR_ORIGIN[0])], crs="EPSG:4326"
    )
    destinations = geopandas.GeoDataFrame(
        {"id": ["d"]}, geometry=[Point(24.9520, 60.1795)], crs="EPSG:4326"
    )
    frame = TravelCostMatrix(
        car_park_network,
        origins,
        destinations,
        DEPARTURE_TIME,
        street_policy=policy,
        max_walking_time=8,
        output_time_units="seconds",
    )
    assert list(frame["from_id"]) == ["o"] and list(frame["to_id"]) == ["d"]
    assert "facility_id" not in frame.columns
    itinerary = DetailedItineraries(
        car_park_network,
        origins,
        destinations,
        DEPARTURE_TIME,
        street_policy=policy,
        max_walking_time=8,
    )
    # The itineraries enumerate ride-count options; the matrix cell is
    # the time-fastest of them.
    durations = (
        itinerary.groupby("option")["arrival_s"].max()
        - itinerary.groupby("option")["departure_s"].min()
    )
    journey = itinerary[itinerary["option"] == durations.idxmin()]
    drive = journey[journey["mode"] == "car_park"]
    kilometres = drive["network_distance_m"].iloc[0] / 1000.0
    # The cell's grams are the whole journey's — the transit legs plus
    # the drive priced as ICE per-vehicle over the occupancy — exactly
    # as the itineraries price them.
    assert frame["emissions"].iloc[0] == pytest.approx(
        float(journey["emissions"].sum()), rel=1e-9
    )
    assert float(journey["emissions"].sum()) > kilometres * 162.0 / 2.0
    assert frame["street_distance_m"].iloc[0] == pytest.approx(
        drive["distance_m"].iloc[0], rel=1e-6
    )
    assert frame["fee"].iloc[0] == pytest.approx(3.0)
    walks = journey[journey["mode"] == "walk"]
    assert frame["walk_distance_m"].iloc[0] == pytest.approx(
        walks["distance_m"].sum(), rel=1e-6
    )
    assert frame["travel_time"].iloc[0] == int(durations.min())


def test_a_walk_won_cell_prices_zero(car_park_network):
    from cafein import TravelCostMatrix

    policy = _pasila_policy()
    origins = geopandas.GeoDataFrame(
        {"id": ["near"]}, geometry=[Point(24.9320, 60.1690)], crs="EPSG:4326"
    )
    destinations = geopandas.GeoDataFrame(
        {"id": ["close"]}, geometry=[Point(24.9316, 60.1688)], crs="EPSG:4326"
    )
    frame = TravelCostMatrix(
        car_park_network,
        origins,
        destinations,
        DEPARTURE_TIME,
        street_policy=policy,
        output_time_units="seconds",
    )
    row = frame.iloc[0]
    assert row["transfers"] == 0
    assert row["emissions"] == 0.0
    assert row["street_distance_m"] == 0.0
    assert row["fee"] == 0.0
    assert row["walk_distance_m"] > 0.0


def test_accessibility_scores_through_the_car_plane(car_park_network):
    from cafein import Accessibility

    policy = _pasila_policy()
    origins = geopandas.GeoDataFrame(
        {"id": ["far"]}, geometry=[Point(FAR_ORIGIN[1], FAR_ORIGIN[0])], crs="EPSG:4326"
    )
    destinations = geopandas.GeoDataFrame(
        {"id": ["a", "b"], "jobs": [100.0, 40.0]},
        geometry=[Point(24.9520, 60.1795), Point(24.9316, 60.1688)],
        crs="EPSG:4326",
    )
    # Both time axes: departure and the arrive-by deadline.
    for when in (
        {"departure": DEPARTURE_TIME},
        {"arrival": "2022-02-22 09:30:00"},
    ):
        walked = Accessibility(
            car_park_network,
            origins,
            destinations,
            opportunities="jobs",
            budgets=(45.0,),
            max_walking_time=8,
            **when,
        )
        # Beyond every stop's walking reach, the origin scores nothing.
        assert float(walked["accessibility"].sum()) == 0.0
        driven = Accessibility(
            car_park_network,
            origins,
            destinations,
            opportunities="jobs",
            budgets=(45.0,),
            max_walking_time=8,
            street_policy=policy,
            **when,
        )
        # The composed car table demonstrably feeds the score.
        assert float(driven["accessibility"].sum()) > 0.0
    # A walking-reachable origin can only gain from the second plane.
    near = geopandas.GeoDataFrame(
        {"id": ["near"]}, geometry=[Point(24.9320, 60.1690)], crs="EPSG:4326"
    )
    base = Accessibility(
        car_park_network,
        near,
        destinations,
        DEPARTURE_TIME,
        opportunities="jobs",
        budgets=(45.0,),
    )
    with_policy = Accessibility(
        car_park_network,
        near,
        destinations,
        DEPARTURE_TIME,
        opportunities="jobs",
        budgets=(45.0,),
        street_policy=policy,
    )
    assert float(with_policy["accessibility"].sum()) >= float(
        base["accessibility"].sum()
    )
    # With the car side present, stop-id inputs refuse on their shape.
    from cafein import Accessibility as _Accessibility

    with pytest.raises(ValueError, match="point-set origins"):
        _Accessibility(
            car_park_network,
            list(car_park_network.stops_gdf["id"].iloc[:1]),
            list(car_park_network.stops_gdf["id"].iloc[:1]),
            DEPARTURE_TIME,
            street_policy=policy,
        )


def test_cost_geometry_stays_multiline_when_the_walk_wins(car_park_network):
    from cafein import TravelCostMatrix

    policy = _pasila_policy()
    origins = geopandas.GeoDataFrame(
        {"id": ["far", "near"]},
        geometry=[Point(FAR_ORIGIN[1], FAR_ORIGIN[0]), Point(24.9320, 60.1690)],
        crs="EPSG:4326",
    )
    destinations = geopandas.GeoDataFrame(
        {"id": ["close"]}, geometry=[Point(24.9316, 60.1688)], crs="EPSG:4326"
    )
    frame = TravelCostMatrix(
        car_park_network,
        origins,
        destinations,
        DEPARTURE_TIME,
        street_policy=policy,
        geometries=True,
        output_time_units="seconds",
    )
    # Ridden and walking-won rows alike keep the cost matrix's
    # MultiLineString contract.
    kinds = {
        geometry.geom_type for geometry in frame["geometry"] if geometry is not None
    }
    assert kinds == {"MultiLineString"}
    walked = frame[(frame["from_id"] == "near") & (frame["transfers"] == 0)]
    assert len(walked) and walked["geometry"].iloc[0] is not None


def test_an_undrivable_origin_is_omitted_with_a_warning(car_park_network):
    from cafein import TravelTimeMatrix

    policy = _pasila_policy()
    origins = geopandas.GeoDataFrame(
        {"id": ["sea", "far"]},
        geometry=[Point(24.8000, 60.1200), Point(FAR_ORIGIN[1], FAR_ORIGIN[0])],
        crs="EPSG:4326",
    )
    destinations = geopandas.GeoDataFrame(
        {"id": ["d"]}, geometry=[Point(24.9520, 60.1795)], crs="EPSG:4326"
    )
    with pytest.warns(UserWarning, match="cannot reach any facility by car"):
        frame = TravelTimeMatrix(
            car_park_network,
            origins,
            destinations,
            DEPARTURE_TIME,
            street_policy=policy,
            max_walking_time=8,
            output_time_units="seconds",
        )
    # The undrivable origin's cells are omitted, never silently walked;
    # the drivable origin still serves.
    assert set(frame["from_id"]) == {"far"}
    # Stop-id forms refuse: the facilities plane needs coordinates.
    with pytest.raises(ValueError, match="point-set origins"):
        TravelTimeMatrix(
            car_park_network,
            list(car_park_network.stops_gdf["id"].iloc[:1]),
            None,
            DEPARTURE_TIME,
            street_policy=policy,
        )


ARRIVAL_TIME = "2022-02-22 09:30:00"
DEADLINE_S = 9 * 3600 + 30 * 60


def _clock(seconds):
    return f"2022-02-22 {seconds // 3600:02d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


def test_arrive_by_journeys_ride_the_car_plane(car_park_network):
    policy = _pasila_policy()
    journeys = car_park_network.route_between_coordinates(
        FAR_ORIGIN,
        (60.1795, 24.9520),
        arrival=ARRIVAL_TIME,
        street_policy=policy,
        max_walking_time=8,
    )
    assert journeys
    best = journeys[0]
    # The winner arrives by the deadline through the full car chain.
    assert best["arrival_s"] <= DEADLINE_S
    assert best["legs"][0]["mode"] == "car_park"
    assert best["legs"][0]["facility_id"] == "pasila"
    assert best["legs"][1]["type"] == "park"
    assert best["legs"][2]["mode"] == "walk"
    # Latest departure first, and each journey is a genuine departure
    # answer: re-routed forward at its own departure, it appears
    # unchanged — the reverse inversion contract.
    departures = [journey["departure_s"] for journey in journeys]
    assert departures == sorted(departures, reverse=True)
    for journey in journeys:
        forward = car_park_network.route_between_coordinates(
            FAR_ORIGIN,
            (60.1795, 24.9520),
            _clock(journey["departure_s"]),
            street_policy=policy,
            max_walking_time=8,
        )
        matched = [
            candidate
            for candidate in forward
            if candidate["departure_s"] == journey["departure_s"]
            and candidate["arrival_s"] == journey["arrival_s"]
            and candidate["rides"] == journey["rides"]
        ]
        assert matched, "the arrive-by journey is not a departure answer"


def test_the_walk_still_wins_nearby_under_the_deadline(car_park_network):
    policy = _pasila_policy()
    journeys = car_park_network.route_between_coordinates(
        (60.1690, 24.9320),
        (60.1688, 24.9316),
        arrival=ARRIVAL_TIME,
        street_policy=policy,
    )
    best = journeys[0]
    # A few metres away, walking wins the latest departure and is
    # placed to arrive exactly at the deadline.
    assert best["rides"] == 0
    assert best["legs"][0]["mode"] == "walk"
    assert best["arrival_s"] == DEADLINE_S


def test_the_arrive_by_cost_matrix_carries_the_car_chain(car_park_network):
    from cafein import DetailedItineraries, TravelCostMatrix

    # Two viable facilities with distinct fees: the cell must carry
    # the fee of the facility the ELECTED journey drove to.
    facilities = geopandas.GeoDataFrame(
        {"id": ["pasila", "haka"], "search_seconds": [240.0, 60.0], "fee": [3.0, 1.5]},
        geometry=[Point(PASILA[1], PASILA[0]), Point(24.9420, 60.2010)],
        crs="EPSG:4326",
    )
    policy = CarParkPolicy(facilities=facilities, vehicle_class="ICE", occupancy=2.0)
    origins = geopandas.GeoDataFrame(
        {"id": ["o"]}, geometry=[Point(FAR_ORIGIN[1], FAR_ORIGIN[0])], crs="EPSG:4326"
    )
    destinations = geopandas.GeoDataFrame(
        {"id": ["d"]}, geometry=[Point(24.9520, 60.1795)], crs="EPSG:4326"
    )
    frame = TravelCostMatrix(
        car_park_network,
        origins,
        destinations,
        arrival=ARRIVAL_TIME,
        street_policy=policy,
        max_walking_time=8,
        output_time_units="seconds",
    )
    assert list(frame["from_id"]) == ["o"] and list(frame["to_id"]) == ["d"]
    itinerary = DetailedItineraries(
        car_park_network,
        origins,
        destinations,
        arrival=ARRIVAL_TIME,
        street_policy=policy,
        max_walking_time=8,
    )
    # The cell is the complete-journey winner — the LATEST departure,
    # the itineraries' leading option — never a faster-but-earlier or
    # more-ridden substitution.
    starts = itinerary.groupby("option")["departure_s"].min()
    journey = itinerary[itinerary["option"] == starts.idxmax()]
    drive = journey[journey["mode"] == "car_park"]
    fees = {"pasila": 3.0, "haka": 1.5}
    assert frame["fee"].iloc[0] == pytest.approx(fees[drive["facility_id"].iloc[0]])
    assert frame["street_distance_m"].iloc[0] == pytest.approx(
        drive["distance_m"].iloc[0], rel=1e-6
    )
    assert frame["emissions"].iloc[0] == pytest.approx(
        float(journey["emissions"].sum()), rel=1e-9
    )
    duration = int(journey["arrival_s"].max() - journey["departure_s"].min())
    assert frame["travel_time"].iloc[0] == duration


def test_a_tied_walk_beats_the_through_stop_car_cell(car_park_network):
    from cafein import TravelCostMatrix
    from cafein.street_network import _resolved_delays

    core = car_park_network._core
    destination = (60.1975, 24.9350)
    # Engineer an exact tie from the fixture's own measured pieces:
    # drive + park + best through-stop walks == the direct walk.
    walk_seconds = int(
        core._walk_matrix([FAR_ORIGIN], [destination], 3.6, 7200.0, 1600.0)[0][0][0]
    )
    model = _resolved_delays("car", True, None, None)
    drive_s = int(
        core._car_park_drive_seconds(
            FAR_ORIGIN[0], FAR_ORIGIN[1], [PASILA], model, 1800.0, 500.0, False
        )[0][0]
    )
    access = core.access_stops(PASILA[0], PASILA[1], 3.6, 600.0, 1600.0)
    egress = core.access_stops(destination[0], destination[1], 3.6, 7200.0, 1600.0)
    through = min(
        int(access[stop]) + int(egress[stop]) for stop in access if stop in egress
    )
    park = walk_seconds - drive_s - through
    assert park >= 2, "the fixture geometry no longer forms a tie"
    all_routes = [route for route, _agency, _kind in car_park_network.routes]
    origins = geopandas.GeoDataFrame(
        {"id": ["far"]}, geometry=[Point(FAR_ORIGIN[1], FAR_ORIGIN[0])], crs="EPSG:4326"
    )
    destinations = geopandas.GeoDataFrame(
        {"id": ["d"]},
        geometry=[Point(destination[1], destination[0])],
        crs="EPSG:4326",
    )

    def cell(search_seconds):
        facilities = geopandas.GeoDataFrame(
            {"id": ["pasila"], "search_seconds": [float(search_seconds)], "fee": [5.0]},
            geometry=[Point(PASILA[1], PASILA[0])],
            crs="EPSG:4326",
        )
        frame = TravelCostMatrix(
            car_park_network,
            origins,
            destinations,
            arrival=ARRIVAL_TIME,
            street_policy=CarParkPolicy(facilities=facilities),
            exclude_routes=all_routes,
            output_time_units="seconds",
        )
        return frame.iloc[0]

    # An exact tie: the walk wins, as on the route surface — no fee,
    # no metres, no grams.
    tied = cell(park)
    assert tied["travel_time"] == walk_seconds
    assert tied["fee"] == 0.0
    assert tied["street_distance_m"] == 0.0
    assert tied["emissions"] == 0.0
    # One second less parking departs strictly later: the car
    # through-chain wins the cell, carrying its fee and grams.
    later = cell(park - 1)
    assert later["travel_time"] == walk_seconds - 1
    assert later["fee"] == pytest.approx(5.0)
    assert later["street_distance_m"] > 0.0
    assert later["emissions"] > 0.0


def test_facility_ties_break_in_declared_order(car_park_network):
    from cafein.network import _car_park_offsets, _walk_options

    # Two facilities at one coordinate with equal search times compose
    # to equal totals at every stop: the declared-first facility wins.
    frame = geopandas.GeoDataFrame(
        {"id": ["first", "second"], "search_seconds": [240.0, 240.0]},
        geometry=[Point(PASILA[1], PASILA[0]), Point(PASILA[1], PASILA[0])],
        crs="EPSG:4326",
    )
    best = _car_park_offsets(
        car_park_network._core,
        FAR_ORIGIN,
        CarParkPolicy(facilities=frame),
        _walk_options(None, None, None),
        False,
    )
    assert best
    assert {token[0] for _total, token in best.values()} == {0}


def test_the_zero_ride_car_chain_serves_the_route(car_park_network):
    from cafein import TravelTimeMatrix

    policy = _pasila_policy()
    destination = (60.1975, 24.9350)
    all_routes = [route for route, _agency, _kind in car_park_network.routes]
    origins = geopandas.GeoDataFrame(
        {"id": ["far"]}, geometry=[Point(FAR_ORIGIN[1], FAR_ORIGIN[0])], crs="EPSG:4326"
    )
    destinations = geopandas.GeoDataFrame(
        {"id": ["d"]},
        geometry=[Point(destination[1], destination[0])],
        crs="EPSG:4326",
    )
    # With transit excluded and walking beyond its budget, the
    # zero-ride car chain — drive, park, walk in, walk out — is the
    # journey, and the matrix cell matches it.
    journeys = car_park_network.route_between_coordinates(
        FAR_ORIGIN,
        destination,
        DEPARTURE_TIME,
        street_policy=policy,
        exclude_routes=all_routes,
        max_walking_time=8,
    )
    assert len(journeys) == 1
    chain = journeys[0]
    assert chain["rides"] == 0
    assert [leg.get("mode") for leg in chain["legs"]] == [
        "car_park",
        None,
        "walk",
        "walk",
    ]
    duration = chain["arrival_s"] - chain["departure_s"]
    frame = TravelTimeMatrix(
        car_park_network,
        origins,
        destinations,
        DEPARTURE_TIME,
        street_policy=policy,
        exclude_routes=all_routes,
        max_walking_time=8,
        output_time_units="seconds",
    )
    assert int(frame["travel_time"].iloc[0]) == duration
    # The same chain under the deadline, departing as late as it can.
    reverse = car_park_network.route_between_coordinates(
        FAR_ORIGIN,
        destination,
        arrival=ARRIVAL_TIME,
        street_policy=policy,
        exclude_routes=all_routes,
        max_walking_time=8,
    )
    assert reverse[0]["rides"] == 0
    assert reverse[0]["arrival_s"] == DEADLINE_S
    assert reverse[0]["arrival_s"] - reverse[0]["departure_s"] == duration
    back = TravelTimeMatrix(
        car_park_network,
        origins,
        destinations,
        arrival=ARRIVAL_TIME,
        street_policy=policy,
        exclude_routes=all_routes,
        max_walking_time=8,
        output_time_units="seconds",
    )
    assert int(back["travel_time"].iloc[0]) == duration


@pytest.mark.parametrize("computer", ["time", "cost"])
def test_the_car_park_matrices_serve_slots_on_both_axes(car_park_network, computer):
    from cafein import TravelCostMatrix, TravelTimeMatrix

    if computer == "time":
        cls = TravelTimeMatrix
        policy = _pasila_policy()
        origin_id, destination_id = "far", "d1"
        extra = {"max_walking_time": 8}
    else:
        cls = TravelCostMatrix
        policy = _pasila_policy(vehicle_class="ICE", occupancy=2.0)
        origin_id, destination_id = "o", "d"
        extra = {}
    origins = geopandas.GeoDataFrame(
        {"id": [origin_id]},
        geometry=[Point(FAR_ORIGIN[1], FAR_ORIGIN[0])],
        crs="EPSG:4326",
    )
    destinations = geopandas.GeoDataFrame(
        {"id": [destination_id]}, geometry=[Point(24.9520, 60.1795)], crs="EPSG:4326"
    )
    for axis, moments in (
        ("departure", ["2022-02-22 08:30:00", "2022-02-22 12:00:00"]),
        ("arrival", ["2022-02-22 09:30:00", "2022-02-22 13:00:00"]),
    ):
        column = f"{axis}_time"
        frame = cls(
            car_park_network,
            origins,
            destinations,
            street_policy=policy,
            **{axis: moments},
            **extra,
        )
        for moment in moments:
            block = frame[frame[column] == moment].drop(columns=column)
            single = cls(
                car_park_network,
                origins,
                destinations,
                street_policy=policy,
                **{axis: moment},
                **extra,
            )
            pd.testing.assert_frame_equal(
                block.reset_index(drop=True), pd.DataFrame(single), check_dtype=False
            )
