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


def test_facilities_validation_refusals():
    with pytest.raises(ValueError, match="must be a GeoDataFrame"):
        CarParkPolicy(facilities=pd.DataFrame({"id": [1]}))
    with pytest.raises(ValueError, match="names no park-and-ride"):
        CarParkPolicy(facilities=_facilities().iloc[:0])
    bare = _facilities()
    bare.crs = None
    with pytest.raises(ValueError, match="must carry a CRS"):
        CarParkPolicy(facilities=bare)
    square = geopandas.GeoDataFrame(
        geometry=[Polygon([(24.9, 60.1), (24.9, 60.2), (25.0, 60.2)])],
        crs="EPSG:4326",
    )
    with pytest.raises(ValueError, match="representative_point"):
        CarParkPolicy(facilities=square)
    with pytest.raises(ValueError, match="duplicates"):
        CarParkPolicy(facilities=_facilities(id=["p", "p"]))
    with pytest.raises(ValueError, match="missing values"):
        CarParkPolicy(facilities=_facilities(id=["p", None]))
    with pytest.raises(ValueError, match="finite and non-negative"):
        CarParkPolicy(facilities=_facilities(search_seconds=[-1.0, 10.0]))
    with pytest.raises(ValueError, match="EUR2017"):
        CarParkPolicy(facilities=_facilities(fee=[-2.0, 0.0]))
    with pytest.raises(ValueError, match="non-numeric values"):
        CarParkPolicy(facilities=_facilities(search_seconds=["fast", 10.0]))
    with pytest.raises(ValueError, match="non-numeric values"):
        CarParkPolicy(facilities=_facilities(fee=["free", 0.0]))


def test_parameter_validation_refusals():
    facilities = _facilities()
    with pytest.raises(ValueError, match="positive time budget"):
        CarParkPolicy(facilities=facilities, max_car_time=0)
    with pytest.raises(ValueError, match="non-negative, finite duration"):
        CarParkPolicy(facilities=facilities, max_facility_walk_time=-5)
    with pytest.raises(TypeError, match="minutes"):
        CarParkPolicy(facilities=facilities, max_car_time="30")
    with pytest.raises(ValueError, match="at least 1"):
        CarParkPolicy(facilities=facilities, occupancy=0.5)
    with pytest.raises(ValueError, match="emission-factor row"):
        CarParkPolicy(facilities=facilities, vehicle_class=3)


def test_the_unwired_surfaces_still_refuse(network):
    from cafein import TravelCostMatrix, TravelTimeMatrix

    policy = CarParkPolicy(facilities=_facilities())
    origins = geopandas.GeoDataFrame({"id": ["a"]}, geometry=[KAMPPI], crs="EPSG:4326")
    destinations = geopandas.GeoDataFrame(
        {"id": ["b"]}, geometry=[HAKANIEMI], crs="EPSG:4326"
    )
    with pytest.raises(NotImplementedError, match="the matrix computers"):
        TravelCostMatrix(
            network,
            origins,
            destinations,
            "2022-02-22 08:30:00",
            street_policy=policy,
        )
    with pytest.raises(NotImplementedError, match="the matrix computers"):
        TravelTimeMatrix(
            network,
            origins,
            destinations,
            "2022-02-22 08:30:00",
            street_policy=policy,
        )
    # Stop-id origins skip the point-frame policy fold; the refusal
    # must fire by name there too.
    stop_ids = list(network.stops_gdf["id"].iloc[:1])
    with pytest.raises(NotImplementedError, match="the matrix computers"):
        TravelTimeMatrix(
            network,
            stop_ids,
            stop_ids,
            "2022-02-22 08:30:00",
            street_policy=policy,
        )
    # The wired surfaces demand the car side by name on a network
    # without it.
    with pytest.raises(ValueError, match="multimodal car side"):
        network.route_between_coordinates(
            (60.1690, 24.9320),
            (60.1795, 24.9520),
            "2022-02-22 08:30:00",
            street_policy=policy,
        )
    with pytest.raises(ValueError, match="multimodal car side"):
        network.travel_times_from_coordinate(
            (60.1690, 24.9320), "2022-02-22 08:30:00", street_policy=policy
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
    offsets, tokens = _car_park_table(
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
    with pytest.raises(ValueError, match="departure window"):
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
    with pytest.raises(NotImplementedError, match="arrive-by stage"):
        car_park_network.route_between_coordinates(
            FAR_ORIGIN,
            (60.1795, 24.9520),
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
