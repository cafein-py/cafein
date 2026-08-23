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


def test_every_surface_refuses_the_policy_for_now(network):
    from cafein import (
        DetailedItineraries,
        TravelCostMatrix,
        TravelTimeMatrix,
    )

    policy = CarParkPolicy(facilities=_facilities())
    origins = geopandas.GeoDataFrame({"id": ["a"]}, geometry=[KAMPPI], crs="EPSG:4326")
    destinations = geopandas.GeoDataFrame(
        {"id": ["b"]}, geometry=[HAKANIEMI], crs="EPSG:4326"
    )
    with pytest.raises(NotImplementedError, match="next stage"):
        network.route_between_coordinates(
            (60.1690, 24.9320),
            (60.1795, 24.9520),
            "2022-02-22 08:30:00",
            street_policy=policy,
        )
    with pytest.raises(NotImplementedError, match="next stage"):
        network.travel_times_from_coordinate(
            (60.1690, 24.9320), "2022-02-22 08:30:00", street_policy=policy
        )
    with pytest.raises(NotImplementedError, match="next stage"):
        DetailedItineraries(
            network,
            origins,
            destinations,
            "2022-02-22 08:30:00",
            street_policy=policy,
        )
    with pytest.raises(NotImplementedError, match="next stage"):
        TravelCostMatrix(
            network,
            origins,
            destinations,
            "2022-02-22 08:30:00",
            street_policy=policy,
        )
    with pytest.raises(NotImplementedError, match="next stage"):
        TravelTimeMatrix(
            network,
            origins,
            destinations,
            "2022-02-22 08:30:00",
            street_policy=policy,
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
