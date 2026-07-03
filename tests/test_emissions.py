"""Emission factors and journey annotation."""

import json

import pandas as pd
import pytest

from cafein import emissions


class StubNetwork:
    routes = [
        ("R-BUS", "HSL", 3),
        ("R-TRAM", "HSL", 0),
        ("R-FANCY", "OP2", 3),
        ("R-FERRY", "HSL", 4),
        ("R-EXTENDED", "HSL", 109),
        ("42", "HSL", 3),
        ("NA", "HSL", 3),
    ]


def transit_leg(distance, trip="T1", route="R-BUS"):
    return {
        "type": "transit",
        "trip_id": trip,
        "route_id": route,
        "distance": distance,
    }


def journey(*legs):
    return {"legs": [{"type": "access"}, *legs, {"type": "egress"}]}


def test_vehicle_class_factors_pin_the_published_values():
    # Dey, Marín-Flores & Tenkanen (2026), Tables 4-5: life-cycle totals
    # in g CO2e/pkm, ITF LCA calibrated to the Finnish electricity mix.
    totals = emissions.vehicle_class_factors().sum(axis=1)
    assert totals["bus-ICE"] == 92
    assert totals["bus-BEV"] == 29
    assert totals["metro-urban-train"] == 25
    assert totals["car-ICE"] == 162
    assert totals["car-BEV"] == 70


def test_route_level_rows_from_vehicle_classes():
    # The GEMMAT pattern: an electrified line gets the bus-BEV factors.
    electric = emissions.vehicle_class_factors().loc["bus-BEV"].to_dict()
    factors = pd.DataFrame([{"route_id": "R-BUS", **electric}])
    trip = journey(transit_leg(1000.0))
    (annotated,) = emissions.annotate([trip], StubNetwork(), factors)
    assert annotated["legs"][1]["emissions"] == pytest.approx(29.0)


def test_default_factors_apply_per_mode():
    trip = journey(transit_leg(2000.0), transit_leg(1000.0, route="R-TRAM"))
    (annotated,) = emissions.annotate([trip], StubNetwork())
    bus, tram = annotated["legs"][1], annotated["legs"][2]
    # ITF life-cycle totals: ICE bus 92, urban rail 25 g CO2e/pkm.
    assert bus["emissions"] == pytest.approx(2.0 * 92)
    assert tram["emissions"] == pytest.approx(1.0 * 25)
    assert annotated["emissions"] == pytest.approx(184 + 25)
    assert annotated["legs"][0]["emissions"] == 0.0
    assert annotated["legs"][-1]["emissions"] == 0.0


def test_extended_route_types_fall_back_to_their_base_mode():
    trip = journey(transit_leg(1000.0, route="R-EXTENDED"))  # 109: rail
    (annotated,) = emissions.annotate([trip], StubNetwork())
    assert annotated["legs"][1]["emissions"] == pytest.approx(25.0)


def test_the_resolution_ladder_is_most_specific_wins():
    factors = pd.DataFrame(
        [
            {"route_type": 3, "fuel": 50.0},
            {"agency_id": "OP2", "route_type": 3, "fuel": 20.0},
            {"route_id": "R-BUS", "fuel": 10.0},
            {"trip_id": "T-ELECTRIC", "fuel": 1.0},
        ]
    )
    trip = journey(
        transit_leg(1000.0),  # route override: 10
        transit_leg(1000.0, trip="T-ELECTRIC"),  # trip override: 1
        transit_leg(1000.0, route="R-FANCY"),  # agency+mode: 20
        transit_leg(1000.0, route="R-TRAM"),  # untouched default: 25
    )
    (annotated,) = emissions.annotate([trip], StubNetwork(), factors)
    values = [leg["emissions"] for leg in annotated["legs"][1:-1]]
    assert values == pytest.approx([10.0, 1.0, 20.0, 25.0])


def test_user_rows_override_defaults_at_equal_specificity():
    factors = pd.DataFrame([{"route_type": 3, "fuel": 30.0}])
    trip = journey(transit_leg(1000.0))
    (annotated,) = emissions.annotate([trip], StubNetwork(), factors)
    assert annotated["legs"][1]["emissions"] == pytest.approx(30.0)


def test_an_all_empty_row_is_the_global_default():
    factors = pd.DataFrame([{"vehicle": 7.0}])
    trip = journey(transit_leg(1000.0, route="R-FERRY"))
    (annotated,) = emissions.annotate([trip], StubNetwork(), factors)
    assert annotated["legs"][1]["emissions"] == pytest.approx(7.0)


def test_unmatched_modes_warn_and_stay_unresolved():
    trip = journey(transit_leg(1000.0, route="R-FERRY"), transit_leg(1000.0))
    with pytest.warns(UserWarning, match=r"route_type\(s\) \[4\]"):
        (annotated,) = emissions.annotate([trip], StubNetwork())
    assert annotated["legs"][1]["emissions"] is None
    assert annotated["legs"][2]["emissions"] == pytest.approx(92.0)
    assert annotated["emissions"] is None


def test_component_selection_narrows_the_scope():
    trip = journey(transit_leg(1000.0))
    (annotated,) = emissions.annotate(
        [trip], StubNetwork(), components=["fuel", "operations"]
    )
    # Operational scope of the ICE bus: 72 (fuel) + 8 (operations).
    assert annotated["legs"][1]["emissions"] == pytest.approx(80.0)
    with pytest.raises(ValueError, match="unknown component"):
        emissions.annotate([journey()], StubNetwork(), components=["karma"])


def test_blank_key_strings_mean_not_applicable(tmp_path):
    path = tmp_path / "factors.json"
    path.write_text(
        json.dumps([{"trip_id": "", "route_id": " ", "route_type": "", "fuel": 7.0}])
    )
    loaded = emissions.load_factors(path)
    assert loaded[emissions.KEY_COLUMNS].isna().all().all()
    # A row with no keys at all is the global default.
    trip = journey(transit_leg(1000.0, route="R-FERRY"))
    (annotated,) = emissions.annotate([trip], StubNetwork(), path)
    assert annotated["legs"][1]["emissions"] == pytest.approx(7.0)


def test_numeric_looking_ids_in_csv_tables_still_match(tmp_path):
    path = tmp_path / "factors.csv"
    pd.DataFrame([{"route_id": 42, "fuel": 11.0}]).to_csv(path, index=False)
    trip = journey(transit_leg(1000.0, route="42"))
    (annotated,) = emissions.annotate([trip], StubNetwork(), path)
    assert annotated["legs"][1]["emissions"] == pytest.approx(11.0)


def test_missing_distances_raise_a_clear_error():
    trip = journey({"type": "transit", "trip_id": "T1", "route_id": "R-BUS"})
    with pytest.raises(ValueError, match="no distances"):
        emissions.annotate([trip], StubNetwork())


ROWS = [
    {"route_id": "R-BUS", "fuel": 10.0, "vehicle": 2.0},
    {"route_type": 0, "operations": 5.0},
]


def test_factor_tables_load_identically_from_all_formats(tmp_path):
    from_frame = emissions.load_factors(pd.DataFrame(ROWS))
    csv = tmp_path / "factors.csv"
    pd.DataFrame(ROWS).to_csv(csv, index=False)
    json_file = tmp_path / "factors.json"
    json_file.write_text(json.dumps(ROWS))
    for loaded in [emissions.load_factors(csv), emissions.load_factors(json_file)]:
        pd.testing.assert_frame_equal(
            loaded.fillna(-1), from_frame.fillna(-1), check_dtype=False
        )


def test_factor_tables_load_from_yaml(tmp_path):
    yaml = pytest.importorskip("yaml")
    path = tmp_path / "factors.yml"
    path.write_text(yaml.safe_dump(ROWS))
    loaded = emissions.load_factors(path)
    pd.testing.assert_frame_equal(
        loaded.fillna(-1),
        emissions.load_factors(pd.DataFrame(ROWS)).fillna(-1),
        check_dtype=False,
    )


def test_float_coerced_ids_in_mixed_records_still_match(tmp_path):
    # Mixing keyed and unkeyed records makes pandas coerce numeric ids
    # to floats; they must still match the string ids legs carry.
    records = [{"route_id": 42, "fuel": 11.0}, {"route_type": 0, "operations": 5.0}]
    path = tmp_path / "factors.json"
    path.write_text(json.dumps(records))
    trip = journey(transit_leg(1000.0, route="42"))
    for source in [path, pd.DataFrame.from_records(records)]:
        (annotated,) = emissions.annotate([trip], StubNetwork(), source)
        assert annotated["legs"][1]["emissions"] == pytest.approx(11.0)


def test_non_ascii_ids_load_from_utf8_files(tmp_path):
    path = tmp_path / "factors.json"
    path.write_text(
        json.dumps([{"route_id": "linja-Ä", "fuel": 3.0}]), encoding="utf-8"
    )
    loaded = emissions.load_factors(path)
    assert loaded["route_id"][0] == "linja-Ä"


def test_na_named_ids_in_csv_tables_survive(tmp_path):
    # "NA" is a legal GTFS identifier, not a missing value.
    path = tmp_path / "factors.csv"
    path.write_text("route_id,fuel\nNA,5.0\n")
    trip = journey(transit_leg(1000.0, route="NA"))
    (annotated,) = emissions.annotate([trip], StubNetwork(), path)
    assert annotated["legs"][1]["emissions"] == pytest.approx(5.0)


def test_rows_without_component_values_are_rejected():
    with pytest.raises(ValueError, match="no component values"):
        emissions.load_factors(
            pd.DataFrame([{"route_id": "R-BUS", "fuel": None, "vehicle": None}])
        )


def test_route_type_keys_must_be_integers():
    for bad in [3.5, -1]:
        with pytest.raises(ValueError, match="non-negative integers"):
            emissions.load_factors(pd.DataFrame([{"route_type": bad, "fuel": 1.0}]))


def test_invalid_factor_tables_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="unknown factor-table column"):
        emissions.load_factors(pd.DataFrame([{"route_type": 3, "co2": 1.0}]))
    with pytest.raises(ValueError, match="at least one component column"):
        emissions.load_factors(pd.DataFrame([{"route_type": 3}]))
    with pytest.raises(ValueError, match="negative"):
        emissions.load_factors(pd.DataFrame([{"route_type": 3, "fuel": -1.0}]))
    with pytest.raises(ValueError, match="unsupported factor-table format"):
        emissions.load_factors(tmp_path / "factors.toml")
    bad = tmp_path / "factors.json"
    bad.write_text('{"fuel": 1.0}')
    with pytest.raises(ValueError, match="list of mappings"):
        emissions.load_factors(bad)


def test_helsinki_k_train_journey_emissions(network):
    # Korso -> Käpylä on the K train: 16.786 km (raw shape_dist tables)
    # at the urban-rail factor of 25 g CO2e/pkm.
    journeys = network.route_between_stops(
        "4810551", "1250551", "2022-02-22", "08:30:00"
    )
    (annotated,) = network.annotate_emissions([journeys[0]])
    transit = annotated["legs"][1]
    assert transit["emissions"] == pytest.approx(16.786 * 25, rel=0.001)
    assert annotated["emissions"] == pytest.approx(16.786 * 25, rel=0.001)
