"""Emission factors, journey annotation, and carbon budgets."""

import json
import warnings

import pandas as pd
import pytest

from cafein import TravelCostMatrix, costs, emissions
from cafein.matrices import _factor_tables
from conftest import point_frame

OUT, BACK = "2022-02-22 08:30", "2022-02-22 16:30"


class StubNetwork:
    routes = [
        ("R-BUS", "HSL", 3),
        ("R-TRAM", "HSL", 0),
        ("R-FANCY", "OP2", 3),
        ("R-FERRY", "HSL", 4),
        ("R-FUNICULAR", "HSL", 7),
        ("R-EXTENDED", "HSL", 109),
        ("42", "HSL", 3),
        ("NA", "HSL", 3),
    ]


def transit_leg(distance, trip="T1", route="R-BUS"):
    return {
        "type": "transit",
        "trip_id": trip,
        "route_id": route,
        "distance_m": distance,
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
    trip = journey(
        transit_leg(2000.0),
        transit_leg(1000.0, route="R-TRAM"),
        transit_leg(1000.0, route="R-EXTENDED"),  # 109: extended, rail base
        transit_leg(1000.0, route="R-FERRY"),
    )
    (annotated,) = emissions.annotate([trip], StubNetwork())
    bus, tram, extended, ferry = annotated["legs"][1:5]
    # ITF life-cycle totals: ICE bus 92, urban rail 25 g CO2e/pkm; the
    # Suomenlinna-derived ferry 333.
    assert bus["emissions"] == pytest.approx(2.0 * 92)
    assert tram["emissions"] == pytest.approx(1.0 * 25)
    # An extended route type falls back to its base mode's factor.
    assert extended["emissions"] == pytest.approx(25.0)
    assert ferry["emissions"] == pytest.approx(333.0)
    assert annotated["emissions"] == pytest.approx(184 + 25 + 25 + 333)
    assert annotated["legs"][0]["emissions"] == 0.0
    assert annotated["legs"][-1]["emissions"] == 0.0


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
    trip = journey(transit_leg(1000.0, route="R-FUNICULAR"))
    (annotated,) = emissions.annotate([trip], StubNetwork(), factors)
    assert annotated["legs"][1]["emissions"] == pytest.approx(7.0)


def test_unmatched_modes_warn_and_stay_unresolved():
    trip = journey(transit_leg(1000.0, route="R-FUNICULAR"), transit_leg(1000.0))
    with pytest.warns(UserWarning, match=r"route_type\(s\) \[7\]"):
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
    trip = journey(transit_leg(1000.0, route="R-FUNICULAR"))
    (annotated,) = emissions.annotate([trip], StubNetwork(), path)
    assert annotated["legs"][1]["emissions"] == pytest.approx(7.0)


def test_missing_distances_raise_a_clear_error():
    trip = journey({"type": "transit", "trip_id": "T1", "route_id": "R-BUS"})
    with pytest.raises(ValueError, match="no distances"):
        emissions.annotate([trip], StubNetwork())


ROWS = [
    {"route_id": "R-BUS", "fuel": 10.0, "vehicle": 2.0},
    {"route_type": 0, "operations": 5.0},
]


@pytest.mark.parametrize("fmt", ["csv", "json", "yaml", "rows", "columns", "row"])
def test_factor_tables_load_identically_from_all_formats(tmp_path, fmt):
    rows = ROWS[:1] if fmt == "row" else ROWS
    from_frame = emissions.load_factors(pd.DataFrame(rows))
    if fmt == "csv":
        source = tmp_path / "factors.csv"
        pd.DataFrame(ROWS).to_csv(source, index=False)
    elif fmt == "json":
        source = tmp_path / "factors.json"
        source.write_text(json.dumps(ROWS))
    elif fmt == "yaml":
        yaml = pytest.importorskip("yaml")
        source = tmp_path / "factors.yml"
        source.write_text(yaml.safe_dump(ROWS))
    else:
        # A list of row mappings, a mapping of columns, one row mapping.
        source = {
            "rows": ROWS,
            "columns": pd.DataFrame(ROWS).to_dict("list"),
            "row": ROWS[0],
        }[fmt]
    loaded = emissions.load_factors(source)
    pd.testing.assert_frame_equal(
        loaded.fillna(-1), from_frame.fillna(-1), check_dtype=False
    )


@pytest.mark.parametrize(
    "load, rows",
    [
        (
            emissions.load_street_factors,
            [{"street_mode": "car", "vehicle_class": "BEV", "fuel": 16.0}],
        ),
        (
            costs.load_street_costs,
            [
                {
                    "perspective": "private",
                    "street_mode": "car",
                    "component": "fuel",
                    "cost_per_km": 0.1,
                }
            ],
        ),
        # The street-policy split reads the same forms before routing and
        # hands either arm the table it read.
        (
            lambda source: _factor_tables(source)[1],
            [{"street_mode": "car", "fuel": 1.0}],
        ),
        (
            lambda source: _factor_tables(source)[0],
            [{"route_type": 4, "fuel": 1.0}],
        ),
    ],
    ids=["street-factors", "street-costs", "policy-split", "policy-split-transit"],
)
def test_street_and_cost_tables_take_mappings(load, rows):
    expected = load(pd.DataFrame(rows))
    for source in (rows, rows[0], pd.DataFrame(rows).to_dict("list")):
        pd.testing.assert_frame_equal(load(source), expected)


@pytest.mark.parametrize(
    "case, route, expected",
    [
        # A numeric-looking id in a CSV table still matches the string
        # route id the legs carry.
        ("numeric_csv", "42", 11.0),
        # Mixing keyed and unkeyed records makes pandas coerce numeric
        # ids to floats; they must still match the string ids legs carry.
        ("float_coerced_json", "42", 11.0),
        # "NA" is a legal GTFS identifier, not a missing value.
        ("na_csv", "NA", 5.0),
    ],
)
def test_coerced_ids_in_factor_tables_still_match(tmp_path, case, route, expected):
    if case == "numeric_csv":
        sources = [tmp_path / "factors.csv"]
        pd.DataFrame([{"route_id": 42, "fuel": 11.0}]).to_csv(sources[0], index=False)
    elif case == "float_coerced_json":
        records = [
            {"route_id": 42, "fuel": 11.0},
            {"route_type": 0, "operations": 5.0},
        ]
        path = tmp_path / "factors.json"
        path.write_text(json.dumps(records))
        sources = [path, pd.DataFrame.from_records(records)]
    else:
        sources = [tmp_path / "factors.csv"]
        sources[0].write_text("route_id,fuel\nNA,5.0\n")
    trip = journey(transit_leg(1000.0, route=route))
    for source in sources:
        (annotated,) = emissions.annotate([trip], StubNetwork(), source)
        assert annotated["legs"][1]["emissions"] == pytest.approx(expected)


def test_non_ascii_ids_load_from_utf8_files(tmp_path):
    path = tmp_path / "factors.json"
    path.write_text(
        json.dumps([{"route_id": "linja-Ä", "fuel": 3.0}]), encoding="utf-8"
    )
    loaded = emissions.load_factors(path)
    assert loaded["route_id"][0] == "linja-Ä"


@pytest.mark.parametrize(
    "load, keys",
    [
        (emissions.load_factors, {"route_type": 3}),
        (
            emissions.load_street_factors,
            {"street_mode": "car", "vehicle_class": "ICE", "service_model": "private"},
        ),
    ],
)
def test_provenance_columns_survive_and_unknown_ones_warn(load, keys, tmp_path):
    # A cafein.lca export's extra columns ride along unchanged — through a
    # CSV too, where a numeric-looking value must stay text — any other
    # column (a non-string label included) is dropped with a warning, and
    # a basis must be a known one.
    provenance = {
        "mode": "car_ice",
        "basis": "passenger_km",
        "scenario": "001",
        "scenario_sha256": "ab" * 32,
        "case": "central",
        "cafein_lca_version": "0.1.0",
    }
    row = {**keys, "vehicle": 1.0, "fuel": 2.0, **provenance, "note": "typed", 7: 0}
    path = tmp_path / "factors.csv"
    pd.DataFrame([row]).to_csv(path, index=False)
    for source in (pd.DataFrame([row]), path):
        with pytest.warns(UserWarning, match=r"unknown .*column\(s\): 7, note"):
            loaded = load(source)
        assert not {"note", 7, "7"} & set(loaded.columns)
        assert loaded.iloc[0][list(provenance)].to_dict() == provenance
    with pytest.raises(ValueError, match="basis must be"):
        load(pd.DataFrame([{**keys, "fuel": 1.0, "basis": "per_trip"}]))


@pytest.mark.parametrize(
    "load, keys, resolve",
    [
        (
            emissions.load_factors,
            {"route_type": 3},
            lambda table: emissions._Resolver(table).resolve(None, None, None, 3),
        ),
        (
            emissions.load_street_factors,
            {"street_mode": "car", "vehicle_class": "ICE", "service_model": "private"},
            lambda table: emissions.street_factor(
                "car", factors=table, vehicle_class="ICE"
            ),
        ),
    ],
)
def test_a_total_column_is_kept_but_never_summed(load, keys, resolve):
    # cafein.lca writes a human-readable ``total`` column; cafein keeps it
    # without warning but computes the factor from the components alone.
    components = {"vehicle": 1.0, "fuel": 2.0, "infrastructure": 3.0, "operations": 4.0}
    plain = {**keys, **components}
    # A deliberately wrong total: if it were ever summed the factor would jump.
    with_total = {**plain, "total": 999.0}
    for row in (plain, with_total):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            loaded = load(pd.DataFrame([row]))
        assert not any("unknown" in str(w.message) for w in caught)
        assert ("total" in loaded.columns) == ("total" in row)
    plain_factor = resolve(load(pd.DataFrame([plain])))
    total_factor = resolve(load(pd.DataFrame([with_total])))
    assert plain_factor == pytest.approx(10.0)
    assert total_factor == pytest.approx(plain_factor)


def test_invalid_factor_tables_are_rejected(tmp_path):
    for source, match in (
        # Transit legs have no occupancy at query time: passenger-km only.
        (
            pd.DataFrame([{"route_type": 3, "fuel": 1.0, "basis": "vehicle_km"}]),
            "basis must be 'passenger_km'",
        ),
        (pd.DataFrame([{"route_type": 3}]), "at least one component column"),
        (pd.DataFrame([{"route_type": 3, "fuel": -1.0}]), "negative"),
        (tmp_path / "factors.toml", "unsupported factor-table format"),
        # A row with every component empty carries no factors at all.
        (
            pd.DataFrame([{"route_id": "R-BUS", "fuel": None, "vehicle": None}]),
            "no component values",
        ),
        # route_type keys must be non-negative integers.
        (pd.DataFrame([{"route_type": 3.5, "fuel": 1.0}]), "non-negative integers"),
        (pd.DataFrame([{"route_type": -1, "fuel": 1.0}]), "non-negative integers"),
        # A mapping is one row or columns, never a guess between them.
        ({"route_type": [3, 0], "fuel": 1.0}, "not both"),
        ({"route_type": [3, 0], "fuel": [1.0]}, "equal lengths"),
    ):
        with pytest.raises(ValueError, match=match):
            emissions.load_factors(source)
    for source, match in (
        ([{"route_type": 3, "fuel": 1.0}, 3], "one mapping per row"),
        (3, "cannot load emission factors from int"),
        # Columns are ordered: a set has no order, a mapping gives keys.
        ({"route_type": {3, 0}, "fuel": [1.0, 2.0]}, "ordered sequences, not set"),
        ({"route_type": {"a": 3}, "fuel": [1.0]}, "ordered sequences, not dict"),
    ):
        with pytest.raises(TypeError, match=match):
            emissions.load_factors(source)
    bad = tmp_path / "factors.json"
    bad.write_text('{"fuel": 1.0}')
    with pytest.raises(ValueError, match="list of mappings"):
        emissions.load_factors(bad)


def test_helsinki_k_train_journey_emissions(network):
    # Korso -> Käpylä on the K train: 16.786 km (raw shape_dist tables)
    # at the urban-rail factor of 25 g CO2e/pkm.
    journeys = network.route_between_stops("4810551", "1250551", "2022-02-22 08:30:00")
    (annotated,) = network.annotate_emissions([journeys[0]])
    transit = annotated["legs"][1]
    assert transit["emissions"] == pytest.approx(16.786 * 25, rel=0.001)
    assert annotated["emissions"] == pytest.approx(16.786 * 25, rel=0.001)


def test_budget_columns_read_as_multiples_of_the_budget():
    counted = pd.Series([400.0, 1000.0, 2400.0, float("nan")])
    columns = emissions._budget_columns(counted, 1000.0)
    assert columns["budget_share"].tolist()[:3] == [0.4, 1.0, 2.4]
    assert columns["budget_left"].tolist()[:3] == [600.0, 0.0, -1400.0]
    assert columns["within_budget"].tolist() == [True, True, False, pd.NA]
    assert columns["budget_share"].isna().tolist() == [False, False, False, True]
    assert columns["budget_left"].isna().tolist() == [False, False, False, True]


@pytest.mark.parametrize(
    "homes, round_trip, trips, chunk, components",
    [
        ([("home", "1100602")], False, 1, None, None),
        # Passed as a one-shot generator: both directions must see it.
        ([("home", "1100602")], True, 2, None, ("fuel", "operations")),
        # Chunk 1 of 2 keeps "home" alone: the way back returns only there.
        ([("near-work", "1040280"), ("home", "1100602")], True, 1, (1, 2), None),
    ],
)
def test_budget_routes_each_direction(
    network_with_footpaths, homes, round_trip, trips, chunk, components
):
    net = network_with_footpaths
    home = point_frame(net, [("home", "1100602")])
    work = point_frame(net, [("work", "1040280")])
    result = emissions.budget(
        net,
        point_frame(net, homes),
        work,
        OUT,
        round_trip=round_trip,
        return_departure=BACK if round_trip else None,
        trips=trips,
        chunk=chunk,
        components=None if components is None else iter(components),
    )
    there = pd.DataFrame(TravelCostMatrix(net, home, work, OUT, components=components))
    back = pd.DataFrame(TravelCostMatrix(net, work, home, BACK, components=components))
    # The M2 home at 16:30 is its own journey, not the morning's reversed.
    assert back["travel_time"].tolist() != there["travel_time"].tolist()
    returned = back.drop(columns=["from_id", "to_id"]).add_prefix("return_")
    budget = ["budget_share", "budget_left", "within_budget"]
    pd.testing.assert_frame_equal(result[there.columns], there)
    if round_trip:
        assert list(result.columns) == [*there.columns, *returned.columns, *budget]
        pd.testing.assert_frame_equal(result[returned.columns], returned)
        counted = trips * (there["emissions"] + back["emissions"])
    else:
        assert list(result.columns) == [*there.columns, *budget]
        counted = trips * there["emissions"]
    assert result["budget_left"].tolist() == pytest.approx((1000 - counted).tolist())


def test_round_trip_warns_about_pairs_reachable_one_way(network):
    # Stop to stop, the K train takes Korso's 4810551 to Käpylä's 1250551,
    # but nothing leaves 1250551 for 4810551.
    with pytest.warns(UserWarning, match=r"1 pair\(s\) reachable only one way"):
        result = emissions.budget(
            network,
            ["4810551"],
            ["1250551"],
            OUT,
            round_trip=True,
            return_departure=BACK,
        )
    assert result.empty


@pytest.mark.parametrize(
    "kwargs, error, match",
    [
        ({"budget": 0}, ValueError, "budget must be a positive"),
        ({"trips": 0}, ValueError, "trips must be at least 1"),
        ({"round_trip": 1}, TypeError, "round_trip must be True or False"),
        ({"return_departure": BACK}, ValueError, "only with round_trip"),
        ({"round_trip": True}, ValueError, "exactly when departure"),
        (
            {"round_trip": True, "departure": None, "return_departure": BACK},
            ValueError,
            "exactly when departure",
        ),
        (
            {"round_trip": True, "return_departure": BACK, "arrival": BACK},
            ValueError,
            "not arrival",
        ),
        (
            {"round_trip": True, "return_departure": [BACK]},
            ValueError,
            "one return_departure moment",
        ),
        (
            {"round_trip": True, "return_departure": BACK, "candidates": "sweep"},
            ValueError,
            "sweep",
        ),
        (
            {"round_trip": True, "return_departure": "evening"},
            ValueError,
            "return_departure takes",
        ),
    ],
)
def test_budget_refuses_before_routing(monkeypatch, kwargs, error, match):
    def routed(*args, **kwargs):
        raise AssertionError("routed before refusing")

    monkeypatch.setattr("cafein.matrices.TravelCostMatrix", routed)
    with pytest.raises(error, match=match):
        emissions.budget(None, ["a"], ["b"], **{"departure": OUT, **kwargs})
