"""The trip-indicator value tables and their per-query resolution."""

import math
import warnings

import pandas as pd
import pytest

from cafein import indicators


def test_shipped_tables_source_every_value_and_order_their_cases():
    assert list(indicators.co2_prices().price) == [60.0, 100.0, 189.0]
    rates = indicators.activity_rates()
    walk = rates[(rates.street_mode == "walk") & (rates.case == "middle")]
    assert walk.met_hours_per_km.item() == pytest.approx(4.0 / 5.3)
    assert (rates.met_hours_per_km == rates.met / rates.speed_kmh).all()
    for table, keys, values in [
        (rates, ["street_mode"], ["met_hours_per_km"]),
        (
            indicators.particle_factors(),
            ["vehicle_class", "road_class"],
            ["pm10", "pm2_5"],
        ),
    ]:
        assert (table.source != "").eq(table.published).all()
        for _, group in table.groupby(keys):
            by_case = group.set_index("case").loc[list(indicators.CASES), values]
            assert (by_case.diff().dropna() >= 0).all().all()
            assert group.set_index("case").loc["middle", "published"]
            # A case no source gives repeats the middle values.
            unpublished = group[~group.published]
            assert (unpublished[values] == by_case.loc["middle"].to_numpy()).all().all()


@pytest.mark.parametrize(
    "arguments, error, message",
    [
        ({"co2_price": 100}, ValueError, "pass indicators="),
        ({"indicators": "pm10", "co2_price": 100}, ValueError, "climate_cost"),
        ({"indicators": ["pm10"], "activity_rates": "low"}, ValueError, "physical"),
        ({"indicators": ["noise"]}, ValueError, "non-empty selection"),
        ({"indicators": []}, ValueError, "non-empty selection"),
        ({"indicators": ["pm10", "pm10"]}, ValueError, "non-empty selection"),
        ({"indicators": "climate_cost", "co2_price": "central"}, ValueError, "one of"),
        ({"indicators": "climate_cost", "co2_price": ""}, ValueError, "one of"),
        ({"indicators": "climate_cost", "co2_price": -1}, ValueError, "non-negative"),
        (
            {"indicators": "physical_activity", "activity_rates": {"car": 0.1}},
            ValueError,
            "names 'car'",
        ),
        (
            {"indicators": "physical_activity", "activity_rates": {"walk": "0.7"}},
            TypeError,
            "number",
        ),
        (
            {"indicators": "physical_activity", "activity_rates": [0.7]},
            TypeError,
            "one mapping per row",
        ),
        (
            {
                "indicators": "physical_activity",
                "activity_rates": [{"street_mode": "car", "met_hours_per_km": 0.1}],
            },
            ValueError,
            "unknown street_mode",
        ),
        (
            {
                "indicators": "physical_activity",
                "activity_rates": [{"street_mode": "walk", "met_hours_per_km": 0.7}]
                * 2,
            },
            ValueError,
            "duplicate",
        ),
        (
            {
                "indicators": "pm10",
                "particle_factors": [{"vehicle_class": "BEV", "pm10": 1.0}],
            },
            ValueError,
            "misses column",
        ),
        (
            {
                "indicators": "pm2_5",
                "particle_factors": [
                    {
                        "vehicle_class": "BEV",
                        "road_class": "street",
                        "pm10": 1.0,
                        "pm2_5": 0.5,
                    }
                ],
            },
            ValueError,
            "road_class",
        ),
        (
            {
                "indicators": "pm10",
                "particle_factors": [
                    {
                        "vehicle_class": "BEV",
                        "road_class": "urban",
                        "pm10": float("nan"),
                        "pm2_5": 0.5,
                    }
                ],
            },
            ValueError,
            "finite",
        ),
    ],
)
def test_query_settings_are_validated_by_name(arguments, error, message):
    settings = {
        "indicators": None,
        "co2_price": None,
        "activity_rates": None,
        "particle_factors": None,
        **arguments,
    }
    with pytest.raises(error, match=message):
        indicators.resolve_query(**settings)


def test_cases_and_own_values_resolve_with_their_warnings():
    assert indicators.resolve_query(None, None, None, None) is None
    default = indicators.resolve_query(list(indicators.INDICATORS), None, None, None)
    assert default.climate_cost(162.0) == pytest.approx(0.0162)
    assert default.activity_rate("e_bike") == pytest.approx(5.5 / 17.0)
    assert default.particle_factor("BEV", "urban") == (32.1, 14.8)

    low = indicators.resolve_query(("physical_activity", "pm10"), None, "low", "high")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert low.activity_rate("walk") == low.activity_rate("walk") == 4.0 / 5.3
        assert low.activity_rate("e_bike") == pytest.approx(5.9 / 21.5)
        assert low.activity_rate("car") == 0.0
        assert math.isnan(low.activity_rate("e_scooter"))
        assert low.particle_factor("BEV", "rural") == (30.5, 13.5)
        assert low.particle_factor("ICE", "urban") == (40.4, 19.1)
        assert all(math.isnan(value) for value in low.particle_factor("HEV", "urban"))
    messages = [str(warning.message) for warning in caught]
    # One warning per mode or class: walk, e_scooter, ICE, HEV.
    assert len(messages) == 4
    assert any("no source gives a low value for 'walk'" in m for m in messages)
    assert any("'e_scooter'" in m and "unresolved" in m for m in messages)

    own = indicators.resolve_query(
        ["climate_cost", "physical_activity", "pm2_5"],
        150,
        {"e_scooter": 0.1},
        pd.DataFrame(
            [{"vehicle_class": "HEV", "road_class": "urban", "pm10": 35, "pm2_5": 16}]
        ),
    )
    assert own.climate_cost(1e6) == 150.0
    assert own.activity_rate("e_scooter") == 0.1
    assert own.activity_rate("bicycle") == pytest.approx(6.8 / 14.0)
    assert own.particle_factor("HEV", "urban") == (35.0, 16.0)
    assert own.particle_factor("ICE", "motorway") == (26.4, 12.6)
    table = pd.DataFrame([{"street_mode": "wheelchair", "met_hours_per_km": 0.9}])
    tabled = indicators.resolve_query("physical_activity", None, table, None)
    assert tabled.activity_rate("wheelchair") == 0.9


def test_journey_totals_keep_an_unresolved_leg_unresolved():
    from cafein import compare_to_fastest

    legs = pd.DataFrame(
        {
            "from_id": ["a"] * 3,
            "to_id": ["b"] * 3,
            "option": [0, 0, 1],
            "segment": [0, 1, 0],
            "leg_type": ["walk", "e_scooter", "walk"],
            "travel_time_s": [60, 120, 400],
            "physical_activity": [0.1, math.nan, 0.5],
        }
    )
    totals = compare_to_fastest(legs).set_index("option").physical_activity
    assert math.isnan(totals[0]) and totals[1] == 0.5
