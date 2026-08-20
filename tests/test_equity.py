"""cafein.equity: the weighted-distribution core and inequality set."""

import numpy as np
import pandas as pd
import pytest

from cafein import _distribution, equity

DEPARTURE = "2022-02-22 08:30:00"

SCALARS = {
    "gini_index": lambda frame, **kw: equity.gini_index(frame, **kw),
    "share_ratio": lambda frame, **kw: equity.share_ratio(frame, **kw),
    "generalized_entropy": lambda frame, **kw: equity.generalized_entropy(frame, **kw),
    "atkinson": lambda frame, **kw: equity.atkinson(frame, **kw),
    "kolm": lambda frame, **kw: equity.kolm(frame, **kw),
    "hoover": lambda frame, **kw: equity.hoover(frame, **kw),
}


def _frame(values, weights=None, **columns):
    frame = pd.DataFrame({"accessibility": values})
    if weights is not None:
        frame["pop"] = weights
    for name, column in columns.items():
        frame[name] = column
    return frame


def _mad_gini(values, weights):
    # The independent mean-absolute-difference form of the Gini.
    values, weights = np.asarray(values, float), np.asarray(weights, float)
    spread = np.abs(values[:, None] - values[None, :])
    pairs = weights[:, None] * weights[None, :]
    mean = np.dot(values, weights) / weights.sum()
    return float((pairs * spread).sum() / (2.0 * weights.sum() ** 2 * mean))


def test_two_value_gini_and_lorenz_golden():
    frame = _frame([0.0, 1.0])
    assert equity.gini_index(frame) == pytest.approx(0.5)
    curve = equity.lorenz_curve(frame)
    assert curve["population_share"].tolist() == [0.0, 0.5, 1.0]
    assert curve["value_share"].tolist() == [0.0, 0.0, 1.0]


def test_weighted_gini_matches_the_mean_absolute_difference_form():
    values, weights = [1.0, 2.0, 3.0, 4.0, 10.0], [1.0, 2.0, 3.0, 2.0, 1.0]
    frame = _frame(values, weights)
    assert equity.gini_index(frame, population="pop") == pytest.approx(
        _mad_gini(values, weights), rel=1e-12
    )


def test_gini_is_the_integral_of_its_own_lorenz_curve():
    frame = _frame([1.0, 2.0, 7.0], [2.0, 1.0, 3.0])
    curve = equity.lorenz_curve(frame, population="pop")
    area = _distribution.polyline_area(
        curve["population_share"].to_numpy(), curve["value_share"].to_numpy()
    )
    assert equity.gini_index(frame, population="pop") == pytest.approx(
        1.0 - 2.0 * area, rel=1e-12
    )


def test_equal_distributions_score_zero_everywhere():
    frame = _frame([3.0, 3.0, 3.0])
    for name, measure in SCALARS.items():
        if name == "share_ratio":
            assert measure(frame) == pytest.approx(1.0)
        else:
            assert measure(frame) == pytest.approx(0.0, abs=1e-12)


def test_presets_equal_their_family_calls_exactly():
    frame = _frame([1.0, 2.0, 3.0, 4.0], group=["a", "a", "b", "b"])
    assert equity.theil_t(frame) == equity.generalized_entropy(frame, alpha=1)
    assert equity.mld(frame) == equity.generalized_entropy(frame, alpha=0)
    pd.testing.assert_frame_equal(
        equity.theil_t(frame, groups="group"),
        equity.generalized_entropy(frame, alpha=1, groups="group"),
    )


def test_the_decomposition_sums_to_the_total():
    frame = _frame(
        [1.0, 2.0, 5.0, 9.0],
        [1.0, 2.0, 1.0, 3.0],
        group=["a", "a", "b", "b"],
    )
    for alpha in (0, 1, 2, 0.5):
        parts = equity.generalized_entropy(
            frame, alpha=alpha, groups="group", population="pop"
        )
        assert isinstance(parts, pd.DataFrame) and len(parts) == 1
        assert parts["total"][0] == pytest.approx(
            parts["between"][0] + parts["within"][0], rel=1e-12
        )
        assert parts["total"][0] == pytest.approx(
            equity.generalized_entropy(frame, alpha=alpha, population="pop"),
            rel=1e-12,
        )


def test_palma_is_the_named_share_ratio_preset():
    frame = _frame([1.0, 2.0, 3.0, 4.0, 5.0], income=[10, 20, 30, 40, 50])
    assert equity.palma_ratio(frame, income="income") == pytest.approx(
        equity.share_ratio(frame, top=0.10, bottom=0.40, income="income")
    )
    with pytest.raises(ValueError, match="palma_ratio requires income="):
        equity.palma_ratio(frame, income=None)


def test_a_weight_of_two_equals_the_duplicated_row():
    weighted = _frame([1.0, 2.0, 5.0], [1.0, 2.0, 1.0])
    duplicated = _frame([1.0, 2.0, 2.0, 5.0])
    for name, measure in SCALARS.items():
        assert measure(weighted, population="pop") == pytest.approx(
            measure(duplicated), rel=1e-12
        ), name
    assert _distribution.weighted_median(
        np.array([1.0, 2.0, 5.0]), np.array([1.0, 2.0, 1.0])
    ) == _distribution.weighted_median(np.array([1.0, 2.0, 2.0, 5.0]), np.ones(4))


def test_relative_indices_are_scale_invariant_and_kolm_translates():
    frame = _frame([1.0, 2.0, 6.0])
    scaled = _frame([3.0, 6.0, 18.0])
    shifted = _frame([6.0, 7.0, 11.0])
    for name in ("gini_index", "generalized_entropy", "atkinson", "hoover"):
        assert SCALARS[name](frame) == pytest.approx(
            SCALARS[name](scaled), rel=1e-12
        ), name
    assert equity.kolm(frame) == pytest.approx(equity.kolm(shifted), rel=1e-12)
    assert equity.gini_index(shifted) != pytest.approx(equity.gini_index(frame))


def test_grouped_fan_out_equals_manual_per_group_calls():
    frame = pd.DataFrame(
        {
            "opportunity": ["jobs"] * 3 + ["schools"] * 3,
            "budget": [30.0] * 6,
            "percentile": [50] * 6,
            "accessibility": [1.0, 2.0, 3.0, 4.0, 5.0, 12.0],
        }
    )
    result = equity.gini_index(frame)
    assert list(result.columns) == [
        "opportunity",
        "budget",
        "percentile",
        "gini_index",
    ]
    for row in result.itertuples():
        part = frame[frame["opportunity"] == row.opportunity]
        expected = equity.gini_index(part[["accessibility"]])
        assert isinstance(expected, float)
        assert row.gini_index == pytest.approx(expected)
    forced = equity.gini_index(frame, group_columns=[])
    assert isinstance(forced, float)
    with pytest.raises(TypeError, match="bare"):
        equity.gini_index(frame, group_columns="opportunity")


def test_percentile_columns_fan_out():
    frame = pd.DataFrame(
        {
            "percentile": [25, 25, 75, 75],
            "accessibility": [1.0, 3.0, 2.0, 2.0],
        }
    )
    result = equity.gini_index(frame)
    assert result["percentile"].tolist() == [25, 75]
    assert result["gini_index"][1] == pytest.approx(0.0)


def test_the_sociodemographic_join_contract():
    data = pd.DataFrame({"from_id": [1, 2, 3], "accessibility": [1.0, 2.0, 3.0]})
    table = pd.DataFrame({"id": [1, 2, 3], "pop": [1.0, 1.0, 2.0]})
    expected = equity.gini_index(
        _frame([1.0, 2.0, 3.0], [1.0, 1.0, 2.0]), population="pop"
    )
    assert equity.gini_index(
        data, sociodemographic_data=table, population="pop"
    ) == pytest.approx(expected)
    missing = pd.DataFrame({"id": [1, 2], "pop": [1.0, 1.0]})
    with pytest.raises(ValueError, match=r"misses 1 origin id\(s\): 3"):
        equity.gini_index(data, sociodemographic_data=missing, population="pop")
    dropped = equity.gini_index(
        data, sociodemographic_data=missing, population="pop", dropna=True
    )
    assert dropped == pytest.approx(equity.gini_index(_frame([1.0, 2.0])))
    overlapping = pd.DataFrame({"id": [1, 2, 3], "accessibility": [0, 0, 0]})
    with pytest.raises(ValueError, match="already exist"):
        equity.gini_index(data, sociodemographic_data=overlapping)


def test_the_delta_recipe_survives_reordering_and_refuses_gaps():
    keys = ["from_id", "opportunity", "budget"]
    before = pd.DataFrame(
        {
            "from_id": ["a", "b", "c"],
            "opportunity": ["jobs"] * 3,
            "budget": [30.0] * 3,
            "accessibility": [10.0, 20.0, 30.0],
        }
    )
    after = before.iloc[[2, 0, 1]].reset_index(drop=True)
    after = after.assign(accessibility=[35.0, 12.0, 18.0])

    def delta_frame(before, after):
        merged = before.merge(
            after,
            on=keys,
            suffixes=("_before", "_after"),
            how="outer",
            indicator=True,
            validate="one_to_one",
        )
        if (merged["_merge"] != "both").any():
            raise ValueError("the scenarios cover different origins")
        delta = merged[keys].copy()
        delta["accessibility"] = (
            merged["accessibility_after"] - merged["accessibility_before"]
        )
        return delta

    delta = delta_frame(before, after)
    by_id = delta.set_index("from_id")["accessibility"]
    assert by_id["a"] == 2.0 and by_id["b"] == -2.0 and by_id["c"] == 5.0
    with pytest.raises(ValueError, match="different origins"):
        delta_frame(before, after[after["from_id"] != "b"])
    with pytest.raises(ValueError, match="kolm"):
        equity.gini_index(delta)
    spread = equity.kolm(delta)
    assert (spread["kolm"] > 0.0).all()


def test_zeros_refuse_in_the_log_family_naming_gini():
    frame = _frame([0.0, 1.0, 2.0])
    for measure in (equity.generalized_entropy, equity.atkinson):
        with pytest.raises(ValueError, match="gini_index tolerates zeros"):
            measure(frame)
    assert equity.gini_index(frame) > 0.0


def test_share_ratio_zero_cases():
    poor_zero = _frame([0.0, 0.0, 10.0, 10.0], income=[1, 2, 3, 4])
    assert equity.share_ratio(
        poor_zero, top=0.10, bottom=0.40, income="income"
    ) == float("inf")
    rich_zero = _frame([5.0, 5.0, 0.0, 0.0], income=[1, 2, 3, 4])
    assert equity.share_ratio(
        rich_zero, top=0.10, bottom=0.40, income="income"
    ) == pytest.approx(0.0)
    both_zero = _frame([0.0, 7.0, 7.0, 0.0], income=[1, 2, 3, 4])
    with pytest.raises(ValueError, match="0/0"):
        equity.share_ratio(both_zero, top=0.10, bottom=0.10, income="income")


def test_a_tied_block_spanning_the_cut_is_order_invariant():
    # The bottom half of the population ends inside the income-tied
    # block (weight 3, cut 2.0): every tied row contributes 2/3 of its
    # weight, so the bottom mean is (10+20+30)·(2/3)/2 = 20 whatever
    # the rows' input order; the top quarter is exactly the rich row.
    frame = _frame([10.0, 20.0, 30.0, 40.0], income=[1, 1, 1, 2])
    for order in ([0, 1, 2, 3], [1, 0, 2, 3], [2, 1, 0, 3], [3, 2, 0, 1]):
        permuted = frame.iloc[order].reset_index(drop=True)
        assert equity.share_ratio(
            permuted, top=0.25, bottom=0.50, income="income"
        ) == pytest.approx(40.0 / 20.0, rel=1e-12)


def test_cut_boundaries_are_exact():
    frame = _frame([1.0, 2.0])
    assert equity.share_ratio(frame, top=0.5, bottom=0.5) == pytest.approx(2.0)
    assert (
        _distribution.weighted_median(np.array([1.0, 2.0]), np.array([1.0, 1.0])) == 1.0
    )
    assert _distribution.weighted_median(np.array([1.0, 2.0, 3.0]), np.ones(3)) == 2.0


def test_missing_data_excludes_the_row_and_its_weight():
    with_nan = _frame([1.0, 2.0, np.nan], [1.0, 1.0, 5.0])
    assert equity.gini_index(with_nan, population="pop") == pytest.approx(
        equity.gini_index(_frame([1.0, 2.0]))
    )
    nan_weight = _frame([1.0, 2.0, 9.0], [1.0, 1.0, np.nan])
    assert equity.gini_index(nan_weight, population="pop") == pytest.approx(
        equity.gini_index(_frame([1.0, 2.0]))
    )
    for bad in (-1.0, np.inf):
        with pytest.raises(ValueError, match="finite and non-negative"):
            equity.gini_index(_frame([1.0, 2.0], [1.0, bad]), population="pop")
    # An unconsumed income column never affects the value measures;
    # share_ratio(income=) consumes it and drops the row.
    with_income_nan = _frame([1.0, 2.0, 3.0], income=[1.0, np.nan, 3.0])
    assert equity.gini_index(with_income_nan) == pytest.approx(
        equity.gini_index(_frame([1.0, 2.0, 3.0]))
    )
    assert equity.share_ratio(
        with_income_nan, top=0.5, bottom=0.5, income="income"
    ) == pytest.approx(equity.share_ratio(_frame([1.0, 3.0]), top=0.5, bottom=0.5))
    # A NaN decomposition label drops its row from the decomposition.
    grouped = _frame([1.0, 2.0, 8.0], group=["a", "a", None])
    parts = equity.generalized_entropy(grouped, groups="group")
    assert parts["total"][0] == pytest.approx(
        equity.generalized_entropy(_frame([1.0, 2.0]))
    )


def test_empty_effective_distributions_refuse_by_group():
    frame = pd.DataFrame(
        {
            "opportunity": ["jobs", "jobs", "schools"],
            "accessibility": [1.0, 2.0, np.nan],
        }
    )
    with pytest.raises(ValueError, match="schools"):
        equity.gini_index(frame)
    with pytest.raises(ValueError, match="no usable rows"):
        equity.gini_index(_frame([np.nan, np.nan]))
    zero_weight = pd.DataFrame(
        {
            "opportunity": ["jobs", "schools"],
            "accessibility": [1.0, 2.0],
            "pop": [1.0, 0.0],
        }
    )
    with pytest.raises(ValueError, match="schools"):
        equity.gini_index(zero_weight, population="pop")


def test_return_shapes_follow_the_contract():
    bare = _frame([1.0, 2.0])
    assert isinstance(equity.gini_index(bare), float)
    grouped = equity.gini_index(
        pd.DataFrame({"budget": [30.0, 30.0], "accessibility": [1.0, 2.0]})
    )
    assert isinstance(grouped, pd.DataFrame) and len(grouped) == 1
    parts = equity.generalized_entropy(
        _frame([1.0, 2.0], group=["a", "b"]), groups="group"
    )
    assert isinstance(parts, pd.DataFrame)
    assert list(parts.columns) == ["total", "between", "within"]


def test_integer_ids_join_without_casting():
    data = pd.DataFrame(
        {
            "from_id": pd.array([10, 20], dtype="int32"),
            "accessibility": [1.0, 3.0],
        }
    )
    table = pd.DataFrame(
        {"from_id": pd.array([10, 20], dtype="int32"), "pop": [1.0, 2.0]}
    )
    assert equity.gini_index(
        data, sociodemographic_data=table, population="pop"
    ) == pytest.approx(
        equity.gini_index(_frame([1.0, 3.0], [1.0, 2.0]), population="pop")
    )


def test_the_accessibility_methods_delegate(network):
    from cafein import Accessibility

    stops = [stop for stop, lat, lon in network.stops if lat is not None]
    frame = Accessibility(
        network, stops[1000:1006], stops[1000:1040], DEPARTURE, budgets=(15.0,)
    )
    rng = np.random.default_rng(7)
    people = pd.DataFrame(
        {
            "id": frame["from_id"].unique(),
            "pop": rng.integers(50, 500, frame["from_id"].nunique()).astype(float),
        }
    )
    method = frame.gini_index(sociodemographic_data=people, population="pop")
    function = equity.gini_index(frame, sociodemographic_data=people, population="pop")
    if isinstance(method, pd.DataFrame):
        pd.testing.assert_frame_equal(method, function)
    else:
        assert method == pytest.approx(function)
    curve = frame.lorenz_curve(sociodemographic_data=people, population="pop")
    assert {"population_share", "value_share"} <= set(curve.columns)
    people = people.assign(income=np.linspace(20000, 60000, len(people)))
    ci_method = frame.concentration_index(
        income="income", sociodemographic_data=people, population="pop"
    )
    ci_function = equity.concentration_index(
        frame, income="income", sociodemographic_data=people, population="pop"
    )
    if isinstance(ci_method, pd.DataFrame):
        pd.testing.assert_frame_equal(ci_method, ci_function)
    else:
        assert ci_method == pytest.approx(ci_function)
    fgt_method = frame.fgt_poverty(
        poverty_line="60% of median",
        sociodemographic_data=people,
        population="pop",
    )
    fgt_function = equity.fgt_poverty(
        frame,
        poverty_line="60% of median",
        sociodemographic_data=people,
        population="pop",
    )
    if isinstance(fgt_method, pd.DataFrame):
        pd.testing.assert_frame_equal(fgt_method, fgt_function)
    else:
        assert fgt_method == pytest.approx(fgt_function)


def test_dropna_never_silently_loses_a_group():
    data = pd.DataFrame(
        {
            "from_id": [1, 2, 3],
            "opportunity": ["jobs", "jobs", "schools"],
            "accessibility": [1.0, 2.0, 3.0],
        }
    )
    # Origin 3 is the schools group's only row; dropping it via
    # dropna must refuse rather than lose the group.
    table = pd.DataFrame({"id": [1, 2], "pop": [1.0, 1.0]})
    with pytest.raises(ValueError, match="schools"):
        equity.gini_index(
            data, sociodemographic_data=table, population="pop", dropna=True
        )
    nobody = pd.DataFrame({"id": [9], "pop": [1.0]})
    with pytest.raises(ValueError, match="no usable rows"):
        equity.gini_index(
            data[["from_id", "accessibility"]],
            sociodemographic_data=nobody,
            population="pop",
            dropna=True,
        )


def test_reserved_and_colliding_column_names_stay_safe():
    # A user column may be named like the internal weight column; the
    # values must survive untouched.
    tricky = pd.DataFrame({"__cafein_weight__": [0.0, 1.0]})
    assert equity.gini_index(tricky, value="__cafein_weight__") == pytest.approx(0.5)
    # A group column named after the result column refuses instead of
    # silently overwriting either.
    collide = pd.DataFrame({"gini_index": ["a", "b"], "accessibility": [1.0, 2.0]})
    with pytest.raises(ValueError, match="collide"):
        equity.gini_index(collide, group_columns=["gini_index"])
    curve_collide = pd.DataFrame(
        {"value_share": ["a", "b"], "accessibility": [1.0, 2.0]}
    )
    with pytest.raises(ValueError, match="collide"):
        equity.lorenz_curve(curve_collide, group_columns=["value_share"])


def test_infinite_values_refuse_and_huge_weights_do_not_overflow():
    with pytest.raises(ValueError, match="finite values"):
        equity.gini_index(_frame([1.0, np.inf]))
    with pytest.raises(ValueError, match="finite values"):
        equity.share_ratio(
            _frame([1.0, 2.0], income=[1.0, np.inf]),
            top=0.5,
            bottom=0.5,
            income="income",
        )
    huge = _frame([1.0, 2.0, 5.0], [0.8e70, 1.6e70, 0.8e70])
    small = _frame([1.0, 2.0, 5.0], [1.0, 2.0, 1.0])
    for name, measure in SCALARS.items():
        assert measure(huge, population="pop") == pytest.approx(
            measure(small, population="pop"), rel=1e-12
        ), name


def test_definition_formula_goldens_on_an_unequal_weighted_distribution():
    values, weights = [1.0, 3.0, 9.0], [2.0, 1.0, 3.0]
    frame = _frame(values, weights)
    total = sum(weights)
    mean = sum(v * w for v, w in zip(values, weights)) / total
    for alpha in (0.5, 2.0):
        expected = (
            sum(w / total * (v / mean) ** alpha for v, w in zip(values, weights)) - 1.0
        ) / (alpha * (alpha - 1.0))
        assert equity.generalized_entropy(
            frame, alpha=alpha, population="pop"
        ) == pytest.approx(expected, rel=1e-12)
    theil = sum(
        (w / total) * (v / mean) * np.log(v / mean) for v, w in zip(values, weights)
    )
    assert equity.theil_t(frame, population="pop") == pytest.approx(theil, rel=1e-12)
    log_deviation = sum((w / total) * np.log(mean / v) for v, w in zip(values, weights))
    assert equity.mld(frame, population="pop") == pytest.approx(
        log_deviation, rel=1e-12
    )
    for epsilon in (0.5, 2.0):
        equivalent = sum(
            (w / total) * v ** (1.0 - epsilon) for v, w in zip(values, weights)
        ) ** (1.0 / (1.0 - epsilon))
        assert equity.atkinson(
            frame, epsilon=epsilon, population="pop"
        ) == pytest.approx(1.0 - equivalent / mean, rel=1e-12)
    geometric = np.exp(sum((w / total) * np.log(v) for v, w in zip(values, weights)))
    assert equity.atkinson(frame, epsilon=1, population="pop") == pytest.approx(
        1.0 - geometric / mean, rel=1e-12
    )
    kappa = 0.7
    kolm = (
        np.log(
            sum(
                (w / total) * np.exp(kappa * (mean - v))
                for v, w in zip(values, weights)
            )
        )
        / kappa
    )
    assert equity.kolm(frame, kappa=kappa, population="pop") == pytest.approx(
        kolm, rel=1e-12
    )
    hoover = sum(w * abs(v - mean) for v, w in zip(values, weights)) / (
        2.0 * sum(v * w for v, w in zip(values, weights))
    )
    assert equity.hoover(frame, population="pop") == pytest.approx(hoover, rel=1e-12)


def test_a_sociodemographic_identifier_name_does_not_regroup():
    data = pd.DataFrame({"from_id": [1, 2], "accessibility": [1.0, 3.0]})
    table = pd.DataFrame({"id": [1, 2], "budget": [30.0, 60.0], "pop": [1.0, 1.0]})
    result = equity.gini_index(data, sociodemographic_data=table, population="pop")
    assert isinstance(result, float)
    # An explicit override may still group by the joined column.
    grouped = equity.gini_index(
        data,
        sociodemographic_data=table,
        population="pop",
        group_columns=["budget"],
    )
    assert isinstance(grouped, pd.DataFrame) and len(grouped) == 2


def test_weights_rescale_per_group_not_globally():
    huge_and_tiny = pd.DataFrame(
        {
            "opportunity": ["a"] * 3 + ["b"] * 3,
            "accessibility": [1.0, 2.0, 5.0] * 2,
            "pop": [0.8e70, 1.6e70, 0.8e70, 0.8e-70, 1.6e-70, 0.8e-70],
        }
    )
    result = equity.gini_index(huge_and_tiny, population="pop")
    expected = equity.gini_index(
        _frame([1.0, 2.0, 5.0], [1.0, 2.0, 1.0]),
        population="pop",
        group_columns=[],
    )
    assert result["gini_index"].tolist() == pytest.approx([expected] * 2)
    spanning = _frame([1.0, 2.0], [1e-308, 1e308])
    with pytest.raises(ValueError, match="supported numeric range"):
        equity.gini_index(spanning, population="pop")


def test_huge_values_do_not_overflow_the_scale_invariant_measures():
    huge = _frame([0.4e70, 0.8e70, 1.6e70], [1.0, 2.0, 1.0])
    small = _frame([1.0, 2.0, 4.0], [1.0, 2.0, 1.0])
    for name in (
        "gini_index",
        "share_ratio",
        "generalized_entropy",
        "atkinson",
        "hoover",
    ):
        assert SCALARS[name](huge, population="pop") == pytest.approx(
            SCALARS[name](small, population="pop"), rel=1e-12
        ), name
    curve = equity.lorenz_curve(huge, population="pop")
    assert np.isfinite(curve["value_share"]).all()


def test_dropna_with_a_joined_group_column_drops_cleanly():
    data = pd.DataFrame({"from_id": [1, 2, 3], "accessibility": [1.0, 2.0, 3.0]})
    table = pd.DataFrame({"id": [1, 2], "zone": ["west", "east"], "pop": [1.0, 1.0]})
    # Origin 3 is unmatched; its synthetic NaN zone must not be
    # counted as an expected group once dropna sanctions the drop.
    result = equity.gini_index(
        data,
        sociodemographic_data=table,
        population="pop",
        group_columns=["zone"],
        dropna=True,
    )
    assert result["zone"].tolist() == ["east", "west"]


def test_mixed_type_group_labels_decompose():
    frame = _frame([1.0, 2.0, 3.0, 4.0], group=[1, 1, "x", "x"])
    parts = equity.generalized_entropy(frame, groups="group")
    assert parts["total"][0] == pytest.approx(
        parts["between"][0] + parts["within"][0], rel=1e-12
    )


def test_extreme_parameters_stay_representable():
    # Large finite aversion: the equivalent value approaches the
    # minimum; the direct power mean overflows but the index is plain.
    frame = _frame([1.0, 2.0])
    epsilon = 1026.0
    equivalent = 0.5 ** (1.0 / (1.0 - epsilon))
    assert equity.atkinson(frame, epsilon=epsilon) == pytest.approx(
        1.0 - equivalent / 1.5, rel=1e-9
    )
    # Opposite float extremes under a tiny kappa: kappa applies before
    # the difference, so nothing overflows on the way to a finite index.
    # Beyond the envelope the refusal names the supported range.
    fmax = np.finfo(float).max
    spread = _frame([-fmax, fmax], [1.0, 2.0])
    with pytest.raises(ValueError, match="supported numeric range"):
        equity.kolm(spread, population="pop", kappa=1e-6)
    # Just inside it, a lopsided tiny mass still computes.
    tiny = _frame([0.0, 1e-70], [1.0, 1e-70])
    assert equity.gini_index(tiny, population="pop") == pytest.approx(
        equity.gini_index(_frame([0.0, 1.0], [1.0, 1e-70]), population="pop")
    )
    with pytest.raises(ValueError, match="supported numeric range"):
        equity.generalized_entropy(_frame([5e-324, 1e308]))


def test_alpha_rides_the_limit_branches_at_the_float_edges():
    frame = _frame([1.0, 3.0, 9.0], [2.0, 1.0, 3.0])
    tiny = np.nextafter(0.0, 1.0)
    near_one = np.nextafter(1.0, 2.0)
    assert equity.generalized_entropy(
        frame, alpha=tiny, population="pop"
    ) == pytest.approx(equity.mld(frame, population="pop"), rel=1e-9)
    assert equity.generalized_entropy(
        frame, alpha=near_one, population="pop"
    ) == pytest.approx(equity.theil_t(frame, population="pop"), rel=1e-9)


def test_extreme_aversion_parameters_reach_their_limits():
    frame = _frame([0.1, 1.0], [1.0, 1.0])
    # Aversion at the float ceiling: the equivalent value is the
    # minimum, so the index is 1 - min/mean, not NaN.
    with pytest.raises(ValueError, match="at most 1e6"):
        equity.atkinson(frame, epsilon=1e308)
    equivalent = 0.1 * 0.5 ** (1.0 / (1.0 - 1e6))
    assert equity.atkinson(frame, epsilon=1e6) == pytest.approx(
        1.0 - equivalent / 0.55, rel=1e-9
    )
    # Huge equal values under a large kappa: zero inequality, not
    # inf - inf.
    equal = _frame([1e70, 1e70])
    assert equity.kolm(equal, kappa=10.0) == 0.0


def test_an_infinite_value_on_a_zero_weight_row_is_excluded_first():
    frame = _frame([1.0, np.inf, 2.0], [1.0, 0.0, 1.0])
    assert equity.gini_index(frame, population="pop") == pytest.approx(
        equity.gini_index(_frame([1.0, 2.0]))
    )


def test_a_group_column_named_palma_ratio_refuses():
    frame = pd.DataFrame(
        {
            "palma_ratio": ["a", "a", "b", "b"],
            "accessibility": [1.0, 2.0, 3.0, 4.0],
            "income": [1.0, 2.0, 3.0, 4.0],
        }
    )
    with pytest.raises(ValueError, match="collide.*palma_ratio"):
        equity.palma_ratio(frame, income="income", group_columns=["palma_ratio"])


def test_log_domain_forms_survive_subnormal_against_ordinary_values():
    frame = _frame([1e-70, 1.0])
    deviations = [np.log(v / 0.5) for v in (1e-70, 1.0)]
    assert equity.mld(frame) == pytest.approx(
        -sum(0.5 * d for d in deviations), rel=1e-9
    )
    theil = sum(0.5 * (v / 0.5) * np.log(v / 0.5) for v in (1e-70, 1.0))
    assert equity.theil_t(frame) == pytest.approx(theil, rel=1e-9)
    assert np.isfinite(equity.generalized_entropy(frame, alpha=0.5))
    # Epsilon just outside the snap band on the same extreme range.
    result = equity.atkinson(frame, epsilon=1.0 + 1e-6)
    assert 0.0 <= result <= 1.0 and np.isfinite(result)


def test_kolm_handles_opposite_extremes_on_both_sides_of_kappa_one():
    spread = _frame([-1e70, 1e70])
    for kappa in (1.0, 2.0):
        result = equity.kolm(spread, kappa=kappa)
        assert np.isfinite(result)
        assert result == pytest.approx(1e70, rel=1e-6)
    still = equity.kolm(_frame([-1e70, 1e70], [1.0, 2.0]), population="pop", kappa=1e-6)
    assert np.isfinite(still) and still > 0


def test_nan_identifier_labels_are_excluded_not_grouped():
    frame = pd.DataFrame(
        {
            "opportunity": ["jobs", "jobs", None],
            "accessibility": [1.0, 2.0, 9.0],
        }
    )
    result = equity.gini_index(frame)
    assert result["opportunity"].tolist() == ["jobs"]
    assert result["gini_index"][0] == pytest.approx(
        equity.gini_index(_frame([1.0, 2.0]))
    )


def test_palma_group_columns_named_after_either_ratio():
    frame = pd.DataFrame(
        {
            "share_ratio": ["a", "a", "b", "b"],
            "accessibility": [1.0, 2.0, 3.0, 4.0],
            "income": [1.0, 2.0, 3.0, 4.0],
        }
    )
    # The temporary name never surfaces: grouping by share_ratio works.
    result = equity.palma_ratio(frame, income="income", group_columns=["share_ratio"])
    assert list(result.columns) == ["share_ratio", "palma_ratio"]
    with pytest.raises(ValueError, match="collide"):
        equity.share_ratio(frame, income="income", group_columns=["share_ratio"])


def test_the_remaining_float_horizons():
    unequal = _frame([1.0, 3.0, 9.0], [2.0, 1.0, 3.0])
    # Alpha beyond the parameter envelope refuses by name; at the
    # ceiling the index may honestly overflow to inf, never NaN.
    with pytest.raises(ValueError, match="at most 1e6"):
        equity.generalized_entropy(unequal, alpha=1e300, population="pop")
    ceiling = equity.generalized_entropy(unequal, alpha=1e6, population="pop")
    assert not np.isnan(ceiling)
    # A lopsided tiny share beside a full one under ceiling aversion.
    tiny_share = _frame([0.1, 1.0], [1e-70, 1.0])
    result = equity.atkinson(tiny_share, population="pop", epsilon=1e6)
    assert 0.0 <= result <= 1.0 and np.isfinite(result)
    # A subnormal kappa keeps the second cumulant: (kappa/2)*Var.
    frame = _frame([0.0, 1.0])
    kappa = 1e-308
    assert equity.kolm(frame, kappa=kappa) == pytest.approx(
        kappa * 0.25 / 2.0, rel=1e-6
    )
    # A subnormal population share cut: the bottom tail mean is the
    # poorest row's value, not an underflown zero.
    cut = float(np.nextafter(0, 1))
    assert equity.share_ratio(
        _frame([0.25, 1.0]), top=0.5, bottom=cut
    ) == pytest.approx(1.0 / 0.25, rel=1e-12)


def test_round_seven_float_corners():
    # A minimum-subnormal weight beside full ones under a huge
    # negative alpha: the low row's log term dominates to overflow.
    frame = _frame([0.1, 1.0, 1.0], [1e-70, 1.0, 1.0])
    result = equity.generalized_entropy(frame, alpha=-1000, population="pop")
    assert result == float("inf")
    # Kolm keeps a tiny share at the minimum under a large kappa.
    tiny_min = _frame([0.0, 1.0], [1e-20, 1.0])
    kolm = equity.kolm(tiny_min, population="pop", kappa=100.0)
    expected = (
        (1.0 * 1e-20 + 1.0) / (1e-20 + 1.0)
        - 1.0
        + (
            np.log((1e-20 * np.exp(0.0) + 1.0 * np.exp(-100.0)) / (1e-20 + 1.0)) / 100.0
            + 1.0
        )
    )
    assert np.isfinite(kolm)
    assert kolm == pytest.approx(expected, rel=1e-9)
    # The decomposition resolves an underflowed factor against an
    # overflowed block index without NaN.
    grouped = _frame(
        [1e-5, 1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 1.0],
        group=["low", "low", "high", "high"],
    )
    parts = equity.generalized_entropy(
        grouped, alpha=2000.0, groups="group", population="pop"
    )
    assert not parts.isna().any().any()
    # Subnormal-scale products inside a tail block keep their mean.
    frame = _frame(
        [1e-70, 2e-70],
        [1.0, 1.0],
        income=[1.0, 2.0],
    )
    assert equity.share_ratio(
        frame, top=0.5, bottom=0.5, income="income", population="pop"
    ) == pytest.approx(2.0, rel=1e-9)
    with pytest.raises(ValueError, match="supported numeric range"):
        equity.share_ratio(
            _frame([1e-200, 1.0], income=[1.0, 2.0]),
            top=0.5,
            bottom=0.5,
            income="income",
        )
    # Complex data refuses by name instead of silently casting.
    with pytest.raises(ValueError, match="real values"):
        equity.gini_index(pd.DataFrame({"accessibility": [1 + 2j, 3 + 0j]}))


def test_object_payloads_and_big_parameters_refuse_by_name():
    from decimal import Decimal

    below = pd.DataFrame({"accessibility": [Decimal("1e-400"), Decimal(1)]})
    with pytest.raises(ValueError, match="supported numeric range"):
        equity.gini_index(below)
    big = pd.DataFrame({"accessibility": pd.Series([10**400, 1], dtype=object)})
    with pytest.raises(ValueError, match="supported numeric range"):
        equity.gini_index(big)
    frame = _frame([1.0, 2.0])
    with pytest.raises(ValueError, match="at most 1e6"):
        equity.generalized_entropy(frame, alpha=10**400)
    with pytest.raises(ValueError, match="at most 1e6"):
        equity.atkinson(frame, epsilon=10**400)
    with pytest.raises(ValueError, match="at most 1e6"):
        equity.kolm(frame, kappa=10**400)


def test_an_underflowed_factor_keeps_its_within_contribution():
    # The low group's factor underflows to zero at this alpha while
    # its block index stays an ordinary float; the decomposition
    # identity must still hold.
    grouped = _frame(
        [0.5, 1.0, 1.0, 2.0],
        group=["low", "low", "high", "high"],
    )
    parts = equity.generalized_entropy(grouped, alpha=-3000.0, groups="group")
    assert not parts.isna().any().any()
    total = equity.generalized_entropy(grouped, alpha=-3000.0)
    if np.isfinite(total):
        assert parts["total"][0] == pytest.approx(
            parts["between"][0] + parts["within"][0], rel=1e-9
        )


def test_concentration_index_goldens_and_identities():
    frame = _frame([1.0, 2.0, 3.0, 4.0], income=[10, 20, 30, 40])
    # Perfect rank correlation: hand value and the covariance form.
    ranks = np.array([1 / 8, 3 / 8, 5 / 8, 7 / 8])
    mean = 2.5
    covariance = float(
        np.dot(np.full(4, 0.25), (np.arange(1, 5) - mean) * (ranks - 0.5))
    )
    standard = equity.concentration_index(frame, income="income")
    assert standard == pytest.approx(0.25, rel=1e-12)
    assert standard == pytest.approx(2.0 * covariance / mean, rel=1e-12)
    absolute = equity.concentration_index(frame, income="income", variant="absolute")
    assert absolute == pytest.approx(2.0 * covariance, rel=1e-12)
    assert absolute / mean == pytest.approx(standard, rel=1e-12)
    # The scalar integrates its own curve.
    curve = equity.concentration_curve(frame, income="income")
    area = _distribution.polyline_area(
        curve["population_share"].to_numpy(), curve["value_share"].to_numpy()
    )
    assert standard == pytest.approx(1.0 - 2.0 * area, rel=1e-12)
    # Pro-poor concentration flips the sign.
    reverse = _frame([4.0, 3.0, 2.0, 1.0], income=[10, 20, 30, 40])
    assert equity.concentration_index(reverse, income="income") == pytest.approx(
        -0.25, rel=1e-12
    )
    # The absolute variant is linear in the values and delta-safe.
    tripled = _frame([3.0, 6.0, 9.0, 12.0], income=[10, 20, 30, 40])
    assert equity.concentration_index(
        tripled, income="income", variant="absolute"
    ) == pytest.approx(3.0 * absolute, rel=1e-12)
    delta = _frame([-1.0, 2.0, -3.0, 4.0], income=[10, 20, 30, 40])
    assert np.isfinite(
        equity.concentration_index(delta, income="income", variant="absolute")
    )
    with pytest.raises(ValueError, match="absolute"):
        equity.concentration_index(delta, income="income")


def test_concentration_ties_collapse_to_one_chord():
    frame = _frame([10.0, 20.0, 30.0, 40.0], income=[1, 1, 1, 2])
    baseline = equity.concentration_index(frame, income="income")
    baseline_curve = equity.concentration_curve(frame, income="income")
    # One vertex per tied block: origin + tie block + the rich row.
    assert len(baseline_curve) == 3
    for order in ([1, 0, 2, 3], [2, 1, 0, 3]):
        permuted = frame.iloc[order].reset_index(drop=True)
        assert equity.concentration_index(permuted, income="income") == pytest.approx(
            baseline, rel=1e-12
        )
        pd.testing.assert_frame_equal(
            equity.concentration_curve(permuted, income="income"),
            baseline_curve,
        )


def test_corrected_concentration_variants():
    frame = _frame([1.0, 2.0, 3.0, 4.0], income=[10, 20, 30, 40])
    standard = 0.25
    mean = 2.5
    erreygers = equity.concentration_index(
        frame, income="income", variant="erreygers", bounds=(0, 10)
    )
    assert erreygers == pytest.approx(4.0 * mean * standard / 10.0, rel=1e-12)
    wagstaff = equity.concentration_index(
        frame, income="income", variant="wagstaff", bounds=(0, 10)
    )
    assert wagstaff == pytest.approx(
        standard * mean * 10.0 / ((10.0 - mean) * mean), rel=1e-12
    )
    # One global bounds serves groups with different observed extrema.
    grouped = pd.DataFrame(
        {
            "opportunity": ["a"] * 4 + ["b"] * 4,
            "accessibility": [1.0, 2.0, 3.0, 4.0, 2.0, 4.0, 6.0, 8.0],
            "income": [10, 20, 30, 40] * 2,
        }
    )
    result = equity.concentration_index(
        grouped, income="income", variant="erreygers", bounds=(0, 10)
    )
    by_group = dict(zip(result["opportunity"], result["concentration_index"]))
    assert by_group["a"] == pytest.approx(4.0 * 2.5 * 0.25 / 10.0, rel=1e-12)
    assert by_group["b"] == pytest.approx(4.0 * 5.0 * 0.25 / 10.0, rel=1e-12)


def test_concentration_refusals():
    frame = _frame([1.0, 2.0], income=[1.0, 2.0])
    with pytest.raises(ValueError, match="requires bounds="):
        equity.concentration_index(frame, income="income", variant="erreygers")
    with pytest.raises(ValueError, match="serves the erreygers"):
        equity.concentration_index(frame, income="income", bounds=(0, 10))
    with pytest.raises(ValueError, match="lower < upper"):
        equity.concentration_index(
            frame, income="income", variant="wagstaff", bounds=(10, 0)
        )
    with pytest.raises(ValueError, match="outside bounds="):
        equity.concentration_index(
            frame, income="income", variant="erreygers", bounds=(0, 1.5)
        )
    saturated = _frame([5.0, 5.0], income=[1.0, 2.0])
    with pytest.raises(ValueError, match="wagstaff.*undefined"):
        equity.concentration_index(
            saturated, income="income", variant="wagstaff", bounds=(0, 5)
        )
    with pytest.raises(ValueError, match="supported numeric range"):
        equity.concentration_index(
            frame, income="income", variant="wagstaff", bounds=(-1e308, 1e308)
        )
    with pytest.raises(ValueError, match="must be standard"):
        equity.concentration_index(frame, income="income", variant="nope")
    with pytest.raises(ValueError, match="requires income="):
        equity.concentration_index(frame, income=None)


def test_suits_goldens_and_domain():
    # Access proportional to income integrates to zero.
    proportional = _frame([1.0, 2.0, 3.0, 4.0], income=[10, 20, 30, 40])
    assert equity.suits(proportional, income="income") == pytest.approx(0.0, abs=1e-12)
    # Equal access against unequal income is pro-poor: hand trapezoid.
    equal = _frame([1.0, 1.0, 1.0, 1.0], income=[10, 20, 30, 40])
    axis = np.array([0.0, 0.1, 0.3, 0.6, 1.0])
    mass = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    expected = 1.0 - 2.0 * _distribution.polyline_area(axis, mass)
    assert equity.suits(equal, income="income") == pytest.approx(expected, rel=1e-12)
    assert expected < 0
    # Within-tie permutation invariance (income ties, distinct values).
    tied = _frame([10.0, 20.0, 30.0], income=[1, 1, 2])
    baseline = equity.suits(tied, income="income")
    permuted = tied.iloc[[1, 0, 2]].reset_index(drop=True)
    assert equity.suits(permuted, income="income") == pytest.approx(baseline, rel=1e-12)
    with pytest.raises(ValueError, match="non-negative incomes"):
        equity.suits(_frame([1.0, 2.0], income=[-1.0, 2.0]), income="income")
    with pytest.raises(ValueError, match="positive weighted income total"):
        equity.suits(_frame([1.0, 2.0], income=[0.0, 0.0]), income="income")


def test_concentration_family_shares_the_calling_convention():
    weighted = _frame([1.0, 2.0, 5.0], [1.0, 2.0, 1.0], income=[3.0, 2.0, 1.0])
    duplicated = _frame([1.0, 2.0, 2.0, 5.0], income=[3.0, 2.0, 2.0, 1.0])
    for call in (
        lambda f, **kw: equity.concentration_index(f, income="income", **kw),
        lambda f, **kw: equity.concentration_index(
            f, income="income", variant="absolute", **kw
        ),
        lambda f, **kw: equity.concentration_index(
            f, income="income", variant="erreygers", bounds=(0, 10), **kw
        ),
        lambda f, **kw: equity.concentration_index(
            f, income="income", variant="wagstaff", bounds=(0, 10), **kw
        ),
        lambda f, **kw: equity.suits(f, income="income", **kw),
    ):
        assert call(weighted, population="pop") == pytest.approx(
            call(duplicated), rel=1e-12
        )
    # The duplicated row shares its income, so it merges into the same
    # tied block: the curve FRAMES are exactly equal.
    pd.testing.assert_frame_equal(
        equity.concentration_curve(weighted, income="income", population="pop"),
        equity.concentration_curve(duplicated, income="income"),
    )
    # NaN income drops its row, weight included.
    with_nan = _frame([1.0, 2.0, 9.0], income=[1.0, 2.0, np.nan])
    assert equity.concentration_index(with_nan, income="income") == pytest.approx(
        equity.concentration_index(
            _frame([1.0, 2.0], income=[1.0, 2.0]), income="income"
        )
    )
    # Grouped fan-out carries the identifier columns.
    grouped = pd.DataFrame(
        {
            "budget": [30.0, 30.0, 60.0, 60.0],
            "accessibility": [1.0, 2.0, 3.0, 4.0],
            "income": [1.0, 2.0, 1.0, 2.0],
        }
    )
    result = equity.suits(grouped, income="income")
    assert list(result.columns) == ["budget", "suits"]


def test_fgt_goldens_on_both_tails():
    frame = _frame([1.0, 2.0, 3.0, 4.0])
    assert equity.fgt_poverty(frame, poverty_line=2.5) == pytest.approx(0.5)
    assert equity.fgt_poverty(frame, poverty_line=2.5, alpha=1) == pytest.approx(
        (1.5 / 2.5 + 0.5 / 2.5) / 4.0, rel=1e-12
    )
    assert equity.fgt_poverty(frame, poverty_line=2.5, alpha=2) == pytest.approx(
        ((1.5 / 2.5) ** 2 + (0.5 / 2.5) ** 2) / 4.0, rel=1e-12
    )
    above = equity.fgt_poverty(frame, poverty_line=2.5, deprived="above", alpha=1)
    assert above == pytest.approx((0.5 / 2.5 + 1.5 / 2.5) / 4.0, rel=1e-12)
    # A value exactly on the line is not poor, on either tail.
    boundary = _frame([2.5, 4.0])
    assert equity.fgt_poverty(boundary, poverty_line=2.5) == 0.0
    assert (
        equity.fgt_poverty(_frame([1.0, 2.5]), poverty_line=2.5, deprived="above")
        == 0.0
    )


def test_relative_poverty_lines_use_the_weighted_median_per_group():
    frame = _frame([1.0, 2.0, 3.0, 4.0])
    # The type-1 weighted median is 2, so the line is 1.2: one poor row.
    assert equity.fgt_poverty(frame, poverty_line="60% of median") == pytest.approx(
        0.25
    )
    grouped = pd.DataFrame(
        {
            "opportunity": ["a"] * 4 + ["b"] * 4,
            "accessibility": [1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0],
        }
    )
    result = equity.fgt_poverty(grouped, poverty_line="60% of median")
    # Each group draws its own line, so the scaled group matches.
    assert result["fgt_poverty"].tolist() == pytest.approx([0.25, 0.25])
    with pytest.raises(ValueError, match="60% of median"):
        equity.fgt_poverty(frame, poverty_line="median times 0.6")
    with pytest.raises(ValueError, match="strictly positive poverty line"):
        equity.fgt_poverty(frame, poverty_line=-1.0)
    with pytest.raises(ValueError, match="at most 1e6"):
        equity.fgt_poverty(frame, poverty_line=1.0, alpha=-1)
    with pytest.raises(ValueError, match='"below" or "above"'):
        equity.fgt_poverty(frame, poverty_line=1.0, deprived="under")


def test_cost_burden_golden_and_detail():
    frame = pd.DataFrame(
        {
            "from_id": ["a", "b"],
            "spend": [10.0, 30.0],
            "earnings": [100.0, 100.0],
        }
    )
    result = equity.cost_burden(frame, cost="spend", income="earnings")
    assert isinstance(result, pd.DataFrame) and len(result) == 1
    # Strictly above the 10 % rule: the 0.1 burden is not burdened.
    assert result["cost_burden"][0] == pytest.approx(0.5)
    assert result["mean_burden"][0] == pytest.approx(0.2)
    detail = equity.cost_burden(frame, cost="spend", income="earnings", detail=True)
    assert detail["burden"].tolist() == pytest.approx([0.1, 0.3])
    assert detail["cost_burdened"].tolist() == [False, True]
    assert detail["from_id"].tolist() == ["a", "b"]
    with pytest.raises(ValueError, match="strictly positive incomes.*b"):
        equity.cost_burden(
            frame.assign(earnings=[100.0, 0.0]), cost="spend", income="earnings"
        )
    with pytest.raises(ValueError, match="non-negative costs"):
        equity.cost_burden(
            frame.assign(spend=[-1.0, 30.0]), cost="spend", income="earnings"
        )


def test_residual_income_is_per_origin():
    frame = pd.DataFrame(
        {
            "from_id": ["a", "b"],
            "spend": [10.0, 130.0],
            "earnings": [100.0, 100.0],
        }
    )
    result = equity.residual_income(frame, cost="spend", income="earnings")
    assert result["residual_income"].tolist() == pytest.approx([90.0, -30.0])
    assert result["from_id"].tolist() == ["a", "b"]
    # A negative cost — a subsidy — is legal here (only the burden
    # ratios require non-negative costs) and raises the residual.
    subsidised = equity.residual_income(
        frame.assign(spend=[-10.0, 130.0]), cost="spend", income="earnings"
    )
    assert subsidised["residual_income"].tolist() == pytest.approx([110.0, -30.0])
    # Grouping by from_id itself never duplicates the carried column.
    keyed = equity.residual_income(
        frame, cost="spend", income="earnings", group_columns=["from_id"]
    )
    assert list(keyed.columns) == ["from_id", "residual_income"]


def test_lihc_goldens():
    frame = pd.DataFrame(
        {
            "spend": [10.0, 30.0, 20.0, 40.0],
            "earnings": [100.0, 50.0, 100.0, 60.0],
        }
    )
    result = equity.lihc(frame, cost="spend", income="earnings", poverty_line=25.0)
    # Median cost (type-1) is 20; residuals [90, 20, 80, 20].
    assert result["lihc"][0] == pytest.approx(0.5)
    assert result["high_costs"][0] == pytest.approx(0.5)
    assert result["low_residual"][0] == pytest.approx(0.5)
    relative = equity.lihc(
        frame, cost="spend", income="earnings", poverty_line="60% of median"
    )
    # 60 % of the median residual (20) is 12: nobody sits below it.
    assert relative["lihc"][0] == 0.0
    assert relative["low_residual"][0] == 0.0
    detail = equity.lihc(
        frame, cost="spend", income="earnings", poverty_line=25.0, detail=True
    )
    assert detail["lihc"].tolist() == [False, True, False, True]
    assert detail["costs_above_median"].tolist() == [False, True, False, True]
    # A relative line against a negative median residual refuses.
    broke = frame.assign(earnings=[15.0, 25.0, 15.0, 30.0])
    with pytest.raises(ValueError, match="inherits the sign"):
        equity.lihc(
            broke, cost="spend", income="earnings", poverty_line="60% of median"
        )


def test_poverty_family_shares_the_calling_convention():
    weighted = _frame([1.0, 2.0, 5.0], [1.0, 2.0, 1.0])
    duplicated = _frame([1.0, 2.0, 2.0, 5.0])
    for kwargs in ({"alpha": 0}, {"alpha": 1}, {"alpha": 2}):
        assert equity.fgt_poverty(
            weighted, poverty_line=2.5, population="pop", **kwargs
        ) == pytest.approx(
            equity.fgt_poverty(duplicated, poverty_line=2.5, **kwargs), rel=1e-12
        )
    burdened = pd.DataFrame(
        {
            "budget": [30.0, 30.0, 60.0, 60.0],
            "spend": [10.0, 30.0, 10.0, 30.0],
            "earnings": [100.0, 100.0, 50.0, 50.0],
        }
    )
    result = equity.cost_burden(burdened, cost="spend", income="earnings")
    assert list(result.columns) == ["budget", "cost_burden", "mean_burden"]
    assert len(result) == 2
    # Negative accessibility values refuse with the delta guidance.
    with pytest.raises(ValueError, match="kolm"):
        equity.fgt_poverty(_frame([-1.0, 2.0]), poverty_line=1.0)


def test_the_burden_family_weighting_and_missing_data():
    weighted = pd.DataFrame(
        {
            "spend": [10.0, 30.0, 40.0],
            "earnings": [100.0, 100.0, 50.0],
            "pop": [1.0, 2.0, 1.0],
        }
    )
    duplicated = pd.DataFrame(
        {
            "spend": [10.0, 30.0, 30.0, 40.0],
            "earnings": [100.0, 100.0, 100.0, 50.0],
        }
    )
    pd.testing.assert_frame_equal(
        equity.cost_burden(weighted, cost="spend", income="earnings", population="pop"),
        equity.cost_burden(duplicated, cost="spend", income="earnings"),
    )
    pd.testing.assert_frame_equal(
        equity.lihc(
            weighted,
            cost="spend",
            income="earnings",
            poverty_line=65.0,
            population="pop",
        ),
        equity.lihc(duplicated, cost="spend", income="earnings", poverty_line=65.0),
    )
    # Hand-checked weighted summary: burdens [0.1, 0.3, 0.8] with
    # weights [1, 2, 1] -> strict-10% headcount 3/4, mean 0.375.
    summary = equity.cost_burden(
        weighted, cost="spend", income="earnings", population="pop"
    )
    assert summary["cost_burden"][0] == pytest.approx(0.75)
    assert summary["mean_burden"][0] == pytest.approx(
        (0.1 + 2 * 0.3 + 0.8) / 4.0, rel=1e-12
    )
    # A NaN cost or income drops its row, weight included, from every
    # burden computation.
    with_nan_cost = weighted.assign(spend=[10.0, np.nan, 40.0])
    trimmed = pd.DataFrame(
        {"spend": [10.0, 40.0], "earnings": [100.0, 50.0], "pop": [1.0, 1.0]}
    )
    pd.testing.assert_frame_equal(
        equity.cost_burden(
            with_nan_cost, cost="spend", income="earnings", population="pop"
        ),
        equity.cost_burden(trimmed, cost="spend", income="earnings", population="pop"),
    )
    with_nan_income = weighted.assign(earnings=[100.0, np.nan, 50.0])
    assert (
        len(equity.residual_income(with_nan_income, cost="spend", income="earnings"))
        == 2
    )
    pd.testing.assert_frame_equal(
        equity.lihc(
            with_nan_income,
            cost="spend",
            income="earnings",
            poverty_line=65.0,
            population="pop",
        ),
        equity.lihc(
            trimmed,
            cost="spend",
            income="earnings",
            poverty_line=65.0,
            population="pop",
        ),
    )
