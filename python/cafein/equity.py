"""Equity indices over accessibility distributions.

Frame-in functions following the IPEA ``{accessibility}`` package's
inequality-and-poverty toolbox: pass an :class:`~cafein.Accessibility`
frame (or any frame with a per-origin value column), optionally a
sociodemographic table joined on the origin ids, and read the
population-weighted index back — a plain float, or a tidy frame with
one row per identifier group (``opportunity``, ``budget``,
``percentile``) when the input carries several distributions at once.
The same functions are exposed as methods on ``Accessibility``.

The indices are descriptive: they state how access is distributed,
not whether that distribution is just — the normative judgment enters
through the parameters (the Atkinson ``epsilon``, poverty lines,
cutoffs). For choosing them, see the transport-justice literature
(Martens 2016; Pereira, Schwanen & Banister 2017).

Scenario differences ride the same functions. Build the Δ frame with
a keyed merge — never positional subtraction, which misaligns rows
and corrupts identifier columns::

    keys = ["from_id", "opportunity", "budget"]  # + "percentile" when windowed
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

Most indices are undefined on the negative values a Δ frame carries
and refuse; ``kolm`` (translation-invariant) and
``concentration_index(variant="absolute")`` are the Δ-safe readings.

A typical analysis follows the IPEA vignette's flow — compute
accessibility, join the sociodemographics, then read the indices off
one frame::

    reachable = cafein.Accessibility(
        network, zones, jobs, "2022-02-22 08:30", budgets=(30.0,)
    )
    demo = ...  # id, pop, income, region — one row per origin zone

    reachable.gini_index(sociodemographic_data=demo, population="pop")
    reachable.palma_ratio(
        income="income", sociodemographic_data=demo, population="pop"
    )
    reachable.theil_t(
        groups="region", sociodemographic_data=demo, population="pop"
    )
    reachable.concentration_index(
        income="income", sociodemographic_data=demo, population="pop"
    )
    reachable.fgt_poverty(
        poverty_line="60% of median",
        sociodemographic_data=demo,
        population="pop",
    )
    reachable.lihc(
        cost="transport_cost",
        income="income",
        poverty_line="60% of median",
        sociodemographic_data=demo,
        population="pop",
    )
    reachable.alkire_foster(
        dimensions={"accessibility": 10000, "transport_cost": (">", 250)},
        k=2,
        sociodemographic_data=demo,
        population="pop",
    )
"""

import re

import numpy as np
import pandas as pd

from cafein import _distribution

_IDENTIFIER_COLUMNS = ("opportunity", "budget", "percentile")

_NEGATIVE_MESSAGE = (
    "{name} needs non-negative values; for difference frames use the "
    "delta-safe readings, kolm or concentration_index(variant='absolute')"
)
_ZERO_MESSAGE = (
    "{name} needs strictly positive values and the distribution "
    "contains zeros; gini_index tolerates zeros"
)


def _column_list(frame, columns, label):
    if isinstance(columns, str):
        raise TypeError(
            f"{label} takes a list of column names, not the bare " f"string {columns!r}"
        )
    columns = list(columns)
    for column in columns:
        if column not in frame.columns:
            raise ValueError(f"{label} names a missing column: {column!r}")
    return columns


def _joined(data, sociodemographic_data):
    """The left join onto ``from_id`` and its matched mask; dropping
    the unmatched rows is the caller's decision, taken only after the
    expected identifier groups are captured."""
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a DataFrame")
    if sociodemographic_data is None:
        return data, np.ones(len(data), dtype=bool)
    if not isinstance(sociodemographic_data, pd.DataFrame):
        raise TypeError("sociodemographic_data must be a DataFrame")
    key = "id" if "id" in sociodemographic_data.columns else "from_id"
    if key not in sociodemographic_data.columns:
        raise ValueError("sociodemographic_data needs an id or from_id column")
    if "from_id" not in data.columns:
        raise ValueError("data needs a from_id column to join sociodemographic_data")
    payload = sociodemographic_data.columns.difference([key])
    overlap = payload.intersection(data.columns)
    if len(overlap):
        raise ValueError(
            "sociodemographic_data columns already exist in data: "
            + ", ".join(map(str, overlap))
        )
    table = sociodemographic_data.rename(columns={key: "from_id"})
    joined = data.merge(table, on="from_id", how="left", validate="m:1")
    matched = data["from_id"].isin(table["from_id"]).to_numpy()
    return joined, matched


def _as_float(name, column, series):
    """Loss-aware float conversion: an object payload (Decimal, big
    int) beyond float64 either raises on conversion or collapses a
    nonzero to zero — both are the envelope refusal, never silence."""
    try:
        array = series.to_numpy(dtype=float, na_value=np.nan)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError(
            f"{name} needs the nonzero magnitudes in {column!r} inside "
            "the supported numeric range (1e-75 to 1e75)"
        ) from error
    if series.dtype == object:
        collapsed = (array == 0) & series.notna().to_numpy()
        if collapsed.any() and (series[collapsed] != 0).any():
            raise ValueError(
                f"{name} needs the nonzero magnitudes in {column!r} inside "
                "the supported numeric range (1e-75 to 1e75)"
            )
    return array


def _prepared(
    name,
    data,
    value,
    sociodemographic_data,
    population,
    group_columns,
    dropna,
    consumed=(),
    labels=(),
):
    """Join, validate, and complete-case filter the input.

    Returns ``(frame, group columns, weight column)``: the frame
    carries the consumed columns plus a collision-proof float weight
    column, rescaled by its maximum so arbitrarily large legal
    weights never overflow a reduction. A row with NaN in ANY
    consumed column is dropped, its weight leaving every denominator
    with it; an infinite value in a numeric column refuses. No
    identifier group may end empty, whatever removed its rows.
    """
    frame, matched = _joined(data, sociodemographic_data)
    if group_columns is None:
        # Identifiers come from the VALUE frame: an unrelated
        # sociodemographic column named like one must not regroup.
        groups = [c for c in _IDENTIFIER_COLUMNS if c in data.columns]
    else:
        groups = _column_list(frame, group_columns, "group_columns")
    numeric = [value]
    if population is not None:
        numeric.append(population)
    numeric += [c for c in consumed if c is not None]
    label_columns = [c for c in labels if c is not None]
    for column in dict.fromkeys(numeric + label_columns):
        if column not in frame.columns:
            raise ValueError(f"{name} needs the column {column!r}")
    if not matched.all() and not dropna:
        missing = frame.loc[~matched, "from_id"].unique()
        shown = ", ".join(map(str, missing[:5]))
        raise ValueError(
            f"sociodemographic_data misses {len(missing)} origin "
            f"id(s): {shown}"
            + ("…" if len(missing) > 5 else "")
            + "; pass dropna=True to drop them"
        )
    if not matched.all():
        frame = frame[matched]
    # Groups named by the VALUE frame are protected across a dropna
    # join-drop (their expected set enumerates from the value frame
    # itself); groups living in the joined payload are meaningless on
    # unmatched rows and enumerate after the sanctioned drop. A NaN
    # group label is missing data — excluded like every other
    # missing consumed column, never an implicit group.
    expected = None
    if groups:
        source = data if all(c in data.columns for c in groups) else frame
        labeled = source
        for column in groups:
            labeled = labeled[labeled[column].notna()]
        expected = list(
            labeled.groupby(groups, dropna=False, observed=True, sort=True).groups
        )
    weight_column = "__cafein_weight__"
    while weight_column in frame.columns:
        weight_column += "_"
    for column in dict.fromkeys(numeric):
        if pd.api.types.is_complex_dtype(frame[column]):
            raise ValueError(f"{name} needs real values in {column!r}")
    if population is None:
        weight = np.ones(len(frame))
    else:
        weight = _as_float(name, population, frame[population])
        finite = weight[~np.isnan(weight)]
        if ((finite < 0) | np.isinf(finite)).any():
            raise ValueError(
                "population weights must be finite and non-negative; "
                "a NaN weight is missing data and drops its row"
            )
    frame = frame.assign(**{weight_column: weight})
    kept = frame[weight_column] > 0
    for column in dict.fromkeys(numeric + label_columns + groups):
        kept &= frame[column].notna().to_numpy()
    filtered = frame[kept]
    converted = {}
    for column in dict.fromkeys(numeric):
        converted[column] = _as_float(name, column, filtered[column])
        magnitudes = np.abs(converted[column])
        if np.isinf(magnitudes).any():
            raise ValueError(f"{name} needs finite values in {column!r}")
        nonzero = magnitudes[magnitudes > 0]
        if len(nonzero) and (nonzero.min() < 1e-75 or nonzero.max() > 1e75):
            raise ValueError(
                f"{name} needs the nonzero magnitudes in {column!r} inside "
                "the supported numeric range (1e-75 to 1e75)"
            )
    # Assigned by label, not **kwargs: a column label need not be a
    # string.
    filtered = filtered.copy()
    for column, array in converted.items():
        filtered[column] = array
    if not len(filtered):
        raise ValueError(
            f"{name} has no usable rows left after excluding missing "
            "and zero-weight rows"
        )
    if groups:
        left = set(
            filtered.groupby(groups, dropna=False, observed=True, sort=True).groups
        )
        lost = [key for key in expected if key not in left]
        if lost:
            raise ValueError(
                f"{name} has no usable rows left in the group "
                f"{lost[0]!r} after excluding missing and zero-weight rows"
            )
    return filtered, groups, weight_column


def _fan_out(frame, groups, compute, columns):
    """Run ``compute`` per identifier group.

    Single-result measures (one column) return a float for an
    ungrouped frame; every multi-result summary returns a one-row
    frame instead. Grouped inputs return one row per group.
    """
    collisions = set(groups) & set(columns)
    if collisions:
        raise ValueError(
            "group_columns collide with the result column(s): "
            + ", ".join(sorted(collisions))
        )
    if not groups:
        result = compute(frame)
        if len(columns) == 1:
            return result[columns[0]]
        return pd.DataFrame([result], columns=columns)
    rows = []
    for keys, part in frame.groupby(groups, dropna=False, observed=True, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        rows.append({**dict(zip(groups, keys)), **compute(part)})
    return pd.DataFrame(rows, columns=groups + columns)


def _values_weights(frame, value, weight_column):
    """The part's value and weight arrays, weights rescaled by their
    own peak: every measure is weight-scale invariant, raw sums of
    legal huge weights would overflow, and one group's scale must not
    leak into another's. A positive weight the rescale underflows is
    a dynamic range float64 cannot honor."""
    values = frame[value].to_numpy(dtype=float)
    weights = frame[weight_column].to_numpy(dtype=float)
    weights = weights / weights.max()
    if (weights == 0).any():
        raise ValueError("population weights span too wide a range to compute together")
    return values, weights


def _rescaled(values):
    """Scale-invariant measures divide by the peak magnitude up front
    so legal huge values never overflow a reduction; a nonzero value
    the rescale underflows is a dynamic range float64 cannot honor."""
    peak = np.abs(values).max()
    if peak == 0:
        return values
    rescaled = values / peak
    if ((rescaled == 0) & (values != 0)).any():
        raise ValueError("values span too wide a range to compute together")
    return rescaled


def _require_non_negative(name, values, weights):
    if (values < 0).any():
        raise ValueError(_NEGATIVE_MESSAGE.format(name=name))
    if np.dot(values, weights) <= 0:
        raise ValueError(f"{name} needs a strictly positive weighted value total")


def _require_positive(name, values):
    if (values < 0).any():
        raise ValueError(_NEGATIVE_MESSAGE.format(name=name))
    if (values == 0).any():
        raise ValueError(_ZERO_MESSAGE.format(name=name))


def gini_index(
    data,
    *,
    value="accessibility",
    sociodemographic_data=None,
    population=None,
    group_columns=None,
    dropna=False,
):
    """The population-weighted Gini index of the value distribution,
    0 (perfect equality) to 1 — the trapezoidal integral of the same
    vertex polyline :func:`lorenz_curve` returns."""
    frame, groups, weight = _prepared(
        "gini_index",
        data,
        value,
        sociodemographic_data,
        population,
        group_columns,
        dropna,
    )

    def compute(part):
        values, weights = _values_weights(part, value, weight)
        values = _rescaled(values)
        _require_non_negative("gini_index", values, weights)
        shares, mass = _distribution.lorenz_vertices(values, weights)
        area = _distribution.polyline_area(shares, mass)
        return {"gini_index": 1.0 - 2.0 * area}

    return _fan_out(frame, groups, compute, ["gini_index"])


def lorenz_curve(
    data,
    *,
    value="accessibility",
    sociodemographic_data=None,
    population=None,
    group_columns=None,
    dropna=False,
):
    """The Lorenz curve as a plottable frame — ``population_share``
    against ``value_share``, one vertex per row plus the origin, per
    identifier group. :func:`gini_index` integrates exactly this
    polyline."""
    frame, groups, weight = _prepared(
        "lorenz_curve",
        data,
        value,
        sociodemographic_data,
        population,
        group_columns,
        dropna,
    )
    collisions = set(groups) & {"population_share", "value_share"}
    if collisions:
        raise ValueError(
            "group_columns collide with the result column(s): "
            + ", ".join(sorted(collisions))
        )
    parts = []
    grouped = (
        frame.groupby(groups, dropna=False, observed=True, sort=True)
        if groups
        else [((), frame)]
    )
    for keys, part in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        values, weights = _values_weights(part, value, weight)
        values = _rescaled(values)
        _require_non_negative("lorenz_curve", values, weights)
        shares, mass = _distribution.lorenz_vertices(values, weights)
        vertex = pd.DataFrame(dict(zip(groups, keys)), index=range(len(shares)))
        vertex["population_share"] = shares
        vertex["value_share"] = mass
        parts.append(vertex)
    return pd.concat(parts, ignore_index=True)


def share_ratio(
    data,
    *,
    top=0.10,
    bottom=0.40,
    income=None,
    value="accessibility",
    sociodemographic_data=None,
    population=None,
    group_columns=None,
    dropna=False,
):
    """The mean accessibility of the top ``top`` population share over
    the bottom ``bottom`` share's, ranked by ``income=`` when given
    and by the value itself otherwise. The boundary of a cut is split
    fractionally — allocated evenly across a whole block of tied
    ranking values — so the ratio is exact at any cut and independent
    of input order."""
    return _quantile_share_ratio(
        "share_ratio",
        data,
        top=top,
        bottom=bottom,
        income=income,
        value=value,
        sociodemographic_data=sociodemographic_data,
        population=population,
        group_columns=group_columns,
        dropna=dropna,
    )


def _quantile_share_ratio(
    name,
    data,
    *,
    top,
    bottom,
    income,
    value,
    sociodemographic_data,
    population,
    group_columns,
    dropna,
):
    for label, share in (("top", top), ("bottom", bottom)):
        if not np.isfinite(share) or not 0 < share < 1:
            raise ValueError(f"{label} must be a share in (0, 1)")
    frame, groups, weight = _prepared(
        name,
        data,
        value,
        sociodemographic_data,
        population,
        group_columns,
        dropna,
        consumed=(income,),
    )

    def compute(part):
        values, weights = _values_weights(part, value, weight)
        values = _rescaled(values)
        _require_non_negative(name, values, weights)
        ranking = part[income].to_numpy(dtype=float) if income is not None else values
        rich = _distribution.tail_mean(values, weights, ranking, top, "top")
        poor = _distribution.tail_mean(values, weights, ranking, bottom, "bottom")
        if poor == 0 and rich == 0:
            raise ValueError(f"{name} is 0/0 here: both tail means are zero")
        if poor == 0:
            return {name: float("inf")}
        return {name: rich / poor}

    return _fan_out(frame, groups, compute, [name])


def palma_ratio(
    data,
    *,
    income,
    value="accessibility",
    sociodemographic_data=None,
    population=None,
    group_columns=None,
    dropna=False,
):
    """The Palma ratio — the income-ranked richest 10 %'s mean
    accessibility over the poorest 40 %'s;
    ``share_ratio(top=0.10, bottom=0.40)`` with ``income=`` required
    (without an income ranking the cut would silently change
    meaning)."""
    if income is None:
        raise ValueError(
            "palma_ratio requires income=; share_ratio serves value-ranked cuts"
        )
    return _quantile_share_ratio(
        "palma_ratio",
        data,
        top=0.10,
        bottom=0.40,
        income=income,
        value=value,
        sociodemographic_data=sociodemographic_data,
        population=population,
        group_columns=group_columns,
        dropna=dropna,
    )


def generalized_entropy(
    data,
    *,
    alpha=1,
    groups=None,
    value="accessibility",
    sociodemographic_data=None,
    population=None,
    group_columns=None,
    dropna=False,
):
    """The generalized entropy index GE(α) — ``alpha=1`` is Theil T,
    ``alpha=0`` the mean log deviation, ``alpha=2`` half the squared
    coefficient of variation. With ``groups=`` (a sociodemographic
    grouping column) the return decomposes into ``total``,
    ``between``, and ``within`` columns; ``total = between + within``
    exactly."""
    try:
        alpha = float(alpha)
    except (OverflowError, TypeError):
        alpha = float("inf")
    if not np.isfinite(alpha) or abs(alpha) > 1e6:
        raise ValueError("alpha must be a finite number with magnitude at most 1e6")
    frame, identifier, weight = _prepared(
        "generalized_entropy",
        data,
        value,
        sociodemographic_data,
        population,
        group_columns,
        dropna,
        labels=(groups,),
    )

    def measured(values, weights):
        # Entirely in the log domain: a value/mean ratio can overflow
        # — and a materialized subnormal share round to zero — while
        # their logarithms, and the index, stay exact. Returns the
        # index beside the general branch's log power sum, so the
        # decomposition can resolve joint extremes in log space.
        _require_positive("generalized_entropy", values)
        total = weights.sum()
        share = weights / total
        log_share = np.log(weights) - np.log(total)
        mean = np.dot(values, weights) / total
        deviation = np.log(values) - np.log(mean)
        # Alphas within an ulp-scale band of the limits ride the
        # exact limit formulas: the general form cancels there.
        if abs(alpha) <= 1e-9:
            return float(-np.dot(share, deviation)), None
        if abs(alpha - 1.0) <= 1e-9:
            # Each share_i * v_i / mean is at most 1, so the Theil
            # terms are safe once formed in the log domain.
            weighted_ratio = np.exp(log_share + deviation)
            return float(np.sum(weighted_ratio * deviation)), None
        # The power sum with expm1, so neither a large alpha
        # overflows nor a small one cancels.
        exponent = alpha * deviation + log_share
        peak = exponent.max()
        if not np.isfinite(peak):
            return float("inf"), float("inf")
        log_power = peak + np.log(np.exp(exponent - peak).sum())
        numerator = np.expm1(log_power)
        denominator = alpha * (alpha - 1.0)
        if np.isinf(numerator) and np.isinf(denominator):
            # Both overflow while their ratio is plain: resolve it in
            # the log domain (expm1 is exp to within an ulp up here).
            value_ = np.exp(log_power - np.log(abs(alpha)) - np.log(abs(alpha - 1.0)))
            return float(value_), float(log_power)
        return float(numerator / denominator), float(log_power)

    def index(values, weights):
        return measured(values, weights)[0]

    def compute(part):
        values, weights = _values_weights(part, value, weight)
        values = _rescaled(values)
        total = index(values, weights)
        if groups is None:
            return {"generalized_entropy": total}
        mean = np.dot(values, weights) / weights.sum()
        positions = part.groupby(groups, dropna=False, observed=True, sort=True).indices
        blocks = [(values[rows], weights[rows]) for rows in positions.values()]
        means = np.array([np.dot(v, w) / w.sum() for v, w in blocks])
        sizes = np.array([w.sum() for _, w in blocks])
        between = index(means, sizes)
        log_factor = np.log(sizes) - np.log(sizes.sum())
        log_factor += alpha * (np.log(means) - np.log(mean))
        measures = [measured(v, w) for v, w in blocks]
        log_denominator = (
            np.log(abs(alpha)) + np.log(abs(alpha - 1.0))
            if abs(alpha) > 1e-9 and abs(alpha - 1.0) > 1e-9
            else None
        )
        within = 0.0
        for factor_log, (part, log_power) in zip(log_factor, measures):
            if part == 0.0:
                continue
            plain = np.exp(factor_log) * part
            if np.isfinite(plain) and plain != 0.0:
                within += plain
            elif np.isfinite(part):
                # The factor underflowed while the block index is an
                # ordinary float: their joint magnitude resolves as
                # one exponent.
                within += np.exp(factor_log + np.log(part))
            elif log_power is not None:
                # An underflowed factor against an overflowed block.
                within += np.exp(factor_log + log_power - log_denominator)
        within = float(within)
        return {"total": total, "between": between, "within": within}

    columns = (
        ["generalized_entropy"] if groups is None else ["total", "between", "within"]
    )
    return _fan_out(frame, identifier, compute, columns)


def theil_t(
    data,
    *,
    groups=None,
    value="accessibility",
    sociodemographic_data=None,
    population=None,
    group_columns=None,
    dropna=False,
):
    """Theil T — ``generalized_entropy(alpha=1)`` exactly, every
    shared argument (``groups=`` included) forwarded."""
    return generalized_entropy(
        data,
        alpha=1,
        groups=groups,
        value=value,
        sociodemographic_data=sociodemographic_data,
        population=population,
        group_columns=group_columns,
        dropna=dropna,
    )


def mld(
    data,
    *,
    groups=None,
    value="accessibility",
    sociodemographic_data=None,
    population=None,
    group_columns=None,
    dropna=False,
):
    """The mean log deviation (Theil L) —
    ``generalized_entropy(alpha=0)`` exactly, every shared argument
    forwarded."""
    return generalized_entropy(
        data,
        alpha=0,
        groups=groups,
        value=value,
        sociodemographic_data=sociodemographic_data,
        population=population,
        group_columns=group_columns,
        dropna=dropna,
    )


def atkinson(
    data,
    *,
    epsilon=1,
    value="accessibility",
    sociodemographic_data=None,
    population=None,
    group_columns=None,
    dropna=False,
):
    """The Atkinson index with inequality aversion ``epsilon`` > 0 —
    the welfare share lost to inequality under the analyst's own
    weighting of the worst-off."""
    try:
        epsilon = float(epsilon)
    except (OverflowError, TypeError):
        epsilon = float("inf")
    if not np.isfinite(epsilon) or epsilon <= 0 or epsilon > 1e6:
        raise ValueError("epsilon must be a positive number at most 1e6")
    frame, groups, weight = _prepared(
        "atkinson",
        data,
        value,
        sociodemographic_data,
        population,
        group_columns,
        dropna,
    )

    def compute(part):
        values, weights = _values_weights(part, value, weight)
        values = _rescaled(values)
        _require_positive("atkinson", values)
        share = weights / weights.sum()
        mean = np.dot(share, values)
        if abs(epsilon - 1.0) <= 1e-9:
            log_equivalent = np.dot(share, np.log(values))
        else:
            # The generalized mean anchored at its dominating value:
            # the exponent forms as a log DIFFERENCE (a plain ratio
            # can overflow first) and stays non-positive, so no
            # aversion parameter overflows a bounded index.
            sign = 1.0 - epsilon
            anchor = values.min() if sign < 0 else values.max()
            exponent = sign * (np.log(values) - np.log(anchor))
            log_terms = np.log(weights) - np.log(weights.sum()) + exponent
            peak = log_terms.max()
            log_mean_term = peak + np.log(np.exp(log_terms - peak).sum())
            log_equivalent = np.log(anchor) + log_mean_term / sign
        return {"atkinson": float(1.0 - np.exp(log_equivalent - np.log(mean)))}

    return _fan_out(frame, groups, compute, ["atkinson"])


def kolm(
    data,
    *,
    kappa=1,
    value="accessibility",
    sociodemographic_data=None,
    population=None,
    group_columns=None,
    dropna=False,
):
    """The Kolm index with absolute inequality aversion ``kappa`` > 0
    — translation-invariant: a uniform absolute gain leaves it
    unchanged, the absolute reading the relative indices cannot give.
    Defined on any real values (difference frames included)."""
    try:
        kappa = float(kappa)
    except (OverflowError, TypeError):
        kappa = float("inf")
    if not np.isfinite(kappa) or kappa <= 0 or kappa > 1e6:
        raise ValueError("kappa must be a positive number at most 1e6")
    frame, groups, weight = _prepared(
        "kolm",
        data,
        value,
        sociodemographic_data,
        population,
        group_columns,
        dropna,
    )

    def compute(part):
        values, weights = _values_weights(part, value, weight)
        share = weights / weights.sum()
        # Computed on unit-normalized values: K_kappa(v) is exactly
        # scale * K_(kappa*scale)(v/scale), and on [-1, 1] no
        # centering, exponent, or mean can overflow. expm1/log1p keep
        # tiny effective kappas exact; an effective kappa beyond the
        # float ceiling pins the log term to its zero limit.
        scale = np.abs(values).max()
        if scale == 0:
            return {"kolm": 0.0}
        unit = values / scale
        low = unit.min()
        gap = np.dot(share, unit - low)
        effective = kappa * scale
        if np.isinf(effective):
            log_term = 0.0
        elif effective < 1e-8:
            # First order cancels the gap exactly; the index IS the
            # second cumulant down here.
            center = np.dot(share, unit)
            variance = np.dot(share, (unit - center) ** 2)
            return {"kolm": float(scale * effective * variance / 2.0)}
        else:
            log_terms = (
                np.log(weights) - np.log(weights.sum()) - effective * (unit - low)
            )
            peak = log_terms.max()
            log_term = (peak + np.log(np.exp(log_terms - peak).sum())) / effective
        return {"kolm": float(scale * (gap + log_term))}

    return _fan_out(frame, groups, compute, ["kolm"])


def hoover(
    data,
    *,
    value="accessibility",
    sociodemographic_data=None,
    population=None,
    group_columns=None,
    dropna=False,
):
    """The Hoover (Pietra) index — the share of total accessibility
    that would have to be redistributed for perfect equality."""
    frame, groups, weight = _prepared(
        "hoover",
        data,
        value,
        sociodemographic_data,
        population,
        group_columns,
        dropna,
    )

    def compute(part):
        values, weights = _values_weights(part, value, weight)
        values = _rescaled(values)
        _require_non_negative("hoover", values, weights)
        mean = np.dot(values, weights) / weights.sum()
        spread = np.dot(weights, np.abs(values - mean))
        return {"hoover": float(spread / (2.0 * np.dot(weights, values)))}

    return _fan_out(frame, groups, compute, ["hoover"])


def concentration_index(
    data,
    *,
    income,
    variant="standard",
    bounds=None,
    value="accessibility",
    sociodemographic_data=None,
    population=None,
    group_columns=None,
    dropna=False,
):
    """The concentration index of the value distribution over the
    income ranking — positive when access concentrates among the
    rich, negative among the poor.

    ``variant`` selects the normalization: ``"standard"`` (the
    relative index, the trapezoidal integral of the same polyline
    :func:`concentration_curve` returns), ``"erreygers"`` and
    ``"wagstaff"`` (the corrected indices for bounded outcomes — both
    REQUIRE ``bounds=(lower, upper)``, the outcome's theoretical
    range, never inferred from the data), and ``"absolute"`` (the
    generalized index, twice the weighted covariance of value and
    fractional rank — defined on any real values, difference frames
    included, and takes no bounds).
    """
    if variant not in ("standard", "erreygers", "wagstaff", "absolute"):
        raise ValueError("variant must be standard, erreygers, wagstaff, or absolute")
    if income is None:
        raise ValueError("concentration_index requires income=")
    corrected = variant in ("erreygers", "wagstaff")
    if corrected:
        try:
            lower, upper = (float(edge) for edge in bounds)
        except (TypeError, ValueError, OverflowError):
            raise ValueError(
                f"the {variant} variant requires bounds=(lower, upper), "
                "the outcome's theoretical range"
            ) from None
        if not (np.isfinite(lower) and np.isfinite(upper)) or lower >= upper:
            raise ValueError("bounds must be finite with lower < upper")
        if max(abs(lower), abs(upper)) > 1e75:
            raise ValueError(
                "bounds must sit inside the supported numeric range "
                "(magnitudes at most 1e75)"
            )
    elif bounds is not None:
        raise ValueError("bounds= serves the erreygers and wagstaff variants")
    frame, groups, weight = _prepared(
        "concentration_index",
        data,
        value,
        sociodemographic_data,
        population,
        group_columns,
        dropna,
        consumed=(income,),
    )

    def compute(part):
        values, weights = _values_weights(part, value, weight)
        ranking = part[income].to_numpy(dtype=float)
        share = weights / weights.sum()
        mean = np.dot(share, values)
        if variant == "absolute":
            ranks = _distribution.fractional_ranks(weights, ranking)
            centered = ranks - np.dot(share, ranks)
            return {
                "concentration_index": float(
                    2.0 * np.dot(share, (values - mean) * centered)
                )
            }
        _require_non_negative("concentration_index", values, weights)
        if corrected and ((values < lower) | (values > upper)).any():
            raise ValueError(
                "concentration_index found values outside bounds=; the "
                "bounds must cover the outcome's whole range"
            )
        rescaled = _rescaled(values)
        population_share, mass = _distribution.concentration_vertices(
            rescaled, weights, ranking
        )
        standard = 1.0 - 2.0 * _distribution.polyline_area(population_share, mass)
        if variant == "standard":
            return {"concentration_index": float(standard)}
        if variant == "erreygers":
            corrected_value = 4.0 * mean * standard / (upper - lower)
            return {"concentration_index": float(corrected_value)}
        if mean == lower or mean == upper:
            raise ValueError(
                "the wagstaff variant is undefined when the weighted mean "
                "equals a bound"
            )
        corrected_value = (
            standard * mean * (upper - lower) / ((upper - mean) * (mean - lower))
        )
        return {"concentration_index": float(corrected_value)}

    return _fan_out(frame, groups, compute, ["concentration_index"])


def concentration_curve(
    data,
    *,
    income,
    value="accessibility",
    sociodemographic_data=None,
    population=None,
    group_columns=None,
    dropna=False,
):
    """The concentration curve as a plottable frame —
    ``population_share`` (income-ranked) against ``value_share``, one
    vertex per tied-income block plus the origin, per identifier
    group. ``concentration_index(variant="standard")`` integrates
    exactly this polyline."""
    if income is None:
        raise ValueError("concentration_curve requires income=")
    frame, groups, weight = _prepared(
        "concentration_curve",
        data,
        value,
        sociodemographic_data,
        population,
        group_columns,
        dropna,
        consumed=(income,),
    )
    collisions = set(groups) & {"population_share", "value_share"}
    if collisions:
        raise ValueError(
            "group_columns collide with the result column(s): "
            + ", ".join(sorted(collisions))
        )
    parts = []
    grouped = (
        frame.groupby(groups, dropna=False, observed=True, sort=True)
        if groups
        else [((), frame)]
    )
    for keys, part in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        values, weights = _values_weights(part, value, weight)
        values = _rescaled(values)
        _require_non_negative("concentration_curve", values, weights)
        ranking = part[income].to_numpy(dtype=float)
        shares, mass = _distribution.concentration_vertices(values, weights, ranking)
        vertex = pd.DataFrame(dict(zip(groups, keys)), index=range(len(shares)))
        vertex["population_share"] = shares
        vertex["value_share"] = mass
        parts.append(vertex)
    return pd.concat(parts, ignore_index=True)


def suits(
    data,
    *,
    income,
    value="accessibility",
    sociodemographic_data=None,
    population=None,
    group_columns=None,
    dropna=False,
):
    """The Suits index — cumulative accessibility share integrated
    against cumulative INCOME share, positive when access
    concentrates among the rich. Requires finite non-negative incomes
    with a strictly positive weighted total (a negative income would
    make the accumulation axis non-monotonic); tied incomes collapse
    to one chord, so the index is order-invariant within ties."""
    if income is None:
        raise ValueError("suits requires income=")
    frame, groups, weight = _prepared(
        "suits",
        data,
        value,
        sociodemographic_data,
        population,
        group_columns,
        dropna,
        consumed=(income,),
    )

    def compute(part):
        values, weights = _values_weights(part, value, weight)
        values = _rescaled(values)
        _require_non_negative("suits", values, weights)
        ranking = part[income].to_numpy(dtype=float)
        if (ranking < 0).any():
            raise ValueError(
                "suits needs non-negative incomes: its accumulation axis "
                "is cumulative income share"
            )
        if np.dot(weights, ranking) <= 0:
            raise ValueError("suits needs a strictly positive weighted income total")
        axis, mass = _distribution.suits_vertices(values, weights, ranking)
        return {"suits": float(1.0 - 2.0 * _distribution.polyline_area(axis, mass))}

    return _fan_out(frame, groups, compute, ["suits"])


_RELATIVE_LINE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*%\s*of\s*median\s*$", re.I)


def _resolved_poverty_line(name, specification, values, weights):
    """A poverty line as a number, or the relative convention
    ``"60% of median"`` against the population-weighted (type-1)
    median of THIS distribution — per identifier group, so a grouped
    call draws each group's own relative line."""
    if isinstance(specification, str):
        match = _RELATIVE_LINE.match(specification)
        if match is None:
            raise ValueError(
                f'{name} poverty_line strings take the form "60% of median"'
            )
        line = (
            float(match.group(1))
            / 100.0
            * _distribution.weighted_median(values, weights)
        )
    else:
        try:
            line = float(specification)
        except (TypeError, ValueError, OverflowError):
            line = float("nan")
    if not np.isfinite(line) or line <= 0 or line > 1e75:
        raise ValueError(
            f"{name} needs a strictly positive poverty line inside the "
            "supported numeric range; a relative line inherits the sign "
            "of its median"
        )
    return line


def fgt_poverty(
    data,
    *,
    poverty_line,
    alpha=0,
    deprived="below",
    value="accessibility",
    sociodemographic_data=None,
    population=None,
    group_columns=None,
    dropna=False,
):
    """The Foster–Greer–Thorbecke poverty measure FGT(α) — ``alpha=0``
    the headcount ratio, ``1`` the poverty gap, ``2`` its severity.

    ``deprived`` states the tail: ``"below"`` (poor when the value
    falls short of the line, normalized gap ``(line − value)/line``)
    or ``"above"`` (poor when the value exceeds it, gap
    ``(value − line)/line``), both censored at zero — a value exactly
    on the line is not poor. ``poverty_line`` is absolute in the
    value's own units, or relative as ``"60% of median"``."""
    try:
        alpha = float(alpha)
    except (OverflowError, TypeError):
        alpha = float("inf")
    if not np.isfinite(alpha) or alpha < 0 or alpha > 1e6:
        raise ValueError("alpha must be a non-negative number at most 1e6")
    if deprived not in ("below", "above"):
        raise ValueError('deprived must be "below" or "above"')
    frame, groups, weight = _prepared(
        "fgt_poverty",
        data,
        value,
        sociodemographic_data,
        population,
        group_columns,
        dropna,
    )

    def compute(part):
        values, weights = _values_weights(part, value, weight)
        if (values < 0).any():
            raise ValueError(_NEGATIVE_MESSAGE.format(name="fgt_poverty"))
        line = _resolved_poverty_line("fgt_poverty", poverty_line, values, weights)
        signed = line - values if deprived == "below" else values - line
        gap = np.maximum(signed / line, 0.0)
        share = weights / weights.sum()
        if alpha == 0:
            return {"fgt_poverty": float(np.dot(share, gap > 0))}
        return {"fgt_poverty": float(np.dot(share, gap**alpha))}

    return _fan_out(frame, groups, compute, ["fgt_poverty"])


def _burden_inputs(name, part, cost, income, weight, non_negative_costs=True):
    costs = part[cost].to_numpy(dtype=float)
    incomes = part[income].to_numpy(dtype=float)
    weights = part[weight].to_numpy(dtype=float)
    weights = weights / weights.max()
    if (weights == 0).any():
        raise ValueError("population weights span too wide a range to compute together")
    if non_negative_costs and (costs < 0).any():
        raise ValueError(f"{name} needs non-negative costs in {cost!r}")
    bad = incomes <= 0
    if bad.any():
        rows = (
            part.loc[bad, "from_id"].unique()[:5]
            if "from_id" in part.columns
            else part.index[bad][:5]
        )
        shown = ", ".join(map(str, rows))
        raise ValueError(
            f"{name} needs strictly positive incomes in {income!r}; "
            f"{int(bad.sum())} row(s) violate that (e.g. {shown}) — "
            "bottom-coding or excluding them is a pre-processing decision"
        )
    return costs, incomes, weights


def _per_origin(part, groups, computed):
    """A per-origin result frame: the origin ids and identifier
    columns beside the computed columns."""
    carried = list(
        dict.fromkeys([c for c in ("from_id",) if c in part.columns] + list(groups))
    )
    collisions = set(carried) & set(computed)
    if collisions:
        raise ValueError(
            "group_columns collide with the result column(s): "
            + ", ".join(sorted(collisions))
        )
    frame = part[carried].reset_index(drop=True)
    for column, values in computed.items():
        frame[column] = values
    return frame


def cost_burden(
    data,
    *,
    cost,
    income,
    threshold=0.10,
    detail=False,
    sociodemographic_data=None,
    population=None,
    group_columns=None,
    dropna=False,
):
    """The transport cost burden — each origin's cost as a share of
    its income. Returns the population-weighted headcount STRICTLY
    above ``threshold`` (the classic 10 % rule) and the weighted mean
    burden per identifier group; ``detail=True`` returns the
    per-origin frame (``burden``, ``cost_burdened``) instead."""
    try:
        threshold = float(threshold)
    except (OverflowError, TypeError):
        threshold = float("inf")
    if not np.isfinite(threshold) or threshold <= 0 or threshold > 1e6:
        raise ValueError("threshold must be a positive share at most 1e6")
    frame, groups, weight = _prepared(
        "cost_burden",
        data,
        cost,
        sociodemographic_data,
        population,
        group_columns,
        dropna,
        consumed=(income,),
    )
    if detail:
        costs, incomes, _ = _burden_inputs("cost_burden", frame, cost, income, weight)
        burden = costs / incomes
        return _per_origin(
            frame,
            groups,
            {"burden": burden, "cost_burdened": burden > threshold},
        )

    def compute(part):
        costs, incomes, weights = _burden_inputs(
            "cost_burden", part, cost, income, weight
        )
        burden = costs / incomes
        share = weights / weights.sum()
        return {
            "cost_burden": float(np.dot(share, burden > threshold)),
            "mean_burden": float(np.dot(share, burden)),
        }

    return _fan_out(frame, groups, compute, ["cost_burden", "mean_burden"])


def residual_income(
    data,
    *,
    cost,
    income,
    sociodemographic_data=None,
    population=None,
    group_columns=None,
    dropna=False,
):
    """Income left after required transport costs, per origin — the
    continuous companion to :func:`lihc`. Inherently a per-origin
    frame; residuals may legitimately be negative."""
    frame, groups, weight = _prepared(
        "residual_income",
        data,
        cost,
        sociodemographic_data,
        population,
        group_columns,
        dropna,
        consumed=(income,),
    )
    # A negative cost (a subsidy) is a legal residual-income input;
    # only the burden ratios require non-negative costs.
    costs, incomes, _ = _burden_inputs(
        "residual_income", frame, cost, income, weight, non_negative_costs=False
    )
    return _per_origin(frame, groups, {"residual_income": incomes - costs})


def lihc(
    data,
    *,
    cost,
    income,
    poverty_line,
    detail=False,
    sociodemographic_data=None,
    population=None,
    group_columns=None,
    dropna=False,
):
    """The Low Income, High Costs indicator in its transport
    adaptation: an origin is transport-poor only if its required
    costs sit STRICTLY above the population-weighted median cost AND
    its residual income after them falls strictly below
    ``poverty_line`` (a number, or ``"60% of median"`` of the
    residual-income distribution). Returns the weighted headcount and
    the two marginal rates per identifier group; ``detail=True``
    returns the per-origin quadrant classification instead."""
    frame, groups, weight = _prepared(
        "lihc",
        data,
        cost,
        sociodemographic_data,
        population,
        group_columns,
        dropna,
        consumed=(income,),
    )

    def classify(part):
        costs, incomes, weights = _burden_inputs("lihc", part, cost, income, weight)
        residual = incomes - costs
        line = _resolved_poverty_line("lihc", poverty_line, residual, weights)
        high = costs > _distribution.weighted_median(costs, weights)
        low = residual < line
        return high, low, weights

    if detail:
        grouped = (
            frame.groupby(groups, dropna=False, observed=True, sort=True)
            if groups
            else [((), frame)]
        )
        parts = []
        for _, part in grouped:
            high, low, _ = classify(part)
            parts.append(
                _per_origin(
                    part,
                    groups,
                    {
                        "costs_above_median": high,
                        "residual_below_line": low,
                        "lihc": high & low,
                    },
                )
            )
        return pd.concat(parts, ignore_index=True)

    def compute(part):
        high, low, weights = classify(part)
        share = weights / weights.sum()
        return {
            "lihc": float(np.dot(share, high & low)),
            "high_costs": float(np.dot(share, high)),
            "low_residual": float(np.dot(share, low)),
        }

    return _fan_out(frame, groups, compute, ["lihc", "high_costs", "low_residual"])


_CUTOFF_OPS = {
    "<": np.less,
    "<=": np.less_equal,
    ">": np.greater,
    ">=": np.greater_equal,
}


def _deprivation_specs(dimensions):
    """The Alkire–Foster dimension dict, validated: column → (op,
    cutoff), a bare number meaning deprived BELOW the cutoff."""
    if not isinstance(dimensions, dict) or not dimensions:
        raise ValueError("dimensions must be a non-empty dict of column → cutoff")
    if None in dimensions:
        # None marks an absent optional column internally; a real
        # column labeled None cannot ride the validation pipeline.
        raise ValueError("a dimension column label must not be None")
    specs = {}
    for column, cutoff in dimensions.items():
        if isinstance(cutoff, tuple):
            try:
                operator, edge = cutoff
            except ValueError:
                operator, edge = None, None
            if operator not in _CUTOFF_OPS:
                raise ValueError(
                    f"the cutoff for {column!r} must be a number or an "
                    '(op, number) tuple with op in "<", "<=", ">", ">="'
                )
        else:
            operator, edge = "<", cutoff
        try:
            edge = float(edge)
        except (TypeError, ValueError, OverflowError):
            edge = float("nan")
        if not np.isfinite(edge) or abs(edge) > 1e75:
            raise ValueError(
                f"the cutoff for {column!r} must be a finite number inside "
                "the supported numeric range"
            )
        specs[column] = (operator, edge)
    names = [f"{column}_deprived" for column in specs]
    if len(set(names)) != len(names):
        raise ValueError(
            "dimension columns stringify to colliding detail names; "
            "rename them so each column keeps its own _deprived column"
        )
    return specs


def alkire_foster(
    data,
    *,
    dimensions,
    k,
    detail=False,
    sociodemographic_data=None,
    population=None,
    group_columns=None,
    dropna=False,
):
    """The Alkire–Foster multidimensional poverty measure M0 over
    user-declared dimensions with equal weights.

    ``dimensions`` maps a column to its deprivation cutoff: a bare
    number means deprived BELOW the cutoff (the accessibility
    reading); an ``(op, number)`` tuple with op in ``"<"``, ``"<="``,
    ``">"``, ``">="`` states the direction explicitly (cost and time
    burdens deprive above). ``k`` is the dual cutoff — an int counts
    dimensions, a float in (0, 1] a share of them — and an origin is
    multidimensionally poor when it is deprived in at least ``k``
    dimensions. Returns ``m0`` (= headcount × intensity),
    ``headcount``, and ``intensity`` (the average deprivation share
    among the poor; NaN when nobody is poor) per identifier group;
    ``detail=True`` returns the per-origin deprivation matrix with
    the deprivation and censored counts beside their shares instead."""
    specs = _deprivation_specs(dimensions)
    columns = list(specs)
    count = len(columns)
    if isinstance(k, bool) or not isinstance(k, (int, float)):
        raise ValueError("k must be a dimension count or a share in (0, 1]")
    if isinstance(k, int):
        if not 1 <= k <= count:
            raise ValueError(
                f"an integer k counts dimensions: it must be between 1 " f"and {count}"
            )
        threshold = k / count
    else:
        if not np.isfinite(k) or not 0 < k <= 1:
            raise ValueError("a float k is a share of dimensions in (0, 1]")
        threshold = k
    frame, groups, weight = _prepared(
        "alkire_foster",
        data,
        columns[0],
        sociodemographic_data,
        population,
        group_columns,
        dropna,
        consumed=tuple(columns[1:]),
    )

    def deprivations(part):
        matrix = np.column_stack(
            [
                _CUTOFF_OPS[operator](part[column].to_numpy(dtype=float), edge)
                for column, (operator, edge) in specs.items()
            ]
        )
        shares = matrix.mean(axis=1)
        poor = shares >= threshold
        return matrix, shares, poor

    if detail:
        grouped = (
            frame.groupby(groups, dropna=False, observed=True, sort=True)
            if groups
            else [((), frame)]
        )
        parts = []
        for _, part in grouped:
            matrix, shares, poor = deprivations(part)
            computed = {
                f"{column}_deprived": matrix[:, position]
                for position, column in enumerate(columns)
            }
            computed["deprivation_count"] = matrix.sum(axis=1)
            computed["deprivation_share"] = shares
            computed["poor"] = poor
            computed["censored_count"] = np.where(poor, matrix.sum(axis=1), 0)
            computed["censored_share"] = np.where(poor, shares, 0.0)
            parts.append(_per_origin(part, groups, computed))
        return pd.concat(parts, ignore_index=True)

    def compute(part):
        _, shares, poor = deprivations(part)
        weights = part[weight].to_numpy(dtype=float)
        weights = weights / weights.max()
        if (weights == 0).any():
            raise ValueError(
                "population weights span too wide a range to compute together"
            )
        share = weights / weights.sum()
        headcount = float(np.dot(share, poor))
        m0 = float(np.dot(share, np.where(poor, shares, 0.0)))
        intensity = m0 / headcount if headcount > 0 else float("nan")
        return {"m0": m0, "headcount": headcount, "intensity": intensity}

    return _fan_out(frame, groups, compute, ["m0", "headcount", "intensity"])
