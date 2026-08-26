"""cafein against IPEA's R ``accessibility`` package: the reference numbers
of its inequality-and-poverty vignette on the sample data it ships, the
closed-form decay weights ``Accessibility`` shares with its gravity
measure, and — with R available — the comparison script itself."""

import math
import os
import pathlib
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from cafein import equity

DATA = pathlib.Path(__file__).parent / "data"
SCRIPT = (
    pathlib.Path(__file__).parent.parent
    / "scripts"
    / "compare_vs_ipea_accessibility.py"
)
DEPARTURE = "2022-02-22 08:30"
PACKAGE_VERSION = "1.5.0"
CUTOFF = 30
POVERTY_LINE = 50000
# Printed by the package's inequality_and_poverty vignette (version
# PACKAGE_VERSION): jobs reachable within 30 minutes on its sample.
GINI = 0.4715251
PALMA = 3.800465
CONCENTRATION_STANDARD = 0.2865013
CONCENTRATION_CORRECTED = 0.3346494
THEIL = 0.3616631
THEIL_BETWEEN = 0.1280753
THEIL_WITHIN = 0.2335878
FGT = {0: 0.3923817, 1: 0.1776241, 2: 0.1010123}
# Hmisc::wtd.quantile(income_per_capita, weights = population, c(0.4, 0.9)),
# the thresholds the package's palma_ratio assigns whole rows at.
PALMA_Q40 = 1208.6
PALMA_Q90 = 3741.6
INCOME = "income_per_capita"


EXPORT = "python scripts/compare_vs_ipea_accessibility.py --export-fixtures"


def _fixture(name, recovery=EXPORT, fetched=False):
    """A test-data path, or a skip naming how to produce it. Only the
    fetched fixtures (``scripts/fetch_test_data.py``) honour
    ``CAFEIN_REQUIRE_TEST_DATA``; the CSVs exported from the R package
    need an R environment, which the CI runners do not have."""
    path = DATA / name
    if not path.exists():
        message = f"{path} missing; run `{recovery}`"
        if fetched and os.environ.get("CAFEIN_REQUIRE_TEST_DATA"):
            pytest.fail(message)
        pytest.skip(message)
    return path


@pytest.fixture(scope="module")
def ipea():
    """The sample land-use table and the cutoff-30 access frame, built
    with the package's own rule: jobs at ``travel_time <= 30`` summed
    per origin, origins without a reached row at 0."""
    matrix = pd.read_csv(
        _fixture("ipea_travel_matrix.csv"), dtype={"from_id": str, "to_id": str}
    )
    land_use = pd.read_csv(_fixture("ipea_land_use.csv"), dtype={"id": str})
    origins = sorted(set(matrix["from_id"]))
    reached = matrix[matrix["travel_time"] <= CUTOFF].merge(
        land_use[["id", "jobs"]], left_on="to_id", right_on="id"
    )
    access = reached.groupby("from_id")["jobs"].sum().reindex(origins, fill_value=0)
    frame = pd.DataFrame({"from_id": origins, "accessibility": access.values})
    return frame, land_use


def _shared(land_use):
    return dict(sociodemographic_data=land_use, population="population")


def test_the_cutoff_replica_matches_the_package_frame(ipea):
    frame, land_use = ipea
    # cumulative_cutoff(cutoff = 30) returns 898 origins, 28 of them at 0.
    assert len(frame) == 898
    assert int((frame["accessibility"] == 0).sum()) == 28
    # The NA-income rows carry no population, so no weighted index sees them.
    assert (land_use.loc[land_use[INCOME].isna(), "population"] == 0).all()


def test_gini_matches_the_vignette(ipea):
    frame, land_use = ipea
    assert equity.gini_index(frame, **_shared(land_use)) == pytest.approx(
        GINI, abs=1e-7
    )


def test_theil_and_its_decomposition_match_the_vignette(ipea):
    frame, land_use = ipea
    # The package drops zero-access rows; cafein refuses them, so the
    # filter is explicit here.
    positive = frame[frame["accessibility"] > 0]
    assert equity.theil_t(positive, **_shared(land_use)) == pytest.approx(
        THEIL, abs=1e-7
    )
    with_decile = land_use.loc[land_use["income_decile"].notna(), "id"]
    grouped = equity.theil_t(
        positive[positive["from_id"].isin(with_decile)],
        groups="income_decile",
        **_shared(land_use),
    )
    assert grouped["total"].iloc[0] == pytest.approx(THEIL, abs=1e-7)
    assert grouped["between"].iloc[0] == pytest.approx(THEIL_BETWEEN, abs=1e-7)
    assert grouped["within"].iloc[0] == pytest.approx(THEIL_WITHIN, abs=1e-7)


@pytest.mark.parametrize("alpha", [0, 1, 2])
def test_fgt_matches_the_vignette(ipea, alpha):
    frame, land_use = ipea
    assert equity.fgt_poverty(
        frame, poverty_line=POVERTY_LINE, alpha=alpha, **_shared(land_use)
    ) == pytest.approx(FGT[alpha], abs=1e-7)


def test_palma_differs_from_the_package_by_its_boundary_rule(ipea):
    frame, land_use = ipea
    joined = frame.merge(land_use, left_on="from_id", right_on="id")
    poorest = joined[joined[INCOME] <= PALMA_Q40]
    wealthiest = joined[joined[INCOME] > PALMA_Q90]
    whole_row = np.average(
        wealthiest["accessibility"], weights=wealthiest["population"]
    ) / np.average(poorest["accessibility"], weights=poorest["population"])
    # Whole rows at Hmisc's thresholds reproduce the vignette exactly ...
    assert whole_row == pytest.approx(PALMA, abs=1e-6)
    # ... while cafein splits the boundary rows so the groups hold exactly
    # 10% and 40% of the population — a definitional difference of ~0.6%.
    ours = equity.palma_ratio(frame, income=INCOME, **_shared(land_use))
    assert ours == pytest.approx(3.779304, abs=1e-6)
    assert abs(ours - PALMA) / PALMA == pytest.approx(0.00557, abs=1e-4)


def _concentration_replica(joined, tie_averaged):
    """The package's covariance form with Lerman–Yitzhaki ranks: on the
    stable income order as data.table leaves it, or with every tied block
    at its midpoint rank."""
    ordered = joined.iloc[np.argsort(joined[INCOME].values, kind="stable")]
    share = ordered["population"] / ordered["population"].sum()
    cumulative = share.cumsum()
    if tie_averaged:
        income = ordered[INCOME]
        start = (cumulative - share).groupby(income, dropna=False).transform("min")
        end = cumulative.groupby(income, dropna=False).transform("max")
        rank = (start + end) / 2
    else:
        rank = cumulative - share / 2
    mean = np.average(ordered["accessibility"], weights=share)
    return float(
        2 * ((rank - 0.5) * (ordered["accessibility"] - mean) * share).sum() / mean
    )


def test_concentration_index_is_tie_invariant_where_the_package_is_not(ipea):
    frame, land_use = ipea
    joined = frame.merge(land_use, left_on="from_id", right_on="id")
    # 30 rows share an income with another row; the package ranks them in
    # row order, so its value depends on that order.
    assert (
        int(joined[joined["population"] > 0][INCOME].duplicated(keep=False).sum()) == 30
    )
    assert _concentration_replica(joined, tie_averaged=False) == pytest.approx(
        CONCENTRATION_STANDARD, abs=1e-7
    )
    standard = equity.concentration_index(frame, income=INCOME, **_shared(land_use))
    assert standard == pytest.approx(
        _concentration_replica(joined, tie_averaged=True), rel=1e-9
    )
    assert abs(standard - CONCENTRATION_STANDARD) < 3e-5
    # Erreygers is the standard index times 4·mean/(upper − lower); the
    # package infers the bounds from the data, cafein takes them explicitly.
    bounds = (float(frame["accessibility"].min()), float(frame["accessibility"].max()))
    mean = np.average(joined["accessibility"], weights=joined["population"])
    factor = 4 * mean / (bounds[1] - bounds[0])
    assert _concentration_replica(joined, tie_averaged=False) * factor == pytest.approx(
        CONCENTRATION_CORRECTED, abs=1e-7
    )
    corrected = equity.concentration_index(
        frame, income=INCOME, variant="erreygers", bounds=bounds, **_shared(land_use)
    )
    assert corrected == pytest.approx(standard * factor, rel=1e-9)
    assert abs(corrected - CONCENTRATION_CORRECTED) < 4e-5


def _sample(network, count_origins, count_destinations, seed=7):
    rng = np.random.default_rng(seed)
    stops = [stop for stop, _, _ in network.stops]
    origins = list(rng.choice(stops, count_origins, replace=False))
    destinations = list(rng.choice(stops, count_destinations, replace=False))
    table = pd.DataFrame(
        {"id": destinations, "jobs": rng.integers(1, 101, len(destinations))}
    )
    return origins, table


def test_whole_second_decay_weights_match_the_closed_forms(network):
    from cafein.accessibility import Accessibility, NearestDestinations

    origins, table = _sample(network, 10, 30)
    costs = NearestDestinations(
        network,
        origins,
        list(table["id"]),
        DEPARTURE,
        k=len(table),
        output_time_units="seconds",
    ).merge(table, left_on="destination_id", right_on="id")
    seconds = costs["cost"]

    def expect(weights):
        return (
            (weights * costs["jobs"])
            .groupby(costs["from_id"])
            .sum()
            .reindex(origins, fill_value=0.0)
        )

    def ours(**decay):
        result = Accessibility(
            network, origins, table, DEPARTURE, opportunities="jobs", **decay
        )
        return result.set_index("from_id")["accessibility"].reindex(origins)

    horizon = float(seconds.max()) / 60 + 1
    exponential = ours(
        budgets=(horizon,), decay="exponential", decay_params={"half_life": 7.0}
    )
    assert np.allclose(
        exponential, expect(np.exp(-math.log(2) * seconds / 420.0)), rtol=1e-12
    )
    logistic = ours(budgets=(30.0,), decay="logistic", decay_params={"scale": 5.0})
    assert np.allclose(
        logistic,
        expect(np.where(seconds <= 1800, 1 / (1 + np.exp((seconds - 1800) / 300)), 0)),
        rtol=1e-12,
    )
    step = ours(budgets=(30.0,))
    assert np.array_equal(step, expect((seconds <= 1800).astype(float)))


def _harness():
    import importlib.util

    spec = importlib.util.spec_from_file_location("compare_vs_ipea", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "ipea, cafein, agrees, valid",
    [
        (1.0, 1.0 + 1e-12, True, True),
        (math.inf, math.inf, True, True),
        (math.inf, -math.inf, False, False),
        (math.inf, 5.0, False, False),
        (math.nan, 1.0, False, False),
        (1.0, math.nan, False, False),
    ],
)
def test_the_harness_never_accepts_non_finite_disagreements(
    ipea, cafein, agrees, valid
):
    comparison = _harness().Comparison()
    comparison.add("part", "measure", "-", "-", ipea, cafein, "differs")
    (row,) = comparison.rows
    assert row["agrees"] is agrees
    assert row["valid"] is valid
    # Only a finite disagreement may pass as an expected difference; any
    # NaN or mismatched infinity fails the run whatever the expectation.
    assert bool(comparison.failures()) is (not valid)


def test_the_harness_summarizes_groups_without_a_finite_difference():
    comparison = _harness().Comparison()
    comparison.add("part", "measure", "-", "a", math.nan, 1.0, "identical")
    comparison.add("part", "measure", "-", "b", 2.0, math.nan, "identical")
    summary = comparison.summary()
    assert len(summary) == 1
    assert summary["max_abs_diff"].iloc[0] == math.inf
    assert summary["status"].iloc[0] == "UNEXPECTED"
    assert summary["worst_key"].iloc[0] in {"a", "b"}


def _rscript():
    rscript = os.environ.get("CAFEIN_RSCRIPT") or shutil.which("Rscript")
    if rscript is None:
        pytest.skip("no Rscript (set CAFEIN_RSCRIPT)")
    loads = subprocess.run(
        [rscript, "-e", 'cat(as.character(packageVersion("accessibility")))'],
        capture_output=True,
        text=True,
        check=False,
    )
    if loads.returncode != 0:
        pytest.skip(f"{rscript} has no accessibility package")
    version = loads.stdout.strip().splitlines()[-1]
    if version != PACKAGE_VERSION:
        pytest.skip(f"accessibility {version} installed, not {PACKAGE_VERSION}")
    return rscript


def test_the_comparison_script_agrees_with_the_package(tmp_path):
    _fixture("helsinki_gtfs.zip", "python scripts/fetch_test_data.py", fetched=True)
    rscript = _rscript()
    output = tmp_path / "rows.csv"
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--part",
            "accessibility",
            "--origins",
            "10",
            "--destinations",
            "30",
            "--rscript",
            rscript,
            "--csv",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    rows = pd.read_csv(output)
    identical = rows[rows["expected"] == "identical"]
    assert identical["agrees"].all()
    covered = identical.groupby(["measure", "variant"])["key"].agg(set)
    assert set(covered[("pair_costs", "matrix vs dispatch")]) == {
        "pairs left_only",
        "pairs right_only",
        "differing pairs",
    }
    origins = covered[("cumulative_cutoff", "15 min")]
    assert len(origins) == 10
    for measure, variants in {
        "cumulative_cutoff": {"15 min", "30 min", "45 min"},
        "cost_to_closest": {"n=1", "n=2", "n=3"},
    }.items():
        for variant in variants:
            assert covered[(measure, variant)] == origins
    gravity = {variant for measure, variant in covered.index if measure == "gravity"}
    assert {variant.split(" (")[0] for variant in gravity} == {
        "exponential whole",
        "logistic whole",
    }
    for variant in gravity:
        assert covered[("gravity", variant)] == origins
