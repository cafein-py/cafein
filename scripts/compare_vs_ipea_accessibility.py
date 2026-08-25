#!/usr/bin/env python3

"""Compare cafein's accessibility and equity measures against IPEA's R
``accessibility`` package on identical inputs.

Part A (``equity``) runs the R package's inequality and poverty indices on
the sample data it ships (a Belo Horizonte travel matrix and land-use
table) and cafein's ``equity`` functions on the same frames. Part B
(``accessibility``) routes cafein's Helsinki test network, exports the
travel-time matrix in seconds to R, and compares the package's
cumulative-cutoff, gravity, and cost-to-closest measures against
``Accessibility`` and ``NearestDestinations`` on the same origins and
destinations — after asserting, pair for pair, that both engines see the
same costs. Every comparison prints both values and their difference; the
exit status is non-zero when a pair expected to be identical is not.

    python scripts/compare_vs_ipea_accessibility.py                  # both parts
    python scripts/compare_vs_ipea_accessibility.py --part equity
    python scripts/compare_vs_ipea_accessibility.py --csv results.csv
    python scripts/compare_vs_ipea_accessibility.py --export-fixtures

Requirements: cafein installed with its compiled core; an R environment
with the ``accessibility`` package, exactly 1.5.0 — point ``--rscript`` (or
``CAFEIN_RSCRIPT``) at its ``Rscript``. Part B needs the Helsinki test
data from ``python scripts/fetch_test_data.py``. ``--export-fixtures``
writes the R package's sample tables to ``tests/data`` as the CSV fixtures
``tests/test_ipea_accessibility.py`` reads.
"""

import argparse
import math
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

REPOSITORY = pathlib.Path(__file__).resolve().parent.parent
DATA = REPOSITORY / "tests" / "data"
GTFS = DATA / "helsinki_gtfs.zip"
FIXTURES = {
    "ipea_travel_matrix.csv": "travel_matrix",
    "ipea_land_use.csv": "land_use",
}

DEPARTURE = "2022-02-22 08:30"
EQUITY_CUTOFF = 30
POVERTY_LINE = 50000
CUTOFFS_MIN = (15.0, 30.0, 45.0)
# Decay parameters: one whole-second value per family (cafein rounds time
# parameters to whole seconds) and the fractional literature value.
HALF_LIVES_MIN = {"whole": 7.0, "fractional": math.log(2) / 0.1}
SCALES_MIN = {"whole": 5.0, "fractional": 10.0 * math.sqrt(3) / math.pi}
LOGISTIC_BUDGET_MIN = 30.0
LINEAR_WIDTH_MIN = 2 * LOGISTIC_BUDGET_MIN
REL_TOLERANCE = 1e-9
ABS_TOLERANCE = 1e-9

R_PRELUDE = """
suppressMessages(library(accessibility))
suppressMessages(library(data.table))
stopifnot(packageVersion("accessibility") == "@@VERSION@@")
staging <- Sys.getenv("CAFEIN_STAGING")
staged <- function(name) file.path(staging, name)
"""

R_EXPORT = R_PRELUDE + """
extdata <- system.file("extdata", package = "accessibility")
travel_matrix <- readRDS(file.path(extdata, "travel_matrix.rds"))
land_use <- readRDS(file.path(extdata, "land_use_data.rds"))
fwrite(travel_matrix, staged("ipea_travel_matrix.csv"))
fwrite(land_use, staged("ipea_land_use.csv"))
"""

R_EQUITY = R_EXPORT + """
access <- cumulative_cutoff(
    travel_matrix, land_use, opportunity = "jobs", travel_cost = "travel_time",
    cutoff = @@CUTOFF@@
)
fwrite(access, staged("access.csv"))
quantiles <- Hmisc::wtd.quantile(
    land_use$income_per_capita, weights = land_use$population, probs = c(0.4, 0.9)
)
index_args <- list(
    sociodemographic_data = land_use, opportunity = "jobs", population = "population"
)
call_index <- function(f, ...) do.call(f, c(list(access), index_args, list(...)))
no_na <- access[id %in% land_use$id[!is.na(land_use$income_decile)]]
grouped <- do.call(
    theil_t, c(list(no_na), index_args, list(socioeconomic_groups = "income_decile"))
)
fgt <- call_index(fgt_poverty, poverty_line = @@LINE@@)
rows <- data.table(
    measure = c(
        "gini_index", "palma_ratio", "palma_q40", "palma_q90",
        "concentration_standard", "concentration_corrected", "theil_t",
        "theil_total", "theil_between", "theil_within", "fgt0", "fgt1", "fgt2",
        "access_min", "access_max"
    ),
    value = c(
        call_index(gini_index)$gini_index,
        call_index(palma_ratio, income = "income_per_capita")$palma_ratio,
        quantiles[["40%"]], quantiles[["90%"]],
        call_index(
            concentration_index, income = "income_per_capita", type = "standard"
        )$concentration_index,
        call_index(
            concentration_index, income = "income_per_capita", type = "corrected"
        )$concentration_index,
        call_index(theil_t)$theil_t,
        grouped$summary[component == "total", value],
        grouped$summary[component == "between_group", value],
        grouped$summary[component == "within_group", value],
        fgt$FGT0, fgt$FGT1, fgt$FGT2,
        min(access$jobs), max(access$jobs)
    )
)
fwrite(rows, staged("equity.csv"))
"""

R_ACCESSIBILITY = R_PRELUDE + """
matrix <- fread(
    staged("matrix.csv"), colClasses = c(from_id = "character", to_id = "character")
)
destinations <- fread(staged("destinations.csv"), colClasses = c(id = "character"))
within <- matrix[travel_time <= @@LOGISTIC_BUDGET@@]
cutoff <- cumulative_cutoff(
    matrix, destinations, opportunity = "jobs", travel_cost = "travel_time",
    cutoff = c(@@CUTOFFS@@)
)
fwrite(cutoff, staged("cutoff.csv"))
run_gravity <- function(data, decay, name) {
    result <- gravity(
        data, destinations, opportunity = "jobs", travel_cost = "travel_time",
        decay_function = decay
    )
    result[, variant := name]
    result
}
gravity_rows <- rbind(
    run_gravity(matrix, decay_exponential(decay_value = @@X_WHOLE@@), "exp_whole"),
    run_gravity(matrix, decay_exponential(decay_value = @@X_FRAC@@), "exp_fractional"),
    run_gravity(
        within, decay_logistic(cutoff = @@LOGISTIC_BUDGET@@, sd = @@SD_WHOLE@@),
        "logistic_whole"
    ),
    run_gravity(
        within, decay_logistic(cutoff = @@LOGISTIC_BUDGET@@, sd = @@SD_FRAC@@),
        "logistic_fractional"
    ),
    run_gravity(within, decay_linear(cutoff = @@LOGISTIC_BUDGET@@), "linear")
)
fwrite(gravity_rows, staged("gravity.csv"))
closest <- cost_to_closest(
    matrix, destinations, opportunity = "unit", travel_cost = "travel_time",
    n = c(1, 2, 3)
)
fwrite(closest, staged("closest.csv"))
"""


def render(template, **values):
    for key, value in values.items():
        template = template.replace(f"@@{key}@@", str(value))
    return template


def run_r(rscript, staging, script_text, name):
    script = staging / f"{name}.R"
    script.write_text(script_text)
    completed = subprocess.run(
        [rscript, str(script)],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "CAFEIN_STAGING": str(staging)},
    )
    if completed.returncode != 0:
        sys.exit(f"R failed in {script}:\n{completed.stderr}")


PACKAGE_VERSION = "1.5.0"


def check_r(rscript):
    """The installed package version, which must be the one every
    formula and reference number here belongs to."""
    completed = subprocess.run(
        [rscript, "-e", 'cat(as.character(packageVersion("accessibility")))'],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        sys.exit(
            f"{rscript} cannot load the accessibility package:\n{completed.stderr}"
        )
    version = completed.stdout.strip().splitlines()[-1]
    if version != PACKAGE_VERSION:
        sys.exit(
            f"accessibility {version} installed; this comparison is written "
            f"against {PACKAGE_VERSION}"
        )
    return version


def check_cafein():
    """The cafein that runs must be this repository's package and its
    compiled extension — not an installed copy or another checkout."""
    import cafein
    import cafein._cafein as core

    package = pathlib.Path(cafein.__file__).resolve()
    extension = pathlib.Path(core.__file__).resolve()
    root = REPOSITORY / "python"
    for path in (package, extension):
        if root not in path.parents:
            sys.exit(f"cafein resolves to {path}, outside {root}")
    # The package version is the workspace's (`dynamic` in pyproject).
    expected = re.search(
        r'^version = "([^"]+)"', (REPOSITORY / "Cargo.toml").read_text(), re.M
    ).group(1)
    if cafein.__version__ != expected:
        sys.exit(f"cafein {cafein.__version__} runs; this checkout is {expected}")
    return cafein.__version__


class Comparison:
    """The accumulated comparison rows and their verdicts."""

    def __init__(self):
        self.rows = []

    def add(self, part, measure, variant, key, ipea, cafein, expected, exact=False):
        """One comparison row; ``exact`` demands equality (integer-valued
        measures), otherwise the float tolerances apply."""
        ipea = float(ipea)
        cafein = float(cafein)
        finite = math.isfinite(ipea) and math.isfinite(cafein)
        # Valid pairs are two finite numbers or the same signed infinity
        # (an unreached destination on both sides); anything else is a
        # failure whatever the expectation, as is any NaN.
        valid = finite or (math.isinf(ipea) and ipea == cafein)
        if not finite:
            ok = valid
            absolute = relative = 0.0 if ok else math.inf
        else:
            absolute = abs(ipea - cafein)
            scale = max(abs(ipea), abs(cafein))
            relative = absolute / scale if scale > 0 else 0.0
            if exact:
                ok = valid and ipea == cafein
            else:
                ok = valid and (absolute <= ABS_TOLERANCE or relative <= REL_TOLERANCE)
        self.rows.append(
            {
                "part": part,
                "measure": measure,
                "variant": variant,
                "key": key,
                "ipea": ipea,
                "cafein": cafein,
                "abs_diff": absolute,
                "rel_diff": relative,
                "expected": expected,
                "agrees": ok,
                "valid": valid,
            }
        )

    def add_series(self, part, measure, variant, ipea, cafein, expected, exact=False):
        """Per-key rows for two Series aligned on their index."""
        import pandas as pd

        aligned = pd.concat([ipea.rename("ipea"), cafein.rename("cafein")], axis=1)
        for key, row in aligned.iterrows():
            self.add(
                part, measure, variant, key, row["ipea"], row["cafein"], expected, exact
            )

    def frame(self):
        import pandas as pd

        return pd.DataFrame(self.rows)

    def summary(self):
        import pandas as pd

        frame = self.frame()
        grouped = frame.groupby(["part", "measure", "variant"], sort=False)
        summary = grouped.agg(
            keys=("key", "size"),
            max_abs_diff=("abs_diff", "max"),
            max_rel_diff=("rel_diff", "max"),
            expected=("expected", "first"),
            agree=("agrees", "sum"),
        ).reset_index()
        # Both engines' values where the measure differs most.
        worst = frame.loc[grouped["abs_diff"].idxmax(), ["key", "ipea", "cafein"]]
        worst.columns = ["worst_key", "ipea_at_worst", "cafein_at_worst"]
        summary = pd.concat([summary, worst.reset_index(drop=True)], axis=1)
        summary["status"] = [
            (
                "ok"
                if (row.agree == row.keys) == (row.expected == "identical")
                else "UNEXPECTED"
            )
            for row in summary.itertuples()
        ]
        return summary

    def failures(self):
        """Rows expected identical that differ, and rows with a NaN on
        either side — no expectation makes a non-number acceptable."""
        return [
            row
            for row in self.rows
            if not row["valid"]
            or (row["expected"] == "identical" and not row["agrees"])
        ]


def concentration_replica(joined, income, tie_averaged):
    """The package's covariance form with Lerman–Yitzhaki ranks: in the
    stable income order data.table leaves (``tie_averaged=False``, the
    package's rule) or with every tied block at its midpoint rank
    (``True``, cafein's rule)."""
    import numpy as np

    ordered = joined.iloc[np.argsort(joined[income].values, kind="stable")]
    share = ordered["population"] / ordered["population"].sum()
    cumulative = share.cumsum()
    if tie_averaged:
        blocks = ordered[income]
        start = (cumulative - share).groupby(blocks, dropna=False).transform("min")
        end = cumulative.groupby(blocks, dropna=False).transform("max")
        rank = (start + end) / 2
    else:
        rank = cumulative - share / 2
    mean = np.average(ordered["accessibility"], weights=share)
    return float(
        2 * ((rank - 0.5) * (ordered["accessibility"] - mean) * share).sum() / mean
    )


def cutoff_access(matrix, destinations, opportunity, cutoff, origins):
    """The IPEA cumulative-cutoff rule in pandas: opportunities at
    ``travel_time <= cutoff`` summed per origin, absent origins 0."""
    reached = matrix[matrix["travel_time"] <= cutoff].merge(
        destinations[["id", opportunity]], left_on="to_id", right_on="id"
    )
    return reached.groupby("from_id")[opportunity].sum().reindex(origins, fill_value=0)


def equity_part(rscript, staging, comparison):
    import numpy as np
    import pandas as pd

    from cafein import equity

    run_r(
        rscript,
        staging,
        render(
            R_EQUITY, VERSION=PACKAGE_VERSION, CUTOFF=EQUITY_CUTOFF, LINE=POVERTY_LINE
        ),
        "equity",
    )
    matrix = pd.read_csv(
        staging / "ipea_travel_matrix.csv", dtype={"from_id": str, "to_id": str}
    )
    land_use = pd.read_csv(staging / "ipea_land_use.csv", dtype={"id": str})
    access = pd.read_csv(staging / "access.csv", dtype={"id": str})
    theirs = pd.read_csv(staging / "equity.csv").set_index("measure")["value"]

    # The step rule itself: R's access frame against the pandas replica.
    origins = sorted(set(matrix["from_id"]))
    replica = cutoff_access(matrix, land_use, "jobs", EQUITY_CUTOFF, origins)
    ipea_access = access.set_index("id")["jobs"]
    comparison.add(
        "equity",
        "cutoff_origins",
        "-",
        "-",
        len(ipea_access),
        len(replica),
        "identical",
        exact=True,
    )
    comparison.add_series(
        "equity", "cutoff_access", "-", ipea_access, replica, "identical", exact=True
    )

    frame = access.rename(columns={"id": "from_id", "jobs": "accessibility"})
    shared = dict(sociodemographic_data=land_use, population="population")
    income = "income_per_capita"
    na_rows = land_use[land_use[income].isna()]
    assert (na_rows["population"] == 0).all(), "NA-income rows carry population"

    comparison.add(
        "equity",
        "gini_index",
        "-",
        "-",
        theirs["gini_index"],
        equity.gini_index(frame, **shared),
        "identical",
    )
    # Concentration indices: the package ranks tied incomes in row order,
    # cafein gives a tied block one midpoint rank, so on data with ties
    # each side is checked against its own replica and the gap between
    # them is expected.
    joined = frame.merge(land_use, left_on="from_id", right_on="id")
    ties = int(
        joined.loc[joined["population"] > 0, income].duplicated(keep=False).sum()
    )
    standard = equity.concentration_index(frame, income=income, **shared)
    bounds = (float(theirs["access_min"]), float(theirs["access_max"]))
    corrected = equity.concentration_index(
        frame, income=income, variant="erreygers", bounds=bounds, **shared
    )
    factor = 4 * np.average(joined["accessibility"], weights=joined["population"])
    factor /= bounds[1] - bounds[0]
    row_order = concentration_replica(joined, income, tie_averaged=False)
    tie_averaged = concentration_replica(joined, income, tie_averaged=True)
    for variant, scale, ipea, ours in (
        ("standard", 1.0, theirs["concentration_standard"], standard),
        (
            "corrected/erreygers(bounds=min,max)",
            factor,
            theirs["concentration_corrected"],
            corrected,
        ),
    ):
        comparison.add(
            "equity",
            "concentration_index",
            f"{variant}: IPEA vs row-order replica",
            "-",
            ipea,
            row_order * scale,
            "identical",
        )
        comparison.add(
            "equity",
            "concentration_index",
            f"{variant}: cafein vs tie-averaged replica",
            "-",
            tie_averaged * scale,
            ours,
            "identical",
        )
        comparison.add(
            "equity",
            "concentration_index",
            f"{variant}: IPEA vs cafein ({ties} tied incomes)",
            "-",
            ipea,
            ours,
            "differs" if ties else "identical",
        )
    positive = frame[frame["accessibility"] > 0]
    comparison.add(
        "equity",
        "theil_t",
        "positive access",
        "-",
        theirs["theil_t"],
        equity.theil_t(positive, **shared),
        "identical",
    )
    with_decile = land_use.loc[land_use["income_decile"].notna(), "id"]
    grouped = equity.theil_t(
        positive[positive["from_id"].isin(with_decile)],
        groups="income_decile",
        **shared,
    )
    for component in ("total", "between", "within"):
        comparison.add(
            "equity",
            "theil_t",
            f"{component} by income_decile",
            "-",
            theirs[f"theil_{component}"],
            grouped[component].iloc[0],
            "identical",
        )
    for alpha in (0, 1, 2):
        comparison.add(
            "equity",
            "fgt_poverty",
            f"alpha={alpha}",
            "-",
            theirs[f"fgt{alpha}"],
            equity.fgt_poverty(frame, poverty_line=POVERTY_LINE, alpha=alpha, **shared),
            "identical",
        )
    # Palma: cafein splits the boundary rows fractionally at the weighted
    # quantiles; the package assigns whole rows at Hmisc's thresholds.
    joined = frame.merge(land_use, left_on="from_id", right_on="id")
    poorest = joined[joined[income] <= theirs["palma_q40"]]
    wealthiest = joined[joined[income] > theirs["palma_q90"]]
    whole_row = np.average(
        wealthiest["accessibility"], weights=wealthiest["population"]
    ) / np.average(poorest["accessibility"], weights=poorest["population"])
    comparison.add(
        "equity",
        "palma_ratio",
        "IPEA whole-row rule replica",
        "-",
        theirs["palma_ratio"],
        whole_row,
        "identical",
    )
    comparison.add(
        "equity",
        "palma_ratio",
        "cafein fractional split",
        "-",
        theirs["palma_ratio"],
        equity.palma_ratio(frame, income=income, **shared),
        "differs",
    )


def sample_stops(network, count_origins, count_destinations, seed):
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(seed)
    stops = [stop for stop, _, _ in network.stops]
    origins = list(rng.choice(stops, count_origins, replace=False))
    destinations = list(rng.choice(stops, count_destinations, replace=False))
    table = pd.DataFrame(
        {
            "id": destinations,
            "jobs": rng.integers(1, 101, len(destinations)),
            "unit": 1,
        }
    )
    return origins, table


def accessibility_part(rscript, staging, comparison, count_origins, count_dests, seed):
    import numpy as np
    import pandas as pd

    from cafein import TransportNetwork, TravelTimeMatrix
    from cafein.accessibility import Accessibility, NearestDestinations

    if not GTFS.exists():
        sys.exit(f"{GTFS} missing; run python scripts/fetch_test_data.py")
    network = TransportNetwork.from_gtfs([str(GTFS)])
    origins, table = sample_stops(network, count_origins, count_dests, seed)
    destinations = list(table["id"])

    matrix = TravelTimeMatrix(
        network, origins, departure=DEPARTURE, output_time_units="seconds"
    )
    matrix = matrix[matrix["to_id"].isin(destinations)].reset_index(drop=True)
    # The per-pair cost check: the accessibility dispatch's costs, pair for
    # pair, against the exported matrix.
    every = NearestDestinations(
        network,
        origins,
        destinations,
        DEPARTURE,
        k=len(destinations),
        output_time_units="seconds",
    ).rename(columns={"destination_id": "to_id"})
    pairs = matrix.merge(every, on=["from_id", "to_id"], how="outer", indicator=True)
    both = pairs[pairs["_merge"] == "both"]
    gate = {
        "pairs left_only": int((pairs["_merge"] == "left_only").sum()),
        "pairs right_only": int((pairs["_merge"] == "right_only").sum()),
        "differing pairs": int((both["travel_time"] != both["cost"]).sum()),
    }
    for key, count in gate.items():
        comparison.add(
            "accessibility",
            "pair_costs",
            "matrix vs dispatch",
            key,
            0,
            count,
            "identical",
            exact=True,
        )
    if any(gate.values()):
        # Aggregates over costs the engines disagree on would compare
        # routing, not aggregation: stop here.
        sys.exit(f"per-pair costs differ between matrix and dispatch: {gate}")

    matrix.to_csv(staging / "matrix.csv", index=False)
    table.to_csv(staging / "destinations.csv", index=False)
    budget_s = LOGISTIC_BUDGET_MIN * 60.0
    x_whole = math.log(2) / (HALF_LIVES_MIN["whole"] * 60.0)
    x_frac = math.log(2) / (HALF_LIVES_MIN["fractional"] * 60.0)
    sd_whole = SCALES_MIN["whole"] * 60.0 * math.pi / math.sqrt(3)
    sd_frac = SCALES_MIN["fractional"] * 60.0 * math.pi / math.sqrt(3)
    run_r(
        rscript,
        staging,
        render(
            R_ACCESSIBILITY,
            VERSION=PACKAGE_VERSION,
            CUTOFFS=", ".join(f"{cutoff * 60:.0f}" for cutoff in CUTOFFS_MIN),
            LOGISTIC_BUDGET=f"{budget_s:.0f}",
            X_WHOLE=repr(x_whole),
            X_FRAC=repr(x_frac),
            SD_WHOLE=repr(sd_whole),
            SD_FRAC=repr(sd_frac),
        ),
        "accessibility",
    )
    cutoff = pd.read_csv(staging / "cutoff.csv", dtype={"id": str})
    gravity = pd.read_csv(staging / "gravity.csv", dtype={"id": str})
    closest = pd.read_csv(staging / "closest.csv", dtype={"id": str})

    def theirs(frame, value, fill):
        return frame.set_index("id")[value].reindex(origins, fill_value=fill)

    step = Accessibility(
        network, origins, table, DEPARTURE, opportunities="jobs", budgets=CUTOFFS_MIN
    )
    for budget in CUTOFFS_MIN:
        ours = step[step["budget"] == budget].set_index("from_id")["accessibility"]
        rows = cutoff[cutoff["travel_time"] == budget * 60]
        comparison.add_series(
            "accessibility",
            "cumulative_cutoff",
            f"{budget:.0f} min",
            theirs(rows, "jobs", 0),
            ours.reindex(origins),
            "identical",
            exact=True,
        )

    horizon = float(matrix["travel_time"].max()) / 60.0 + 1.0
    for kind, half_life in HALF_LIVES_MIN.items():
        ours = Accessibility(
            network,
            origins,
            table,
            DEPARTURE,
            opportunities="jobs",
            budgets=(horizon,),
            decay="exponential",
            decay_params={"half_life": half_life},
        )
        comparison.add_series(
            "accessibility",
            "gravity",
            f"exponential {kind} (half_life {half_life * 60:.3f} s)",
            theirs(gravity[gravity["variant"] == f"exp_{kind}"], "jobs", 0),
            ours.set_index("from_id")["accessibility"].reindex(origins),
            "identical" if kind == "whole" else "differs",
        )
    for kind, scale in SCALES_MIN.items():
        ours = Accessibility(
            network,
            origins,
            table,
            DEPARTURE,
            opportunities="jobs",
            budgets=(LOGISTIC_BUDGET_MIN,),
            decay="logistic",
            decay_params={"scale": scale},
        )
        factor = 1.0 + math.exp(-budget_s / (scale * 60.0))
        comparison.add_series(
            "accessibility",
            "gravity",
            f"logistic {kind} (scale {scale * 60:.3f} s, rescaled)",
            theirs(gravity[gravity["variant"] == f"logistic_{kind}"], "jobs", 0),
            ours.set_index("from_id")["accessibility"].reindex(origins) * factor,
            "identical" if kind == "whole" else "differs",
        )
    ours = Accessibility(
        network,
        origins,
        table,
        DEPARTURE,
        opportunities="jobs",
        budgets=(LOGISTIC_BUDGET_MIN,),
        decay="linear",
        decay_params={"width": LINEAR_WIDTH_MIN},
    )
    comparison.add_series(
        "accessibility",
        "gravity",
        f"linear (width {LINEAR_WIDTH_MIN:.0f} min vs 1 - t/cutoff)",
        theirs(gravity[gravity["variant"] == "linear"], "jobs", 0),
        ours.set_index("from_id")["accessibility"].reindex(origins),
        "differs",
    )

    nearest = NearestDestinations(
        network, origins, destinations, DEPARTURE, k=3, output_time_units="seconds"
    )
    for rank in (1, 2, 3):
        ours = (
            nearest[nearest["rank"] == rank]
            .set_index("from_id")["cost"]
            .reindex(origins, fill_value=np.inf)
        )
        comparison.add_series(
            "accessibility",
            "cost_to_closest",
            f"n={rank}",
            theirs(closest[closest["n"] == rank], "travel_time", np.inf),
            ours,
            "identical",
            exact=True,
        )


def export_fixtures(rscript, staging):
    run_r(rscript, staging, render(R_EXPORT, VERSION=PACKAGE_VERSION), "export")
    DATA.mkdir(parents=True, exist_ok=True)
    # The destination directory is resolved once; each fixture is written
    # through the descriptor mkstemp returned and renamed into place with
    # full paths, which is atomic and portable (Windows has no dir_fd).
    directory = DATA.resolve(strict=True)
    for name in FIXTURES:
        target = directory / name
        if target.is_symlink():
            sys.exit(f"{target} is a symlink; refusing to write through it")
        handle, temporary = tempfile.mkstemp(dir=directory, prefix=f".{name}.")
        try:
            with os.fdopen(handle, "wb") as sink:
                with open(staging / name, "rb") as source:
                    shutil.copyfileobj(source, sink)
            os.replace(temporary, target)
        except BaseException:
            try:
                os.unlink(temporary)
            except OSError:
                pass
            raise
        print(f"wrote {target}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--part", choices=("equity", "accessibility", "all"), default="all"
    )
    parser.add_argument("--origins", type=int, default=40)
    parser.add_argument("--destinations", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--csv", type=pathlib.Path, help="write every row here")
    parser.add_argument(
        "--export-fixtures",
        action="store_true",
        help="write the R package's sample tables to tests/data and exit",
    )
    parser.add_argument(
        "--rscript",
        default=os.environ.get("CAFEIN_RSCRIPT", "Rscript"),
        help="Rscript of an R environment with the accessibility package",
    )
    arguments = parser.parse_args()

    version = check_r(arguments.rscript)
    print(f"R accessibility package {version} via {arguments.rscript}")
    with tempfile.TemporaryDirectory(prefix="cafein-ipea-") as directory:
        staging = pathlib.Path(directory)
        if arguments.export_fixtures:
            export_fixtures(arguments.rscript, staging)
            return
        print(f"cafein {check_cafein()} from {REPOSITORY}")
        comparison = Comparison()
        if arguments.part in ("equity", "all"):
            equity_part(arguments.rscript, staging, comparison)
        if arguments.part in ("accessibility", "all"):
            accessibility_part(
                arguments.rscript,
                staging,
                comparison,
                arguments.origins,
                arguments.destinations,
                arguments.seed,
            )

    import pandas as pd

    with pd.option_context("display.width", 200, "display.max_rows", 500):
        print(comparison.summary().to_string(index=False))
        scalars = comparison.frame()
        scalars = scalars[scalars["key"] == "-"]
        if len(scalars):
            print()
            print(
                scalars[
                    ["measure", "variant", "ipea", "cafein", "abs_diff", "rel_diff"]
                ].to_string(index=False, float_format=lambda v: f"{v:.10g}")
            )
    if arguments.csv:
        comparison.frame().to_csv(arguments.csv, index=False)
        print(f"rows written to {arguments.csv}")
    failures = comparison.failures()
    if failures:
        sys.exit(f"{len(failures)} comparison(s) failed (expected identical, or NaN)")


if __name__ == "__main__":
    main()
