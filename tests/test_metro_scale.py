"""Metro-scale runs over cafein.sampledata.helsinki's downloaded data.

Skipped without an installed, pinned sampledata package; the recurring
installed-package job runs them with ``CAFEIN_REQUIRE_SAMPLEDATA=1``.
"""

import csv
import datetime
import io
import zipfile

import pytest

pytest.importorskip("cafein._cafein")

from cafein import StreetNetwork, TransportNetwork, TravelTimeMatrix  # noqa: E402

pytestmark = pytest.mark.metro_scale


def _service_date(gtfs_path):
    """A Tuesday near the middle of the feed's own service span.

    Read from the pinned feed rather than the wall clock, so an
    immutable data release keeps passing however much time passes.
    """
    dates = []
    with zipfile.ZipFile(gtfs_path) as archive:
        names = set(archive.namelist())
        for name, columns in (
            ("calendar.txt", ("start_date", "end_date")),
            ("calendar_dates.txt", ("date",)),
        ):
            if name not in names or archive.getinfo(name).file_size > (8 << 20):
                continue
            rows = csv.DictReader(io.StringIO(archive.read(name).decode("utf-8-sig")))
            for row in rows:
                for column in columns:
                    value = (row.get(column) or "").strip()
                    if len(value) == 8 and value.isdigit():
                        dates.append(value)
    if not dates:
        pytest.fail("the feed declares no service dates")
    dates.sort()
    parse = lambda v: datetime.date(int(v[:4]), int(v[4:6]), int(v[6:]))  # noqa: E731
    start, end = parse(dates[0]), parse(dates[-1])
    middle = start + (end - start) / 2
    tuesday = middle + datetime.timedelta(days=(1 - middle.weekday()) % 7)
    if tuesday > end:
        tuesday = middle - datetime.timedelta(days=(middle.weekday() - 1) % 7)
    return tuesday.isoformat()


@pytest.fixture(scope="module")
def metro_network(helsinki_metro_data):
    """The full HSL network with the capital-region walking streets."""
    return TransportNetwork.from_gtfs(
        [str(helsinki_metro_data.gtfs)],
        osm_pbf=str(helsinki_metro_data.osm_pbf),
    )


@pytest.fixture(scope="module")
def metro_streets(helsinki_metro_data):
    """The capital-region street network with elevations."""
    return StreetNetwork.from_osm(
        str(helsinki_metro_data.osm_pbf), dem=str(helsinki_metro_data.dem)
    )


def test_the_metro_network_exceeds_the_sample_fixture(
    metro_network, metro_streets, network, helsinki_streets
):
    # The point of the package: strictly more than the pinned r5py
    # sample — more stops in the feed, more streets in the extract.
    assert len(list(metro_network.stops)) > len(list(network.stops))
    assert metro_streets.edge_count > helsinki_streets.edge_count


def test_transit_routes_at_metro_scale(metro_network, helsinki_metro_data):
    date = _service_date(helsinki_metro_data.gtfs)
    stops = [stop for stop, lat, lon in metro_network.stops if lat is not None]
    # An even spread: HSL stop ids sort an unserved-station block first,
    # so the head of the list alone would probe no served stop.
    sampled = stops[:: max(1, len(stops) // 25)][:25]
    for origin in sampled:
        times = metro_network.travel_times_from_stop(origin, f"{date} 08:30:00")
        if any(stop != origin and seconds > 0 for stop, seconds in times.items()):
            break
    else:
        pytest.fail(f"no transit service reachable from 25 sampled stops on {date}")
    # One small stop-to-stop matrix over the sampled origins.
    matrix = metro_network.travel_time_matrix(sampled[:5], f"{date} 08:30:00")
    assert len(matrix) > 0


def test_streets_route_across_the_capital_region(metro_streets):
    # Central Helsinki to Tikkurila, Vantaa: a ride the small sample
    # extract cannot even contain.
    ride = metro_streets.travel_time(
        (60.1699, 24.9384), (60.2934, 25.0378), mode="bicycle"
    )
    assert ride is not None and ride > 0
    origins = [(60.1699, 24.9384), (60.2055, 24.6559), (60.2934, 25.0378)]
    import geopandas

    frame = geopandas.GeoDataFrame(
        {"id": ["kamppi", "espoo", "tikkurila"]},
        geometry=geopandas.points_from_xy(
            [lon for lat, lon in origins], [lat for lat, lon in origins]
        ),
        crs="EPSG:4326",
    )
    matrix = TravelTimeMatrix(metro_streets, frame, transport_mode="bicycle")
    assert (matrix.from_id != matrix.to_id).any()


def test_the_population_grid_reads_and_counts(helsinki_metro_data):
    geopandas = pytest.importorskip("geopandas")

    grid = geopandas.read_file(
        helsinki_metro_data.population_grid, layer="population_grid"
    )
    assert grid.crs.to_epsg() == 3067
    assert (grid["asukkaita"].dropna() > 0).any()
    assert len(grid) > 3000


def test_metro_footpaths_stay_bounded(metro_network, helsinki_metro_data):
    # Issue #249: the transitive closure turned the one-component
    # capital-region walking graph into a near-complete transfer set
    # (50 M edges, ~3000 per stop) and manufactured hours-long transfer
    # legs. Each transfer is now one street walk within the cutoff.
    from cafein.streets import MAX_WALKING_TIME

    stops = len(list(metro_network.stops))
    assert metro_network.transfer_count < 100 * stops
    durations = [duration for _, _, duration in metro_network._core._transfer_edges()]
    assert durations and max(durations) <= MAX_WALKING_TIME

    date = _service_date(helsinki_metro_data.gtfs)
    served = [stop for stop, lat, lon in metro_network.stops if lat is not None]
    sampled = served[:: max(1, len(served) // 25)][:25]
    walked = 0
    for origin in sampled:
        for destination in sampled:
            if origin == destination:
                continue
            for journey in metro_network.route_between_stops(
                origin, destination, f"{date} 08:30:00"
            ):
                for leg in journey["legs"]:
                    if leg["type"] != "transfer":
                        continue
                    walked += 1
                    assert leg["arrival_s"] - leg["departure_s"] <= MAX_WALKING_TIME
    assert walked, "the sampled journeys walked no transfer at all"


def test_zone_fares_price_the_measured_pairs_exactly(
    metro_network, helsinki_metro_data
):
    # Issue #246's measured pairs: today's fare fold reports 5.00 for
    # both, but BC-only journeys exist — the exact zone engine must
    # find them at 3.30.
    from cafein import fares as fares_module
    from cafein.frontier import fare_frontier

    date = _service_date(helsinki_metro_data.gtfs)
    structure = fares_module.zone_fare_structure(
        str(helsinki_metro_data.gtfs), rules="zones"
    )
    frame = fare_frontier(
        metro_network,
        ["9214203", "4340212"],
        ["1000109"],
        f"{date} 08:30:00",
        30,
        structure,
        cutoffs=[3.30, 4.50, 5.00],
        max_travel_time=180,
        # The event discipline, per the plan's acceptance; the derived
        # mid-span Tuesday is the plan's pinned witness date for the
        # pinned sampledata release.
        departure_time_step=None,
    )
    # The pinned trade-off curve: 3.30 buys the trip, 4.50 buys it
    # faster, and no cutoff ever needs the fold's 5.00 — it is
    # dominated everywhere.
    for origin in ("9214203", "4340212"):
        rows = frame[frame["from_id"] == origin]
        pinned = {3.30: 3.30, 4.50: 4.50, 5.00: 4.50}
        assert {
            row["cutoff"]: round(row["fare"], 2) for _, row in rows.iterrows()
        } == pinned, origin
    # And the public matrix rides the same exact engine: both cells
    # price at the frontier's 3.30 — the fare-blind fold this call
    # refined away reported 5.00 on the identical window.
    from cafein import TravelCostMatrix

    matrix = TravelCostMatrix(
        metro_network,
        ["9214203", "4340212"],
        ["1000109"],
        f"{date} 08:30:00",
        optimize="fare",
        departure_time_window=30,
        max_travel_time=180,
        fares=structure,
        geometries=True,
        output_time_units="seconds",
    )
    assert sorted(round(fare, 2) for fare in matrix["fare"]) == [3.30, 3.30]
    # The fold reported 5.00 here, so both cells were beaten by the
    # exact engine and every cost column below comes from the
    # reconstructed winning chain, not the fold's journey. Values are
    # pinned against the pinned sampledata release.
    pinned = {
        "9214203": (3521, 2, 23259.0, 1031.0, 2043.48),
        "4340212": (4465, 2, 28655.0, 691.3, 2277.21),
    }
    for _, cell in matrix.iterrows():
        seconds, transfers, transit, walk, grams = pinned[cell["from_id"]]
        assert cell["travel_time"] == seconds
        assert cell["transfers"] == transfers
        assert cell["transit_distance_m"] == pytest.approx(transit, abs=1.0)
        assert cell["walk_distance_m"] == pytest.approx(walk, abs=1.0)
        assert cell["emissions"] == pytest.approx(grams, abs=0.5)
        assert cell["geometry"] is not None
    # The default output reports the same cells in whole minutes
    # rounded to the nearest: 3521 s is 59, 4465 s is 74.
    rounded = TravelCostMatrix(
        metro_network,
        ["9214203", "4340212"],
        ["1000109"],
        f"{date} 08:30:00",
        optimize="fare",
        departure_time_window=30,
        max_travel_time=180,
        fares=structure,
    )
    by_origin = {cell["from_id"]: cell for _, cell in rounded.iterrows()}
    assert by_origin["9214203"]["travel_time"] == 59
    assert by_origin["4340212"]["travel_time"] == 74


def test_accessibility_products_ride_the_sampledata_pois(
    metro_network, helsinki_metro_data
):
    # The design's acceptance pair: libraries reachable within 30 PT
    # minutes, and the nearest swimming hall, from region-spread
    # origins over the sampledata POI layers.
    import geopandas

    from cafein import Accessibility, Catchment, NearestDestinations
    from cafein.sampledata.helsinki import pois

    date = _service_date(helsinki_metro_data.gtfs)
    departure = f"{date} 08:30:00"
    libraries = geopandas.read_file(pois.library).rename(columns={"osm_id": "id"})
    halls = geopandas.read_file(pois.swimming_hall).rename(columns={"osm_id": "id"})
    origins = geopandas.GeoDataFrame(
        {"id": ["kamppi", "espoo", "tikkurila"]},
        geometry=geopandas.points_from_xy(
            [24.9384, 24.6559, 25.0378], [60.1699, 60.2055, 60.2934]
        ),
        crs="EPSG:4326",
    )
    reachable = Accessibility(
        metro_network, origins, libraries, departure, budgets=(15.0, 30.0)
    )
    counts = reachable.pivot(index="from_id", columns="budget", values="accessibility")
    # Central Helsinki reaches libraries inside 15 minutes; every
    # origin's 30-minute count dominates its 15-minute count.
    assert counts.loc["kamppi", 15.0] > 0
    assert (counts[30.0] >= counts[15.0]).all()
    assert counts[30.0].sum() > counts[15.0].sum()
    nearest = NearestDestinations(
        metro_network,
        origins,
        halls,
        departure,
        k=1,
        max_cost=60,
        output_time_units="seconds",
    )
    by_origin = {row.from_id: row.cost for row in nearest.itertuples()}
    # Every origin has a nearest hall within the hour, door to door —
    # the horizon filters on exact costs, and the seconds mode proves
    # it without rounding slack.
    assert set(by_origin) == {"kamppi", "espoo", "tikkurila"}
    assert all(cost <= 3600 for cost in by_origin.values())
    catchment = Catchment(metro_network, origins.iloc[:1], departure, budgets=(15.0,))
    assert len(catchment) == 1
    assert catchment.geometry.iloc[0].area > 0


def test_equity_indices_over_a_metro_accessibility_run(
    metro_network, helsinki_metro_data
):
    # The inequality set on a real metro-scale accessibility frame
    # joined to synthetic sociodemographics: every index finite, the
    # tidy shapes right.
    import geopandas
    import numpy as np
    import pandas as pd

    from cafein import Accessibility, equity
    from cafein.sampledata.helsinki import pois

    date = _service_date(helsinki_metro_data.gtfs)
    libraries = geopandas.read_file(pois.library).rename(columns={"osm_id": "id"})
    origins = geopandas.GeoDataFrame(
        {"id": ["kamppi", "espoo", "tikkurila", "malmi"]},
        geometry=geopandas.points_from_xy(
            [24.9384, 24.6559, 25.0378, 25.0110],
            [60.1699, 60.2055, 60.2934, 60.2500],
        ),
        crs="EPSG:4326",
    )
    reachable = Accessibility(
        metro_network, origins, libraries, f"{date} 08:30:00", budgets=(30.0,)
    )
    rng = np.random.default_rng(11)
    people = pd.DataFrame(
        {
            "id": origins["id"],
            "pop": rng.integers(1000, 9000, len(origins)).astype(float),
            "income": rng.integers(20000, 60000, len(origins)).astype(float),
        }
    )
    people["zone"] = ["west", "west", "east", "east"]
    kwargs = {"sociodemographic_data": people, "population": "pop"}
    results = {
        "gini_index": reachable.gini_index(**kwargs),
        "share_ratio": reachable.share_ratio(top=0.2, bottom=0.2, **kwargs),
        "palma_ratio": reachable.palma_ratio(income="income", **kwargs),
        "generalized_entropy": reachable.generalized_entropy(alpha=2, **kwargs),
        "theil_t": reachable.theil_t(**kwargs),
        "mld": reachable.mld(**kwargs),
        "atkinson": reachable.atkinson(epsilon=0.5, **kwargs),
        "kolm": reachable.kolm(**kwargs),
        "hoover": reachable.hoover(**kwargs),
    }
    for name, result in results.items():
        if isinstance(result, pd.DataFrame):
            assert {"opportunity", "budget"} <= set(result.columns), name
            values = result.iloc[:, -1].to_numpy()
        else:
            values = [result]
        assert np.isfinite(values).all(), name
    people["transport_cost"] = rng.uniform(50, 400, len(people))
    identifiers = {"opportunity", "budget"}
    burden = reachable.cost_burden(cost="transport_cost", income="income", **kwargs)
    assert identifiers <= set(burden.columns) and len(burden) == 1
    assert np.isfinite(burden[["cost_burden", "mean_burden"]].to_numpy()).all()
    residual = reachable.residual_income(
        cost="transport_cost", income="income", **kwargs
    )
    assert len(residual) == len(reachable)
    assert np.isfinite(residual["residual_income"].to_numpy()).all()
    hardship = reachable.lihc(
        cost="transport_cost", income="income", poverty_line=10000.0, **kwargs
    )
    assert identifiers <= set(hardship.columns) and len(hardship) == 1
    assert np.isfinite(
        hardship[["lihc", "high_costs", "low_residual"]].to_numpy()
    ).all()
    multidim = reachable.alkire_foster(
        dimensions={
            "accessibility": float(np.median(reachable["accessibility"])),
            "transport_cost": (">", 200.0),
        },
        k=1,
        **kwargs,
    )
    assert {"m0", "headcount", "intensity"} <= set(multidim.columns)
    assert np.isfinite(multidim[["m0", "headcount", "intensity"]].to_numpy()).all()
    poverty = reachable.fgt_poverty(poverty_line="60% of median", **kwargs)
    assert isinstance(poverty, pd.DataFrame)
    assert identifiers <= set(poverty.columns) and len(poverty) == 1
    assert np.isfinite(poverty["fgt_poverty"].to_numpy()).all()
    concentration = reachable.concentration_index(income="income", **kwargs)
    progressivity = reachable.suits(income="income", **kwargs)
    for extra in (concentration, progressivity):
        column = extra.columns[-1]
        assert np.isfinite(extra[column].to_numpy()).all()
    parts = reachable.theil_t(groups="zone", **kwargs)
    last = parts[["total", "between", "within"]].iloc[-1]
    assert last["total"] == pytest.approx(last["between"] + last["within"])
    curve = reachable.lorenz_curve(**kwargs)
    assert {"population_share", "value_share"} <= set(curve.columns)
    assert curve["value_share"].iloc[-1] == 1.0


def test_park_and_ride_serves_the_metro_region(helsinki_metro_data):
    geopandas = pytest.importorskip("geopandas")
    from shapely.geometry import Point

    from cafein import Accessibility
    from cafein.policy import CarParkPolicy
    from cafein.streets import park_and_ride_facilities

    # A real facility frame from the region's own park_ride tagging.
    facilities = park_and_ride_facilities(str(helsinki_metro_data.osm_pbf))
    assert len(facilities) >= 1
    policy = CarParkPolicy(facilities=facilities)
    network = TransportNetwork.from_gtfs(
        [str(helsinki_metro_data.gtfs)],
        osm_pbf=str(helsinki_metro_data.osm_pbf),
        street_modes=("walk", "car"),
        country="FI",
    )
    departure = f"{_service_date(str(helsinki_metro_data.gtfs))} 08:30:00"
    origins = geopandas.GeoDataFrame(
        {"id": ["espoo", "vantaa"]},
        geometry=[Point(24.6559, 60.2055), Point(25.0378, 60.2934)],
        crs="EPSG:4326",
    )
    destinations = geopandas.GeoDataFrame(
        {"id": ["centre"], "jobs": [1000.0]},
        geometry=[Point(24.9414, 60.1719)],
        crs="EPSG:4326",
    )
    frame = TravelTimeMatrix(
        network,
        origins,
        destinations,
        departure,
        street_policy=policy,
        output_time_units="seconds",
    )
    # Every suburb reaches the centre through some facility, finitely.
    assert set(frame["from_id"]) == {"espoo", "vantaa"}
    assert (frame["travel_time"] > 0).all()
    scores = Accessibility(
        network,
        origins,
        destinations,
        departure,
        opportunities="jobs",
        budgets=(90.0,),
        street_policy=policy,
    )
    values = scores["accessibility"].to_numpy()
    assert len(values) == 2 and (values >= 0).all()
    assert values.max() > 0.0
    # The same pair under the arrival axis.
    deadline = f"{_service_date(str(helsinki_metro_data.gtfs))} 09:30:00"
    arrived = Accessibility(
        network,
        origins,
        destinations,
        arrival=deadline,
        opportunities="jobs",
        budgets=(90.0,),
        street_policy=policy,
    )
    arrived_values = arrived["accessibility"].to_numpy()
    assert len(arrived_values) == 2 and (arrived_values >= 0).all()
    assert arrived_values.max() > 0.0
