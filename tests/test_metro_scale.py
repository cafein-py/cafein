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
        times = metro_network.travel_times_from_stop(origin, date, "08:30:00")
        if any(stop != origin and seconds > 0 for stop, seconds in times.items()):
            break
    else:
        pytest.fail(f"no transit service reachable from 25 sampled stops on {date}")
    # One small stop-to-stop matrix over the sampled origins.
    matrix = metro_network.travel_time_matrix(sampled[:5], date, "08:30:00")
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
                origin, destination, date, "08:30:00"
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
        date,
        "08:30:00",
        1800,
        structure,
        cutoffs=[3.30, 4.50, 5.00],
        max_duration=3 * 3600,
        # The event discipline, per the plan's acceptance; the derived
        # mid-span Tuesday is the plan's pinned witness date for the
        # pinned sampledata release.
        departure_step=None,
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
    # And the fare-blind fold still reports 5.00 on the identical
    # window — the 3.30 is the engine's win, not a departure the fold
    # could also see.
    from cafein import TravelCostMatrix

    fold = TravelCostMatrix(
        metro_network,
        ["9214203", "4340212"],
        ["1000109"],
        date,
        "08:30:00",
        optimize="fare",
        window=1800,
        fares=structure,
    )
    assert sorted(round(fare, 2) for fare in fold["fare"]) == [5.00, 5.00]
