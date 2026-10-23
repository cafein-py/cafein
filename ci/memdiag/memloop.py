"""Repeat cafein computations on one network and print the Linux memory after
each call: identical arguments, so memory that keeps rising is retained per
call, and memory that rises once is a one-time cache."""

import gc
import os
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from memprof_plugin import _FIELDS, _read_kb_fields  # noqa: E402

import geopandas as gpd  # noqa: E402
from shapely.geometry import Point  # noqa: E402

from cafein import (  # noqa: E402
    DetailedItineraries,
    StreetLegPolicy,
    TransportNetwork,
    TravelCostMatrix,
    TravelTimeMatrix,
    VehiclePolicy,
)


def memory_mb():
    """Anonymous and file-backed RSS in MB, after a gc."""
    gc.collect()
    status = _read_kb_fields("/proc/self/status", _FIELDS)
    return status["RssAnon"] / 1024, status["RssFile"] / 1024


def report(label):
    """Print the memory after ``label``; returns the anonymous RSS in MB."""
    anon, file = memory_mb()
    print(f"MEMLOOP {label}: anon {anon:.0f} MB, file {file:.0f} MB", flush=True)
    return anon


DATA = "tests/data"
rng = np.random.default_rng(7)
lats = rng.uniform(60.160, 60.185, 20)
lons = rng.uniform(24.920, 24.960, 20)
points = gpd.GeoDataFrame(
    {"id": [f"p{i}" for i in range(20)]},
    geometry=[Point(x, y) for x, y in zip(lons, lats)],
    crs="EPSG:4326",
)
DEPARTURE = "2022-02-22 08:30:00"
bike = StreetLegPolicy(
    access={"walk": 1200, "bicycle": 900},
    egress={"walk": 1200},
    vehicles={"bicycle": VehiclePolicy(source="own", side="origin", facilities="any_stop")},
)

report("start")
t = time.perf_counter()
network = TransportNetwork.from_gtfs(
    [f"{DATA}/helsinki_gtfs.zip"],
    osm_pbf=f"{DATA}/kantakaupunki.osm.pbf",
    street_modes=("walk", "bicycle", "e_scooter", "wheelchair"),
)
report(f"after build ({time.perf_counter() - t:.0f} s)")

scenarios = {
    "ttm": lambda: TravelTimeMatrix(network, points, points, DEPARTURE),
    "ttm_policy": lambda: TravelTimeMatrix(
        network, points, points, DEPARTURE, street_policy=bike
    ),
    "cost": lambda: TravelCostMatrix(network, points, points, DEPARTURE),
    "itineraries": lambda: DetailedItineraries(
        network, points.head(5), points.tail(5), DEPARTURE
    ),
}
for name, call in scenarios.items():
    values = []
    for i in range(20):
        result = call()
        del result
        values.append(report(f"{name} call {i + 1}"))
    print(
        f"MEMLOOP SUMMARY {name}: call 1 {values[0]:.0f} MB, call 20 {values[-1]:.0f} MB, "
        f"calls 11-20 {values[-1] - values[9]:+.0f} MB"
    )

folder = tempfile.mkdtemp()
values = []
for i in range(10):
    path = os.path.join(folder, f"n{i}.cafein")
    network.save(path)
    loaded = TransportNetwork.load(path)
    TravelTimeMatrix(loaded, points.head(3), points.head(3), DEPARTURE)
    del loaded
    os.remove(path)
    values.append(report(f"save+load+query cycle {i + 1}"))
print(
    f"MEMLOOP SUMMARY saveload: cycle 1 {values[0]:.0f} MB, cycle 10 {values[-1]:.0f} MB"
)
