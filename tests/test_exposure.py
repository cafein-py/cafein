"""Exposure layer ingestion onto the street network."""

import math

import numpy as np
import pytest

from cafein import exposure
from cafein.exposure import (
    Exposure,
    midpoint_samples,
    polygon_edge_products,
    sample_products,
)

shapely = pytest.importorskip("shapely")
from shapely.geometry import LineString, Point, Polygon, box  # noqa: E402

# --- helpers: exact polygon products -----------------------------------------


def test_polygon_products_are_exact_and_zero_filled():
    line = LineString([(0, 0), (100, 0)])
    zone = box(0, -10, 50, 10)  # covers the first half
    mean, coverage, maximum, shares = polygon_edge_products(
        line, [(55.0, zone)], thresholds=(55, 60)
    )
    assert mean == pytest.approx(27.5)
    assert coverage == pytest.approx(0.5)
    assert maximum == 55.0
    assert shares[55] == pytest.approx(0.5)  # at the bound counts
    assert shares[60] == 0.0


def test_polygon_overlaps_resolve_by_maximum():
    line = LineString([(0, 0), (100, 0)])
    low = box(0, -10, 100, 10)  # value 50 everywhere
    high = box(25, -10, 75, 10)  # value 70 over the middle half
    mean, coverage, maximum, shares = polygon_edge_products(
        line, [(50.0, low), (70.0, high)], thresholds=(60,)
    )
    # 25 m at 50, 50 m at 70, 25 m at 50
    assert mean == pytest.approx((50 * 50 + 70 * 50) / 100)
    assert coverage == pytest.approx(1.0)
    assert maximum == 70.0
    assert shares[60] == pytest.approx(0.5)


def test_polygon_split_invariance_is_exact():
    whole = LineString([(0, 0), (100, 0)])
    first = LineString([(0, 0), (60, 0)])
    second = LineString([(60, 0), (100, 0)])
    zone = box(30, -10, 80, 10)
    pieces = [(55.0, zone)]
    whole_mean, whole_cov, _, _ = polygon_edge_products(whole, pieces)
    m1, c1, _, _ = polygon_edge_products(first, pieces)
    m2, c2, _, _ = polygon_edge_products(second, pieces)
    # integrated dose (mean x length) and covered length both add up
    assert m1 * 60 + m2 * 40 == pytest.approx(whole_mean * 100)
    assert c1 * 60 + c2 * 40 == pytest.approx(whole_cov * 100)


def test_an_invalid_bowtie_polygon_is_repaired(street_network=None):
    # self-intersecting "bowtie": make_valid splits it into two
    # triangles; the overlay then works and covers both crossings.
    # the lobes pinch at (5, 5); probe through the upper lobe
    line = LineString([(0, 7), (20, 7)])
    bowtie = Polygon([(0, 0), (10, 10), (0, 10), (10, 0)])
    from shapely import make_valid

    mean, coverage, maximum, _ = polygon_edge_products(
        line, [(60.0, make_valid(bowtie))]
    )
    assert maximum == 60.0
    assert 0 < coverage < 1
    assert mean == pytest.approx(60.0 * coverage)


def test_a_boundary_touch_contributes_nothing():
    line = LineString([(0, 0), (100, 0)])
    touching = Polygon([(50, 0), (60, 10), (40, 10)])  # touches at one point
    mean, coverage, maximum, _ = polygon_edge_products(line, [(99.0, touching)])
    assert mean == 0.0
    assert coverage == 0.0
    assert math.isnan(maximum)


# --- helpers: sampled products ----------------------------------------------


def test_midpoint_samples_partition_the_length():
    line = LineString([(0, 0), (100, 0)])
    points = midpoint_samples(line, step=25.0)
    assert len(points) == 4
    assert [p.x for p in points] == pytest.approx([12.5, 37.5, 62.5, 87.5])


def test_sample_products_zero_fill_and_threshold():
    values = [10.0, np.nan, 30.0, np.nan]
    mean, coverage, maximum, shares = sample_products(values, thresholds=(30,))
    assert mean == pytest.approx(10.0)  # (10 + 0 + 30 + 0) / 4
    assert coverage == pytest.approx(0.5)
    assert maximum == 30.0
    assert shares[30] == pytest.approx(0.25)


def test_fully_uncovered_samples_have_nan_max():
    mean, coverage, maximum, shares = sample_products([np.nan, np.nan], thresholds=(5,))
    assert mean == 0.0 and coverage == 0.0 and shares[5] == 0.0
    assert math.isnan(maximum)


def test_sampled_split_invariance_holds_to_tolerance():
    # A synthetic gradient sampled on the whole line vs its halves:
    # integrated dose agrees within the sampling resolution.
    def field(point):
        return point.x**2  # nonlinear, so midpoint sampling is inexact

    whole = LineString([(0, 0), (100, 0)])
    first = LineString([(0, 0), (60, 0)])
    second = LineString([(60, 0), (100, 0)])

    def dose(line):
        values = [field(p) for p in midpoint_samples(line)]
        mean, _, _, _ = sample_products(values)
        return mean * line.length

    assert dose(first) + dose(second) == pytest.approx(dose(whole), rel=0.05)


def test_equidistant_lines_reduce_to_the_maximum():
    geopandas = pytest.importorskip("geopandas")
    from cafein.exposure import _VALUE, _ingest_lines

    # one sample point exactly between two lines of differing value
    samples = ([Point(0, 0)], np.asarray([0]))
    lines = geopandas.GeoDataFrame(
        {_VALUE: [30.0, 70.0]},
        geometry=[
            LineString([(-10, -50), (-10, 50)]),
            LineString([(10, -50), (10, 50)]),
        ],
        crs="EPSG:32635",
    )
    products = _ingest_lines(lines, samples, (50,), 1)
    assert products["mean"][0] == 70.0
    assert products["max"][0] == 70.0
    assert products["shares"][50][0] == 1.0


def _edge_series(lines):
    geopandas = pytest.importorskip("geopandas")
    return geopandas.GeoSeries(lines, crs="EPSG:32635")


def _integrated(products, edges):
    lengths = np.asarray([line.length for line in edges])
    dose = float((products["mean"] * lengths).sum())
    covered = float((products["coverage"] * lengths).sum())
    return dose, covered


def test_polygon_ingestion_is_split_invariant():
    geopandas = pytest.importorskip("geopandas")
    from cafein.exposure import _VALUE, _ingest_polygons

    zone = box(30, -10, 80, 10)  # coverage boundary inside the edge
    frame = geopandas.GeoDataFrame({_VALUE: [55.0]}, geometry=[zone], crs="EPSG:32635")
    whole = [LineString([(0, 0), (100, 0)])]
    split = [LineString([(0, 0), (60, 0)]), LineString([(60, 0), (100, 0)])]
    a = _ingest_polygons(frame, _edge_series(whole), (55,))
    b = _ingest_polygons(frame, _edge_series(split), (55,))
    assert _integrated(a, whole) == pytest.approx(_integrated(b, split))


def test_line_ingestion_is_split_invariant_to_tolerance():
    geopandas = pytest.importorskip("geopandas")
    from cafein.exposure import _VALUE, _edge_samples, _ingest_lines

    # the source line runs beside only PART of the edge — a coverage
    # boundary — so the tolerance genuinely bites
    source = geopandas.GeoDataFrame(
        {_VALUE: [40.0]},
        geometry=[LineString([(20, 5), (55, 5)])],
        crs="EPSG:32635",
    )
    whole = [LineString([(0, 0), (100, 0)])]
    split = [LineString([(0, 0), (60, 0)]), LineString([(60, 0), (100, 0)])]
    a = _ingest_lines(source, _edge_samples(_edge_series(whole)), (), 1)
    b = _ingest_lines(source, _edge_samples(_edge_series(split)), (), 2)
    dose_a, covered_a = _integrated(a, whole)
    dose_b, covered_b = _integrated(b, split)
    assert dose_b == pytest.approx(dose_a, rel=0.15)
    assert covered_b == pytest.approx(covered_a, rel=0.15)
    assert 0 < covered_a < 100


def test_raster_ingestion_is_split_invariant_to_tolerance(tmp_path):
    rasterio = pytest.importorskip("rasterio")
    pytest.importorskip("rioxarray")
    from rasterio.transform import from_bounds

    from cafein.exposure import _edge_samples, _ingest_raster

    # values only over x in [0, 50]: a coverage boundary mid-edge
    path = tmp_path / "half.tif"
    data = np.full((1, 8, 8), np.nan, dtype="float32")
    data[:, :, :4] = 7.0
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=8,
        width=8,
        count=1,
        dtype="float32",
        crs="EPSG:32635",
        transform=from_bounds(0, -50, 100, 50, 8, 8),
    ) as sink:
        sink.write(data)
        sink.descriptions = ("X [u]",)
    whole = [LineString([(0, 0), (100, 0)])]
    split = [LineString([(0, 0), (60, 0)]), LineString([(60, 0), (100, 0)])]
    a = _ingest_raster(
        "x",
        str(path),
        "X",
        "EPSG:32635",
        _edge_samples(_edge_series(whole)),
        (),
        1,
    )
    b = _ingest_raster(
        "x",
        str(path),
        "X",
        "EPSG:32635",
        _edge_samples(_edge_series(split)),
        (),
        2,
    )
    dose_a, covered_a = _integrated(a, whole)
    dose_b, covered_b = _integrated(b, split)
    # coarse 25 m sampling against a sharp value boundary: the
    # estimate's honest error band
    assert dose_b == pytest.approx(dose_a, rel=0.25)
    assert covered_b == pytest.approx(covered_a, rel=0.25)
    assert 0 < covered_a < 100


def test_axis_aligned_edges_keep_partial_polygon_coverage():
    # horizontal and vertical edges have degenerate envelopes; the
    # padded clip must not drop their genuine coverage.
    geopandas = pytest.importorskip("geopandas")
    from cafein.exposure import _VALUE, _ingest_polygons

    zone = box(0, -10, 50, 10)
    frame = geopandas.GeoDataFrame({_VALUE: [55.0]}, geometry=[zone], crs="EPSG:32635")
    horizontal = _edge_series([LineString([(0, 0), (100, 0)])])
    products = _ingest_polygons(frame, horizontal, ())
    assert products["coverage"][0] == pytest.approx(0.5)
    vertical_zone = box(-10, 0, 10, 50)
    vframe = geopandas.GeoDataFrame(
        {_VALUE: [55.0]}, geometry=[vertical_zone], crs="EPSG:32635"
    )
    vertical = _edge_series([LineString([(0, 0), (0, 100)])])
    products = _ingest_polygons(vframe, vertical, ())
    assert products["coverage"][0] == pytest.approx(0.5)


def test_touch_only_polygons_produce_zero_coverage():
    geopandas = pytest.importorskip("geopandas")
    from cafein.exposure import _VALUE, _ingest_polygons

    touching = Polygon([(50, 0), (60, 10), (40, 10)])
    frame = geopandas.GeoDataFrame(
        {_VALUE: [99.0]}, geometry=[touching], crs="EPSG:32635"
    )
    edges = _edge_series([LineString([(0, 0), (100, 0)])])
    products = _ingest_polygons(frame, edges, ())
    assert products["coverage"][0] == 0.0
    assert math.isnan(products["max"][0])


def test_lineal_make_valid_remnants_carry_no_area_exposure():
    from cafein.exposure import _polygonal
    from shapely import make_valid
    from shapely.geometry import GeometryCollection

    sliver = Polygon([(0, 0), (100, 0), (0, 0)])  # zero-area
    repaired = _polygonal(make_valid(sliver))
    assert repaired.is_empty or repaired.area > 0
    mixed = GeometryCollection([box(0, -10, 30, 10), LineString([(30, 0), (100, 0)])])
    kept = _polygonal(mixed)
    assert kept.area == pytest.approx(box(0, -10, 30, 10).area)
    line = LineString([(0, 0), (100, 0)])
    mean, coverage, _, _ = polygon_edge_products(line, [(60.0, kept)])
    assert coverage == pytest.approx(0.3)
    assert mean == pytest.approx(60.0 * 0.3)


def test_rasterized_default_matches_exact_on_synthetic_zones(street_network):
    geopandas = pytest.importorskip("geopandas")
    pytest.importorskip("rasterio")
    west, south, east, north = _extent(street_network)
    midx = (west + east) / 2
    half = geopandas.GeoDataFrame(
        {"level": [61.0]},
        geometry=[box(west - 0.01, south - 0.01, midx, north + 0.01)],
        crs="EPSG:4326",
    )
    edges = street_network.streets_gdf
    try:
        Exposure(street_network, a=(half, "level"))  # default: rasterized
        burned = edges["a"].copy()
        _drop_layer_columns(edges, "a", ())
        Exposure(street_network, a=(half, "level"), rasterize=None)
        exact = edges["a"].copy()
        agree = np.isclose(burned.values, exact.values, atol=1.0)
        assert agree.mean() > 0.97  # boundary cells differ, the rest match
    finally:
        _drop_layer_columns(edges, "a", ())


def test_strip_tiled_burning_matches_untiled(street_network, monkeypatch):
    geopandas = pytest.importorskip("geopandas")
    pytest.importorskip("rasterio")
    from cafein import exposure as exposure_module

    west, south, east, north = _extent(street_network)
    zones = geopandas.GeoDataFrame(
        {"level": [61.0]},
        geometry=[box(west - 0.01, south - 0.01, east + 0.01, north + 0.01)],
        crs="EPSG:4326",
    )
    edges = street_network.streets_gdf
    try:
        Exposure(street_network, a=(zones, "level"), rasterize=8.0)
        untiled = edges["a"].copy()
        _drop_layer_columns(edges, "a", ())
        monkeypatch.setattr(exposure_module, "BURN_STRIP_CELLS", 10_000)
        Exposure(street_network, a=(zones, "level"), rasterize=8.0)
        tiled = edges["a"].copy()
        assert tiled.values == pytest.approx(untiled.values)
    finally:
        _drop_layer_columns(edges, "a", ())


def test_rasterize_validation(street_network):
    geopandas = pytest.importorskip("geopandas")
    west, south, east, north = _extent(street_network)
    zones = geopandas.GeoDataFrame(
        {"level": [61.0]},
        geometry=[box(west, south, east, north)],
        crs="EPSG:4326",
    )
    for bad in (0, -1.0, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="positive cell size"):
            Exposure(street_network, a=(zones, "level"), rasterize=bad)


def test_self_crossing_edges_claim_by_sampling():
    from cafein.exposure import _claimed_lengths

    # a figure-eight edge: both traversals through the crossing must
    # keep their own coverage (interval projection would collapse them)
    eight = LineString(
        [
            (0, 0),
            (10, 0),
            (10, 10),
            (0, 10),
            (0, 0),
            (-10, 0),
            (-10, -10),
            (0, -10),
            (0, 0),
        ]
    )
    full = LineString(eight.coords)
    lengths = _claimed_lengths([(60.0, full)], eight)
    total = sum(l for _, l in lengths)
    assert total == pytest.approx(eight.length, rel=0.02)


def test_an_absurdly_fine_rasterize_is_refused(street_network, monkeypatch):
    pytest.importorskip("rasterio")
    geopandas = pytest.importorskip("geopandas")
    from cafein import exposure as exposure_module

    west, south, east, north = _extent(street_network)
    zones = geopandas.GeoDataFrame(
        {"level": [61.0]},
        geometry=[box(west - 0.01, south - 0.01, east + 0.01, north + 0.01)],
        crs="EPSG:4326",
    )
    monkeypatch.setattr(exposure_module, "BURN_STRIP_CELLS", 1000)
    with pytest.raises(ValueError, match="strip budget"):
        Exposure(street_network, a=(zones, "level"), rasterize=0.5)


def test_closed_edges_claim_by_midpoint_parameterization():
    from cafein.exposure import _claimed_lengths

    # a closed square edge fully covered by two overlapping pieces:
    # endpoint projection would collapse to zero-length intervals
    ring = LineString([(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)])
    lengths = _claimed_lengths(
        [(70.0, ring), (75.0, LineString([(0, 0), (10, 0)]))], ring
    )
    total = sum(l for _, l in lengths)
    assert total == pytest.approx(ring.length)
    assert lengths[0][0] == 75.0  # worst wins on the shared stretch


def test_nearly_coincident_nested_zones_stay_bounded():
    # the real-data condition: the inner zone's linework is offset by
    # coordinate noise, which defeats GEOS union/difference — interval
    # claiming must still keep coverage at 1
    geopandas = pytest.importorskip("geopandas")
    from cafein.exposure import _VALUE, _ingest_polygons

    frame = geopandas.GeoDataFrame(
        {_VALUE: [70.0, 75.0]},
        geometry=[
            box(0, -10, 100, 10),
            box(1e-7, -10 + 1e-9, 40 + 1e-7, 10 - 1e-9),
        ],
        crs="EPSG:32635",
    )
    edges = _edge_series([LineString([(0, 0), (100, 0)])])
    products = _ingest_polygons(frame, edges, ())
    assert products["coverage"][0] <= 1 + 1e-9
    assert products["coverage"][0] == pytest.approx(1.0)
    assert products["mean"][0] == pytest.approx(75 * 0.4 + 70 * 0.6, rel=1e-4)


def test_overlapping_zones_claim_in_interval_space():
    # The real noise layer nests zones (a 75 dB band inside a 70 dB
    # zone covering the whole edge); nearly-coincident linework defeats
    # GEOS union/difference, so claiming runs on 1-D intervals. The
    # regression: coverage stayed <= 1 and worst-wins holds.
    geopandas = pytest.importorskip("geopandas")
    from cafein.exposure import _VALUE, _ingest_polygons

    frame = geopandas.GeoDataFrame(
        {_VALUE: [70.0, 75.0]},
        geometry=[box(0, -10, 100, 10), box(0, -10, 40, 10)],
        crs="EPSG:32635",
    )
    edges = _edge_series([LineString([(0, 0), (100, 0)])])
    products = _ingest_polygons(frame, edges, (75,))
    assert products["coverage"][0] == pytest.approx(1.0)
    assert products["mean"][0] == pytest.approx(75 * 0.4 + 70 * 0.6)
    assert products["max"][0] == 75.0
    assert products["shares"][75][0] == pytest.approx(0.4)


def test_an_invalid_polygon_straight_into_the_overlay_is_survivable():
    # _ingest_vector repairs with make_valid, but the overlay itself
    # must also survive dirty geometry (clip output can be dirty too).
    geopandas = pytest.importorskip("geopandas")
    from cafein.exposure import _VALUE, _ingest_polygons

    bowtie = Polygon([(0, 0), (100, 70), (0, 70), (100, 0)])
    frame = geopandas.GeoDataFrame(
        {_VALUE: [60.0]}, geometry=[bowtie], crs="EPSG:32635"
    )
    edges = _edge_series([LineString([(0, 20), (100, 20)])])
    products = _ingest_polygons(frame, edges, ())
    assert np.isfinite(products["mean"][0])
    assert 0 <= products["coverage"][0] <= 1


def test_an_easting_northing_grid_resolves_and_transposes(tmp_path):
    pytest.importorskip("rioxarray")
    numpy = pytest.importorskip("numpy")
    xarray = pytest.importorskip("xarray")
    from cafein.exposure import _open_band

    # spatial dims named easting/northing AND ordered (x, y): the rio
    # accessor identifies them and the array lands as (y, x)
    grid = xarray.Dataset(
        {
            "X": (
                ("easting", "northing"),
                numpy.arange(8 * 4, dtype="float32").reshape(8, 4),
            )
        },
        coords={
            "easting": numpy.linspace(0.5, 7.5, 8),
            "northing": numpy.linspace(0.5, 3.5, 4),
        },
    )
    grid = grid.rio.set_spatial_dims(x_dim="easting", y_dim="northing")
    grid = grid.rio.write_crs("EPSG:32635")
    band, transform, crs, nodata = _open_band("x", grid, "X")
    assert band.shape == (4, 8)  # (y, x)
    assert crs is not None


def test_a_scaled_masked_raster_decodes_physical_values(street_network, tmp_path):
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_bounds

    west, south, east, north = _extent(street_network)
    path = tmp_path / "scaled.tif"
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=8,
        width=8,
        count=1,
        dtype="int16",
        crs="EPSG:4326",
        nodata=-9999,
        transform=from_bounds(
            west - 0.01, south - 0.01, east + 0.01, north + 0.01, 8, 8
        ),
    ) as sink:
        sink.write(np.full((1, 8, 8), 35, dtype="int16"))
        sink.scales = (0.1,)
        sink.descriptions = ("X [u]",)
    edges = street_network.streets_gdf
    try:
        Exposure(street_network, x=(str(path), "X"))
        # 35 raw x 0.1 scale = 3.5 physical
        assert edges["x"].values == pytest.approx(3.5)
    finally:
        _drop_layer_columns(edges, "x", ())


def test_a_named_dataarray_without_descriptions_is_accepted(tmp_path):
    pytest.importorskip("rioxarray")
    numpy = pytest.importorskip("numpy")
    xarray = pytest.importorskip("xarray")
    from cafein.exposure import _open_band

    array = xarray.DataArray(
        numpy.full((4, 8), 2.5, dtype="float32"),
        dims=("y", "x"),
        coords={"y": numpy.linspace(3.5, 0.5, 4), "x": numpy.linspace(0.5, 7.5, 8)},
        name="X",
    )
    array = array.rio.write_crs("EPSG:32635")
    band, transform, crs, nodata = _open_band("x", array, "X")
    assert band.shape == (4, 8)


def test_a_one_row_dataset_grid_is_accepted(tmp_path):
    pytest.importorskip("rioxarray")
    numpy = pytest.importorskip("numpy")
    xarray = pytest.importorskip("xarray")
    from cafein.exposure import _open_band

    grid = xarray.Dataset(
        {"X": (("band", "y", "x"), numpy.full((1, 1, 8), 2.0, dtype="float32"))},
        coords={"band": [1], "y": [0.5], "x": numpy.linspace(0.5, 7.5, 8)},
    )
    grid = grid.rio.write_crs("EPSG:32635")
    band, transform, crs, nodata = _open_band("x", grid, "X")
    assert band.shape == (1, 8)
    assert crs is not None


def test_near_identical_thresholds_render_distinct_columns():
    from cafein.exposure import _threshold_suffix

    a = _threshold_suffix(1.0000001)
    b = _threshold_suffix(1.0000002)
    assert a != b
    assert _threshold_suffix(55) == "55"
    assert _threshold_suffix(7.5) == "7_5"
    assert _threshold_suffix(-0.0) == "0"


# --- the class over the Helsinki network -------------------------------------


@pytest.fixture(scope="module")
def street_network(network_with_footpaths):
    return network_with_footpaths


def _extent(network):
    bounds = network.streets_gdf.total_bounds
    return bounds  # (west, south, east, north), EPSG:4326


def test_constant_polygon_layer_covers_every_edge(street_network):
    pytest.importorskip("rasterio")
    geopandas = pytest.importorskip("geopandas")
    west, south, east, north = _extent(street_network)
    zones = geopandas.GeoDataFrame(
        {"level": [61.0]},
        geometry=[box(west - 0.01, south - 0.01, east + 0.01, north + 0.01)],
        crs="EPSG:4326",
    )
    ex = Exposure(
        street_network, noise=(zones, "level"), thresholds={"noise": (55, 65)}
    )
    edges = street_network.streets_gdf
    try:
        assert ex.layers == ("noise",)
        assert ex.thresholds("noise") == (55.0, 65.0)
        assert edges["noise"].values == pytest.approx(61.0)
        assert edges["noise_coverage"].values == pytest.approx(1.0)
        assert edges["noise_max"].values == pytest.approx(61.0)
        assert edges["noise_share_above_55"].values == pytest.approx(1.0)
        assert edges["noise_share_above_65"].values == pytest.approx(0.0)
    finally:
        _drop_layer_columns(edges, "noise", (55, 65))


def test_differing_crs_vectors_reproject_to_identical_results(street_network):
    pytest.importorskip("rasterio")
    geopandas = pytest.importorskip("geopandas")
    west, south, east, north = _extent(street_network)
    midx = (west + east) / 2
    half = geopandas.GeoDataFrame(
        {"level": [50.0]},
        geometry=[box(west - 0.01, south - 0.01, midx, north + 0.01)],
        crs="EPSG:4326",
    )
    edges = street_network.streets_gdf
    try:
        Exposure(street_network, a=(half, "level"))
        geographic = edges["a"].copy()
        _drop_layer_columns(edges, "a", ())
        Exposure(street_network, a=(half.to_crs("EPSG:3067"), "level"))
        projected = edges["a"].copy()
        assert projected.values == pytest.approx(geographic.values, abs=1e-6)
        assert 0 < (projected.values > 0).sum() < len(projected)
    finally:
        _drop_layer_columns(edges, "a", ())


def test_line_layer_matches_only_nearby_edges(street_network):
    geopandas = pytest.importorskip("geopandas")
    edges = street_network.streets_gdf
    # take a real edge's geometry as the "GVI segment": its own edge
    # must match; edges far away must not.
    template = edges.geometry.iloc[0]
    lines = geopandas.GeoDataFrame(
        {"gvi": [40.0]}, geometry=[template], crs="EPSG:4326"
    )
    try:
        Exposure(street_network, greenery=(lines, "gvi"))
        assert edges["greenery"].iloc[0] == pytest.approx(40.0, abs=1e-6)
        assert edges["greenery_coverage"].iloc[0] == pytest.approx(1.0)
        covered = edges["greenery_coverage"].values > 0
        assert 0 < covered.sum() < len(edges)
    finally:
        _drop_layer_columns(edges, "greenery", ())


def test_raster_layer_samples_by_band_name(street_network, tmp_path):
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_bounds

    west, south, east, north = _extent(street_network)
    path = tmp_path / "layer.tif"
    height = width = 32
    data = np.stack(
        [
            np.full((height, width), 3.5, dtype="float32"),
            np.full((height, width), 9.0, dtype="float32"),
        ]
    )
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=2,
        dtype="float32",
        crs="EPSG:4326",
        transform=from_bounds(
            west - 0.01, south - 0.01, east + 0.01, north + 0.01, width, height
        ),
    ) as sink:
        sink.write(data)
        sink.descriptions = ("PM25Concentration [ug/m3]", "Other [x]")
    edges = street_network.streets_gdf
    try:
        Exposure(
            street_network,
            pollution=(str(path), "PM25Concentration"),
            thresholds={"pollution": 3},
        )
        assert edges["pollution"].values == pytest.approx(3.5)
        assert edges["pollution_coverage"].values == pytest.approx(1.0)
        assert edges["pollution_share_above_3"].values == pytest.approx(1.0)
    finally:
        _drop_layer_columns(edges, "pollution", (3,))


def test_refusals(street_network, tmp_path):
    pytest.importorskip("rasterio")
    geopandas = pytest.importorskip("geopandas")
    west, south, east, north = _extent(street_network)
    world = box(west - 0.01, south - 0.01, east + 0.01, north + 0.01)
    zones = geopandas.GeoDataFrame({"level": [61.0]}, geometry=[world], crs="EPSG:4326")
    with pytest.raises(ValueError, match="at least one layer"):
        Exposure(street_network)
    with pytest.raises(ValueError, match="not a lowercase identifier"):
        Exposure(street_network, **{"Noise": (zones, "level")})
    with pytest.raises(ValueError, match="already exists"):
        Exposure(street_network, highway=(zones, "level"))
    with pytest.raises(ValueError, match="both derive"):
        Exposure(street_network, x=(zones, "level"), x_coverage=(zones, "level"))
    with pytest.raises(ValueError, match="unknown layer"):
        Exposure(street_network, noise=(zones, "level"), thresholds={"pollution": 3})
    with pytest.raises(ValueError, match="must be finite"):
        Exposure(
            street_network, noise=(zones, "level"), thresholds={"noise": float("inf")}
        )
    with pytest.raises(ValueError, match="no column"):
        Exposure(street_network, noise=(zones, "wrong"))
    with pytest.raises(ValueError, match="declares no CRS"):
        Exposure(
            street_network,
            noise=(geopandas.GeoDataFrame({"level": [1.0]}, geometry=[world]), "level"),
        )
    with pytest.raises(ValueError, match="non-finite value"):
        bad = geopandas.GeoDataFrame(
            {"level": [float("nan")]}, geometry=[world], crs="EPSG:4326"
        )
        Exposure(street_network, noise=(bad, "level"))
    with pytest.raises(ValueError, match="unsupported geometry"):
        points = geopandas.GeoDataFrame(
            {"level": [1.0]}, geometry=[Point(west, south)], crs="EPSG:4326"
        )
        Exposure(street_network, noise=(points, "level"))
    with pytest.raises(ValueError, match="covers no street edge"):
        far = geopandas.GeoDataFrame(
            {"level": [61.0]}, geometry=[box(0, 0, 1, 1)], crs="EPSG:4326"
        )
        Exposure(street_network, noise=(far, "level"))
    with pytest.raises(ValueError, match=r"\(source, value\) pair"):
        Exposure(street_network, noise=zones)


def test_a_failing_layer_leaves_streets_gdf_untouched(street_network):
    pytest.importorskip("rasterio")
    geopandas = pytest.importorskip("geopandas")
    west, south, east, north = _extent(street_network)
    world = box(west - 0.01, south - 0.01, east + 0.01, north + 0.01)
    zones = geopandas.GeoDataFrame({"level": [61.0]}, geometry=[world], crs="EPSG:4326")
    edges = street_network.streets_gdf
    before = set(edges.columns)
    with pytest.raises(ValueError, match="no column"):
        Exposure(street_network, a=(zones, "level"), b=(zones, "wrong"))
    assert set(edges.columns) == before
    # a clean retry with the same names succeeds
    Exposure(street_network, a=(zones, "level"))
    _drop_layer_columns(edges, "a", ())


def test_a_filtered_frame_with_retained_indices_ingests_correctly(
    street_network,
):
    # sjoin returns labels: a pre-filtered source frame keeping its
    # original index must not be indexed positionally.
    pytest.importorskip("rasterio")
    geopandas = pytest.importorskip("geopandas")
    west, south, east, north = _extent(street_network)
    world = box(west - 0.01, south - 0.01, east + 0.01, north + 0.01)
    stacked = geopandas.GeoDataFrame(
        {
            "level": [10.0, 61.0],
            "metric": ["Ln", "Lden"],
        },
        geometry=[world, world],
        crs="EPSG:4326",
    )
    filtered = stacked[stacked["metric"] == "Lden"]  # index label 1 only
    edges = street_network.streets_gdf
    try:
        Exposure(street_network, noise=(filtered, "level"))
        assert edges["noise"].values == pytest.approx(61.0)
    finally:
        _drop_layer_columns(edges, "noise", ())


def test_threshold_share_columns_join_the_collision_checks(street_network):
    pytest.importorskip("rasterio")
    geopandas = pytest.importorskip("geopandas")
    west, south, east, north = _extent(street_network)
    world = box(west - 0.01, south - 0.01, east + 0.01, north + 0.01)
    zones = geopandas.GeoDataFrame({"level": [61.0]}, geometry=[world], crs="EPSG:4326")
    with pytest.raises(ValueError, match="both derive"):
        Exposure(
            street_network,
            x=(zones, "level"),
            x_share_above_55=(zones, "level"),
            thresholds={"x": 55},
        )


def test_raster_nan_is_uncovered_and_any_infinity_refuses(street_network, tmp_path):
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_bounds

    west, south, east, north = _extent(street_network)
    height = width = 32
    # NaN over the western half: those edges lose coverage, silently.
    patchy = np.full((height, width), 4.0, dtype="float32")
    patchy[:, : width // 2] = np.nan
    path = tmp_path / "patchy.tif"
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=from_bounds(
            west - 0.01, south - 0.01, east + 0.01, north + 0.01, width, height
        ),
    ) as sink:
        sink.write(patchy[None, ...])
        sink.descriptions = ("X [u]",)
    edges = street_network.streets_gdf
    try:
        Exposure(street_network, x=(str(path), "X"))
        coverage = edges["x_coverage"].values
        assert (coverage == 0).any() and (coverage == 1).any()
        assert np.isfinite(edges["x"].values).all()
    finally:
        _drop_layer_columns(edges, "x", ())
    # An infinity anywhere in the band refuses — even in a corner cell
    # far outside every street, which no sample ever reads.
    infected = np.full((height, width), 4.0, dtype="float32")
    infected[0, 0] = np.inf
    bad = tmp_path / "infected.tif"
    with rasterio.open(
        bad,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=from_bounds(
            west - 1.0, south - 1.0, east + 0.01, north + 0.01, width, height
        ),
    ) as sink:
        sink.write(infected[None, ...])
        sink.descriptions = ("X [u]",)
    with pytest.raises(ValueError, match="infinite"):
        Exposure(street_network, x=(str(bad), "X"))


def test_the_helsinki_sampledata_layers_ingest(street_network):
    helsinki = pytest.importorskip("cafein.sampledata.helsinki")
    geopandas = pytest.importorskip("geopandas")
    pytest.importorskip("rioxarray")
    try:
        noise_path = helsinki.noise
        air_path = helsinki.air_quality
        green_path = helsinki.green_view
    except Exception as error:  # no network, unreleased data, …
        pytest.skip(f"sampledata unavailable: {error}")
    zones = geopandas.read_file(noise_path)
    road_lden = zones[(zones["source"] == "road") & (zones["metric"] == "Lden")]
    # A central spatial subset keeps this optional test light: real
    # bytes and schema, bounded overlay work (the full noise layer
    # costs minutes against a metropolitan edge set).
    west, south, east, north = _extent(street_network)
    center_x, center_y = (west + east) / 2, (south + north) / 2
    box_3067 = (
        geopandas.GeoSeries([Point(center_x, center_y)], crs="EPSG:4326")
        .to_crs(road_lden.crs)
        .buffer(1000)
        .total_bounds
    )
    road_lden = road_lden.cx[box_3067[0] : box_3067[2], box_3067[1] : box_3067[3]]
    roads = geopandas.read_file(green_path, layer="roads")
    edges = street_network.streets_gdf
    try:
        ex = Exposure(
            street_network,
            noise=(road_lden, "db_low"),
            pollution=(str(air_path), "PM25Concentration"),
            greenery=(roads, "Comb_GVI"),
            thresholds={"noise": 55},
        )
        assert ex.layers == ("noise", "pollution", "greenery")
        for column in ("noise", "pollution", "greenery"):
            assert (edges[f"{column}_coverage"].values > 0).any()
            assert np.isfinite(edges[column].values).all()
    finally:
        _drop_layer_columns(edges, "noise", (55,))
        _drop_layer_columns(edges, "pollution", ())
        _drop_layer_columns(edges, "greenery", ())


def test_a_crs_less_raster_is_refused(street_network, tmp_path):
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_bounds

    west, south, east, north = _extent(street_network)
    path = tmp_path / "no_crs.tif"
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=4,
        width=4,
        count=1,
        dtype="float32",
        transform=from_bounds(west, south, east, north, 4, 4),
    ) as sink:
        sink.write(np.full((1, 4, 4), 1.0, dtype="float32"))
        sink.descriptions = ("X [u]",)
    with pytest.raises(ValueError, match="declares no CRS"):
        Exposure(street_network, x=(str(path), "X"))


def test_unknown_band_lists_the_available_ones(street_network, tmp_path):
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_bounds

    west, south, east, north = _extent(street_network)
    path = tmp_path / "bands.tif"
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=4,
        width=4,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=from_bounds(west, south, east, north, 4, 4),
    ) as sink:
        sink.write(np.full((1, 4, 4), 1.0, dtype="float32"))
        sink.descriptions = ("AQIndex [1]",)
    with pytest.raises(ValueError, match="no band 'Nope'"):
        Exposure(street_network, x=(str(path), "Nope"))


def _drop_layer_columns(edges, name, thresholds):
    for column in exposure._derived_columns(name, tuple(float(x) for x in thresholds)):
        if column in edges.columns:
            del edges[column]
