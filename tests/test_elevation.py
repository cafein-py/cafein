"""DEM sampling and the street network's elevation intake."""

import numpy as np
import pytest

pytest.importorskip("cafein._cafein")

from cafein import StreetNetwork  # noqa: E402
from cafein import elevation  # noqa: E402

KAMPPI, HAKANIEMI = (60.1690, 24.9320), (60.1795, 24.9520)


def ramp(lons, lats):
    """An analytic west-east ramp: 1 m of climb per 0.001° of longitude."""
    return ((np.asarray(lons) - 24.8) * 1000.0).astype("float32")


@pytest.fixture(scope="module")
def elevated(kantakaupunki_pbf):
    return StreetNetwork.from_osm(str(kantakaupunki_pbf), dem=ramp)


def test_sample_dem_through_a_callable():
    values = elevation.sample_dem([24.9, 25.0], [60.1, 60.2], ramp)
    assert values.dtype == np.float32
    assert values == pytest.approx([100.0, 200.0])
    # Raster nodata quirks and float32 overflow both surface as inf; the
    # stored array knows only one unavailable sentinel.
    values = elevation.sample_dem(
        [24.9, 25.0, 25.1],
        [60.1, 60.2, 60.3],
        lambda lons, lats: [np.inf, 4.0e38, 7.0],
    )
    assert np.isnan(values[0]) and np.isnan(values[1])
    assert values[2] == 7.0


def test_a_misshaped_callable_result_is_rejected():
    with pytest.raises(ValueError, match="returned"):
        elevation.sample_dem([24.9, 25.0], [60.1, 60.2], lambda lons, lats: [1.0])


def test_a_callable_cannot_mutate_the_geometry():
    # The callable sees read-only copies, so the builder's coordinate arrays
    # can never be changed through the sampling callback.
    coordinates = np.array([[24.9, 60.1], [25.0, 60.2]])

    def vandal(lons, lats):
        with pytest.raises(ValueError):
            lons[0] = 0.0
        return ramp(lons, lats)

    elevation.sample_dem(coordinates[:, 0], coordinates[:, 1], vandal)
    assert coordinates[0, 0] == 24.9


@pytest.mark.parametrize(
    "offsets, lons, values, expected_inferred, expected",
    [
        # Three uneven segments; the interior vertex sits a quarter of the
        # way along, so it takes a quarter of the climb — not half, as
        # index-based interpolation would give.
        pytest.param(
            [0, 4],
            [24.0, 24.001, 24.002, 24.004],
            [0.0, 55.0, 77.0, 100.0],
            1,
            [0.0, 25.0, 50.0, 100.0],
            id="interpolates-by-distance",
        ),
        # No interior, no rewrite — and no place in the inferred count.
        pytest.param(
            [0, 2],
            [24.0, 24.001],
            [10.0, 20.0],
            0,
            [10.0, 20.0],
            id="two-coordinate-structure",
        ),
        # A structure edge with one unavailable endpoint cannot be
        # inferred: the interior becomes NaN, and the finite endpoint
        # keeps its sampled value.
        pytest.param(
            [0, 3],
            [24.0, 24.001, 24.002],
            [10.0, 55.0, np.nan],
            1,
            [10.0, np.nan, np.nan],
            id="finite-endpoint-next-to-nodata",
        ),
    ],
)
def test_infer_structures_interpolates_by_distance(
    offsets, lons, values, expected_inferred, expected
):
    offsets = np.array(offsets)
    lons = np.array(lons)
    lats = np.full(len(lons), 60.0)
    values = np.array(values, dtype=np.float32)
    inferred = elevation.infer_structures(offsets, lons, lats, values, [0])
    assert inferred == expected_inferred
    assert values == pytest.approx(expected, nan_ok=True)


def test_coverage_counts_the_finite_share():
    assert elevation.coverage(np.array([], dtype=np.float32)) == 0.0
    assert elevation.coverage(np.array([1.0, np.nan, 3.0, 4.0])) == 0.75


def test_from_osm_with_a_dem_installs_metadata(elevated):
    meta = elevated.elevation_metadata
    assert meta["source"] == "callable"
    assert meta["sampling_interval"] == 25.0
    assert meta["nodata_policy"] == "nan"
    assert meta["coverage"] == 1.0
    assert meta["inferred_edges"] > 0  # central Helsinki has bridges
    # Flat routing is untouched by carrying elevations.
    assert elevated.travel_time(KAMPPI, HAKANIEMI, mode="bicycle") is not None


def test_stored_elevations_match_the_ramp(elevated):
    # The values the Rust core actually stores, checked against the analytic
    # ramp at each stored coordinate's own longitude. Bridge and tunnel
    # interiors are endpoint-interpolated by distance rather than sampled, so
    # they may drift where a structure curves — a small minority, and bounded
    # by the structure's own climb.
    values = np.asarray(elevated._coordinate_elevations, dtype=np.float32)
    lons = np.asarray(elevated._coordinates)[:, 0]
    assert len(values) == len(lons) > 0
    assert np.isfinite(values).all()
    error = np.abs(values - (lons - 24.8) * 1000.0)
    assert (error < 0.05).mean() > 0.98
    assert error.max() < 50.0


def test_routing_is_unchanged_for_slope_free_modes(elevated, helsinki_streets):
    # Same extract with and without the DEM: profiles without a slope model
    # must route identically — only the bicycle carries one. The ramp
    # tops out near 1.8 %, so the wheelchair's 8 % cap never bites and
    # its routing matches the capless build exactly.
    pairs = [
        (KAMPPI, HAKANIEMI),
        (HAKANIEMI, KAMPPI),
        ((60.1580, 24.9350), KAMPPI),
    ]
    for mode in ("walk", "e_bike", "e_scooter", "wheelchair"):
        for origin, destination in pairs:
            with_dem = elevated.travel_time(origin, destination, mode=mode)
            without = helsinki_streets.travel_time(origin, destination, mode=mode)
            assert with_dem == without
            assert with_dem is not None


def test_cycling_charges_the_climb_on_an_elevated_network(elevated, helsinki_streets):
    # The ramp climbs east, so eastbound cycling pays the slope penalty —
    # every path east must gain at least the net elevation — and westbound
    # earns the descent credit.
    east = elevated.travel_time(KAMPPI, HAKANIEMI, mode="bicycle")
    west = elevated.travel_time(HAKANIEMI, KAMPPI, mode="bicycle")
    flat_east = helsinki_streets.travel_time(KAMPPI, HAKANIEMI, mode="bicycle")
    flat_west = helsinki_streets.travel_time(HAKANIEMI, KAMPPI, mode="bicycle")
    assert east > flat_east
    assert west < flat_west
    assert east > west


def test_a_coarse_dem_interval_is_capped_at_the_stored_segment_limit(
    kantakaupunki_pbf,
):
    # The stored geometry is densified under ~100 m regardless, so a coarser
    # request would let densification interpolate structure interiors after
    # inference already counted. The cap makes the two builds identical, and
    # the metadata records the interval actually used.
    from cafein._cafein import STREET_MAX_SEGMENT_METERS

    limit = STREET_MAX_SEGMENT_METERS - 1.0
    coarse = StreetNetwork.from_osm(
        str(kantakaupunki_pbf), dem=ramp, dem_interval=5000.0
    )
    capped = StreetNetwork.from_osm(
        str(kantakaupunki_pbf), dem=ramp, dem_interval=limit
    )
    assert coarse.elevation_metadata == capped.elevation_metadata
    assert coarse.elevation_metadata["sampling_interval"] == limit
    assert np.array_equal(
        np.asarray(coarse._coordinate_elevations, dtype=np.float32),
        np.asarray(capped._coordinate_elevations, dtype=np.float32),
        equal_nan=True,
    )


@pytest.mark.parametrize(
    "elevations, metadata, match",
    [
        ([5.0, 6.0], ("callable", 0.0, "nan", 1.0, 0), "elevation"),
        ([5.0, 6.0], ("callable", float("inf"), "nan", 1.0, 0), "elevation"),
        ([5.0, 6.0], ("callable", 25.0, "nan", 1.5, 0), "elevation"),
        ([5.0, 6.0], ("callable", 25.0, "nan", 1.0, 2), "elevation"),
        # Metadata and elevations only travel together.
        (None, ("callable", 25.0, "nan", 1.0, 0), "together"),
    ],
)
def test_bogus_elevation_metadata_is_rejected(elevations, metadata, match):
    from cafein._cafein import StreetNetwork as Core

    zeros = [0]
    with pytest.raises(ValueError, match=match):
        Core(
            2,
            [(0, 1, 100.0)],
            [0, 2],
            [24.93, 24.932],
            [60.169, 60.169],
            zeros,
            zeros,
            zeros,
            zeros,
            [1],
            [1],
            zeros,
            zeros,
            elevations,
            metadata,
        )


def test_stored_elevations_and_metadata_round_trip(elevated, tmp_path):
    built = np.asarray(elevated._coordinate_elevations, dtype=np.float32)
    path = tmp_path / "elevated.cafein"
    elevated.save(path)
    for mmap in (False, True):
        loaded = StreetNetwork.load(path, mmap=mmap)
        stored = np.asarray(loaded._coordinate_elevations, dtype=np.float32)
        assert np.array_equal(stored, built, equal_nan=True)
        assert loaded.elevation_metadata == elevated.elevation_metadata
        assert loaded.travel_time(KAMPPI, HAKANIEMI, mode="bicycle") == (
            elevated.travel_time(KAMPPI, HAKANIEMI, mode="bicycle")
        )


def test_without_a_dem_there_is_no_metadata(helsinki_streets):
    assert helsinki_streets.elevation_metadata is None
    assert helsinki_streets._coordinate_elevations is None


def test_nodata_shows_up_in_coverage_and_the_stored_values(kantakaupunki_pbf):
    def patchy(lons, lats):
        values = ramp(lons, lats)
        values[np.asarray(lons) > 24.95] = np.nan
        return values

    net = StreetNetwork.from_osm(str(kantakaupunki_pbf), dem=patchy)
    meta = net.elevation_metadata
    assert 0.0 < meta["coverage"] < 1.0
    # The NaN band survives into the stored array. A buffer around the cut
    # absorbs structure inference, which can move NaN across it.
    values = np.asarray(net._coordinate_elevations, dtype=np.float32)
    lons = np.asarray(net._coordinates)[:, 0]
    assert np.isnan(values[lons > 24.96]).all()
    assert np.isfinite(values[lons < 24.94]).mean() > 0.99


def test_dem_interval_is_validated(kantakaupunki_pbf):
    with pytest.raises(ValueError, match="dem_interval"):
        StreetNetwork.from_osm(str(kantakaupunki_pbf), dem=ramp, dem_interval=0.0)


def test_the_raster_backend_needs_rioxarray_or_a_real_file(tmp_path):
    # Without the optional dependency the error names the way out; with it,
    # a missing file fails at open rather than being silently empty.
    missing = tmp_path / "missing.tif"
    try:
        import rioxarray  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError, match="rioxarray"):
            elevation.sample_dem([24.9], [60.1], str(missing))
    else:
        with pytest.raises(Exception, match="missing.tif|No such file|not recognized"):
            elevation.sample_dem([24.9], [60.1], str(missing))


def test_a_geotiff_ramp_samples_bilinearly(tmp_path):
    rioxarray = pytest.importorskip("rioxarray")
    xr = pytest.importorskip("xarray")

    # A 1° × 1° raster whose value equals metres of easting from 24°E.
    lons = np.linspace(24.0, 25.0, 101)
    lats = np.linspace(60.5, 59.5, 101)
    grid = np.tile(((lons - 24.0) * 1000.0).astype("float32"), (101, 1))
    raster = xr.DataArray(grid, coords={"y": lats, "x": lons}, dims=("y", "x"))
    raster = raster.rio.write_crs("EPSG:4326")
    path = tmp_path / "ramp.tif"
    raster.rio.to_raster(path)

    values = elevation.sample_dem([24.25, 24.5, 24.905], [60.0, 60.0, 60.0], str(path))
    assert values == pytest.approx([250.0, 500.0, 905.0], abs=0.5)
    del rioxarray


def test_a_scaled_geotiff_decodes_to_physical_elevations(tmp_path):
    pytest.importorskip("rioxarray")
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin

    # Raw int cells with a scale/offset encoding: physical = raw * 0.01 + 100.
    path = tmp_path / "scaled.tif"
    data = np.array([[1000, 2000], [3000, 4000]], dtype="int32")
    with rasterio.open(
        str(path),
        "w",
        driver="GTiff",
        height=2,
        width=2,
        count=1,
        dtype="int32",
        crs="EPSG:4326",
        transform=from_origin(24.0, 61.0, 0.5, 0.5),
    ) as raster:
        raster.write(data, 1)
        raster.scales = (0.01,)
        raster.offsets = (100.0,)

    # Pixel centers of the top row, so interpolation is exact.
    values = elevation.sample_dem([24.25, 24.75], [60.75, 60.75], str(path))
    assert values == pytest.approx([110.0, 120.0])


def test_the_wheelchair_gradient_cap_bites_on_a_steep_dem(
    kantakaupunki_pbf, helsinki_streets
):
    # A near-vertical west-east wall: after the ±100 % slope clamp, any
    # arc drifting more than a few degrees off north-south is steeper
    # than the 8 % cap. The eastward crossing that walking makes directly
    # becomes unreachable on wheels within the street cutoff, while
    # capless walking is untouched by the DEM.
    def cliff(lons, lats):
        return ((np.asarray(lons) - 24.8) * 2_000_000.0).astype("float32")

    steep = StreetNetwork.from_osm(str(kantakaupunki_pbf), dem=cliff)
    walked = steep.travel_time(KAMPPI, HAKANIEMI, mode="walk")
    assert walked is not None
    assert walked == helsinki_streets.travel_time(KAMPPI, HAKANIEMI, mode="walk")
    assert steep.travel_time(KAMPPI, HAKANIEMI, mode="wheelchair") is None
