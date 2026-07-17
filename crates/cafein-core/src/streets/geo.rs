//! Small geodesic helpers shared by the index, the graph build,
//! and snapping.

use super::*;

/// Drops consecutive duplicate points, keeping at least two.
pub(super) fn dedup_consecutive(path: Vec<(f64, f64)>) -> Vec<(f64, f64)> {
    let mut coordinates: Vec<(f64, f64)> = Vec::with_capacity(path.len());
    for point in path {
        if coordinates.last() != Some(&point) {
            coordinates.push(point);
        }
    }
    if coordinates.len() == 1 {
        coordinates.push(coordinates[0]);
    }
    coordinates
}

/// The lon/lat envelope containing every segment within `max_snap_distance` of
/// a query, sized at the query's own latitude. The latitude half-width uses a
/// global minimum metres-per-degree-latitude; the longitude half-width uses
/// the minimum metres-per-degree-longitude over the reachable latitude band,
/// so no truly-nearby segment is clipped at any latitude or snap distance.
pub(super) fn snap_envelope(latitude: f64, longitude: f64, max_snap_distance: f64) -> Envelope {
    // Metres per degree of latitude bottoms out at the equator.
    const MIN_MPD_LAT: f64 = 110_574.0;
    let margin = 1.0 + 1e-6;
    let delta_lat = max_snap_distance / MIN_MPD_LAT * margin;
    let lo_lat = (latitude - delta_lat).clamp(-90.0, 90.0);
    let hi_lat = (latitude + delta_lat).clamp(-90.0, 90.0);
    let min_mpd_lon = meters_per_degree(lo_lat).0.min(meters_per_degree(hi_lat).0);
    let delta_lon = if min_mpd_lon > 1e-9 {
        (max_snap_distance / min_mpd_lon * margin).min(180.0)
    } else {
        180.0
    };
    // Rounded outward onto the fixed-point grid, so the envelope stays a
    // superset of the true one.
    let outward = |degrees: f64, up: bool| -> i32 {
        let scaled = degrees * COORDINATE_SCALE;
        let rounded = if up { scaled.ceil() } else { scaled.floor() };
        rounded.clamp(f64::from(i32::MIN), f64::from(i32::MAX)) as i32
    };
    [
        outward(longitude - delta_lon, false),
        outward(latitude - delta_lat, false),
        outward(longitude + delta_lon, true),
        outward(latitude + delta_lat, true),
    ]
}

/// Shortest signed longitude difference in degrees, wrapped to `[-180, 180]`
/// so a pair straddling the antimeridian measures the short way.
pub(super) fn longitude_delta(from: f64, to: f64) -> f64 {
    let delta = (to - from) % 360.0;
    if delta > 180.0 {
        delta - 360.0
    } else if delta < -180.0 {
        delta + 360.0
    } else {
        delta
    }
}

/// The true geometric length between two lon/lat points, in metres, using a
/// local `cos(latitude)` at their midpoint (exact for a short segment).
pub(super) fn segment_length(lon_a: f64, lat_a: f64, lon_b: f64, lat_b: f64) -> f64 {
    let (mpd_lon, mpd_lat) = meters_per_degree((lat_a + lat_b) / 2.0);
    let dx = longitude_delta(lon_a, lon_b) * mpd_lon;
    let dy = (lat_b - lat_a) * mpd_lat;
    (dx * dx + dy * dy).sqrt()
}

/// Splits every segment longer than `MAX_SEGMENT_METERS` into equal colinear
/// pieces, returning the densified coordinate offsets, geographic coordinates,
/// and per-coordinate cumulative distance from each edge's first point.
pub(super) fn densify(
    coordinate_offsets: &[u32],
    longitudes: &[i32],
    latitudes: &[i32],
) -> (Vec<u32>, Vec<i32>, Vec<i32>, Vec<f32>) {
    let edge_count = coordinate_offsets.len().saturating_sub(1);
    let mut offsets = Vec::with_capacity(coordinate_offsets.len());
    let mut lons = Vec::new();
    let mut lats = Vec::new();
    let mut cumulative = Vec::new();
    // Inserted points re-quantize onto the grid (≤ ~0.8 cm each), so the
    // split targets a hair under the maximum and no sub-segment exceeds it.
    let target = MAX_SEGMENT_METERS - QUANTIZATION_GUARD_METERS;
    offsets.push(0);
    for edge in 0..edge_count {
        let start = coordinate_offsets[edge] as usize;
        let end = coordinate_offsets[edge + 1] as usize;
        lons.push(longitudes[start]);
        lats.push(latitudes[start]);
        cumulative.push(0.0f32);
        let mut running = 0.0f64;
        for point in start..end - 1 {
            let (lon_a, lat_a) = (degrees(longitudes[point]), degrees(latitudes[point]));
            let (lon_b, lat_b) = (
                degrees(longitudes[point + 1]),
                degrees(latitudes[point + 1]),
            );
            // Bound each sub-piece by the largest metres-per-degree over the
            // segment's latitude band, so none exceeds MAX_SEGMENT_METERS even
            // when the segment spans a wide latitude range. Longitude
            // metres-per-degree peaks toward the equator, latitude toward the
            // poles.
            let mut max_mpd_lon = meters_per_degree(lat_a).0.max(meters_per_degree(lat_b).0);
            if (lat_a <= 0.0) != (lat_b <= 0.0) {
                max_mpd_lon = max_mpd_lon.max(meters_per_degree(0.0).0);
            }
            let max_mpd_lat = meters_per_degree(lat_a).1.max(meters_per_degree(lat_b).1);
            let dx = longitude_delta(lon_a, lon_b).abs() * max_mpd_lon;
            let dy = (lat_b - lat_a).abs() * max_mpd_lat;
            let pieces = ((dx * dx + dy * dy).sqrt() / target).ceil().max(1.0) as usize;
            for k in 1..=pieces {
                let t = k as f64 / pieces as f64;
                let lon = if k == pieces {
                    longitudes[point + 1]
                } else {
                    quantize(lon_a + t * (lon_b - lon_a))
                };
                let lat = if k == pieces {
                    latitudes[point + 1]
                } else {
                    quantize(lat_a + t * (lat_b - lat_a))
                };
                let (prev_lon, prev_lat) = (*lons.last().unwrap(), *lats.last().unwrap());
                running += segment_length(
                    degrees(prev_lon),
                    degrees(prev_lat),
                    degrees(lon),
                    degrees(lat),
                );
                lons.push(lon);
                lats.push(lat);
                cumulative.push(running as f32);
            }
        }
        offsets.push(lons.len() as u32);
    }
    (offsets, lons, lats, cumulative)
}

/// Local meters per degree of (longitude, latitude) on the WGS84 spheroid.
pub(super) fn meters_per_degree(latitude: f64) -> (f64, f64) {
    let phi = latitude.to_radians();
    let meters_per_lat = 111_132.954 - 559.822 * (2.0 * phi).cos() + 1.175 * (4.0 * phi).cos();
    let meters_per_lon =
        111_412.84 * phi.cos() - 93.5 * (3.0 * phi).cos() + 0.118 * (5.0 * phi).cos();
    (meters_per_lon, meters_per_lat)
}

/// Conservative rounding of a duration to whole seconds: up, with a small
/// tolerance for floating-point noise.
pub(super) fn seconds(duration: f64) -> u32 {
    (duration - 1e-6).ceil().max(0.0) as u32
}
