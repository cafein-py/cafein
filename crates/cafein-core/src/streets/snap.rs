//! Coordinate snapping onto the nearest street segment.

use super::*;

impl StreetNetwork {
    /// Snaps a coordinate to its nearest edge within `max_snap_distance`
    /// meters through the packed segment index. Non-finite coordinates or a
    /// non-finite or negative allowance never snap.
    pub fn snap(&self, latitude: f64, longitude: f64, max_snap_distance: f64) -> Option<Snap> {
        if !latitude.is_finite()
            || !longitude.is_finite()
            || !max_snap_distance.is_finite()
            || max_snap_distance < 0.0
        {
            return None;
        }
        // Expanding-ring nearest-edge search. Segments are found by envelope
        // intersection — never a degree-Euclidean nearest — and each candidate
        // is re-measured exactly; exact connector ties break by
        // (edge, fraction), so the winner is a function of the built network,
        // not of index internals. Querying the full `max_snap_distance`
        // envelope at once would collect every segment in a city-sized box
        // under a generous allowance, so the query grows outward instead:
        // `snap_envelope(r)` is a superset of the true ±r metre box, so a
        // segment it misses lies strictly farther than `r` — once the best
        // exact connector is within the ring, nothing outside can beat it.
        let mut radius = 32.0_f64.min(max_snap_distance);
        let mut candidates = Vec::new();
        loop {
            let envelope = snap_envelope(latitude, longitude, radius);
            self.query_index_into(&envelope, &mut candidates);
            let mut best: Option<Snap> = None;
            for &(edge, start) in &candidates {
                let (connector, fraction) = self.foot_on_segment(latitude, longitude, edge, start);
                if connector <= max_snap_distance
                    && best.is_none_or(|current| {
                        connector < current.connector
                            || (connector == current.connector
                                && (edge, fraction) < (current.edge, current.fraction))
                    })
                {
                    best = Some(Snap {
                        edge,
                        fraction,
                        connector,
                    });
                }
            }
            if let Some(snap) = best {
                if snap.connector <= radius || radius >= max_snap_distance {
                    return Some(snap);
                }
            }
            if radius >= max_snap_distance {
                return None;
            }
            radius = (radius * 4.0).min(max_snap_distance);
        }
    }

    /// Collects the payloads of every packed-index leaf whose box
    /// intersects the envelope, through the array accessors.
    pub(super) fn query_index_into(&self, envelope: &Envelope, matches: &mut Vec<(u32, u32)>) {
        query_packed_index(
            self.arrays().index_boxes(),
            self.arrays().index_payload(),
            self.level_starts(),
            envelope,
            matches,
        );
    }

    /// The exact connector distance and true-length fraction of a query's
    /// foot on one segment (its first coordinate is `start`), measured in an
    /// equirectangular frame local to the query — exact because segments are
    /// short (densified below `MAX_SEGMENT_METERS`).
    pub(super) fn foot_on_segment(
        &self,
        latitude: f64,
        longitude: f64,
        edge: u32,
        start: u32,
    ) -> (f64, f64) {
        let (a, b) = (start as usize, start as usize + 1);
        let (mpd_lon, mpd_lat) = meters_per_degree(latitude);
        let to_xy = |lon: f64, lat: f64| {
            (
                longitude_delta(longitude, lon) * mpd_lon,
                (lat - latitude) * mpd_lat,
            )
        };
        let (lon_a, lat_a) = self.coordinate(a);
        let (lon_b, lat_b) = self.coordinate(b);
        let (ax, ay) = to_xy(lon_a, lat_a);
        let (bx, by) = to_xy(lon_b, lat_b);
        let (dx, dy) = (bx - ax, by - ay);
        let squared = dx * dx + dy * dy;
        // The query sits at the frame origin, so the foot parameter is
        // ((Q - A)·(B - A)) / |B - A|² with Q = 0.
        let t = if squared > 0.0 {
            ((-ax * dx - ay * dy) / squared).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let (px, py) = (ax + t * dx, ay + t * dy);
        let connector = (px * px + py * py).sqrt();

        let end = self.arrays().coordinate_offsets()[edge as usize + 1] as usize;
        let along = self.along(a) + t * (self.along(b) - self.along(a));
        let total = self.along(end - 1);
        let fraction = if total > 0.0 { along / total } else { 0.0 };
        (connector, fraction)
    }
}
