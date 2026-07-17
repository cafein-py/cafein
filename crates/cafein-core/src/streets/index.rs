//! The packed spatial index over street segments, Hilbert-ordered.

/// Longest stored segment, in meters. Segments are densified below this so a
/// single centre-latitude scale represents each one to well under a
/// millimetre even at high latitude.
pub(super) const MAX_SEGMENT_METERS: f64 = 100.0;

/// Headroom the densifier leaves under `MAX_SEGMENT_METERS`, so re-quantizing
/// the inserted points (≤ ~0.8 cm each) never pushes a segment over the
/// maximum.
pub(super) const QUANTIZATION_GUARD_METERS: f64 = 0.05;

/// Fixed-point coordinate scale: degrees × 10⁷ stored as `i32`
/// (≈ 1.1 cm of latitude per step; ±180° fits comfortably).
pub(super) const COORDINATE_SCALE: f64 = 1e7;

/// Children per packed-index node.
pub(super) const INDEX_NODE_SIZE: usize = 16;

/// A fixed-point degree value from a float one, rounding to the nearest
/// grid step (ties to even).
pub(super) fn quantize(degrees: f64) -> i32 {
    (degrees * COORDINATE_SCALE)
        .round_ties_even()
        .clamp(f64::from(i32::MIN), f64::from(i32::MAX)) as i32
}

/// The float degree value of a fixed-point one.
pub(super) fn degrees(fixed: i32) -> f64 {
    f64::from(fixed) / COORDINATE_SCALE
}

/// A lon/lat bounding box in fixed-point coordinates:
/// `[min_lon, min_lat, max_lon, max_lat]`.
pub(super) type Envelope = [i32; 4];

pub(super) fn envelopes_intersect(a: &Envelope, b: &Envelope) -> bool {
    a[0] <= b[2] && b[0] <= a[2] && a[1] <= b[3] && b[1] <= a[3]
}

/// A packed static spatial index over the densified edge segments: leaf
/// boxes sorted by the Hilbert position of their segment's midpoint, parent
/// levels packed bottom-up over runs of `INDEX_NODE_SIZE` children — flat
/// arrays with an implicit tree layout, built once and never mutated.
/// Queried by envelope intersection only. Boxes are exact in the
/// fixed-point grid (the coordinates are grid points), so only query
/// envelopes need outward rounding.
#[derive(Debug)]
pub(super) struct PackedIndex {
    /// Node boxes: the leaves first (Hilbert order), then each parent
    /// level, the root last.
    pub(super) boxes: Vec<Envelope>,
    /// Two entries per leaf — `edge index, index of the segment's first
    /// coordinate` — parallel to the leaf boxes.
    pub(super) payload: Vec<u32>,
    /// Start of each level in `boxes` (leaves at 0), plus a tail.
    pub(super) level_starts: Vec<u32>,
}

/// Collects the payloads of every leaf whose box intersects the envelope,
/// sorted by payload so callers see a traversal-free order. The arrays
/// are a [`PackedIndex`]'s, whichever backing they live in.
pub(super) fn query_packed_index(
    boxes: &[Envelope],
    payload: &[u32],
    level_starts: &[u32],
    envelope: &Envelope,
    matches: &mut Vec<(u32, u32)>,
) {
    matches.clear();
    if payload.is_empty() {
        return;
    }
    let levels = level_starts.len() - 1;
    // (global node index, level), starting at the root.
    let mut stack = vec![(boxes.len() - 1, levels - 1)];
    while let Some((node, level)) = stack.pop() {
        if !envelopes_intersect(&boxes[node], envelope) {
            continue;
        }
        if level == 0 {
            matches.push((payload[2 * node], payload[2 * node + 1]));
            continue;
        }
        // A node's children sit in the level below, in its own run.
        let position = node - level_starts[level] as usize;
        let start = level_starts[level - 1] as usize + position * INDEX_NODE_SIZE;
        let end = (start + INDEX_NODE_SIZE).min(level_starts[level] as usize);
        for child in start..end {
            stack.push((child, level - 1));
        }
    }
    matches.sort_unstable();
}

/// Builds the packed index over a densified fixed-point polyline set.
pub(super) fn build_index(coordinate_offsets: &[u32], lons: &[i32], lats: &[i32]) -> PackedIndex {
    // One item per consecutive coordinate pair, keyed by the Hilbert
    // position of its midpoint on a grid over the extract; ties broken by
    // the payload so the order is a pure function of the geometry.
    let bounds = coordinate_bounds(lons, lats);
    let mut items: Vec<(u64, (u32, u32), Envelope)> = Vec::new();
    for edge in 0..coordinate_offsets.len().saturating_sub(1) {
        let start = coordinate_offsets[edge] as usize;
        let end = coordinate_offsets[edge + 1] as usize;
        for segment in start..end - 1 {
            let (lon_a, lat_a) = (lons[segment], lats[segment]);
            let (lon_b, lat_b) = (lons[segment + 1], lats[segment + 1]);
            let key = hilbert(
                grid_position(
                    ((i64::from(lon_a) + i64::from(lon_b)) / 2) as i32,
                    bounds[0],
                    bounds[2],
                ),
                grid_position(
                    ((i64::from(lat_a) + i64::from(lat_b)) / 2) as i32,
                    bounds[1],
                    bounds[3],
                ),
            );
            items.push((
                key,
                (edge as u32, segment as u32),
                [
                    lon_a.min(lon_b),
                    lat_a.min(lat_b),
                    lon_a.max(lon_b),
                    lat_a.max(lat_b),
                ],
            ));
        }
    }
    items.sort_unstable_by_key(|&(key, payload, _)| (key, payload));

    let count = items.len();
    let level_starts = level_starts_for(count);
    if count == 0 {
        return PackedIndex {
            boxes: Vec::new(),
            payload: Vec::new(),
            level_starts,
        };
    }
    if count == 1 {
        // A single leaf is its own root: one leaf-only level.
        let (_, (edge, segment), envelope) = items[0];
        return PackedIndex {
            boxes: vec![envelope],
            payload: vec![edge, segment],
            level_starts,
        };
    }

    let total = *level_starts.last().unwrap() as usize;
    let mut boxes = Vec::with_capacity(total);
    let mut payload = Vec::with_capacity(count * 2);
    for (_, (edge, segment), envelope) in items {
        boxes.push(envelope);
        payload.push(edge);
        payload.push(segment);
    }
    for level in 1..level_starts.len() - 1 {
        let (start, end) = (
            level_starts[level - 1] as usize,
            level_starts[level] as usize,
        );
        for run in (start..end).step_by(INDEX_NODE_SIZE) {
            let mut merged = boxes[run];
            for child in &boxes[run + 1..(run + INDEX_NODE_SIZE).min(end)] {
                merged[0] = merged[0].min(child[0]);
                merged[1] = merged[1].min(child[1]);
                merged[2] = merged[2].max(child[2]);
                merged[3] = merged[3].max(child[3]);
            }
            boxes.push(merged);
        }
    }
    PackedIndex {
        boxes,
        payload,
        level_starts,
    }
}

/// The level-start offsets of a packed index with `count` leaves: each
/// level's start in the node array (leaves at 0), plus a tail holding the
/// total node count. A pure function of the leaf count, so an adopted
/// index never needs them stored.
pub(super) fn level_starts_for(count: usize) -> Vec<u32> {
    let mut level_starts = vec![0u32];
    let mut total = count;
    let mut level_size = count;
    while level_size > 1 {
        level_starts.push(total as u32);
        level_size = level_size.div_ceil(INDEX_NODE_SIZE);
        total += level_size;
    }
    match count {
        0 => vec![0, 0],
        // A single leaf is its own root: one leaf-only level.
        1 => vec![0, 1],
        _ => {
            level_starts.push(total as u32);
            level_starts
        }
    }
}

/// The `[min_lon, min_lat, max_lon, max_lat]` bounds of a fixed-point
/// coordinate set.
pub(super) fn coordinate_bounds(lons: &[i32], lats: &[i32]) -> Envelope {
    let mut bounds = [i32::MAX, i32::MAX, i32::MIN, i32::MIN];
    for (&lon, &lat) in lons.iter().zip(lats) {
        bounds[0] = bounds[0].min(lon);
        bounds[1] = bounds[1].min(lat);
        bounds[2] = bounds[2].max(lon);
        bounds[3] = bounds[3].max(lat);
    }
    bounds
}

/// A coordinate's cell on a 2¹⁶-wide grid over `[min, max]`.
pub(super) fn grid_position(value: i32, min: i32, max: i32) -> u16 {
    if max <= min {
        return 0;
    }
    let fraction =
        (i64::from(value) - i64::from(min)) as f64 / (i64::from(max) - i64::from(min)) as f64;
    (fraction * f64::from(u16::MAX)).clamp(0.0, f64::from(u16::MAX)) as u16
}

/// A cell's position along the order-16 Hilbert curve (the classic
/// rotate-and-accumulate walk), giving spatially-nearby cells nearby
/// positions.
pub(super) fn hilbert(x: u16, y: u16) -> u64 {
    const N: u64 = 1 << 16;
    let (mut x, mut y) = (u64::from(x), u64::from(y));
    let mut d: u64 = 0;
    let mut s: u64 = N / 2;
    while s > 0 {
        let rx = u64::from(x & s > 0);
        let ry = u64::from(y & s > 0);
        d += s * s * ((3 * rx) ^ ry);
        // Rotate the quadrant so the curve connects.
        if ry == 0 {
            if rx == 1 {
                x = N - 1 - x;
                y = N - 1 - y;
            }
            std::mem::swap(&mut x, &mut y);
        }
        s /= 2;
    }
    d
}
