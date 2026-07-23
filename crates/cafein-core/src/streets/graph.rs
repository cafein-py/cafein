//! Storage (owned or memory-mapped), construction, and the
//! network's structural accessors.

use super::*;

/// An immutable byte store backing mapped street arrays — a read-only
/// memory map of a saved artifact, on the Python side.
///
/// The bytes must never change while any [`StreetNetwork`] holds the
/// store: the network reinterprets them as typed slices, so a mutation
/// would be undefined behaviour, not just a wrong result. Producers
/// replace artifacts by writing a new file and atomically renaming it
/// over the old one — never by editing or truncating in place.
pub trait Backing: Send + Sync + 'static {
    fn bytes(&self) -> &[u8];
}

impl Backing for Vec<u8> {
    fn bytes(&self) -> &[u8] {
        self
    }
}

/// A typed view of a byte range inside a [`Backing`] store, resolved and
/// validated (bounds and alignment) once at adoption.
#[derive(Debug, Clone, Copy)]
pub(super) struct MappedSlice<T> {
    pub(super) ptr: *const T,
    pub(super) len: usize,
}

// The slices point into an immutable `Backing` owned by the same
// `MappedArrays`, so they are as sendable as the store itself.
unsafe impl<T: Send + Sync> Send for MappedSlice<T> {}
unsafe impl<T: Send + Sync> Sync for MappedSlice<T> {}

impl<T> MappedSlice<T> {
    /// Resolves `count` elements at `offset` bytes into `bytes`,
    /// refusing out-of-bounds or misaligned ranges.
    fn new(bytes: &[u8], offset: u64, count: u64) -> Option<MappedSlice<T>> {
        let length = count.checked_mul(std::mem::size_of::<T>() as u64)?;
        let end = offset.checked_add(length)?;
        if end > bytes.len() as u64 {
            return None;
        }
        let ptr = bytes[offset as usize..].as_ptr();
        if !(ptr as usize).is_multiple_of(std::mem::align_of::<T>()) {
            return None;
        }
        Some(MappedSlice {
            ptr: ptr.cast(),
            len: count as usize,
        })
    }

    fn get(&self) -> &[T] {
        // SAFETY: `new` validated bounds and alignment against the
        // backing store, which outlives the slice and never mutates
        // (the `Backing` contract).
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}

/// The persisted street arrays as owned vectors — the build and default
/// load path.
#[derive(Debug)]
pub(super) struct OwnedArrays {
    pub(super) adjacency_offsets: Vec<u32>,
    pub(super) adj_targets: Vec<u32>,
    pub(super) adj_meters: Vec<f64>,
    pub(super) adj_edges: Vec<u32>,
    pub(super) endpoints: Vec<u32>,
    pub(super) lengths: Vec<f64>,
    pub(super) coordinate_offsets: Vec<u32>,
    pub(super) lons: Vec<i32>,
    pub(super) lats: Vec<i32>,
    pub(super) cumulative: Vec<f32>,
    pub(super) index_boxes: Vec<Envelope>,
    pub(super) index_payload: Vec<u32>,
}

/// The persisted street arrays as typed views into a mapped artifact.
pub(super) struct MappedArrays {
    /// Keeps the mapping alive; every slice below points into it.
    pub(super) _backing: std::sync::Arc<dyn Backing>,
    pub(super) adjacency_offsets: MappedSlice<u32>,
    pub(super) adj_targets: MappedSlice<u32>,
    pub(super) adj_meters: MappedSlice<f64>,
    pub(super) adj_edges: MappedSlice<u32>,
    pub(super) endpoints: MappedSlice<u32>,
    pub(super) lengths: MappedSlice<f64>,
    pub(super) coordinate_offsets: MappedSlice<u32>,
    pub(super) lons: MappedSlice<i32>,
    pub(super) lats: MappedSlice<i32>,
    pub(super) cumulative: MappedSlice<f32>,
    pub(super) index_boxes: MappedSlice<Envelope>,
    pub(super) index_payload: MappedSlice<u32>,
}

impl std::fmt::Debug for MappedArrays {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MappedArrays")
            .field("lons", &self.lons.len)
            .field("endpoints", &self.endpoints.len)
            .finish_non_exhaustive()
    }
}

/// The street arrays behind either backing. Query code reads through the
/// slice accessors and never touches a concrete field.
#[derive(Debug)]
pub(super) enum Arrays {
    Owned(OwnedArrays),
    Mapped(MappedArrays),
}

macro_rules! array_accessor {
    ($name:ident, $type:ty) => {
        pub(super) fn $name(&self) -> &[$type] {
            match self {
                Arrays::Owned(arrays) => &arrays.$name,
                Arrays::Mapped(arrays) => arrays.$name.get(),
            }
        }
    };
}

impl Arrays {
    array_accessor!(adjacency_offsets, u32);
    array_accessor!(adj_targets, u32);
    array_accessor!(adj_meters, f64);
    array_accessor!(adj_edges, u32);
    array_accessor!(endpoints, u32);
    array_accessor!(lengths, f64);
    array_accessor!(coordinate_offsets, u32);
    array_accessor!(lons, i32);
    array_accessor!(lats, i32);
    array_accessor!(cumulative, f32);
    array_accessor!(index_boxes, Envelope);
    array_accessor!(index_payload, u32);
}

/// The optional multimodal edge attributes: per-adjacency-slot mode
/// permissions and facility flags, and per-physical-edge class codes. All
/// six arrays are present together (a multimodal build) or the group is
/// absent (walk-only). Always owned — small relative to the CSR, and kept
/// out of the mapped path so it needs no `u8`/`u16` slice mapping yet.
#[derive(Debug, Clone, PartialEq)]
pub struct StreetAttributes {
    /// Mode-permission bits per adjacency slot (`2·edges`).
    pub adj_access: Vec<u8>,
    /// Facility/directional flags per adjacency slot (`2·edges`).
    pub adj_facility: Vec<u8>,
    /// Highway-class code per physical edge.
    pub edge_highway: Vec<u8>,
    /// Surface code per physical edge.
    pub edge_surface: Vec<u8>,
    /// Smoothness code per physical edge.
    pub edge_smoothness: Vec<u8>,
    /// Packed edge flags per physical edge.
    pub edge_flags: Vec<u16>,
}

/// Where each street array lives inside a mapped artifact: byte offsets
/// into the backing store plus element counts, as the artifact's
/// descriptor table records them. The index level starts are not mapped —
/// they are a pure function of the leaf count and are recomputed. The
/// optional multimodal arrays are decoded owned (see [`StreetAttributes`]).
pub struct MappedStreets {
    pub backing: std::sync::Arc<dyn Backing>,
    pub vertex_count: u32,
    pub links: Vec<StoredLink>,
    /// `(byte offset, element count)` per array; offsets are absolute
    /// within the backing bytes.
    pub adjacency_offsets: (u64, u64),
    pub adj_targets: (u64, u64),
    pub adj_meters: (u64, u64),
    pub adj_edges: (u64, u64),
    pub endpoints: (u64, u64),
    pub lengths: (u64, u64),
    pub coordinate_offsets: (u64, u64),
    pub lons: (u64, u64),
    pub lats: (u64, u64),
    pub cumulative: (u64, u64),
    pub index_boxes: (u64, u64),
    pub index_payload: (u64, u64),
    /// The optional multimodal attributes, decoded owned by the loader when
    /// the artifact carries them, else `None` (a walk-only artifact).
    pub attributes: Option<StreetAttributes>,
    /// The optional per-coordinate elevations, decoded owned, else `None`.
    pub elevations: Option<Vec<f32>>,
}

/// The walking street graph with its spatial index and stop links.
///
/// The graph-owned data (CSR, geometry, spatial index, and the run-once
/// contraction) is separated from the GTFS-derived stop links: a later
/// street-only network can hold the [`StreetGraph`] without any timetable,
/// while a `TransportNetwork` pairs it with its [`StopLinks`]. Every existing
/// method reads through the two members as a façade, so the public API is
/// unchanged.
#[derive(Debug)]
pub struct StreetNetwork {
    pub(super) graph: StreetGraph,
    pub(super) stop_links: StopLinks,
}

/// The graph-owned street data. None of it depends on the GTFS stops, so it
/// can back a timetable-free network: the persisted CSR and geometry arrays
/// behind [`Arrays`] (owned vectors or typed views into a mapped artifact;
/// queries are identical over both), the spatial-index level starts, the
/// lazily computed walking symmetry, and the optional run-once contraction.
#[derive(Debug)]
pub(super) struct StreetGraph {
    pub(super) arrays: Arrays,
    /// Start of each level in the index boxes (leaves at 0), plus a
    /// tail; derived from the leaf count, tiny, and always owned.
    pub(super) level_starts: Vec<u32>,
    /// An optional contraction hierarchy accelerating the bounded one-to-many
    /// walking searches (`access_stops`/`stop_transfers`/…). Built on demand by
    /// [`install_hierarchy`](StreetNetwork::install_hierarchy); when absent the
    /// searches use `bounded_dijkstra`. Graph-level state that persists with the
    /// artifact; its stop-side buckets are derived and live on [`StopLinks`],
    /// present exactly when [`StopLinks::buckets`] is (`install_hierarchy_from`
    /// sets both, so [`StreetNetwork::ch`] reads them as a pair).
    pub(super) contraction: Option<crate::ch::ContractionHierarchy>,
    /// Whether the walking adjacency is symmetric, so a search from a stop
    /// gives the same distances as a search to it — the invariant that lets
    /// [`link_pointsets`](StreetNetwork::link_pointsets) link from the stop
    /// side. Computed once, lazily; walking is undirected in the OSM
    /// extraction, so it holds, but an asymmetric graph is marked ineligible
    /// and falls back to the per-point search.
    pub(super) symmetric: std::sync::OnceLock<bool>,
    /// The optional multimodal edge attributes (mode permissions, facility
    /// flags, class codes). `None` on a walk-only build; the current routing
    /// never reads them. Persisted with the artifact when present.
    pub(super) attributes: Option<StreetAttributes>,
    /// The optional per-coordinate elevations, aligned one-to-one with the
    /// geometry `lons`/`lats`/`cumulative`. `None` unless elevation was
    /// enabled; persisted with the artifact when present.
    pub(super) elevations: Option<Vec<f32>>,
}

/// The GTFS-derived stop links: how each snapped stop enters the graph, the
/// vertex→link index, and the contraction's stop-side one-to-many buckets.
/// All of it is a function of the stops, so a street-only graph carries none.
#[derive(Debug)]
pub(super) struct StopLinks {
    /// How each snapped stop enters the graph, endpoints denormalised.
    pub(super) links: Vec<StoredLink>,
    /// `(vertex, link index)` pairs sorted by vertex — every link listed
    /// under both endpoints of its edge — so a search finds the links
    /// near its reached vertices without scanning all links.
    pub(super) vertex_links: Vec<(u32, u32)>,
    /// The contraction's **unbounded** one-to-many buckets over the stops'
    /// link-endpoint vertices (so a query at any finite cutoff is within the
    /// buckets' build cutoff) — the acceleration index for
    /// `reachable_from_snaps`. Derived state rebuilt on load, present exactly
    /// when [`StreetGraph::contraction`] is.
    pub(super) buckets: Option<crate::ch::Buckets>,
}

impl StreetNetwork {
    /// Builds the network from flat edge arrays and stop links.
    ///
    /// `edges` carries `(from, to, meters)` per edge; edge `i`'s geometry
    /// runs from its `from` vertex through coordinates
    /// `coordinate_offsets[i]..coordinate_offsets[i + 1]` of the
    /// longitude/latitude arrays.
    pub fn new(
        vertex_count: u32,
        stop_count: u32,
        edges: &[(u32, u32, f64)],
        coordinate_offsets: &[u32],
        longitudes: &[f64],
        latitudes: &[f64],
        links: Vec<StopLink>,
    ) -> Result<StreetNetwork, StreetError> {
        if coordinate_offsets.len() != edges.len() + 1
            || coordinate_offsets.first() != Some(&0)
            || coordinate_offsets.last() != Some(&(longitudes.len() as u32))
            || longitudes.len() != latitudes.len()
        {
            return Err(StreetError::InvalidOffsets);
        }
        for (index, &(from, to, meters)) in edges.iter().enumerate() {
            let edge = index as u32;
            let start = coordinate_offsets[index];
            let end = coordinate_offsets[index + 1];
            if end < start || longitudes.len() < end as usize {
                return Err(StreetError::InvalidOffsets);
            }
            if end - start < 2 {
                return Err(StreetError::ShortGeometry { edge });
            }
            for position in start as usize..end as usize {
                if !longitudes[position].is_finite() || !latitudes[position].is_finite() {
                    return Err(StreetError::InvalidCoordinates { edge });
                }
            }
            if from >= vertex_count || to >= vertex_count {
                return Err(StreetError::VertexOutOfRange { edge, vertex_count });
            }
            if !meters.is_finite() || meters < 0.0 {
                return Err(StreetError::InvalidLength { edge });
            }
        }
        for (index, link) in links.iter().enumerate() {
            if link.edge as usize >= edges.len() {
                return Err(StreetError::LinkEdgeOutOfRange {
                    link: index,
                    edge_count: edges.len() as u32,
                });
            }
            if link.stop.0 >= stop_count {
                return Err(StreetError::StopOutOfRange {
                    stop: link.stop.0,
                    stop_count,
                });
            }
            if !(0.0..=1.0).contains(&link.fraction)
                || !link.connector.is_finite()
                || link.connector < 0.0
            {
                return Err(StreetError::InvalidLink { link: index });
            }
        }

        // Quantize the geometry onto the fixed-point grid up front, so
        // every derived structure — permutation keys, densified points,
        // cumulative lengths, index boxes — is a pure function of the
        // stored coordinates.
        let fixed_lons: Vec<i32> = longitudes.iter().map(|&lon| quantize(lon)).collect();
        let fixed_lats: Vec<i32> = latitudes.iter().map(|&lat| quantize(lat)).collect();

        // Hilbert-order the edges by their first coordinate and renumber
        // vertices by first appearance in that order, so spatially-nearby
        // streets are nearby in every edge- and vertex-indexed array. The
        // ids are internal; costs and results are unchanged (only
        // exactly-equal ties could resolve differently than under the
        // input order). Hilbert-cell ties break by the edges' own data —
        // endpoints, length, vertices, then the full geometry — never by
        // the input position, so the layout is the same whatever order the
        // edges arrive in (edges identical in every field are
        // interchangeable, and their links follow them either way).
        let bounds = coordinate_bounds(&fixed_lons, &fixed_lats);
        let mut order: Vec<u32> = (0..edges.len() as u32).collect();
        type EdgeKey = (u64, i32, i32, i32, i32, u64, u32, u32);
        let keys: Vec<EdgeKey> = (0..edges.len())
            .map(|edge| {
                let first = coordinate_offsets[edge] as usize;
                let last = coordinate_offsets[edge + 1] as usize - 1;
                let (from, to, meters) = edges[edge];
                (
                    hilbert(
                        grid_position(fixed_lons[first], bounds[0], bounds[2]),
                        grid_position(fixed_lats[first], bounds[1], bounds[3]),
                    ),
                    fixed_lons[first],
                    fixed_lats[first],
                    fixed_lons[last],
                    fixed_lats[last],
                    meters.to_bits(),
                    from,
                    to,
                )
            })
            .collect();
        let geometry_points = |edge: u32| {
            let start = coordinate_offsets[edge as usize] as usize;
            let end = coordinate_offsets[edge as usize + 1] as usize;
            fixed_lons[start..end]
                .iter()
                .zip(&fixed_lats[start..end])
                .map(|(&lon, &lat)| (lon, lat))
        };
        order.sort_unstable_by(|&a, &b| {
            keys[a as usize]
                .cmp(&keys[b as usize])
                .then_with(|| geometry_points(a).cmp(geometry_points(b)))
        });

        let mut edge_map = vec![0u32; edges.len()];
        let mut vertex_map = vec![u32::MAX; vertex_count as usize];
        let mut next_vertex = 0u32;
        let mut permuted_edges = Vec::with_capacity(edges.len());
        let mut permuted_offsets = Vec::with_capacity(edges.len() + 1);
        let mut permuted_lons = Vec::with_capacity(fixed_lons.len());
        let mut permuted_lats = Vec::with_capacity(fixed_lats.len());
        permuted_offsets.push(0u32);
        for (new_edge, &old_edge) in order.iter().enumerate() {
            edge_map[old_edge as usize] = new_edge as u32;
            let (from, to, meters) = edges[old_edge as usize];
            let mut renumber = |vertex: u32| {
                if vertex_map[vertex as usize] == u32::MAX {
                    vertex_map[vertex as usize] = next_vertex;
                    next_vertex += 1;
                }
                vertex_map[vertex as usize]
            };
            permuted_edges.push((renumber(from), renumber(to), meters));
            let start = coordinate_offsets[old_edge as usize] as usize;
            let end = coordinate_offsets[old_edge as usize + 1] as usize;
            permuted_lons.extend_from_slice(&fixed_lons[start..end]);
            permuted_lats.extend_from_slice(&fixed_lats[start..end]);
            permuted_offsets.push(permuted_lons.len() as u32);
        }
        // Vertices no edge touches keep ids after the connected ones.
        for slot in vertex_map.iter_mut() {
            if *slot == u32::MAX {
                *slot = next_vertex;
                next_vertex += 1;
            }
        }
        let edges = permuted_edges;
        let links: Vec<StoredLink> = links
            .into_iter()
            .map(|link| {
                let edge = edge_map[link.edge as usize];
                let (from, to, _) = edges[edge as usize];
                StoredLink {
                    stop: link.stop,
                    edge,
                    fraction: link.fraction,
                    connector: link.connector,
                    from,
                    to,
                }
            })
            .collect();

        let (dense_offsets, lons, lats, cumulative) =
            densify(&permuted_offsets, &permuted_lons, &permuted_lats);

        let mut adjacency_offsets = vec![0u32; vertex_count as usize + 1];
        for &(from, to, _) in &edges {
            adjacency_offsets[from as usize + 1] += 1;
            adjacency_offsets[to as usize + 1] += 1;
        }
        for vertex in 0..vertex_count as usize {
            adjacency_offsets[vertex + 1] += adjacency_offsets[vertex];
        }
        let mut adj_targets = vec![0u32; edges.len() * 2];
        let mut adj_meters = vec![0f64; edges.len() * 2];
        let mut adj_edges = vec![0u32; edges.len() * 2];
        let mut cursor = adjacency_offsets.clone();
        for (index, &(from, to, meters)) in edges.iter().enumerate() {
            for (a, b) in [(from, to), (to, from)] {
                let slot = cursor[a as usize] as usize;
                adj_targets[slot] = b;
                adj_meters[slot] = meters;
                adj_edges[slot] = index as u32;
                cursor[a as usize] += 1;
            }
        }

        let index = build_index(&dense_offsets, &lons, &lats);
        let endpoints: Vec<u32> = edges.iter().flat_map(|&(from, to, _)| [from, to]).collect();
        let vertex_links = build_vertex_links(&links);

        let graph = StreetGraph {
            arrays: Arrays::Owned(OwnedArrays {
                adjacency_offsets,
                adj_targets,
                adj_meters,
                adj_edges,
                endpoints,
                lengths: edges.iter().map(|&(_, _, meters)| meters).collect(),
                coordinate_offsets: dense_offsets,
                lons,
                lats,
                cumulative,
                index_boxes: index.boxes,
                index_payload: index.payload,
            }),
            level_starts: index.level_starts,
            contraction: None,
            symmetric: std::sync::OnceLock::new(),
            attributes: None,
            elevations: None,
        };
        Ok(StreetNetwork {
            graph,
            stop_links: StopLinks {
                links,
                vertex_links,
                buckets: None,
            },
        })
    }

    /// The graph-owned CSR and geometry arrays.
    pub(super) fn arrays(&self) -> &Arrays {
        &self.graph.arrays
    }

    /// The spatial-index level starts.
    pub(super) fn level_starts(&self) -> &[u32] {
        &self.graph.level_starts
    }

    /// The stop links.
    pub(super) fn links(&self) -> &[StoredLink] {
        &self.stop_links.links
    }

    /// The vertex→link index.
    pub(super) fn vertex_links(&self) -> &[(u32, u32)] {
        &self.stop_links.vertex_links
    }

    /// The installed contraction and its stop-side buckets, when both are
    /// present — they always are together, since `install_hierarchy_from` sets
    /// the pair. The bounded one-to-many walking search runs over them.
    pub(super) fn ch(&self) -> Option<(&crate::ch::ContractionHierarchy, &crate::ch::Buckets)> {
        match (&self.graph.contraction, &self.stop_links.buckets) {
            (Some(contraction), Some(buckets)) => Some((contraction, buckets)),
            _ => None,
        }
    }

    /// The installed multimodal edge attributes, when present.
    pub fn street_attributes(&self) -> Option<&StreetAttributes> {
        self.graph.attributes.as_ref()
    }

    /// The installed per-coordinate elevations, when present.
    pub fn elevations(&self) -> Option<&[f32]> {
        self.graph.elevations.as_deref()
    }

    /// Attaches multimodal edge attributes to the graph, replacing any
    /// installed set. Every array must match the graph's shape: the two
    /// adjacency-slot arrays span `2·edges`, the four per-edge arrays span
    /// `edges`; otherwise the attributes are rejected and none are installed.
    pub fn install_street_attributes(
        &mut self,
        attributes: StreetAttributes,
    ) -> Result<(), StreetError> {
        let slots = 2 * self.edge_count() as usize;
        let edges = self.edge_count() as usize;
        if attributes.adj_access.len() != slots
            || attributes.adj_facility.len() != slots
            || attributes.edge_highway.len() != edges
            || attributes.edge_surface.len() != edges
            || attributes.edge_smoothness.len() != edges
            || attributes.edge_flags.len() != edges
        {
            return Err(StreetError::InvalidAttributes);
        }
        self.graph.attributes = Some(attributes);
        Ok(())
    }

    /// Attaches per-coordinate elevations, replacing any installed set. The
    /// array must have one value per stored geometry coordinate.
    pub fn install_elevations(&mut self, elevations: Vec<f32>) -> Result<(), StreetError> {
        if elevations.len() != self.arrays().lons().len() {
            return Err(StreetError::InvalidAttributes);
        }
        self.graph.elevations = Some(elevations);
        Ok(())
    }

    /// Number of street vertices.
    pub fn vertex_count(&self) -> u32 {
        self.arrays().adjacency_offsets().len() as u32 - 1
    }

    /// Number of street edges.
    pub fn edge_count(&self) -> u32 {
        (self.arrays().endpoints().len() / 2) as u32
    }

    /// Number of stored geometry coordinates (the elevation array's length).
    pub fn coordinate_count(&self) -> u32 {
        self.arrays().lons().len() as u32
    }

    /// Number of stop links.
    pub fn link_count(&self) -> usize {
        self.stop_links.links.len()
    }

    /// Whether the arrays are views into a mapped artifact.
    pub fn is_mapped(&self) -> bool {
        matches!(self.graph.arrays, Arrays::Mapped(_))
    }

    /// Whether a contraction-hierarchy index is installed.
    pub fn has_hierarchy(&self) -> bool {
        self.graph.contraction.is_some()
    }

    /// Builds and installs a contraction hierarchy over the walking graph, plus
    /// **unbounded** one-to-many buckets over the stops' link-endpoint vertices,
    /// so the bounded walking searches (`access_stops`/`stop_transfers`/…) run as
    /// hierarchy queries instead of graph sweeps. Heavy (the contraction is
    /// run-once); opt-in, so the default build path is unchanged. Idempotent —
    /// rebuilding replaces the index. The contraction persists with the artifact,
    /// so acceleration survives `save`/`load` without a second contraction.
    pub fn install_hierarchy(&mut self) {
        let hierarchy = crate::ch::ContractionHierarchy::build(
            self.vertex_count(),
            self.arrays().adjacency_offsets(),
            self.arrays().adj_targets(),
            self.arrays().adj_meters(),
        );
        self.install_hierarchy_from(hierarchy);
    }

    /// Installs a prebuilt contraction hierarchy, deriving its one-to-many
    /// buckets from the stops' link-endpoint vertices. Used to restore a
    /// persisted hierarchy on `load`: the run-once contraction is deserialised
    /// and only the buckets — derived state — are rebuilt, exactly as
    /// [`install_hierarchy`](Self::install_hierarchy) builds them for a fresh
    /// contraction, so a loaded network matches a freshly built one.
    pub fn install_hierarchy_from(&mut self, hierarchy: crate::ch::ContractionHierarchy) {
        let mut endpoints: Vec<u32> = self
            .stop_links
            .links
            .iter()
            .flat_map(|link| [link.from, link.to])
            .collect();
        endpoints.sort_unstable();
        endpoints.dedup();
        let buckets = hierarchy.buckets(&endpoints, f64::INFINITY);
        self.graph.contraction = Some(hierarchy);
        self.stop_links.buckets = Some(buckets);
    }

    /// The installed contraction hierarchy, if any — the run-once contraction
    /// result. Persisted with the artifact; the buckets are rebuilt on load.
    pub fn hierarchy(&self) -> Option<&crate::ch::ContractionHierarchy> {
        self.graph.contraction.as_ref()
    }

    /// A fingerprint of this network's walking-graph CSR, matching what a
    /// hierarchy built over it records: a persisted hierarchy binds to this so a
    /// loaded artifact with a mismatched graph is refused.
    pub fn graph_fingerprint(&self) -> u64 {
        crate::ch::csr_fingerprint(
            self.arrays().adjacency_offsets(),
            self.arrays().adj_targets(),
            self.arrays().adj_meters(),
        )
    }

    /// An edge's `(from, to)` endpoint vertices.
    pub(super) fn edge_endpoints(&self, edge: u32) -> (u32, u32) {
        let endpoints = self.arrays().endpoints();
        (
            endpoints[2 * edge as usize],
            endpoints[2 * edge as usize + 1],
        )
    }

    /// A stored coordinate as float degrees.
    pub(super) fn coordinate(&self, position: usize) -> (f64, f64) {
        (
            degrees(self.arrays().lons()[position]),
            degrees(self.arrays().lats()[position]),
        )
    }

    /// A stored cumulative along-distance as f64 meters.
    pub(super) fn along(&self, position: usize) -> f64 {
        f64::from(self.arrays().cumulative()[position])
    }

    /// The network's serializable state.
    pub fn to_parts(&self) -> StreetNetworkParts {
        StreetNetworkParts {
            vertex_count: self.vertex_count(),
            adjacency_offsets: self.arrays().adjacency_offsets().to_vec(),
            adj_targets: self.arrays().adj_targets().to_vec(),
            adj_meters: self.arrays().adj_meters().to_vec(),
            adj_edges: self.arrays().adj_edges().to_vec(),
            endpoints: self.arrays().endpoints().to_vec(),
            lengths: self.arrays().lengths().to_vec(),
            coordinate_offsets: self.arrays().coordinate_offsets().to_vec(),
            lons: self.arrays().lons().to_vec(),
            lats: self.arrays().lats().to_vec(),
            cumulative: self.arrays().cumulative().to_vec(),
            index_boxes: self
                .arrays()
                .index_boxes()
                .iter()
                .flat_map(|envelope| *envelope)
                .collect(),
            index_payload: self.arrays().index_payload().to_vec(),
            index_level_starts: self.graph.level_starts.clone(),
            links: self.stop_links.links.clone(),
            attributes: self.graph.attributes.clone(),
            elevations: self.graph.elevations.clone(),
        }
    }

    /// Adopts a network from its serialized parts — nothing street-sized
    /// is rebuilt (the spatial index arrives as arrays); the one derived
    /// rebuild is the L-sized vertex→link index, from the links'
    /// denormalised endpoints.
    pub fn from_parts(parts: StreetNetworkParts) -> StreetNetwork {
        let vertex_links = build_vertex_links(&parts.links);
        StreetNetwork {
            graph: StreetGraph {
                arrays: Arrays::Owned(OwnedArrays {
                    adjacency_offsets: parts.adjacency_offsets,
                    adj_targets: parts.adj_targets,
                    adj_meters: parts.adj_meters,
                    adj_edges: parts.adj_edges,
                    endpoints: parts.endpoints,
                    lengths: parts.lengths,
                    coordinate_offsets: parts.coordinate_offsets,
                    lons: parts.lons,
                    lats: parts.lats,
                    cumulative: parts.cumulative,
                    index_boxes: parts
                        .index_boxes
                        .chunks_exact(4)
                        .map(|chunk| [chunk[0], chunk[1], chunk[2], chunk[3]])
                        .collect(),
                    index_payload: parts.index_payload,
                }),
                level_starts: parts.index_level_starts,
                contraction: None,
                symmetric: std::sync::OnceLock::new(),
                attributes: parts.attributes,
                elevations: parts.elevations,
            },
            stop_links: StopLinks {
                links: parts.links,
                vertex_links,
                buckets: None,
            },
        }
    }

    /// Adopts a network whose arrays stay typed views into a mapped
    /// artifact — no street-sized bytes are read or copied. The caller
    /// (the artifact loader) has validated the layout against the
    /// descriptor table; the bounds and alignment checks here are the
    /// soundness net for constructing the slices, and content stays as
    /// trusted as the mapping (see [`Backing`]).
    pub fn from_mapped(spec: MappedStreets) -> Result<StreetNetwork, StreetError> {
        let bytes = spec.backing.bytes();
        fn slice<T>(
            bytes: &[u8],
            (offset, count): (u64, u64),
        ) -> Result<MappedSlice<T>, StreetError> {
            MappedSlice::new(bytes, offset, count).ok_or(StreetError::InvalidMapping)
        }
        // Envelopes are stored as flat i32 quadruples.
        if !spec.index_boxes.1.is_multiple_of(4) {
            return Err(StreetError::InvalidMapping);
        }
        let arrays = MappedArrays {
            adjacency_offsets: slice(bytes, spec.adjacency_offsets)?,
            adj_targets: slice(bytes, spec.adj_targets)?,
            adj_meters: slice(bytes, spec.adj_meters)?,
            adj_edges: slice(bytes, spec.adj_edges)?,
            endpoints: slice(bytes, spec.endpoints)?,
            lengths: slice(bytes, spec.lengths)?,
            coordinate_offsets: slice(bytes, spec.coordinate_offsets)?,
            lons: slice(bytes, spec.lons)?,
            lats: slice(bytes, spec.lats)?,
            cumulative: slice(bytes, spec.cumulative)?,
            index_boxes: slice(bytes, (spec.index_boxes.0, spec.index_boxes.1 / 4))?,
            index_payload: slice(bytes, spec.index_payload)?,
            _backing: spec.backing,
        };
        let level_starts = level_starts_for(arrays.index_payload.len / 2);
        if *level_starts.last().unwrap() as usize != arrays.index_boxes.len {
            return Err(StreetError::InvalidMapping);
        }
        let vertex_links = build_vertex_links(&spec.links);
        Ok(StreetNetwork {
            graph: StreetGraph {
                arrays: Arrays::Mapped(arrays),
                level_starts,
                contraction: None,
                symmetric: std::sync::OnceLock::new(),
                attributes: spec.attributes,
                elevations: spec.elevations,
            },
            stop_links: StopLinks {
                links: spec.links,
                vertex_links,
                buckets: None,
            },
        })
    }
}
/// A [`StreetNetwork`]'s state split for serialization: the flat POD
/// arrays a container stores raw, plus the small link records and scalars
/// it stores decoded. The spatial index rides along as plain arrays;
/// nothing is rebuilt on adoption except the L-sized vertex→link index.
#[derive(Debug, PartialEq)]
pub struct StreetNetworkParts {
    pub vertex_count: u32,
    pub adjacency_offsets: Vec<u32>,
    pub adj_targets: Vec<u32>,
    pub adj_meters: Vec<f64>,
    pub adj_edges: Vec<u32>,
    /// Two entries (`from`, `to`) per edge.
    pub endpoints: Vec<u32>,
    pub lengths: Vec<f64>,
    pub coordinate_offsets: Vec<u32>,
    /// Fixed-point degrees × 10⁷.
    pub lons: Vec<i32>,
    pub lats: Vec<i32>,
    pub cumulative: Vec<f32>,
    /// Flattened index boxes, four entries per node.
    pub index_boxes: Vec<i32>,
    /// Flattened leaf payloads, two entries per leaf.
    pub index_payload: Vec<u32>,
    pub index_level_starts: Vec<u32>,
    pub links: Vec<StoredLink>,
    /// The optional multimodal edge attributes, present on a multimodal build.
    pub attributes: Option<StreetAttributes>,
    /// The optional per-coordinate elevations, present when elevation is on.
    pub elevations: Option<Vec<f32>>,
}

/// The `(vertex, link index)` pairs behind [`StreetNetwork::links_at`],
/// sorted by vertex: each link listed under both endpoints of its edge,
/// once when they coincide. The endpoints come from the links themselves,
/// so this rebuilds without touching any street-sized array.
pub(super) fn build_vertex_links(links: &[StoredLink]) -> Vec<(u32, u32)> {
    let mut vertex_links = Vec::with_capacity(links.len() * 2);
    for (index, link) in links.iter().enumerate() {
        vertex_links.push((link.from, index as u32));
        if link.to != link.from {
            vertex_links.push((link.to, index as u32));
        }
    }
    vertex_links.sort_unstable();
    vertex_links
}
