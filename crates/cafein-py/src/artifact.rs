//! The versioned artifact container: save, load, and the
//! memory-mapped adoption path.

use super::*;

#[pymethods]
impl TransportNetwork {
    /// Save the network as a reusable artifact.
    ///
    /// The artifact carries everything queries need — the timetable,
    /// service calendar, transfers, trip distances, leg geometries,
    /// the street network, and any computed accelerators (ULTRA/McULTRA
    /// shortcut sets, walking hierarchy, cached TBTR transfers) — behind
    /// a versioned header, so batch jobs can ``load`` the same file
    /// read-only instead of rebuilding from GTFS and OSM inputs. The payload carries a checksum, so
    /// on-disk corruption is caught at load time. Build diagnostics
    /// (quarantine reports) are not persisted; their warnings belong
    /// to the build. The file is staged beside the destination and
    /// atomically renamed into place, so saving over an artifact never
    /// rewrites it under live mapped readers.
    fn save(&self, py: Python<'_>, path: &str) -> PyResult<()> {
        use std::io::Write;

        let parts = self.streets.as_ref().map(StreetNetwork::to_parts);
        py.allow_threads(|| {
            let (streets_meta, streets_bytes) = match &parts {
                Some(parts) => {
                    let (descriptors, bytes) = encode_streets(parts);
                    (
                        Some(StreetsMeta {
                            vertex_count: parts.vertex_count,
                            links: parts.links.clone(),
                            descriptors,
                        }),
                        bytes,
                    )
                }
                None => (None, Vec::new()),
            };
            let artifact = ArtifactRef {
                feed: &self.feed,
                timetable: &self.build.timetable,
                services: &self.build.services,
                transfers: &self.transfers,
                geometry: &self.geometry,
                leg_geometry: &self.leg_geometry,
                streets: streets_meta,
                ultra_transfers: &self.ultra_transfers,
                ultra_window: self.ultra_window,
                mcultra_transfers: &self.mcultra_transfers,
                mcultra_window: self.mcultra_window,
                mcultra_factor: self.mcultra_factor,
                walking_hierarchy: self.streets.as_ref().and_then(StreetNetwork::hierarchy),
                tbtr_time_transfers: &self.tbtr_time_transfers,
                mctbtr_transfers: &self.mctbtr_transfers,
            };
            let meta = bincode::serialize(&artifact)
                .map_err(|error| PyValueError::new_err(error.to_string()))?;

            // Layout: header | directory | META … pad … | STREETS. The
            // STREETS section starts on `STREETS_ALIGNMENT`, so a mapped
            // load never shares an OS page between the sections; without
            // a street network there is nothing to align (or to pad —
            // padding bytes sit outside every section CRC).
            let version = env!("CARGO_PKG_VERSION").as_bytes();
            let header = 8 + 4 + 2 + version.len() as u64;
            let directory = 4 + 2 * (2 + 8 + 8 + 4) as u64;
            let meta_offset = header + directory;
            let meta_end = meta_offset + meta.len() as u64;
            let streets_offset = if streets_bytes.is_empty() {
                meta_end
            } else {
                meta_end.div_ceil(STREETS_ALIGNMENT) * STREETS_ALIGNMENT
            };

            // Stage into a sibling temp file and atomically rename over
            // the destination: an artifact must never be rewritten in
            // place under live mapped readers, whose mappings keep the
            // replaced inode valid. The name is unique per process and
            // save, and creation is exclusive, so concurrent saves never
            // share a staging path and a stale file or symlink at it
            // fails the save instead of being written through.
            static SAVE_SEQUENCE: std::sync::atomic::AtomicU64 =
                std::sync::atomic::AtomicU64::new(0);
            let sequence = SAVE_SEQUENCE.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let temporary = format!("{path}.tmp-{}-{sequence}", std::process::id());
            let write = || -> PyResult<()> {
                let file = std::fs::OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .open(&temporary)
                    .map_err(io_error)?;
                let mut writer = std::io::BufWriter::new(file);
                writer.write_all(ARTIFACT_MAGIC).map_err(io_error)?;
                writer
                    .write_all(&ARTIFACT_FORMAT.to_le_bytes())
                    .map_err(io_error)?;
                writer
                    .write_all(&(version.len() as u16).to_le_bytes())
                    .map_err(io_error)?;
                writer.write_all(version).map_err(io_error)?;
                writer.write_all(&2u32.to_le_bytes()).map_err(io_error)?;
                for (tag, offset, bytes) in [
                    (SECTION_META, meta_offset, &meta),
                    (SECTION_STREETS, streets_offset, &streets_bytes),
                ] {
                    writer.write_all(&tag.to_le_bytes()).map_err(io_error)?;
                    writer.write_all(&offset.to_le_bytes()).map_err(io_error)?;
                    writer
                        .write_all(&(bytes.len() as u64).to_le_bytes())
                        .map_err(io_error)?;
                    writer
                        .write_all(&crc32(bytes).to_le_bytes())
                        .map_err(io_error)?;
                }
                writer.write_all(&meta).map_err(io_error)?;
                let padding = streets_offset - meta_offset - meta.len() as u64;
                writer
                    .write_all(&vec![0u8; padding as usize])
                    .map_err(io_error)?;
                writer.write_all(&streets_bytes).map_err(io_error)?;
                writer.flush().map_err(io_error)?;
                writer.get_ref().sync_all().map_err(io_error)?;
                // Replacing keeps the destination's permissions, as the
                // old truncate-in-place write did.
                if let Ok(metadata) = std::fs::metadata(path) {
                    writer
                        .get_ref()
                        .set_permissions(metadata.permissions())
                        .map_err(io_error)?;
                }
                std::fs::rename(&temporary, path).map_err(io_error)
            };
            write().inspect_err(|_| {
                let _ = std::fs::remove_file(&temporary);
            })
        })
    }

    /// Load a network saved with ``save``.
    ///
    /// Artifacts written in another format version are refused with a
    /// message naming the writing cafein version, and corrupted
    /// payloads fail their checksum; rebuild from the inputs (or
    /// re-save) with a matching version instead. Artifacts are trusted
    /// input, like pickles: load only files you created.
    ///
    /// ``mmap='auto'`` maps the file and uses the street arrays in
    /// place, falling back to the owned load where mapping is
    /// unavailable; ``'require'`` errors instead of falling back.
    /// ``verify`` toggles the STREETS checksum: default on for owned
    /// loads (the bytes are read anyway), off for mapped loads (the
    /// check would page the whole section in).
    #[staticmethod]
    #[pyo3(signature = (path, mmap = "off", verify = None))]
    fn load(
        py: Python<'_>,
        path: &str,
        mmap: &str,
        verify: Option<bool>,
    ) -> PyResult<TransportNetwork> {
        let mode = match mmap {
            "off" => MmapMode::Off,
            "auto" => MmapMode::Auto,
            "require" => MmapMode::Require,
            other => {
                return Err(PyValueError::new_err(format!(
                    "mmap must be 'off', 'auto', or 'require', not '{other}'"
                )))
            }
        };
        if mode != MmapMode::Off {
            match py.allow_threads(|| load_mapped(path, verify))? {
                Ok(loaded) => return Ok(assemble(loaded)),
                Err(reason) if mode == MmapMode::Require => {
                    return Err(PyValueError::new_err(format!(
                        "'{path}' cannot be memory-mapped ({reason}) and \
                         mmap='require' forbids the owned fallback"
                    )))
                }
                Err(_) => {}
            }
        }
        let loaded = py.allow_threads(|| load_owned(path, verify))?;
        Ok(assemble(loaded))
    }

    /// Whether the street arrays are memory-mapped views of the loaded
    /// artifact.
    #[getter]
    fn mapped(&self) -> bool {
        self.streets.as_ref().is_some_and(StreetNetwork::is_mapped)
    }

    /// STREETS-section bytes the load explicitly read — 0 for a lazy
    /// mapped load. Internal; the laziness tests assert on it.
    #[getter]
    fn _streets_bytes_read(&self) -> u64 {
        self.streets_bytes_read
    }
}
