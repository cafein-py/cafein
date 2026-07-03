//! Shared access to the downloaded test datasets.
//!
//! Tests skip (with a notice on stderr) when the data has not been fetched,
//! unless `CAFEIN_REQUIRE_TEST_DATA` is set — CI sets it so a missing
//! fixture fails loudly instead of skipping every test.

use std::path::PathBuf;

pub fn helsinki_gtfs_path() -> Option<PathBuf> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../data/helsinki_gtfs.zip");
    if path.exists() {
        return Some(path);
    }
    let message = format!(
        "test data missing at {}; run `python scripts/fetch_test_data.py`",
        path.display()
    );
    if std::env::var_os("CAFEIN_REQUIRE_TEST_DATA").is_some() {
        panic!("{message}");
    }
    eprintln!("skipping test: {message}");
    None
}
