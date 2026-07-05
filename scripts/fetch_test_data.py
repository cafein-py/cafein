#!/usr/bin/env python3

"""Download the shared test datasets into tests/data/.

The datasets are the r5py sample data for the Helsinki region
(https://github.com/r5py/r5py.sampledata.helsinki) and r5r's Porto
Alegre fare structure (https://github.com/ipeaGIT/r5r), pinned by
release tag or commit and SHA-256 so that cafein and r5py/r5r test
against byte-identical input files.
"""

import hashlib
import pathlib
import shutil
import sys
import time
import urllib.request

DOWNLOAD_ATTEMPTS = 3

BASE_URL = "https://github.com/r5py/r5py.sampledata.helsinki/raw/v1.1.1/data"

R5R_URL = "https://github.com/ipeaGIT/r5r/raw/eae1aacfc94987bcc06d55e04e43c3d879280d13"

DATASETS = {
    "helsinki_gtfs.zip": (
        f"{BASE_URL}/helsinki_gtfs.zip",
        "8ecccde3e76441b47e90c7f311fc57a8d38df92e9ee592e8f440a9b7e3abf228",
    ),
    "kantakaupunki.osm.pbf": (
        f"{BASE_URL}/kantakaupunki.osm.pbf",
        "94f1a86cb8defaca4b6eea64fba699fde957a848151642b2ad2599bd5ad1e858",
    ),
    "fares_poa.zip": (
        f"{R5R_URL}/r-package/inst/extdata/poa/fares/fares_poa.zip",
        "84cbb9ab01f4f1406aa6c281cd70101b5ad98180a450ad90f304fbdd8a5cc2a0",
    ),
    "poa_eptc.zip": (
        f"{R5R_URL}/r-package/inst/extdata/poa/poa_eptc.zip",
        "90f9e66980efc998d4bf69f13b760b93bb85bf41f884eb6818629cd261a547ba",
    ),
    "poa_trensurb.zip": (
        f"{R5R_URL}/r-package/inst/extdata/poa/poa_trensurb.zip",
        "1c0162e0b7c76604c79037e9e9879f94325989b3775cec854e93fe9473fd99f1",
    ),
    "poa_osm.pbf": (
        f"{R5R_URL}/r-package/inst/extdata/poa/poa_osm.pbf",
        "d0d692b2b13c3ccabb494856966ea40068fbb3b6d5534b70b9234c74f465787d",
    ),
}

DATA_DIRECTORY = pathlib.Path(__file__).parent.parent / "tests" / "data"


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as opened_file:
        for chunk in iter(lambda: opened_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url, destination):
    with urllib.request.urlopen(url) as response, open(destination, "wb") as out:
        shutil.copyfileobj(response, out)


def fetch(file_name, url, expected_sha256):
    destination = DATA_DIRECTORY / file_name
    if destination.exists() and sha256(destination) == expected_sha256:
        print(f"{file_name}: already present")
        return
    for attempt in range(1, DOWNLOAD_ATTEMPTS + 1):
        print(f"{file_name}: downloading from {url} (attempt {attempt})")
        try:
            download(url, destination)
        except OSError as error:
            print(f"{file_name}: download failed ({error})")
        else:
            if sha256(destination) == expected_sha256:
                print(f"{file_name}: ok")
                return
            print(f"{file_name}: checksum mismatch")
        time.sleep(attempt)
    if destination.exists():
        destination.unlink()
    sys.exit(f"{file_name}: could not download a valid copy")


def main():
    DATA_DIRECTORY.mkdir(parents=True, exist_ok=True)
    for file_name, (url, expected_sha256) in DATASETS.items():
        fetch(file_name, url, expected_sha256)


if __name__ == "__main__":
    main()
