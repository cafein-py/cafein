#!/usr/bin/env python3

"""Download the shared test datasets into tests/data/.

The datasets are the r5py sample data for the Helsinki region
(https://github.com/r5py/r5py.sampledata.helsinki), pinned by release tag and
SHA-256 so that cafein and r5py test against byte-identical input files.
"""

import hashlib
import pathlib
import shutil
import sys
import time
import urllib.request

DOWNLOAD_ATTEMPTS = 3

BASE_URL = "https://github.com/r5py/r5py.sampledata.helsinki/raw/v1.1.1/data"

DATASETS = {
    "helsinki_gtfs.zip": (
        f"{BASE_URL}/helsinki_gtfs.zip",
        "8ecccde3e76441b47e90c7f311fc57a8d38df92e9ee592e8f440a9b7e3abf228",
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
