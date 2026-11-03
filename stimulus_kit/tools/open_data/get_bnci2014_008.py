"""Download BNCI 2014-008 (P300 speller, ALS patients) and convert it to EDF+ for the speller.

    uv run --no-sync python stimulus_kit/tools/open_data/get_bnci2014_008.py            # A01
    uv run --no-sync python stimulus_kit/tools/open_data/get_bnci2014_008.py 1 2 3 8    # several
    uv run --no-sync python stimulus_kit/tools/open_data/get_bnci2014_008.py all

Files go to data/bnci2014_008/ (A0x.mat, bnci2014_008_A0x.edf). Every .mat is checked
against its SHA-1, so a mirror cannot slip in a different file. Sources, in order:
the BNCI Horizon 2020 database, then a byte-identical copy on GitHub.
License of the data: CC BY-NC-ND 4.0 — Riccio et al. 2013, doi:10.3389/fnhum.2013.00732.
"""
import argparse
import hashlib
import os
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
from bnci2014_008_to_edf import convert  # noqa: E402

SHA1 = {
    1: "7f7281f3e2ce5c785ad36878c87de2bb08fc7009", 2: "d8c1af1a856c478aadaed5fd1fda627122f0fdb4",
    3: "db72718a85a0311864cafec9992eae5eb306360d", 4: "fd033b260639bc5f9718be6c655b68ddceffa4d2",
    5: "72e7adc787992a3f71e978ff6d17d221a84908ec", 6: "466d23aa4d00fc1d099e85a5947da8ad5cebb39d",
    7: "45c276c551889aea5fb9fe42b68fd52a538636b3", 8: "6139fe8ec478dcc0be489e38009e98e34aa91ffb",
}
SOURCES = [
    "http://bnci-horizon-2020.eu/database/data-sets/008-2014/A{n:02d}.mat",
    "https://github.com/AyanaMussabayeva/P300-Speller-BCI-Research/raw/HEAD/BCI_Horizon2020_dataset/A{n:02d}.mat",
]


def sha1(path):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch(n, out):
    path = out / f"A{n:02d}.mat"
    if path.is_file() and sha1(path) == SHA1[n]:
        print(f"A{n:02d}.mat: already here")
        return path
    for url in SOURCES:
        url = url.format(n=n)
        tmp = path.with_suffix(".part")
        try:
            print(f"A{n:02d}.mat ← {url}", flush=True)
            req = urllib.request.Request(url, headers={"User-Agent": "eeg-speller-data-fetch"})
            with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as f:
                while block := r.read(1 << 20):
                    f.write(block)
        except OSError as exc:
            print(f"  failed: {exc}")
            tmp.unlink(missing_ok=True)
            continue
        if sha1(tmp) != SHA1[n]:
            print("  wrong checksum, trying the next source")
            tmp.unlink(missing_ok=True)
            continue
        os.replace(tmp, path)
        return path
    sys.exit(f"A{n:02d}.mat: no source worked; download it by hand into {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("subjects", nargs="*", default=["1"], help="1..8 or 'all' (default 1)")
    ap.add_argument("--out", default=str(ROOT / "data" / "bnci2014_008"))
    a = ap.parse_args()
    subjects = list(SHA1) if a.subjects == ["all"] else [int(s) for s in a.subjects]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for n in subjects:
        convert(str(fetch(n, out)), str(out))


if __name__ == "__main__":
    main()
