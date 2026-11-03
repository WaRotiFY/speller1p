"""Download BCI Competition II data set IIb (Wadsworth P300 speller, English words) and convert it.

    uv run --no-sync python stimulus_kit/tools/open_data/get_bci2_iib.py

19 runs (AAS010R01 … AAS012R08, ~95 MB) go to data/bci2_iib/ and become one EDF+,
data/bci2_iib/bci2_iib_AAS.edf. Every file is checked against its SHA-1. The files come
from a copy on GitHub (Ravi-ranjan1801/P300-BCI-Speller); the original is the BCI
Competition II page, https://www.bbci.de/competition/ii/ (data set IIb, Wadsworth Center,
J. R. Wolpaw, G. Schalk, D. Krusienski). If the copy disappears, put the 19 files into
data/bci2_iib/ by hand and run this script again.
"""
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
from bci2_iib_to_edf import convert  # noqa: E402
from get_bnci2014_008 import download  # noqa: E402

SOURCE = "https://github.com/Ravi-ranjan1801/P300-BCI-Speller/raw/HEAD/data/{name}"
SHA1 = {
    "AAS010R01.mat": "9a2846e2007fba324ab7e57581cc9ae8e8d75ce9",
    "AAS010R02.mat": "2d6a0800f40107ae4ecc998e1863bc5c5a27bb80",
    "AAS010R03.mat": "06c21db3b3ec942f00b9db04f2bd47c2497ef633",
    "AAS010R04.mat": "935289f8fb3ca2fcfe909e8d82edd7c3ecffc845",
    "AAS010R05.mat": "18abd2873b95fad9ba3460172ce42075a1609fe3",
    "AAS011R01.mat": "daceba58bde668aa4f33766fbe71fcc974832d97",
    "AAS011R02.mat": "0b2a26e67e3d05a84c4a93b595ff1e370a45c922",
    "AAS011R03.mat": "b394548f61965b4560ddd8c0d0e29f492e0fbc6a",
    "AAS011R04.mat": "f92fb0a22413d764e91768a60e1486aad34aedb0",
    "AAS011R05.mat": "9d9b124ee36eef1fc0eeddd7de71944725c02994",
    "AAS011R06.mat": "c5d4d3108059bd1399197a8195d4524bef2c4072",
    "AAS012R01.mat": "c6a379bd4e7a8dbcaf6fa5b99f48153339dcb585",
    "AAS012R02.mat": "ad9a5f6addad814b104451951514af7a9de0cff7",
    "AAS012R03.mat": "90bd265fefc2b8aeaf92783046de9e5eaecaedbe",
    "AAS012R04.mat": "f892acbe001980de5330b38b8ab4e82f57bcd9c7",
    "AAS012R05.mat": "3f54835dec18890886f17dcb8a0bae94000648a0",
    "AAS012R06.mat": "878a2b3ff2c1e8516f13293c7a3bdd9e7b5e4e76",
    "AAS012R07.mat": "8a6ac4f527ae0a79c4e4553af81db142c1428ffd",
    "AAS012R08.mat": "79c159518a134069390b275cf57128917920fb7d",
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ROOT / "data" / "bci2_iib"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    files = [str(download(name, sha, [SOURCE.format(name=name)], out)) for name, sha in SHA1.items()]
    convert(files, str(out))


if __name__ == "__main__":
    main()
