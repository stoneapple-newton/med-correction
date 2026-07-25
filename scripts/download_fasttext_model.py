"""Download the official compressed fastText lid.176 language-ID model."""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path, nargs="?", default=Path("data/models/lid.176.ftz"))
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(URL, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
