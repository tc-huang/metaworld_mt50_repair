"""Download, repair, and validate the pinned MetaWorld MT50 snapshot."""

import argparse
from pathlib import Path

from metaworld_mt50_repair.download import download_source
from metaworld_mt50_repair.repair import repair
from metaworld_mt50_repair.validate import verify


def reproduce(output: Path) -> None:
    """Run the complete reproducible repair workflow."""
    source = download_source()
    repair(source, output)
    verify(source, output)


def main() -> None:
    """Run the complete repair workflow from the command line."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True, help="New directory for the replacement files")
    args = parser.parse_args()
    reproduce(args.output)


if __name__ == "__main__":
    main()
