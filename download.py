"""Download the pinned source snapshot into the LeRobot Hub cache."""

import argparse
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from metaworld_mt50_repair.repair import SOURCE_REVISION, SPEC


def download_source() -> Path:
    """Return the pinned source snapshot, downloading it when needed."""
    dataset = LeRobotDataset(
        repo_id=SPEC["source_repo_id"],
        revision=SOURCE_REVISION,
        download_videos=False,
        video_backend="pyav",
    )
    return Path(dataset.root)


def main() -> None:
    """Run the source download command."""
    argparse.ArgumentParser().parse_args()
    dataset_root = download_source()
    print(dataset_root)


if __name__ == "__main__":
    main()
