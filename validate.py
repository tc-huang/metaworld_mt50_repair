"""Verify the MetaWorld MT50 replacement files against their source snapshot."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from lerobot.datasets.compute_stats import aggregate_feature_stats
from metaworld_mt50_repair.repair import (
    COLLIDING_TASK_INDEX,
    EPISODE_INDEX_STAT_COLUMNS,
    EPISODES_PATH,
    MANIFEST_PATH,
    METADATA_PATHS,
    NEW_TASK,
    OLD_TASK,
    PROJECT_ROOT,
    PROVENANCE_PATHS,
    PUSH_BACK_TASK_ID,
    PUSH_TASK_ID,
    REPAIRED_TASK_INDEX,
    SOURCE_REPO_ID,
    SOURCE_REVISION,
    SPEC,
    inspect_source,
    read_json,
    sha256,
)

AUDITED_TASK_FRAMES = {PUSH_BACK_TASK_ID: 8_888, PUSH_TASK_ID: 3_521}
REPAIRED_TASK_INDICES = {
    PUSH_BACK_TASK_ID: REPAIRED_TASK_INDEX,
    PUSH_TASK_ID: COLLIDING_TASK_INDEX,
}


def verify_manifest(source: Path, output: Path, affected: list[Path], expected_measured: dict) -> None:
    """Verify the repair manifest, output paths, and before/after hashes."""
    manifest = read_json(output / MANIFEST_PATH)
    expected_paths = set(affected) | set(METADATA_PATHS)
    provenance_files = {str(path.relative_to(PROJECT_ROOT)): sha256(path) for path in PROVENANCE_PATHS}
    if (
        manifest["source_repo_id"] != SOURCE_REPO_ID
        or manifest["source_revision"] != SOURCE_REVISION
        or manifest["measured"] != expected_measured
        or manifest["provenance"]["files"] != provenance_files
        or set(manifest["files"]) != {str(path) for path in expected_paths}
    ):
        raise ValueError("Repair manifest does not match the expected file set")
    if {path.relative_to(output) for path in output.rglob("*") if path.is_file()} != expected_paths | {
        MANIFEST_PATH
    }:
        raise ValueError("Repair output contains missing or extra files")
    for path in expected_paths:
        hashes = manifest["files"][str(path)]
        if (
            sha256(source / path) != hashes["source_sha256"]
            or sha256(output / path) != hashes["output_sha256"]
        ):
            raise ValueError(f"File hash mismatch: {path}")


def _verify_replacement(before: pa.Table, after: pa.Table, path: Path) -> tuple[np.ndarray, int]:
    """Verify one replacement changes task_index only for push-back rows."""
    same_schema = before.schema.equals(after.schema, check_metadata=True)
    if not same_schema or not before.drop(["task_index"]).equals(after.drop(["task_index"])):
        raise ValueError(f"Non-task data changed: {path}")

    task_ids = before["task_id"].to_numpy(zero_copy_only=False)
    before_indices = before["task_index"].to_numpy(zero_copy_only=False)
    after_indices = after["task_index"].to_numpy(zero_copy_only=False)
    expected_rows = task_ids == PUSH_BACK_TASK_ID
    changed_rows = before_indices != after_indices
    if not np.array_equal(changed_rows, expected_rows):
        raise ValueError(f"Unexpected task indices: {path}")
    return after_indices, int(changed_rows.sum())


def _repaired_task_counts(task_ids: np.ndarray, task_indices: np.ndarray) -> dict[int, int]:
    """Return audited task counts after verifying their repaired task indices."""
    counts = {}
    for task_id, expected_index in REPAIRED_TASK_INDICES.items():
        rows = task_ids == task_id
        if np.any(task_indices[rows] != expected_index):
            raise ValueError(f"Repaired task_id={task_id} has an unexpected task_index")
        counts[task_id] = int(rows.sum())
    return counts


def verify_parquets(source: Path, output: Path, affected: list[Path]) -> tuple[int, set[int]]:
    """Verify audited frame counts and that replacements change only the intended task index."""
    repaired_counts = dict.fromkeys(AUDITED_TASK_FRAMES, 0)
    affected_paths = set(affected)
    derived_affected_paths: set[Path] = set()
    push_back_episodes: set[int] = set()
    changed = 0

    for source_path in (source / "data").rglob("*.parquet"):
        path = source_path.relative_to(source)
        before = pq.read_table(
            source_path,
            columns=None if path in affected_paths else ["task_id", "task_index", "episode_index"],
        )
        task_ids = before["task_id"].to_numpy(zero_copy_only=False)
        task_indices = before["task_index"].to_numpy(zero_copy_only=False)
        push_back_rows = task_ids == PUSH_BACK_TASK_ID

        if np.any(push_back_rows):
            derived_affected_paths.add(path)
            push_back_episodes.update(
                before["episode_index"].to_numpy(zero_copy_only=False)[push_back_rows].tolist()
            )

        effective_indices = task_indices
        if path in affected_paths:
            after = pq.read_table(output / path)
            effective_indices, changed_in_file = _verify_replacement(before, after, path)
            changed += changed_in_file

        for task_id, count in _repaired_task_counts(task_ids, effective_indices).items():
            repaired_counts[task_id] += count

    if repaired_counts != AUDITED_TASK_FRAMES:
        raise ValueError(
            f"Expected repaired task frame counts {AUDITED_TASK_FRAMES}, found {repaired_counts}"
        )
    if derived_affected_paths != affected_paths:
        raise ValueError("Affected data Parquets disagree with the frame-level task IDs")
    return changed, push_back_episodes


def verify_task_registry(tasks: pd.DataFrame, repaired_tasks: pd.DataFrame) -> None:
    """Verify meta/tasks.parquet contains only the missing task addition."""
    if (
        len(repaired_tasks) != len(tasks) + 1
        or repaired_tasks.loc[NEW_TASK, "task_index"] != REPAIRED_TASK_INDEX
        or not repaired_tasks.drop(NEW_TASK).equals(tasks)
    ):
        raise ValueError("Task registry changed unexpectedly")
    # LeRobot resolves each frame's instruction with tasks.iloc[task_index], so row order is the mapping.
    if repaired_tasks["task_index"].tolist() != list(range(len(repaired_tasks))):
        raise ValueError("Task registry rows are not ordered by task_index")


def verify_episodes(episodes: pa.Table, repaired_episodes: pa.Table, push_back_episodes: set[int]) -> None:
    """Verify only push-back instructions and their task-index statistics changed."""
    mutable_columns = ("tasks", *EPISODE_INDEX_STAT_COLUMNS)
    same_schema = repaired_episodes.schema.equals(episodes.schema, check_metadata=True)
    if not same_schema or not repaired_episodes.drop(mutable_columns).equals(episodes.drop(mutable_columns)):
        raise ValueError("Episode metadata changed unexpectedly")

    columns = ("episode_index", *mutable_columns)
    before_rows = episodes.select(columns).to_pylist()
    after_rows = repaired_episodes.select(columns).to_pylist()
    for before, after in zip(before_rows, after_rows, strict=True):
        episode = before["episode_index"]
        if episode not in push_back_episodes:
            if before != after:
                raise ValueError(f"Unaffected episode {episode} changed unexpectedly")
            continue
        if before["tasks"] != [OLD_TASK] or after["tasks"] != [NEW_TASK]:
            raise ValueError(f"Episode {episode} has an unexpected task-label change")
        for column in EPISODE_INDEX_STAT_COLUMNS:
            if before[column] != [COLLIDING_TASK_INDEX] or after[column] != [REPAIRED_TASK_INDEX]:
                raise ValueError(f"Episode {episode} has an unexpected {column} change")


def verify_episode_task_stats(episodes: pa.Table, tasks: pd.DataFrame) -> None:
    """Check each single-task episode against its registry index and frame count."""
    columns = [
        "episode_index",
        "tasks",
        "length",
        *(f"stats/task_index/{name}" for name in ("min", "max", "mean", "std", "count")),
    ]
    for row in episodes.select(columns).to_pylist():
        labels = row["tasks"]
        if len(labels) != 1 or labels[0] not in tasks.index:
            raise ValueError(f"Episode {row['episode_index']} has an invalid task label")
        index = int(tasks.loc[labels[0], "task_index"])
        expected = {"min": [index], "max": [index], "mean": [float(index)], "std": [0.0]}
        if any(row[f"stats/task_index/{name}"] != value for name, value in expected.items()) or row[
            "stats/task_index/count"
        ] != [row["length"]]:
            raise ValueError(f"Episode {row['episode_index']} task-index statistics are inconsistent")


def verify_info(source: Path, output: Path) -> None:
    """Verify meta/info.json changes only the total task count."""
    info = read_json(source / "meta/info.json")
    info["total_tasks"] = 50
    if read_json(output / "meta/info.json") != info:
        raise ValueError("info.json changed unexpectedly")


def verify_stats(source: Path, output: Path) -> None:
    """Verify meta/stats.json changes no statistics outside task_index."""
    old_stats = read_json(source / "meta/stats.json")
    new_stats = read_json(output / "meta/stats.json")
    if {key: value for key, value in new_stats.items() if key != "task_index"} != {
        key: value for key, value in old_stats.items() if key != "task_index"
    }:
        raise ValueError("Non-task statistics changed")


def aggregated_task_index_stats(episodes: pa.Table) -> dict[str, list[int | float]]:
    """Pool per-episode task-index statistics the way LeRobot rebuilds dataset statistics."""
    names = ("min", "max", "mean", "std", "count")
    columns = {name: episodes[f"stats/task_index/{name}"].to_pylist() for name in names}
    per_episode = [
        {name: np.asarray(columns[name][index]) for name in names} for index in range(episodes.num_rows)
    ]
    return {name: values.tolist() for name, values in aggregate_feature_stats(per_episode).items()}


def verify_stats_consistency(root: Path, repaired_episodes: pa.Table) -> None:
    """Verify meta/stats.json agrees with the per-episode statistics it aggregates."""
    aggregated = aggregated_task_index_stats(repaired_episodes)
    stats = read_json(root / "meta/stats.json")["task_index"]
    if (
        stats["min"] != aggregated["min"]
        or stats["max"] != aggregated["max"]
        or stats["count"] != aggregated["count"]
        or not np.isclose(stats["mean"][0], aggregated["mean"][0], rtol=1e-12, atol=1e-12)
        or not np.isclose(stats["std"][0], aggregated["std"][0], rtol=1e-12, atol=1e-12)
    ):
        raise ValueError(f"task_index statistics disagree with the episode metadata in {root}")


def verify_card(source: Path, output: Path) -> None:
    """Verify README.md contains exactly the expected task-count change."""
    old_value = '"total_tasks": 49,'
    new_value = '"total_tasks": 50,'
    before = (source / "README.md").read_text()
    after = (output / "README.md").read_text()
    if before.count(old_value) != 1 or after != before.replace(old_value, new_value):
        raise ValueError("Dataset card changed unexpectedly")


def verify(source: Path, output: Path) -> None:
    """Check the replacement files against the pinned source snapshot."""
    # Revalidate the source before trusting its metadata as the expected baseline.
    # affected (list[Path]): source-relative data Parquets containing task_id=37 frames.
    # _episodes_37 (set[int]): repair-derived push-back episodes; validation derives its own set.
    # _task_index_stats (dict): repair-derived corrected stats; validation aggregates them independently.
    # tasks (pd.DataFrame): source task registry indexed by instruction.
    # episodes (pa.Table): source episode metadata.
    # _repair_measured (dict): repair-derived counts; validation does not trust them as expected values.
    affected, _episodes_37, _task_index_stats, tasks, episodes, _repair_measured = inspect_source(source)

    # changed (int): number of frame rows whose task_index changed from 40 to 49.
    # push_back_episodes (set[int]): episode indices derived from task_id=37 frame rows.
    changed, push_back_episodes = verify_parquets(source, output, affected)
    independently_measured = {
        "affected_data_files": len(affected),
        "affected_episodes": len(push_back_episodes),
        "affected_frames": changed,
    }

    verify_manifest(source, output, affected, independently_measured)

    repaired_tasks = pd.read_parquet(output / "meta/tasks.parquet")
    repaired_episodes = pq.read_table(output / EPISODES_PATH)

    verify_task_registry(tasks, repaired_tasks)

    verify_episodes(episodes, repaired_episodes, push_back_episodes)
    verify_episode_task_stats(repaired_episodes, repaired_tasks)

    verify_info(source, output)
    verify_stats(source, output)

    verify_stats_consistency(output, repaired_episodes)

    verify_card(source, output)

    # Report the pinned and the derived numbers separately so a reviewer can tell which of them this
    # run asserted against the input manifest and which it measured from the snapshot itself.
    print(f"Verified {SOURCE_REPO_ID} at revision {SOURCE_REVISION}")
    print(
        f"  pinned   {SPEC['data_files']} data Parquets, "
        f"{SPEC['total_frames']} frames, {SPEC['total_episodes']} episodes"
    )
    print(
        f"  measured {changed} relabeled frames in "
        f"{independently_measured['affected_episodes']} episodes, "
        f"{independently_measured['affected_data_files']} rewritten data Parquets"
    )


def main() -> None:
    """Run the local validation command."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Dataset root of the pinned Hub snapshot, as printed by download.py",
    )
    parser.add_argument("--output", type=Path, required=True, help="Directory holding the replacement files")
    args = parser.parse_args()
    verify(args.source, args.output)


if __name__ == "__main__":
    main()
