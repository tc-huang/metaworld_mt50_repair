"""Prepare replacement files that correct the MetaWorld MT50 task labels."""

import argparse
import hashlib
import importlib.metadata
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

SPEC = json.loads(Path(__file__).with_name("repair_manifest.json").read_text())
SOURCE_REPO_ID = SPEC["source_repo_id"]
SOURCE_REVISION = SPEC["source_revision"]
OLD_TASK = "Push the puck to a goal"
NEW_TASK = "Push the puck back toward the robot to the goal"
PUSH_BACK_TASK_ID = 37
PUSH_TASK_ID = 38
COLLIDING_TASK_INDEX = 40
REPAIRED_TASK_INDEX = 49
EPISODES_PATH = Path("meta/episodes/chunk-000/file-000.parquet")
# Per-episode statistics that describe the relabeled task index. Push-back episodes hold a
# single task, so each of these collapses to the repaired index; std and count do not move.
EPISODE_INDEX_STAT_COLUMNS = (
    "stats/task_index/min",
    "stats/task_index/max",
    "stats/task_index/mean",
)
METADATA_PATHS = (
    Path("meta/tasks.parquet"),
    EPISODES_PATH,
    Path("meta/info.json"),
    Path("meta/stats.json"),
    Path("README.md"),
)
MANIFEST_PATH = Path("repair_manifest.json")
# episode_index -> (task_id, task_index, frame count)
Episodes = dict[int, tuple[int, int, int]]
REPAIR_ROOT = Path(__file__).parent
PROJECT_ROOT = REPAIR_ROOT.parent
PROVENANCE_PATHS = (
    REPAIR_ROOT / "download.py",
    REPAIR_ROOT / "repair.py",
    REPAIR_ROOT / "reproduce.py",
    REPAIR_ROOT / "validate.py",
    REPAIR_ROOT / "repair_manifest.json",
    PROJECT_ROOT / "uv.lock",
)


def read_json(path: Path) -> dict:
    """Read a dataset metadata file."""
    return json.loads(path.read_text())


def write_json(path: Path, value: dict) -> None:
    """Write metadata using the source dataset's JSON formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=4, ensure_ascii=False))


def sha256(path: Path) -> str:
    """Hash a file without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def provenance() -> dict:
    """Describe the tool files, lockfile, and runtime that produced the output."""
    return {
        "files": {str(path.relative_to(PROJECT_ROOT)): sha256(path) for path in PROVENANCE_PATHS},
        "runtime": {
            "python": sys.version,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "pyarrow": pa.__version__,
            "huggingface_hub": importlib.metadata.version("huggingface-hub"),
        },
    }


def relabel(table: pa.Table) -> tuple[pa.Table, int]:
    """Change only push-back rows from task index 40 to 49, keeping the column's Arrow type."""
    task_ids = table["task_id"].to_numpy(zero_copy_only=False)
    indices = table["task_index"].to_numpy(zero_copy_only=False)
    mask = task_ids == PUSH_BACK_TASK_ID
    if np.any(indices[mask] != COLLIDING_TASK_INDEX):
        raise ValueError("task_id=37 has an unexpected task_index")

    field = table.schema.field("task_index")
    updated = pa.array(np.where(mask, REPAIRED_TASK_INDEX, indices), type=field.type)
    column_index = table.schema.get_field_index("task_index")
    new_table = table.set_column(column_index, field, updated)
    changed_count = int(mask.sum())
    return new_table, changed_count


def shard_episodes(table: pa.Table) -> Episodes:
    """Summarize one data shard as (task_id, task_index, frame count) per episode."""
    task_ids = table["task_id"].to_numpy(zero_copy_only=False)
    task_indices = table["task_index"].to_numpy(zero_copy_only=False)
    episode_indices = table["episode_index"].to_numpy(zero_copy_only=False)
    summary: Episodes = {}
    for episode in np.unique(episode_indices):
        rows = episode_indices == episode
        labels = set(zip(task_ids[rows].tolist(), task_indices[rows].tolist(), strict=True))
        if len(labels) != 1:
            raise ValueError(f"Episode {episode} mixes task IDs or task indices")
        task_id, task_index = labels.pop()
        summary[int(episode)] = (task_id, task_index, int(rows.sum()))
    return summary


def merge_episodes(episodes: Episodes, shard: Episodes) -> None:
    """Add a shard summary, requiring each episode to live in a single shard."""
    repeated = episodes.keys() & shard.keys()
    if repeated:
        raise ValueError(f"Episodes appear in more than one shard: {sorted(repeated)}")
    episodes.update(shard)


def check_collision(episodes: Episodes) -> None:
    """Confirm the frames still hold exactly the audited 50-task, 49-index collision."""
    if set(episodes) != set(range(SPEC["total_episodes"])):
        raise ValueError(f"Expected episodes 0 to {SPEC['total_episodes'] - 1}")

    frames = sum(frames for _, _, frames in episodes.values())
    if frames != SPEC["total_frames"]:
        raise ValueError(f"Expected {SPEC['total_frames']} frames, found {frames}")

    pairs = {(task_id, task_index) for task_id, task_index, _ in episodes.values()}
    labels = dict(pairs)
    if len(labels) != len(pairs) or set(labels) != set(range(50)) or set(labels.values()) != set(range(49)):
        raise ValueError("Expected 50 task IDs sharing 49 task indices")

    colliding = {task_id for task_id, task_index in labels.items() if task_index == COLLIDING_TASK_INDEX}
    if colliding != {PUSH_BACK_TASK_ID, PUSH_TASK_ID}:
        raise ValueError(f"Expected task IDs 37 and 38 to share task index {COLLIDING_TASK_INDEX}")

    # MT50 holds the same number of demonstrations per task, so the size of the repair follows from
    # the dataset layout instead of being a count this tool has to be told in advance.
    per_task = SPEC["total_episodes"] // 50
    if set(Counter(task_id for task_id, _, _ in episodes.values()).values()) != {per_task}:
        raise ValueError(f"Expected {per_task} episodes for each of the 50 tasks")


def scan(source: Path) -> tuple[list[Path], np.ndarray, np.ndarray, Episodes]:
    """Locate the shards to rewrite and check the audited source collision."""
    files = sorted(path.relative_to(source) for path in (source / "data").rglob("*.parquet"))
    if len(files) != SPEC["data_files"]:
        raise ValueError(f"Expected {SPEC['data_files']} data Parquets, found {len(files)}")

    episodes: Episodes = {}
    affected: list[Path] = []
    all_task_ids = []
    all_indices = []

    for path in files:
        table = pq.read_table(source / path, columns=["task_id", "task_index", "episode_index"])
        all_task_ids.append(table["task_id"].to_numpy(zero_copy_only=False))
        all_indices.append(table["task_index"].to_numpy(zero_copy_only=False))
        shard = shard_episodes(table)
        merge_episodes(episodes, shard)
        if any(task_id == PUSH_BACK_TASK_ID for task_id, _, _ in shard.values()):
            affected.append(path)

    check_collision(episodes)
    return affected, np.concatenate(all_task_ids), np.concatenate(all_indices), episodes


def corrected_stats(task_ids: np.ndarray, indices: np.ndarray) -> dict[str, list[int | float]]:
    """Summarize the task indices as meta/stats.json records them, with push-back moved to 49."""
    corrected = np.where(task_ids == PUSH_BACK_TASK_ID, REPAIRED_TASK_INDEX, indices)
    return {
        "min": [int(corrected.min())],
        "max": [int(corrected.max())],
        "mean": [float(corrected.mean())],
        "std": [float(corrected.std())],
        "count": [len(corrected)],
    }


def inspect_source(source: Path) -> tuple[list[Path], set[int], dict, pd.DataFrame, pa.Table, dict]:
    """Check the pinned snapshot and its task-related metadata."""
    if source.resolve().name != SOURCE_REVISION:
        raise ValueError(f"Source must be the Hugging Face snapshot {SOURCE_REVISION}")

    # affected (list[Path]): source-relative paths of the data shards holding task_id=37 frames.
    # task_ids (np.ndarray): the task_id of every frame, in file order.
    # indices (np.ndarray): the task_index of every frame, in the same order.
    # episode_rows (Episodes): episode_index -> (task_id, task_index, frame count), read from the frames.
    affected, task_ids, indices, episode_rows = scan(source)

    # episodes_37 (set[int]): the push-back episodes.
    episodes_37 = {
        episode for episode, (task_id, _, _) in episode_rows.items() if task_id == PUSH_BACK_TASK_ID
    }

    measured = {
        "affected_data_files": len(affected),
        "affected_episodes": len(episodes_37),
        "affected_frames": sum(episode_rows[episode][2] for episode in episodes_37),
    }

    tasks = pd.read_parquet(source / "meta/tasks.parquet")

    if (
        not tasks.index.is_unique
        or tasks["task_index"].to_list() != list(range(49))
        or OLD_TASK not in tasks.index
        or tasks.loc[OLD_TASK, "task_index"] != COLLIDING_TASK_INDEX
        or NEW_TASK in tasks.index
    ):
        raise ValueError("Source task registry does not match the expected 49 labels")

    episodes = pq.read_table(source / EPISODES_PATH)
    episode_ids = episodes["episode_index"].to_pylist()
    episode_tasks = episodes["tasks"].to_pylist()

    if len(episode_ids) != SPEC["total_episodes"] or set(episode_ids) != set(range(SPEC["total_episodes"])):
        raise ValueError("Source episode metadata is incomplete")

    # Every episode holds a single task, so its recorded task-index statistics must be flat over
    # exactly its own frames. The repair keeps std and count as they are, so they must be right here.
    for position, episode_id in enumerate(episode_ids):
        _, task_index, frame_count = episode_rows[episode_id]
        task_name = episode_tasks[position]
        if (
            frame_count != episodes["length"][position].as_py()
            or episodes["stats/task_index/std"][position].as_py() != [0.0]
            or episodes["stats/task_index/count"][position].as_py() != [frame_count]
            or len(task_name) != 1
            or task_name[0] not in tasks.index
            or tasks.loc[task_name[0], "task_index"] != task_index
        ):
            raise ValueError(f"Source episode {episode_id} disagrees with its frames or task registry")

    info = read_json(source / "meta/info.json")
    if (info["total_tasks"], info["total_episodes"], info["total_frames"]) != (
        49,
        SPEC["total_episodes"],
        SPEC["total_frames"],
    ):
        raise ValueError("Source info.json does not match the audited dataset")

    stats = read_json(source / "meta/stats.json")["task_index"]
    if (
        stats["count"] != [SPEC["total_frames"]]
        or stats["min"] != [0]
        or stats["max"] != [48]
        or not np.isclose(stats["mean"][0], indices.mean(), atol=1e-6)
        or not np.isclose(stats["std"][0], indices.std(), atol=1e-6)
    ):
        raise ValueError("Source task_index stats do not match the audited dataset")

    # The source statistics describe the source frames, so the same summary of the relabeled frames
    # is what meta/stats.json must become. Computed once here, then written by repair and compared
    # by validate, so the two cannot drift apart.
    task_index_stats = corrected_stats(task_ids, indices)

    if (source / "README.md").read_text().count('"total_tasks": 49,') != 1:
        raise ValueError("Dataset card does not contain the expected total_tasks value")

    return affected, episodes_37, task_index_stats, tasks, episodes, measured


def repair_parquets(source: Path, output: Path, affected: list[Path]) -> None:
    """Write relabeled replacements for the affected data Parquets."""
    for path in affected:
        updated, _ = relabel(pq.read_table(source / path))
        destination = output / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(updated, destination)


def repair_task(output: Path, tasks: pd.DataFrame) -> None:
    """Add the missing push-back task to meta/tasks.parquet."""
    new_task = pd.DataFrame(
        {"task_index": [REPAIRED_TASK_INDEX]}, index=pd.Index([NEW_TASK], name=tasks.index.name)
    )
    repaired_tasks = pd.concat([tasks, new_task])
    (output / "meta").mkdir(parents=True, exist_ok=True)
    repaired_tasks.to_parquet(output / "meta/tasks.parquet")


def relabel_episodes(episodes: pa.Table, episodes_37: set[int]) -> pa.Table:
    """Relabel the instruction and the task-index statistics of push-back episodes."""
    push_back_mask = [index in episodes_37 for index in episodes["episode_index"].to_pylist()]

    replacements = {
        "tasks": [
            ([NEW_TASK] if is_push_back else task)
            for task, is_push_back in zip(episodes["tasks"].to_pylist(), push_back_mask, strict=True)
        ]
    }
    for column in EPISODE_INDEX_STAT_COLUMNS:
        pairs = list(zip(episodes[column].to_pylist(), push_back_mask, strict=True))
        if any(cell != [COLLIDING_TASK_INDEX] for cell, is_push_back in pairs if is_push_back):
            raise ValueError(f"{column} has an unexpected value for a task_id=37 episode")
        replacements[column] = [
            ([REPAIRED_TASK_INDEX] if is_push_back else cell) for cell, is_push_back in pairs
        ]

    # Replace each column through its own Arrow type so the Parquet schema is unchanged.
    for column, values in replacements.items():
        field = episodes.schema.field(column)
        episodes = episodes.set_column(
            episodes.schema.get_field_index(column), field, pa.array(values, type=field.type)
        )
    return episodes


def repair_episodes(output: Path, episodes: pa.Table, episodes_37: set[int]) -> None:
    """Write the relabeled episode metadata Parquet."""
    (output / EPISODES_PATH).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(relabel_episodes(episodes, episodes_37), output / EPISODES_PATH)


def repair_info(source: Path, output: Path) -> None:
    """Update the total task count in meta/info.json."""
    info = read_json(source / "meta/info.json")
    info["total_tasks"] = 50
    write_json(output / "meta/info.json", info)


def repair_stats(source: Path, output: Path, task_index_stats: dict) -> None:
    """Update task-index statistics in meta/stats.json."""
    stats = read_json(source / "meta/stats.json")
    stats["task_index"] = task_index_stats
    write_json(output / "meta/stats.json", stats)


def repair_card(source: Path, output: Path) -> None:
    """Update the dataset card with the corrected task mapping."""
    card = corrected_card((source / "README.md").read_text())
    (output / "README.md").write_text(card)


def corrected_card(card: str) -> str:
    """Return the dataset card with the corrected task count."""
    if card.count('"total_tasks": 49,') != 1:
        raise ValueError("Dataset card does not contain exactly one original task count")
    return card.replace('"total_tasks": 49,', '"total_tasks": 50,')


def write_manifest(source: Path, output: Path, affected: list[Path], measured: dict) -> None:
    """Record hashes for all generated replacement files."""
    paths = [*affected, *METADATA_PATHS]
    write_json(
        output / MANIFEST_PATH,
        {
            "source_repo_id": SOURCE_REPO_ID,
            "source_revision": SOURCE_REVISION,
            "measured": measured,
            "provenance": provenance(),
            "files": {
                str(path): {"source_sha256": sha256(source / path), "output_sha256": sha256(output / path)}
                for path in paths
            },
        },
    )


def repair(source: Path, output: Path) -> None:
    """Write only replacement files to a new directory."""
    if output.exists() or output.resolve().is_relative_to(source.resolve()):
        raise ValueError("Output must be a new directory outside the source snapshot")
    if not output.parent.is_dir():
        raise ValueError("Output parent directory must already exist")
    # affected (list[Path]): the data shards to rewrite, as source-relative paths.
    # episodes_37 (set[int]): the push-back episodes whose metadata row changes.
    # task_index_stats (dict): the repaired task_index block for meta/stats.json.
    # tasks (pd.DataFrame): meta/tasks.parquet, indexed by instruction, with a task_index column.
    # episodes (pa.Table): the meta/episodes table whose instructions and statistics move to 49.
    # measured (dict): the repair's scale, derived from the frames and recorded in the manifest.
    affected, episodes_37, task_index_stats, tasks, episodes, measured = inspect_source(source)

    output.mkdir(parents=True)

    repair_parquets(source, output, affected)
    repair_task(output, tasks)
    repair_episodes(output, episodes, episodes_37)
    repair_info(source, output)
    repair_stats(source, output, task_index_stats)
    repair_card(source, output)
    write_manifest(source, output, affected, measured)

    print(f"Prepared {len(affected) + len(METADATA_PATHS)} replacement files in {output}")


def main() -> None:
    """Run the local repair command."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Dataset root of the pinned Hub snapshot, as printed by download.py",
    )
    parser.add_argument("--output", type=Path, required=True, help="New directory for the replacement files")
    args = parser.parse_args()
    repair(args.source, args.output)


if __name__ == "__main__":
    main()
