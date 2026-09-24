"""Focused tests for the MetaWorld MT50 repair transformations."""

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from metaworld_mt50_repair.download import download_source
from metaworld_mt50_repair.repair import (
    COLLIDING_TASK_INDEX,
    EPISODES_PATH,
    MANIFEST_PATH,
    NEW_TASK,
    OLD_TASK,
    PUSH_BACK_TASK_ID,
    PUSH_TASK_ID,
    REPAIRED_TASK_INDEX,
    SPEC,
    Episodes,
    check_collision,
    corrected_card,
    corrected_stats,
    merge_episodes,
    provenance,
    read_json,
    relabel,
    relabel_episodes,
    repair,
    repair_card,
    repair_episodes,
    repair_info,
    repair_parquets,
    repair_stats,
    repair_task,
    shard_episodes,
    write_manifest,
)

OTHER_TASK = "Reach a goal position"
OTHER_TASK_INDEX = 42
EPISODES_PER_TASK = SPEC["total_episodes"] // 50
PUSH_BACK_EPISODE = PUSH_BACK_TASK_ID * EPISODES_PER_TASK
OTHER_EPISODE = 0


def audited_episodes() -> Episodes:
    """Build an episode table holding every fact check_collision requires."""
    spare = [index for index in range(49) if index != COLLIDING_TASK_INDEX]
    labels = {PUSH_BACK_TASK_ID: COLLIDING_TASK_INDEX, PUSH_TASK_ID: COLLIDING_TASK_INDEX}
    for task_id in range(50):
        if task_id not in labels:
            labels[task_id] = spare.pop()

    episodes = {
        episode: (episode // EPISODES_PER_TASK, labels[episode // EPISODES_PER_TASK], 1)
        for episode in range(SPEC["total_episodes"])
    }
    episodes[OTHER_EPISODE] = (0, labels[0], SPEC["total_frames"] - SPEC["total_episodes"] + 1)
    return episodes


def episodes_metadata(task_index: int) -> pa.Table:
    """Build episode metadata where episode 0 is push-back and episode 1 is another task."""
    return pa.table(
        {
            "episode_index": pa.array([0, 1], type=pa.int64()),
            "tasks": pa.array([[OLD_TASK], [OTHER_TASK]], type=pa.list_(pa.string())),
            "length": pa.array([5, 7], type=pa.int64()),
            "stats/task_index/min": pa.array([[task_index], [OTHER_TASK_INDEX]], type=pa.list_(pa.int64())),
            "stats/task_index/max": pa.array([[task_index], [OTHER_TASK_INDEX]], type=pa.list_(pa.int64())),
            "stats/task_index/mean": pa.array(
                [[task_index], [OTHER_TASK_INDEX]], type=pa.list_(pa.float64())
            ),
            "stats/task_index/std": pa.array([[0.0], [0.0]], type=pa.list_(pa.float64())),
            "stats/task_index/count": pa.array([[5], [7]], type=pa.list_(pa.int64())),
        }
    )


def source_registry() -> pd.DataFrame:
    """Build a source-shaped registry: one row per task_index below 49, with OLD_TASK on 40."""
    names = [f"task {index}" for index in range(REPAIRED_TASK_INDEX)]
    names[COLLIDING_TASK_INDEX] = OLD_TASK
    return pd.DataFrame({"task_index": range(REPAIRED_TASK_INDEX)}, index=names)


def test_relabel_changes_only_push_back_rows(tmp_path) -> None:
    """Only ID 37 changes; ID 38, payload bytes, and schema metadata survive a Parquet round trip."""
    table = pa.table(
        {
            "task_id": pa.array([PUSH_BACK_TASK_ID, PUSH_TASK_ID], type=pa.int16()),
            "task_index": pa.array([COLLIDING_TASK_INDEX, COLLIDING_TASK_INDEX], type=pa.int64()),
            "payload": [b"push-back", b"push"],
        },
        metadata={b"huggingface": b"{}"},
    )

    result, count = relabel(table)
    path = tmp_path / "repaired.parquet"
    pq.write_table(result, path)
    loaded = pq.read_table(path)

    assert count == 1
    assert loaded["task_index"].to_pylist() == [REPAIRED_TASK_INDEX, COLLIDING_TASK_INDEX]
    assert loaded.schema.equals(table.schema, check_metadata=True)
    assert loaded.drop(["task_index"]).equals(table.drop(["task_index"]))


def test_relabel_rejects_already_repaired_rows() -> None:
    """A second pass over index 49 fails instead of rewriting it."""
    table = pa.table({"task_id": [PUSH_BACK_TASK_ID], "task_index": [REPAIRED_TASK_INDEX]})

    with pytest.raises(ValueError, match="unexpected task_index"):
        relabel(table)


def test_shard_episodes_rejects_an_episode_with_two_tasks() -> None:
    """A frame-level episode must carry exactly one task ID and one task index."""
    shard = pa.table(
        {
            "task_id": pa.array([PUSH_BACK_TASK_ID, PUSH_TASK_ID], type=pa.int16()),
            "task_index": pa.array([COLLIDING_TASK_INDEX] * 2, type=pa.int64()),
            "episode_index": pa.array([7, 7], type=pa.int64()),
        }
    )

    with pytest.raises(ValueError, match="mixes task IDs or task indices"):
        shard_episodes(shard)


def test_shard_episodes_summarizes_each_episode() -> None:
    """A valid shard is reduced to one label and frame count per episode."""
    shard = pa.table(
        {
            "task_id": [PUSH_BACK_TASK_ID, PUSH_BACK_TASK_ID, PUSH_TASK_ID],
            "task_index": [COLLIDING_TASK_INDEX] * 3,
            "episode_index": [7, 7, 8],
        }
    )

    assert shard_episodes(shard) == {
        7: (PUSH_BACK_TASK_ID, COLLIDING_TASK_INDEX, 2),
        8: (PUSH_TASK_ID, COLLIDING_TASK_INDEX, 1),
    }


def test_merge_episodes_rejects_an_episode_in_two_shards() -> None:
    """A repeated episode across shards is rejected."""
    episodes = {7: (PUSH_BACK_TASK_ID, COLLIDING_TASK_INDEX, 5)}

    merge_episodes(episodes, {8: (PUSH_TASK_ID, COLLIDING_TASK_INDEX, 5)})

    assert set(episodes) == {7, 8}
    with pytest.raises(ValueError, match="appear in more than one shard"):
        merge_episodes(episodes, {7: (PUSH_BACK_TASK_ID, COLLIDING_TASK_INDEX, 5)})


def test_check_collision_accepts_the_audited_source() -> None:
    """The complete audited source shape passes every collision check."""
    check_collision(audited_episodes())


def test_check_collision_rejects_a_missing_episode() -> None:
    """An incomplete episode range is rejected."""
    episodes = audited_episodes()
    del episodes[OTHER_EPISODE]

    with pytest.raises(ValueError, match="Expected episodes"):
        check_collision(episodes)


def test_check_collision_rejects_a_changed_frame_total() -> None:
    """A frame total different from the pinned manifest is rejected."""
    episodes = audited_episodes()
    task_id, task_index, frames = episodes[OTHER_EPISODE]
    episodes[OTHER_EPISODE] = (task_id, task_index, frames + 1)

    with pytest.raises(ValueError, match=f"Expected {SPEC['total_frames']} frames"):
        check_collision(episodes)


def test_check_collision_rejects_a_task_id_with_two_indices() -> None:
    """A task ID mapped to two indices is rejected."""
    episodes = audited_episodes()
    task_id, task_index, frames = episodes[OTHER_EPISODE]
    episodes[OTHER_EPISODE] = (task_id, task_index + 1, frames)

    with pytest.raises(ValueError, match="50 task IDs sharing 49 task indices"):
        check_collision(episodes)


def test_check_collision_rejects_an_already_repaired_source() -> None:
    """A source without the audited collision is rejected."""
    episodes = {
        episode: (task_id, REPAIRED_TASK_INDEX if task_id == PUSH_BACK_TASK_ID else task_index, frames)
        for episode, (task_id, task_index, frames) in audited_episodes().items()
    }

    with pytest.raises(ValueError, match="50 task IDs sharing 49 task indices"):
        check_collision(episodes)


def test_check_collision_rejects_an_unbalanced_task() -> None:
    """A task with the wrong episode count is rejected."""
    episodes = audited_episodes()
    _, _, frames = episodes[PUSH_BACK_EPISODE]
    episodes[PUSH_BACK_EPISODE] = (PUSH_TASK_ID, COLLIDING_TASK_INDEX, frames)

    with pytest.raises(ValueError, match=f"Expected {EPISODES_PER_TASK} episodes for each"):
        check_collision(episodes)


def test_check_collision_rejects_the_wrong_colliding_tasks() -> None:
    """The sole shared index must belong specifically to task IDs 37 and 38."""
    source = audited_episodes()
    task_zero_index = source[OTHER_EPISODE][1]
    replacement_indices = {0: COLLIDING_TASK_INDEX, PUSH_BACK_TASK_ID: task_zero_index}
    episodes = {
        episode: (task_id, replacement_indices.get(task_id, task_index), frames)
        for episode, (task_id, task_index, frames) in source.items()
    }

    with pytest.raises(ValueError, match="Expected task IDs 37 and 38"):
        check_collision(episodes)


def test_relabel_episodes_updates_only_intended_metadata() -> None:
    """Push-back episodes receive the new instruction and index statistics."""
    episodes = episodes_metadata(COLLIDING_TASK_INDEX)

    result = relabel_episodes(episodes, {0})

    assert result.schema == episodes.schema
    assert result["tasks"].to_pylist() == [[NEW_TASK], [OTHER_TASK]]
    assert result["stats/task_index/min"].to_pylist() == [[REPAIRED_TASK_INDEX], [OTHER_TASK_INDEX]]
    assert result["stats/task_index/max"].to_pylist() == [[REPAIRED_TASK_INDEX], [OTHER_TASK_INDEX]]
    assert result["stats/task_index/mean"].to_pylist() == [[REPAIRED_TASK_INDEX], [OTHER_TASK_INDEX]]
    assert result["stats/task_index/std"].equals(episodes["stats/task_index/std"])
    assert result["stats/task_index/count"].equals(episodes["stats/task_index/count"])


def test_relabel_episodes_rejects_already_repaired_statistics() -> None:
    """A second pass over repaired episode statistics is rejected."""
    episodes = episodes_metadata(REPAIRED_TASK_INDEX)

    with pytest.raises(ValueError, match="unexpected value"):
        relabel_episodes(episodes, {0})


def test_corrected_card_replaces_exactly_one_task_count() -> None:
    """The card transformation changes only the embedded task count."""
    assert corrected_card('    "total_tasks": 49,\n```') == '    "total_tasks": 50,\n```'

    with pytest.raises(ValueError, match="exactly one original task count"):
        corrected_card("A card without the embedded info block.")


def test_repair_task_preserves_existing_rows_and_dtype(tmp_path) -> None:
    """The new registry entry preserves existing rows and integer dtype, at the row LeRobot reads."""
    tasks = source_registry()

    repair_task(tmp_path, tasks)
    repaired = pd.read_parquet(tmp_path / "meta/tasks.parquet")

    assert tasks["task_index"].dtype == repaired["task_index"].dtype
    assert repaired.loc[NEW_TASK, "task_index"] == REPAIRED_TASK_INDEX
    assert repaired.iloc[REPAIRED_TASK_INDEX].name == NEW_TASK
    assert repaired.drop(NEW_TASK).equals(tasks)


def test_provenance_hashes_tool_files_and_lockfile() -> None:
    """Provenance identifies the repair code and locked runtime inputs."""
    result = provenance()

    assert "uv.lock" in result["files"]
    assert "metaworld_mt50_repair/repair.py" in result["files"]
    assert all(len(file_hash) == 64 for file_hash in result["files"].values())


def test_download_source_skips_videos(monkeypatch, tmp_path) -> None:
    """The downloader requests the pinned dataset without video files."""
    arguments = {}

    class FakeDataset:
        def __init__(self, **kwargs) -> None:
            arguments.update(kwargs)
            self.root = tmp_path

    monkeypatch.setattr("metaworld_mt50_repair.download.LeRobotDataset", FakeDataset)

    assert download_source() == tmp_path
    assert arguments == {
        "repo_id": SPEC["source_repo_id"],
        "revision": SPEC["source_revision"],
        "download_videos": False,
        "video_backend": "pyav",
    }


def test_corrected_stats_moves_only_push_back_frames() -> None:
    """Corrected statistics move ID 37 while ID 38 remains at index 40."""
    task_ids = np.array([PUSH_BACK_TASK_ID, PUSH_TASK_ID, 0], dtype=np.int16)
    indices = np.array([COLLIDING_TASK_INDEX, COLLIDING_TASK_INDEX, 0], dtype=np.int64)
    expected = np.array([REPAIRED_TASK_INDEX, COLLIDING_TASK_INDEX, 0], dtype=np.int64)

    result = corrected_stats(task_ids, indices)

    assert result["min"] == [0]
    assert result["max"] == [REPAIRED_TASK_INDEX]
    assert result["count"] == [len(expected)]
    assert result["mean"] == pytest.approx([float(expected.mean())])
    assert result["std"] == pytest.approx([float(expected.std())])


@pytest.mark.parametrize(
    ("location", "error"),
    [
        ("existing", "new directory"),
        ("inside_source", "outside the source snapshot"),
        ("missing_parent", "parent directory"),
    ],
)
def test_repair_rejects_unsafe_output_locations(tmp_path, location: str, error: str) -> None:
    """Repair refuses output locations that could overwrite or mix with source data."""
    source = tmp_path / "source"
    source.mkdir()
    if location == "existing":
        output = tmp_path / "existing"
        output.mkdir()
    elif location == "inside_source":
        output = source / "output"
    else:
        output = tmp_path / "missing" / "output"

    with pytest.raises(ValueError, match=error):
        repair(source, output)


def test_repair_does_not_create_output_when_source_inspection_fails(tmp_path) -> None:
    """A rejected source leaves the requested output path untouched."""
    source = tmp_path / "wrong-revision"
    output = tmp_path / "output"
    source.mkdir()

    with pytest.raises(ValueError, match="Source must be"):
        repair(source, output)

    assert not output.exists()


def test_repair_writers_emit_the_complete_replacement_set(tmp_path) -> None:
    """Each writer persists its transformation before the manifest hashes the result."""
    source = tmp_path / "source"
    output = tmp_path / "output"
    data_path = tmp_path.joinpath("data/chunk-000/file-000.parquet").relative_to(tmp_path)
    (source / data_path).parent.mkdir(parents=True)
    source_frames = pa.table(
        {
            "task_id": [PUSH_BACK_TASK_ID, PUSH_TASK_ID],
            "task_index": [COLLIDING_TASK_INDEX, COLLIDING_TASK_INDEX],
            "episode_index": [0, 1],
        }
    )
    pq.write_table(source_frames, source / data_path)

    tasks = pd.DataFrame({"task_index": [COLLIDING_TASK_INDEX]}, index=[OLD_TASK])
    (source / "meta").mkdir(parents=True)
    tasks.to_parquet(source / "meta/tasks.parquet")
    episodes = episodes_metadata(COLLIDING_TASK_INDEX)
    (source / EPISODES_PATH).parent.mkdir(parents=True)
    pq.write_table(episodes, source / EPISODES_PATH)
    (source / "meta/info.json").write_text('{"total_tasks": 49, "other": true}')
    (source / "meta/stats.json").write_text('{"task_index": {"mean": [40.0]}, "other": {"mean": [1.0]}}')
    (source / "README.md").write_text('"total_tasks": 49,\n')

    output.mkdir()
    task_index_stats = {"mean": [44.5]}
    measured = {"affected_data_files": 1, "affected_episodes": 1, "affected_frames": 1}
    repair_parquets(source, output, [data_path])
    repair_task(output, tasks)
    repair_episodes(output, episodes, {0})
    repair_info(source, output)
    repair_stats(source, output, task_index_stats)
    repair_card(source, output)
    write_manifest(source, output, [data_path], measured)

    assert pq.read_table(output / data_path)["task_index"].to_pylist() == [
        REPAIRED_TASK_INDEX,
        COLLIDING_TASK_INDEX,
    ]
    assert pd.read_parquet(output / "meta/tasks.parquet").loc[NEW_TASK, "task_index"] == REPAIRED_TASK_INDEX
    assert pq.read_table(output / EPISODES_PATH)["tasks"].to_pylist()[0] == [NEW_TASK]
    assert read_json(output / "meta/info.json")["total_tasks"] == 50
    assert read_json(output / "meta/stats.json")["task_index"] == task_index_stats
    assert (output / "README.md").read_text() == '"total_tasks": 50,\n'
    manifest = read_json(output / MANIFEST_PATH)
    assert manifest["measured"] == measured
    assert set(manifest["files"]) == {
        str(data_path),
        "meta/tasks.parquet",
        str(EPISODES_PATH),
        "meta/info.json",
        "meta/stats.json",
        "README.md",
    }
