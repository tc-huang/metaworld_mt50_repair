"""Focused tests for the MetaWorld MT50 repair validator."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from metaworld_mt50_repair.repair import (
    COLLIDING_TASK_INDEX,
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
)
from metaworld_mt50_repair.validate import (
    aggregated_task_index_stats,
    verify_card,
    verify_episode_task_stats,
    verify_episodes,
    verify_info,
    verify_manifest,
    verify_parquets,
    verify_stats,
    verify_stats_consistency,
    verify_task_registry,
)

OTHER_TASK = "Reach a goal position"
OTHER_TASK_INDEX = 42
EXPECTED_MEASURED = {
    "affected_data_files": 1,
    "affected_episodes": 1,
    "affected_frames": 8_888,
}


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


def repaired_episodes_metadata() -> pa.Table:
    """Build the expected repaired metadata independently of repair.py."""
    repaired = episodes_metadata(REPAIRED_TASK_INDEX)
    field = repaired.schema.field("tasks")
    return repaired.set_column(
        repaired.schema.get_field_index("tasks"),
        field,
        pa.array([[NEW_TASK], [OTHER_TASK]], type=field.type),
    )


def audited_frame_tables() -> tuple[pa.Table, pa.Table]:
    """Build source and repaired frame tables with the audited task counts and schema metadata."""
    task_ids = np.concatenate(
        [
            np.full(8_888, PUSH_BACK_TASK_ID, dtype=np.int16),
            np.full(3_521, PUSH_TASK_ID, dtype=np.int16),
        ]
    )
    before = pa.table(
        {
            "task_id": task_ids,
            "task_index": np.full(len(task_ids), COLLIDING_TASK_INDEX, dtype=np.int64),
            "episode_index": np.concatenate(
                [np.zeros(8_888, dtype=np.int64), np.ones(3_521, dtype=np.int64)]
            ),
            "payload": np.arange(len(task_ids), dtype=np.int32),
        },
        metadata={b"huggingface": b"{}"},
    )
    repaired_indices = np.where(task_ids == PUSH_BACK_TASK_ID, REPAIRED_TASK_INDEX, COLLIDING_TASK_INDEX)
    after = before.set_column(
        before.schema.get_field_index("task_index"),
        before.schema.field("task_index"),
        pa.array(repaired_indices, type=before.schema.field("task_index").type),
    )
    return before, after


def write_frame_replacement(
    tmp_path: Path, after: pa.Table, before: pa.Table | None = None
) -> tuple[Path, Path, list[Path]]:
    """Write one source shard and its replacement under separate roots."""
    source = tmp_path / "source"
    output = tmp_path / "output"
    relative_path = Path("data/chunk-000/file-000.parquet")
    (source / relative_path).parent.mkdir(parents=True)
    (output / relative_path).parent.mkdir(parents=True)
    if before is None:
        before, _ = audited_frame_tables()
    pq.write_table(before, source / relative_path)
    pq.write_table(after, output / relative_path)
    return source, output, [relative_path]


def digest(path: Path) -> str:
    """Return a test-side SHA-256 digest independent of the repair helper."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_manifest_fixture(tmp_path: Path) -> tuple[Path, Path]:
    """Write a minimal valid replacement set and repair manifest."""
    source = tmp_path / "source"
    output = tmp_path / "output"
    files = {}
    for path in METADATA_PATHS:
        (source / path).parent.mkdir(parents=True, exist_ok=True)
        (output / path).parent.mkdir(parents=True, exist_ok=True)
        (source / path).write_bytes(f"source:{path}".encode())
        (output / path).write_bytes(f"output:{path}".encode())
        files[str(path)] = {
            "source_sha256": digest(source / path),
            "output_sha256": digest(output / path),
        }

    provenance_files = {str(path.relative_to(PROJECT_ROOT)): digest(path) for path in PROVENANCE_PATHS}
    manifest = {
        "source_repo_id": SOURCE_REPO_ID,
        "source_revision": SOURCE_REVISION,
        "measured": EXPECTED_MEASURED,
        "provenance": {"files": provenance_files},
        "files": files,
    }
    (output / MANIFEST_PATH).write_text(json.dumps(manifest))
    return source, output


def test_verify_parquets_accepts_the_audited_replacement(tmp_path) -> None:
    """The validator accepts exact repaired indices and audited task counts."""
    _, after = audited_frame_tables()
    source, output, affected = write_frame_replacement(tmp_path, after)

    changed, episodes = verify_parquets(source, output, affected)

    assert changed == 8_888
    assert episodes == {0}


def test_verify_parquets_rejects_non_task_changes(tmp_path) -> None:
    """A replacement cannot modify payload data alongside task_index."""
    _, after = audited_frame_tables()
    after = after.set_column(
        after.schema.get_field_index("payload"),
        after.schema.field("payload"),
        pa.array(np.full(after.num_rows, -1, dtype=np.int32)),
    )
    source, output, affected = write_frame_replacement(tmp_path, after)

    with pytest.raises(ValueError, match="Non-task data changed"):
        verify_parquets(source, output, affected)


def test_verify_parquets_rejects_dropped_schema_metadata(tmp_path) -> None:
    """A replacement must keep the Hub feature metadata stored in the Parquet schema."""
    _, after = audited_frame_tables()
    source, output, affected = write_frame_replacement(tmp_path, after.replace_schema_metadata(None))

    with pytest.raises(ValueError, match="Non-task data changed"):
        verify_parquets(source, output, affected)


def test_verify_parquets_rejects_the_wrong_task_index_change(tmp_path) -> None:
    """Only push-back rows may change task_index."""
    _, after = audited_frame_tables()
    indices = after["task_index"].to_numpy(zero_copy_only=False).copy()
    indices[-1] = REPAIRED_TASK_INDEX
    after = after.set_column(
        after.schema.get_field_index("task_index"),
        after.schema.field("task_index"),
        pa.array(indices, type=after.schema.field("task_index").type),
    )
    source, output, affected = write_frame_replacement(tmp_path, after)

    with pytest.raises(ValueError, match="Unexpected task indices"):
        verify_parquets(source, output, affected)


def test_verify_parquets_rejects_an_incorrect_final_task_index(tmp_path) -> None:
    """Audited tasks must end on their designated repaired indices."""
    before, after = audited_frame_tables()
    before_indices = before["task_index"].to_numpy(zero_copy_only=False).copy()
    after_indices = after["task_index"].to_numpy(zero_copy_only=False).copy()
    before_indices[-1] = 41
    after_indices[-1] = 41
    before = before.set_column(
        before.schema.get_field_index("task_index"),
        before.schema.field("task_index"),
        pa.array(before_indices, type=before.schema.field("task_index").type),
    )
    after = after.set_column(
        after.schema.get_field_index("task_index"),
        after.schema.field("task_index"),
        pa.array(after_indices, type=after.schema.field("task_index").type),
    )
    source, output, affected = write_frame_replacement(tmp_path, after, before)

    with pytest.raises(ValueError, match="Repaired task_id=38 has an unexpected task_index"):
        verify_parquets(source, output, affected)


def test_verify_parquets_rejects_an_audited_count_mismatch(tmp_path) -> None:
    """The repaired corpus must retain both audited frame counts."""
    before, after = audited_frame_tables()
    before = before.slice(0, before.num_rows - 1)
    after = after.slice(0, after.num_rows - 1)
    source, output, affected = write_frame_replacement(tmp_path, after, before)

    with pytest.raises(ValueError, match="Expected repaired task frame counts"):
        verify_parquets(source, output, affected)


def test_verify_parquets_rejects_an_extra_affected_path(tmp_path) -> None:
    """The affected path set must exactly match shards containing push-back rows."""
    before, after = audited_frame_tables()
    source, output, affected = write_frame_replacement(tmp_path, after, before)
    extra_path = Path("data/chunk-000/file-001.parquet")
    pq.write_table(before.slice(0, 0), source / extra_path)
    pq.write_table(after.slice(0, 0), output / extra_path)
    affected.append(extra_path)

    with pytest.raises(ValueError, match="Affected data Parquets disagree"):
        verify_parquets(source, output, affected)


def test_verify_episodes_accepts_the_expected_metadata_changes() -> None:
    """The exact push-back episode repair is accepted."""
    before = episodes_metadata(COLLIDING_TASK_INDEX)
    after = repaired_episodes_metadata()
    verify_episodes(before, after, {0})


def test_verify_episodes_rejects_a_non_task_metadata_change() -> None:
    """Columns outside the allowed task metadata set must remain unchanged."""
    before = episodes_metadata(COLLIDING_TASK_INDEX)
    after = repaired_episodes_metadata()
    invalid = after.set_column(
        after.schema.get_field_index("length"),
        after.schema.field("length"),
        pa.array([6, 7], type=after.schema.field("length").type),
    )

    with pytest.raises(ValueError, match="Episode metadata changed unexpectedly"):
        verify_episodes(before, invalid, {0})


def test_verify_episodes_rejects_dropped_schema_metadata() -> None:
    """The episode table must keep the pandas metadata stored in its Parquet schema."""
    before = episodes_metadata(COLLIDING_TASK_INDEX).replace_schema_metadata({b"pandas": b"{}"})

    with pytest.raises(ValueError, match="Episode metadata changed unexpectedly"):
        verify_episodes(before, repaired_episodes_metadata(), {0})


@pytest.mark.parametrize(
    ("column", "values", "error"),
    [
        ("tasks", [[NEW_TASK], [NEW_TASK]], "Unaffected episode"),
        ("tasks", [[OTHER_TASK], [OTHER_TASK]], "unexpected task-label change"),
        ("stats/task_index/min", [[48], [OTHER_TASK_INDEX]], "unexpected stats/task_index/min"),
    ],
)
def test_verify_episodes_rejects_unexpected_mutable_column_changes(
    column: str, values: list, error: str
) -> None:
    """Only the exact mutable-column changes are accepted for affected episodes."""
    before = episodes_metadata(COLLIDING_TASK_INDEX)
    after = repaired_episodes_metadata()
    invalid = after.set_column(
        after.schema.get_field_index(column),
        after.schema.field(column),
        pa.array(values, type=after.schema.field(column).type),
    )

    with pytest.raises(ValueError, match=error):
        verify_episodes(before, invalid, {0})


def test_episode_stats_validation_rejects_stale_push_back_metadata() -> None:
    """A repaired instruction cannot retain stale task-index statistics."""
    tasks = pd.DataFrame(
        {"task_index": [COLLIDING_TASK_INDEX, OTHER_TASK_INDEX, REPAIRED_TASK_INDEX]},
        index=[OLD_TASK, OTHER_TASK, NEW_TASK],
    )
    repaired = repaired_episodes_metadata()
    verify_episode_task_stats(repaired, tasks)

    stale = repaired.set_column(
        repaired.schema.get_field_index("stats/task_index/mean"),
        repaired.schema.field("stats/task_index/mean"),
        pa.array(
            [[COLLIDING_TASK_INDEX], [OTHER_TASK_INDEX]],
            type=repaired.schema.field("stats/task_index/mean").type,
        ),
    )
    with pytest.raises(ValueError, match="task-index statistics are inconsistent"):
        verify_episode_task_stats(stale, tasks)


def test_episode_stats_validation_rejects_an_unknown_task_label() -> None:
    """Every episode instruction must exist in the repaired task registry."""
    episodes = repaired_episodes_metadata()
    episodes = episodes.set_column(
        episodes.schema.get_field_index("tasks"),
        episodes.schema.field("tasks"),
        pa.array([[NEW_TASK], ["unknown task"]], type=episodes.schema.field("tasks").type),
    )
    tasks = pd.DataFrame(
        {"task_index": [OTHER_TASK_INDEX, REPAIRED_TASK_INDEX]},
        index=[OTHER_TASK, NEW_TASK],
    )

    with pytest.raises(ValueError, match="invalid task label"):
        verify_episode_task_stats(episodes, tasks)


def test_task_registry_validation_accepts_only_the_missing_addition() -> None:
    """The registry may append the missing task without altering existing rows."""
    tasks = source_registry()
    repaired = pd.concat([tasks, pd.DataFrame({"task_index": [REPAIRED_TASK_INDEX]}, index=[NEW_TASK])])
    verify_task_registry(tasks, repaired)

    repaired.loc["task 0", "task_index"] = 1
    with pytest.raises(ValueError, match="Task registry changed unexpectedly"):
        verify_task_registry(tasks, repaired)


def test_task_registry_validation_rejects_rows_out_of_task_index_order() -> None:
    """The appended task must sit at the row LeRobot reads for its task_index."""
    tasks = source_registry()
    new_task = pd.DataFrame({"task_index": [REPAIRED_TASK_INDEX]}, index=[NEW_TASK])

    with pytest.raises(ValueError, match="not ordered by task_index"):
        verify_task_registry(tasks, pd.concat([new_task, tasks]))


def test_info_validation_accepts_only_the_total_task_change(tmp_path) -> None:
    """info.json may change only total_tasks from the source value to 50."""
    source = tmp_path / "source"
    output = tmp_path / "output"
    (source / "meta").mkdir(parents=True)
    (output / "meta").mkdir(parents=True)
    before = {"total_tasks": 49, "total_frames": 204_806}
    (source / "meta/info.json").write_text(json.dumps(before))
    (output / "meta/info.json").write_text(json.dumps({**before, "total_tasks": 50}))
    verify_info(source, output)

    (output / "meta/info.json").write_text(json.dumps({"total_tasks": 50, "total_frames": 204_805}))
    with pytest.raises(ValueError, match="info.json changed unexpectedly"):
        verify_info(source, output)


def test_stats_validation_rejects_non_task_stat_changes(tmp_path) -> None:
    """stats.json may replace task_index statistics but no other feature statistics."""
    source = tmp_path / "source"
    output = tmp_path / "output"
    (source / "meta").mkdir(parents=True)
    (output / "meta").mkdir(parents=True)
    before = {"task_index": {"mean": [20.0]}, "observation.state": {"mean": [1.0]}}
    after = {"task_index": {"mean": [21.0]}, "observation.state": {"mean": [1.0]}}
    (source / "meta/stats.json").write_text(json.dumps(before))
    (output / "meta/stats.json").write_text(json.dumps(after))
    verify_stats(source, output)

    after["observation.state"]["mean"] = [2.0]
    (output / "meta/stats.json").write_text(json.dumps(after))
    with pytest.raises(ValueError, match="Non-task statistics changed"):
        verify_stats(source, output)


def test_aggregated_task_index_stats_matches_frame_values() -> None:
    """Pooling per-episode statistics reproduces the underlying frame statistics."""
    episodes = repaired_episodes_metadata()
    frames = np.array([REPAIRED_TASK_INDEX] * 5 + [OTHER_TASK_INDEX] * 7, dtype=np.float64)

    result = aggregated_task_index_stats(episodes)

    assert result["min"] == [OTHER_TASK_INDEX]
    assert result["max"] == [REPAIRED_TASK_INDEX]
    assert result["count"] == [len(frames)]
    assert result["mean"] == pytest.approx([float(frames.mean())])
    assert result["std"] == pytest.approx([float(frames.std())])


def test_stats_consistency_rejects_stats_that_disagree_with_episodes(tmp_path) -> None:
    """Dataset task-index statistics must aggregate the repaired episode metadata."""
    output = tmp_path / "output"
    (output / "meta").mkdir(parents=True)
    episodes = repaired_episodes_metadata()
    expected = aggregated_task_index_stats(episodes)
    (output / "meta/stats.json").write_text(json.dumps({"task_index": expected}))
    verify_stats_consistency(output, episodes)

    expected["mean"] = [0.0]
    (output / "meta/stats.json").write_text(json.dumps({"task_index": expected}))
    with pytest.raises(ValueError, match="statistics disagree"):
        verify_stats_consistency(output, episodes)


def test_card_validation_rejects_an_extra_edit(tmp_path) -> None:
    """The card may change only its embedded total task count."""
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    output.mkdir()
    before = 'prefix\n    "total_tasks": 49,\nsuffix\n'
    (source / "README.md").write_text(before)
    expected = before.replace('"total_tasks": 49,', '"total_tasks": 50,')
    (output / "README.md").write_text(expected)
    verify_card(source, output)

    (output / "README.md").write_text(expected + "unexpected\n")
    with pytest.raises(ValueError, match="Dataset card changed unexpectedly"):
        verify_card(source, output)


def test_manifest_validation_checks_file_set_and_hashes(tmp_path) -> None:
    """The manifest binds the exact replacement set to source and output hashes."""
    source, output = write_manifest_fixture(tmp_path)
    verify_manifest(source, output, [], EXPECTED_MEASURED)

    manifest_path = output / MANIFEST_PATH
    manifest = json.loads(manifest_path.read_text())
    manifest["source_revision"] = "wrong-revision"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Repair manifest does not match"):
        verify_manifest(source, output, [], EXPECTED_MEASURED)
    manifest["source_revision"] = SOURCE_REVISION
    manifest_path.write_text(json.dumps(manifest))

    extra = output / "extra.txt"
    extra.write_text("unexpected")
    with pytest.raises(ValueError, match="missing or extra files"):
        verify_manifest(source, output, [], EXPECTED_MEASURED)
    extra.unlink()

    (output / "README.md").write_text("tampered")
    with pytest.raises(ValueError, match="File hash mismatch"):
        verify_manifest(source, output, [], EXPECTED_MEASURED)
