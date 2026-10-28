import csv
import json

import numpy as np
import pytest

from ssatk.ssa.event_windows import EventWindow
from ssatk.ssa.workflows import (
    EventMetricBatch,
    PairAnalysisConfig,
    PairAnalysisResult,
)
from ssatk.io.ssa_events import write_event_metrics, write_event_summary


def _result():
    return PairAnalysisResult(
        object_ids=("alpha", 42),
        windows=(EventWindow((0, 1), 9.0, 13.0, 10.0, 12.0, 2),),
        config=PairAnalysisConfig(),
        start_time=8.0,
        end_time=14.0,
        sample_count=4,
        complete=True,
        diagnostics={"candidate_count": 2},
        provenance={"source": {"name": "synthetic"}},
    )


def test_event_summary_serializes_ids_windows_and_metadata(tmp_path):
    path = write_event_summary(_result(), tmp_path / "nested" / "summary.json")

    payload = json.loads(path.read_text())
    assert payload["completeness"] is True
    assert payload["counts"] == {
        "objects": 2,
        "windows": 1,
        "trigger_samples": 2,
    }
    assert payload["object_ids"] == ["alpha", 42]
    assert payload["windows"][0]["object_ids"] == ["alpha", 42]
    assert payload["windows"][0]["trigger_start"] == 10.0
    assert payload["diagnostics"]["candidate_count"] == 2
    assert payload["provenance"]["source"]["name"] == "synthetic"


def test_event_summary_rejects_ambiguous_ids_without_replacing_target(tmp_path):
    target = tmp_path / "summary.json"
    write_event_summary(_result(), target)
    previous = target.read_bytes()
    # Fixed sampled interval reference; exact GPS-second serialization.
    np.testing.assert_allclose(
        json.loads(previous)["windows"][0]["trigger_start"], 10.0,
        atol=0, rtol=0,
    )
    result = _result()
    result = PairAnalysisResult(
        object_ids=(object(), 42),
        windows=result.windows,
        config=result.config,
        start_time=result.start_time,
        end_time=result.end_time,
        sample_count=result.sample_count,
        complete=result.complete,
    )

    with pytest.raises(TypeError, match="object IDs"):
        write_event_summary(result, target)
    assert target.read_bytes() == previous


def test_event_metrics_streams_times_and_closed_form_metric_values(tmp_path):
    batches = [
        EventMetricBatch(
            window_index=3,
            object_ids=("alpha", 42),
            times=np.array([100.0, 101.0]),
            metrics={
                "angular_separation_rad": np.array([0.1, np.inf]),
                "relative_position_rtn_m": np.array(
                    [[3.0, 4.0, 5.0], [6.0, 7.0, 8.0]]
                ),
            },
            valid=np.array([True, False]),
            trigger=np.array([True, False]),
        )
    ]
    path = write_event_metrics(batches, tmp_path / "metrics.csv")

    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert list(rows[0]) == [
        "time_gps_s", "event_index", "object_id_a", "object_id_b",
        "valid", "trigger", "angular_separation_rad",
        "relative_position_rtn_m_r", "relative_position_rtn_m_t",
        "relative_position_rtn_m_n",
    ]
    assert [float(row["time_gps_s"]) for row in rows] == [100.0, 101.0]
    assert [float(row["relative_position_rtn_m_r"]) for row in rows] == [3.0, 6.0]
    assert [float(row["relative_position_rtn_m_t"]) for row in rows] == [4.0, 7.0]
    assert [float(row["relative_position_rtn_m_n"]) for row in rows] == [5.0, 8.0]
    assert rows[1]["angular_separation_rad"] == ""


def test_relative_paths_use_configured_output_root(tmp_path, monkeypatch):
    monkeypatch.setenv("SSATK_OUTPUT_DIR", str(tmp_path))
    path = write_event_summary(_result(), "../events/summary.json")
    assert path == tmp_path / "events" / "summary.json"
    payload = json.loads(path.read_text())
    # Known padded interval [9,13] GPS seconds, exact serialization.
    np.testing.assert_allclose(
        [payload["windows"][0]["start"], payload["windows"][0]["end"]],
        [9., 13.], atol=0, rtol=0,
    )


def test_metric_generator_failure_preserves_target_and_cleans_tempfile(tmp_path):
    target = tmp_path / "metrics.csv"
    def batches():
        yield EventMetricBatch(
            0, ("alpha", 42), np.array([1.0]),
            {"distance_m": np.array([2.0])},
            np.array([True]), np.array([False]),
        )
        raise RuntimeError("source failed")

    first = next(batches())
    write_event_metrics([first], target)
    previous = target.read_bytes()
    with target.open(newline="") as stream:
        row = next(csv.DictReader(stream))
    # Closed-form 2 m separation at GPS second 1, exact decimal round trip.
    np.testing.assert_allclose(
        [float(row["time_gps_s"]), float(row["distance_m"])],
        [1., 2.], atol=0, rtol=0,
    )
    with pytest.raises(RuntimeError, match="source failed"):
        write_event_metrics(batches(), target)
    assert target.read_bytes() == previous
    assert list(tmp_path.glob(f".{target.name}.*.tmp")) == []
