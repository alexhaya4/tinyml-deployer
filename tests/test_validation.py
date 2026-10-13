"""Quality gates reject incorrect data and invalid measurement streams."""

import copy

import numpy as np
import pytest

from tinyml_deployer.evaluation import evaluate_models, load_samples
from tinyml_deployer.profiler import validate_records
from tinyml_deployer.reporting import write_report


def test_identical_model_has_zero_error(sine_model_path, validation_path):
    result = evaluate_models(sine_model_path, sine_model_path, validation_path, 0)
    assert result.passed and result.max_absolute_error == 0


@pytest.mark.parametrize(
    "data", [np.array([]), np.array([np.nan]), np.array([1], dtype=np.int32)]
)
def test_reject_invalid_dataset(data, tmp_path):
    path = tmp_path / "invalid.npy"
    np.save(path, data)
    with pytest.raises(ValueError):
        load_samples(str(path))


def test_reports_escape_html_and_preserve_existing_files(tmp_path):
    path = tmp_path / "report.html"
    write_report(
        {"name": "<script>alert(1)</script>", "hardware": "not_run"}, str(path)
    )
    assert "<script>" not in path.read_text()
    assert "&lt;script&gt;" in path.read_text()
    with pytest.raises(FileExistsError):
        write_report({"changed": True}, str(path))


@pytest.fixture
def protocol_run():
    # Protocol unit-test data, never used as a hardware measurement report.
    manifest = {
        "schema_version": 1,
        "model_sha256": "a" * 64,
        "dataset_sha256": "b" * 64,
        "target": "esp32s3",
        "esp_idf_version": "5.5.1",
        "samples": 1,
        "repetitions": 2,
        "arena_reserved_bytes": 4096,
        "max_abs_error": 0.01,
    }
    rows = [
        {
            "event": "begin",
            "run_id": "test",
            "model_sha256": "a" * 64,
            "dataset_sha256": "b" * 64,
            "target": "esp32s3",
            "idf_version": "v5.5.1",
        }
    ]
    for repetition, latency in enumerate([10, 20]):
        rows.append(
            {
                "event": "sample",
                "run_id": "test",
                "sample": 0,
                "repetition": repetition,
                "latency_us": latency,
                "max_abs_error": 0.005,
                "arena_used_bytes": 2048,
                "min_free_heap_bytes": 8000,
            }
        )
    rows.append({"event": "done", "run_id": "test"})
    return manifest, rows


def test_complete_protocol_run(protocol_run):
    manifest, rows = protocol_run
    result = validate_records(rows, manifest, "test")
    assert result.median_latency_us == 15
    assert result.observations == 2
    assert result.prediction_agreement_passed


@pytest.mark.parametrize(
    "defect",
    [
        "duplicate",
        "missing",
        "nan",
        "mismatch",
        "error",
        "arena",
        "run_id",
        "idf_version",
    ],
)
def test_bad_protocol_runs_rejected(protocol_run, defect):
    manifest, original = protocol_run
    rows = copy.deepcopy(original)
    if defect == "duplicate":
        rows[2] = rows[1].copy()
    elif defect == "missing":
        rows.pop(1)
    elif defect == "nan":
        rows[1]["latency_us"] = float("nan")
    elif defect == "mismatch":
        rows[0]["model_sha256"] = "wrong"
    elif defect == "error":
        rows[1]["event"] = "error"
    elif defect == "arena":
        rows[1]["arena_used_bytes"] = 8192
    elif defect == "run_id":
        rows[1]["run_id"] = "stale"
    elif defect == "idf_version":
        rows[0]["idf_version"] = "v5.5.10"
    with pytest.raises(ValueError):
        validate_records(rows, manifest, "test")


@pytest.mark.parametrize(
    "field,value",
    [
        ("samples", 0),
        ("repetitions", -1),
        ("max_abs_error", float("nan")),
        ("model_sha256", "invalid"),
    ],
)
def test_invalid_manifest_rejected(protocol_run, field, value):
    from tinyml_deployer.profiler import validate_manifest

    manifest, _ = protocol_run
    manifest[field] = value
    with pytest.raises(ValueError):
        validate_manifest(manifest)


def test_serial_transport_reassembles_partial_records(protocol_run, tmp_path):
    """Exercise pyserial over a PTY; these are protocol fixtures, not device metrics."""
    import json
    import os
    import select
    import sys
    import threading
    import time

    if sys.platform == "win32":
        pytest.skip("POSIX pseudo-terminals are unavailable on Windows")
    import pty

    from tinyml_deployer.profiler import profile_device

    manifest, records = protocol_run
    path = tmp_path / "deployment.json"
    path.write_text(json.dumps(manifest))
    master, slave = pty.openpty()
    errors = []

    def send_records() -> None:
        try:
            readable, _, _ = select.select([master], [], [], 10)
            assert readable, "No profiling request received"
            request = os.read(master, 256).decode().strip()
            nonce = request.split()[1]
            for record in records:
                record["run_id"] = nonce
            payload = ("TINYML " + json.dumps(records[0]) + "\n").encode()
            # Cross the serial read timeout to exercise partial-line handling.
            os.write(master, payload[:20])
            time.sleep(0.65)
            os.write(master, payload[20:])
            for record in records[1:]:
                os.write(master, ("TINYML " + json.dumps(record) + "\n").encode())
        except (OSError, AssertionError, IndexError, UnicodeError) as exc:
            errors.append(exc)

    thread = threading.Thread(target=send_records, daemon=True)
    thread.start()
    try:
        result = profile_device(os.ttyname(slave), str(path), timeout=10)
        thread.join(timeout=10)
        assert not errors
        assert result.observations == 2
    finally:
        os.close(master)
        os.close(slave)
