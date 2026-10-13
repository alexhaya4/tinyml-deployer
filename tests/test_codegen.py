"""Check that generated resolver and raw model bytes match actual models."""

import re
from pathlib import Path

import pytest

from tinyml_deployer.analyzer import analyze_model
from tinyml_deployer.codegen import generate_inference_wrapper, generate_model_data


def test_model_bytes_round_trip(sine_model_path, tmp_path):
    generate_model_data(sine_model_path, str(tmp_path))
    source = (tmp_path / "model_data.cpp").read_text()
    recovered = bytes(
        int(value, 16) for value in re.findall(r"0x([0-9a-f]{2})", source)
    )
    assert recovered == Path(sine_model_path).read_bytes()
    assert "alignas(16)" in source


def test_resolver_comes_from_graph(sine_model_path, tmp_path):
    analysis = analyze_model(sine_model_path, "esp32s3")
    generate_inference_wrapper(analysis, "ESP-IDF", str(tmp_path), 65536)
    source = (tmp_path / "inference.cpp").read_text()
    assert "resolver.AddFullyConnected()" in source
    assert "resolver.AddConv2D()" not in source
    assert "MODEL_INVALID_BUFFER" in source
    assert "VerifyModelBuffer" in source


def test_arena_must_be_explicit(sine_model_path, tmp_path):
    with pytest.raises(ValueError, match="arena"):
        generate_inference_wrapper(
            analyze_model(sine_model_path, "esp32s3"), "ESP-IDF", str(tmp_path)
        )
