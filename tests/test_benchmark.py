"""Tests for tinyml_deployer.benchmark."""

from __future__ import annotations

import pytest

from tinyml_deployer.benchmark import BenchmarkResult, benchmark_all_targets, benchmark_model
from tests.conftest import ALL_TARGET_NAMES


class TestBenchmarkModel:
    """Test benchmark_model() on a single target."""

    @pytest.mark.parametrize("target", ALL_TARGET_NAMES)
    def test_returns_benchmark_result(self, sine_model_path: str, target: str) -> None:
        result = benchmark_model(sine_model_path, target)
        assert isinstance(result, BenchmarkResult)
        assert result.target_name == target

    def test_latency_positive(self, sine_model_path: str) -> None:
        result = benchmark_model(sine_model_path, "esp32")
        assert result.estimated_latency_ms > 0

    def test_throughput_positive(self, sine_model_path: str) -> None:
        result = benchmark_model(sine_model_path, "esp32")
        assert result.throughput_ips > 0

    def test_utilization_positive(self, sine_model_path: str) -> None:
        result = benchmark_model(sine_model_path, "esp32")
        assert result.ops_utilization_pct > 0

    def test_total_macs_positive(self, sine_model_path: str) -> None:
        result = benchmark_model(sine_model_path, "esp32")
        assert result.total_macs > 0

    def test_memory_fields(self, sine_model_path: str) -> None:
        result = benchmark_model(sine_model_path, "esp32")
        assert result.model_size_bytes > 0
        assert result.tensor_arena_bytes > 0
        assert result.fits_in_flash is True
        assert result.fits_in_ram is True

    def test_target_specs_populated(self, sine_model_path: str) -> None:
        result = benchmark_model(sine_model_path, "esp32")
        assert result.clock_mhz == 240
        assert result.cycles_per_mac == 10
        assert result.fpu is False


class TestBenchmarkAllTargets:
    """Test benchmark_all_targets() comparison mode."""

    def test_returns_all_targets(self, sine_model_path: str) -> None:
        results = benchmark_all_targets(sine_model_path)
        assert len(results) == len(ALL_TARGET_NAMES)

    def test_sorted_by_latency(self, sine_model_path: str) -> None:
        results = benchmark_all_targets(sine_model_path)
        latencies = [r.estimated_latency_ms for r in results]
        assert latencies == sorted(latencies)

    def test_all_are_benchmark_results(self, sine_model_path: str) -> None:
        results = benchmark_all_targets(sine_model_path)
        assert all(isinstance(r, BenchmarkResult) for r in results)

    def test_fastest_is_stm32h7(self, sine_model_path: str) -> None:
        results = benchmark_all_targets(sine_model_path)
        # STM32H7 has 480 MHz / 2 cyc = highest MAC throughput
        assert results[0].target_name == "stm32h7"
