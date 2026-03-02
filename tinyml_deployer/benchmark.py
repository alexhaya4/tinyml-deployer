"""Estimate inference performance for a TFLite model on target MCUs."""

from __future__ import annotations

from dataclasses import dataclass

from tinyml_deployer.analyzer import ModelAnalysis, analyze_model
from tinyml_deployer.targets import MCUTarget, TARGETS, get_target


@dataclass
class BenchmarkResult:
    """Estimated performance metrics for a model on a single target."""

    target_name: str
    clock_mhz: int
    cycles_per_mac: int
    fpu: bool
    total_macs: int
    estimated_latency_ms: float
    throughput_ips: float
    ops_utilization_pct: float
    model_size_bytes: int
    tensor_arena_bytes: int
    fits_in_flash: bool
    fits_in_ram: bool
    compatible: bool


def benchmark_model(model_path: str, target_name: str) -> BenchmarkResult:
    """Estimate inference performance of a TFLite model on a target MCU.

    Runs the analyzer to obtain MAC counts and memory estimates, then
    computes latency, throughput, and utilization from the target specs.

    Args:
        model_path: Path to the .tflite model file.
        target_name: Name of the target MCU (e.g. "esp32", "stm32f4").

    Returns:
        BenchmarkResult with all estimated metrics.

    Raises:
        FileNotFoundError: If the model file does not exist.
        ValueError: If the target name is not recognized.
    """
    target = get_target(target_name)
    analysis = analyze_model(model_path, target_name)
    return _build_result(analysis, target)


def benchmark_all_targets(model_path: str) -> list[BenchmarkResult]:
    """Benchmark a model across every registered target.

    Args:
        model_path: Path to the .tflite model file.

    Returns:
        List of BenchmarkResult, one per target, sorted by latency
        (fastest first).
    """
    results: list[BenchmarkResult] = []
    for name in sorted(TARGETS):
        results.append(benchmark_model(model_path, name))
    results.sort(key=lambda r: r.estimated_latency_ms)
    return results


def _build_result(analysis: ModelAnalysis, target: MCUTarget) -> BenchmarkResult:
    """Derive benchmark numbers from an analysis and target spec."""
    total_macs = analysis.total_macs

    # Peak theoretical throughput of the target in MACs per second
    peak_mac_per_sec = target.clock_mhz * 1e6 / target.cycles_per_mac

    if total_macs > 0 and peak_mac_per_sec > 0:
        latency_s = total_macs / peak_mac_per_sec
        latency_ms = latency_s * 1000.0
        throughput = 1.0 / latency_s if latency_s > 0 else 0.0
        # Utilization: fraction of peak compute the model would consume
        # if run back-to-back at 100 % duty cycle.
        utilization = (total_macs / peak_mac_per_sec) * 100.0 if peak_mac_per_sec > 0 else 0.0
    else:
        latency_ms = 0.0
        throughput = 0.0
        utilization = 0.0

    return BenchmarkResult(
        target_name=target.name,
        clock_mhz=target.clock_mhz,
        cycles_per_mac=target.cycles_per_mac,
        fpu=target.fpu,
        total_macs=total_macs,
        estimated_latency_ms=latency_ms,
        throughput_ips=throughput,
        ops_utilization_pct=utilization,
        model_size_bytes=analysis.model_size_bytes,
        tensor_arena_bytes=analysis.tensor_arena_bytes,
        fits_in_flash=analysis.fits_in_flash,
        fits_in_ram=analysis.fits_in_ram,
        compatible=analysis.compatible,
    )
