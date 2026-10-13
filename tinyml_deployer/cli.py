"""CLI entry point for tinyml-deployer."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from tinyml_deployer import __version__

console = Console()


def _format_bytes(n: int) -> str:
    """Format a byte count as a human-readable string."""
    if n >= 1024 * 1024:
        return f"{n / (1024 * 1024):.2f} MB"
    if n >= 1024:
        return f"{n / 1024:.1f} KB"
    return f"{n} B"


def _yes_no(value: bool) -> str:
    return "[green]Yes[/green]" if value else "[red]No[/red]"


@click.group()
@click.version_option(version=__version__, prog_name="tinyml-deployer")
def cli() -> None:
    """Validate TinyML models, generate ESP-IDF firmware, and measure inference."""


@cli.command()
@click.argument("model", type=click.Path(exists=True))
@click.option(
    "--target",
    "-t",
    type=str,
    default="esp32",
    help="Target MCU (e.g. esp32, stm32f4).",
)
def analyze(model: str, target: str) -> None:
    """Analyze a TFLite or ONNX model for target MCU compatibility."""
    from tinyml_deployer.analyzer import analyze_model
    from tinyml_deployer.converters import ensure_tflite, is_onnx_model
    from tinyml_deployer.targets import get_target

    try:
        mcu = get_target(target)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    if is_onnx_model(model):
        console.print("[yellow]Converting ONNX model to TFLite...[/yellow]")

    console.print(f"\nAnalyzing [bold]{model}[/bold] for [cyan]{mcu.name}[/cyan]...\n")

    try:
        with ensure_tflite(model) as tflite_path:
            result = analyze_model(tflite_path, target)
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc

    # -- Model info table --
    info_table = Table(title="Model Info", show_header=False, title_style="bold")
    info_table.add_column("Property", style="dim")
    info_table.add_column("Value")
    info_table.add_row("Path", result.model_path)
    info_table.add_row("Size", _format_bytes(result.model_size_bytes))
    for tensor in result.inputs:
        info_table.add_row("Input", f"{tensor.name}  {tensor.shape}  ({tensor.dtype})")
    for tensor in result.outputs:
        info_table.add_row("Output", f"{tensor.name}  {tensor.shape}  ({tensor.dtype})")
    if result.operators:
        unique_ops = sorted(set(result.operators))
        info_table.add_row("Operators", ", ".join(unique_ops))
        info_table.add_row("Total ops", str(len(result.operators)))
    console.print(info_table)
    console.print()

    # -- Memory estimates table --
    mem_table = Table(title="Memory Estimates", show_header=False, title_style="bold")
    mem_table.add_column("Property", style="dim")
    mem_table.add_column("Value")
    mem_table.add_row(
        "Model bytes only",
        f"{_format_bytes(result.flash_usage_bytes)}  (limit: {mcu.flash_kb} KB)",
    )
    mem_table.add_row("Model below nominal flash", _yes_no(result.fits_in_flash))
    mem_table.add_row(
        "Tensor arena",
        f"{_format_bytes(result.tensor_arena_bytes)}  (limit: {mcu.ram_kb} KB)",
    )
    mem_table.add_row("Estimate below nominal RAM", _yes_no(result.fits_in_ram))
    if result.total_macs > 0:
        mem_table.add_row("Estimated MACs", f"{result.total_macs:,}")
        mem_table.add_row("Estimated latency", f"{result.estimated_latency_ms:.2f} ms")
    console.print(mem_table)
    console.print()

    # -- Compatibility table --
    compat_table = Table(
        title="Op Compatibility", show_header=False, title_style="bold"
    )
    compat_table.add_column("Property", style="dim")
    compat_table.add_column("Value")
    compat_table.add_row("Target", f"{mcu.name} ({mcu.framework})")
    compat_table.add_row("Static preflight", _yes_no(result.compatible))
    if result.unsupported_ops:
        compat_table.add_row("Unsupported ops", ", ".join(result.unsupported_ops))
    console.print(compat_table)
    console.print()

    # -- Warnings --
    if result.warnings:
        warning_text = "\n".join(f"  * {w}" for w in result.warnings)
        console.print(Panel(warning_text, title="Warnings", border_style="yellow"))
    else:
        console.print("[green]No warnings. Model looks good for deployment.[/green]")

    console.print()


@cli.command()
@click.argument("model", type=click.Path(exists=True))
@click.option(
    "--calibration",
    type=click.Path(exists=True, dir_okay=False),
    help="Float32 NPY calibration samples.",
)
@click.option(
    "--validation", required=True, type=click.Path(exists=True, dir_okay=False)
)
@click.option("--max-abs-error", required=True, type=click.FloatRange(min=0))
@click.option("--report", type=click.Path())
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    default=None,
    help="Output path for quantized model.",
)
@click.option(
    "--type",
    "qtype",
    type=click.Choice(["int8", "float16", "dynamic"]),
    default="int8",
    help="Quantization type (default: int8).",
)
def quantize(
    model: str,
    output: str | None,
    qtype: str,
    calibration: str | None,
    validation: str,
    max_abs_error: float,
    report: str | None,
) -> None:
    """Quantize a Keras, SavedModel or ONNX source within an error budget."""
    from tinyml_deployer.evaluation import load_samples
    from tinyml_deployer.quantizer import quantize_model
    from tinyml_deployer.reporting import write_report

    try:
        if report and Path(report).exists():
            raise FileExistsError(f"Report already exists: {report}")
        samples = load_samples(validation)
        representative = None
        if calibration:
            calibration_samples = load_samples(calibration)

            def representative() -> Iterator[list[Any]]:
                for sample in calibration_samples:
                    yield [sample.astype("float32")]

        result = quantize_model(
            model,
            output,
            qtype,
            representative,
            validation_samples=samples,
            max_abs_error=max_abs_error,
        )
        if report:
            write_report(result, report)
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc

    # -- Summary table --
    table = Table(title="Quantization Result", show_header=False, title_style="bold")
    table.add_column("Property", style="dim")
    table.add_column("Value")
    table.add_row("Input model", model)
    table.add_row("Output model", result.output_path)
    table.add_row("Quantization type", result.quantization_type)
    table.add_row("Float TFLite baseline size", _format_bytes(result.original_size))
    table.add_row("Quantized size", _format_bytes(result.quantized_size))
    table.add_row("Compression ratio", f"{result.compression_ratio:.2f}x")
    table.add_row(
        "Maximum prediction error", f"{result.evaluation.max_absolute_error:.8g}"
    )
    table.add_row("Validation samples", str(result.evaluation.samples))
    console.print(table)
    console.print()

    if result.compression_ratio > 1.0:
        console.print("[green]Model successfully compressed.[/green]\n")
    else:
        console.print(
            "[yellow]Quantized model is not smaller than the original.[/yellow]\n"
        )


@cli.command()
@click.argument("model", type=click.Path(exists=True))
@click.option(
    "--target",
    "-t",
    type=str,
    default="esp32",
    help="Target MCU (e.g. esp32, stm32f4).",
)
@click.option(
    "--output", "-o", type=click.Path(), default=None, help="Output project directory."
)
@click.option(
    "--arena-size",
    required=True,
    type=click.IntRange(min=1),
    help="Tensor arena reservation in bytes.",
)
@click.option(
    "--validation", required=True, type=click.Path(exists=True, dir_okay=False)
)
@click.option("--max-abs-error", required=True, type=click.FloatRange(min=0))
@click.option("--repetitions", default=10, type=click.IntRange(1, 10000))
@click.option(
    "--flash-budget",
    default=1048576,
    type=click.IntRange(min=65536),
    help="Application partition bytes.",
)
def deploy(
    model: str,
    target: str,
    output: str | None,
    arena_size: int,
    validation: str,
    max_abs_error: float,
    repetitions: int,
    flash_budget: int,
) -> None:
    """Generate a deployment project for a TFLite or ONNX model."""
    from pathlib import Path

    from tinyml_deployer.converters import ensure_tflite, is_onnx_model
    from tinyml_deployer.deployer import deploy_model

    if output is None:
        stem = Path(model).stem
        output = f"{stem}_{target}_project"

    if is_onnx_model(model):
        console.print("[yellow]Converting ONNX model to TFLite...[/yellow]")

    console.print(
        f"\nDeploying [bold]{model}[/bold] for [cyan]{target}[/cyan] "
        f"to [bold]{output}[/bold]...\n"
    )

    try:
        with ensure_tflite(model) as tflite_path:
            result = deploy_model(
                model_path=tflite_path,
                target_name=target,
                output_dir=output,
                arena_size=arena_size,
                validation_data=validation,
                max_abs_error=max_abs_error,
                repetitions=repetitions,
                flash_budget_bytes=flash_budget,
            )
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc

    # -- Generated files table --
    table = Table(title="Generated Files", title_style="bold")
    table.add_column("File", style="cyan")
    table.add_column("Description", style="dim")
    for gf in result.files:
        table.add_row(gf.path, gf.description)
    console.print(table)
    console.print()

    if result.analysis:
        info_table = Table(title="Model Summary", show_header=False, title_style="bold")
        info_table.add_column("Property", style="dim")
        info_table.add_column("Value")
        info_table.add_row(
            "Model size", _format_bytes(result.analysis.model_size_bytes)
        )
        info_table.add_row(
            "Tensor arena", _format_bytes(result.analysis.tensor_arena_bytes)
        )
        info_table.add_row("Target", result.target_name)
        info_table.add_row("Static preflight", _yes_no(result.analysis.compatible))
        console.print(info_table)
        console.print()

    console.print(f"[green]Project generated at:[/green] {result.output_dir}\n")


@cli.command()
@click.argument("model", type=click.Path(exists=True))
@click.option(
    "--target",
    "-t",
    type=str,
    default="esp32",
    help="Target MCU (e.g. esp32, stm32f4).",
)
@click.option(
    "--compare",
    is_flag=True,
    default=False,
    help="Compare across all supported targets.",
)
def benchmark(model: str, target: str, compare: bool) -> None:
    """Estimate inference performance of a TFLite or ONNX model on target MCU(s)."""
    from tinyml_deployer.benchmark import benchmark_all_targets, benchmark_model
    from tinyml_deployer.converters import ensure_tflite, is_onnx_model

    if is_onnx_model(model):
        console.print("[yellow]Converting ONNX model to TFLite...[/yellow]")

    if compare:
        console.print(f"\nBenchmarking [bold]{model}[/bold] across all targets...\n")
        try:
            with ensure_tflite(model) as tflite_path:
                results = benchmark_all_targets(tflite_path)
        except Exception as exc:
            raise click.ClickException(str(exc)) from exc

        wide = Console(width=100)
        table = Table(
            title="Target Comparison (analytical estimates)",
            title_style="bold",
            show_lines=True,
        )
        table.add_column("Target", style="cyan", no_wrap=True)
        table.add_column("Clock", justify="right", no_wrap=True)
        table.add_column("FPU", no_wrap=True)
        table.add_column("Latency (ms)", justify="right", no_wrap=True)
        table.add_column("Throughput (inf/s)", justify="right", no_wrap=True)
        table.add_column("Duty % at 1/s", justify="right", no_wrap=True)
        table.add_column("Model fits flash", justify="center", no_wrap=True)
        table.add_column("Arena estimate fits", justify="center", no_wrap=True)

        for r in results:
            latency = (
                f"{r.estimated_latency_ms:.4f}" if r.estimated_latency_ms > 0 else "N/A"
            )
            throughput = f"{r.throughput_ips:,.0f}" if r.throughput_ips > 0 else "N/A"
            util = (
                f"{r.ops_utilization_pct:.4f}" if r.ops_utilization_pct > 0 else "N/A"
            )
            table.add_row(
                r.target_name,
                f"{r.clock_mhz} MHz",
                _yes_no(r.fpu),
                latency,
                throughput,
                util,
                _yes_no(r.fits_in_flash),
                _yes_no(r.fits_in_ram),
            )

        wide.print(table)
        console.print()
        if results and results[0].total_macs > 0:
            console.print(
                f"Total MACs per inference: [bold]{results[0].total_macs:,}[/bold]\n"
            )
        return

    # Single-target benchmark
    console.print(f"\nBenchmarking [bold]{model}[/bold] on [cyan]{target}[/cyan]...\n")

    try:
        with ensure_tflite(model) as tflite_path:
            result = benchmark_model(tflite_path, target)
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc

    table = Table(
        title="Benchmark Result (analytical estimate)",
        show_header=False,
        title_style="bold",
    )
    table.add_column("Property", style="dim")
    table.add_column("Value")
    table.add_row("Target", result.target_name)
    table.add_row("Clock speed", f"{result.clock_mhz} MHz")
    table.add_row("Cycles per MAC", str(result.cycles_per_mac))
    table.add_row("FPU", _yes_no(result.fpu))
    table.add_row("Total MACs", f"{result.total_macs:,}")
    if result.estimated_latency_ms > 0:
        table.add_row("Estimated latency", f"{result.estimated_latency_ms:.4f} ms")
        table.add_row("Throughput", f"{result.throughput_ips:,.0f} inferences/s")
        table.add_row(
            "Estimated duty at 1 inference/s", f"{result.ops_utilization_pct:.4f} %"
        )
    else:
        table.add_row("Estimated latency", "N/A (no MACs detected)")
    table.add_row("Model size", _format_bytes(result.model_size_bytes))
    table.add_row("Tensor arena", _format_bytes(result.tensor_arena_bytes))
    table.add_row("Model below nominal flash", _yes_no(result.fits_in_flash))
    table.add_row("Estimate below nominal RAM", _yes_no(result.fits_in_ram))
    console.print(table)
    console.print()


@cli.command()
@click.option("--onnx", is_flag=True, help="Also check ONNX conversion dependencies.")
@click.option("--report", type=click.Path())
def doctor(onnx: bool, report: str | None) -> None:
    """Check installed packages and the ESP-IDF build environment."""
    from tinyml_deployer.doctor import diagnose
    from tinyml_deployer.reporting import write_report

    result = diagnose(onnx=onnx)
    if report:
        write_report(result, report)
    for check in result.checks:
        click.echo(
            f"{check['name']}: {'OK' if check['ok'] else 'MISSING/INCOMPATIBLE'} - {check['detail']}"
        )
    if not result.ready:
        raise click.ClickException(
            "Resolve the failed environment checks before building firmware."
        )


@cli.command()
@click.argument("reference", type=click.Path(exists=True, dir_okay=False))
@click.argument("candidate", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--validation", required=True, type=click.Path(exists=True, dir_okay=False)
)
@click.option("--max-abs-error", required=True, type=click.FloatRange(min=0))
@click.option("--report", required=True, type=click.Path())
def evaluate(
    reference: str, candidate: str, validation: str, max_abs_error: float, report: str
) -> None:
    """Compare TFLite predictions on supplied preprocessed validation samples."""
    from tinyml_deployer.evaluation import evaluate_models
    from tinyml_deployer.reporting import write_report

    try:
        result = evaluate_models(reference, candidate, validation, max_abs_error)
        write_report(result, report)
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"Maximum absolute error: {result.max_absolute_error:.8g}")
    if not result.passed:
        raise click.ClickException(
            "Prediction error exceeds the supplied limit. See the report."
        )


@cli.command()
@click.option("--port", required=True)
@click.option("--manifest", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--report", required=True, type=click.Path())
@click.option("--baud", default=115200, type=click.IntRange(min=1))
@click.option("--timeout", default=120.0, type=click.FloatRange(min=0, min_open=True))
def profile(port: str, manifest: str, report: str, baud: int, timeout: float) -> None:
    """Measure generated firmware through its real serial connection."""
    from tinyml_deployer.profiler import profile_device
    from tinyml_deployer.reporting import write_report

    try:
        if Path(report).exists():
            raise FileExistsError(f"Report already exists: {report}")
        result = profile_device(port, manifest, baud=baud, timeout=timeout)
        write_report(result, report)
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(
        f"Median {result.median_latency_us:.3f} us; p95 {result.p95_latency_us:.3f} us"
    )
    if not result.prediction_agreement_passed:
        raise click.ClickException(
            "Device predictions exceed the supplied error limit. See the report."
        )


@cli.command()
@click.argument("project", type=click.Path(exists=True, file_okay=False))
@click.option(
    "--qemu", default="qemu-system-xtensa", help="Pinned Espressif QEMU executable."
)
@click.option("--timeout", default=120.0, type=click.FloatRange(min=0, min_open=True))
@click.option("--report", required=True, type=click.Path())
def emulate(project: str, qemu: str, timeout: float, report: str) -> None:
    """Execute built ESP32-S3 firmware in QEMU and validate its predictions."""
    from tinyml_deployer.emulator import emulate_project
    from tinyml_deployer.reporting import write_report

    try:
        if Path(report).exists():
            raise FileExistsError(f"Report already exists: {report}")
        result = emulate_project(project, qemu_binary=qemu, timeout=timeout)
        write_report(result, report)
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(
        f"Emulator validation: {result.validation.observations} observations; "
        f"maximum prediction error {result.validation.max_absolute_error:.8g}."
    )
    click.echo(
        "Physical board validation remains pending; emulator timing is not hardware performance."
    )
    if not result.validation.prediction_agreement_passed:
        raise click.ClickException(
            "Emulated predictions exceed the supplied error limit. See the report."
        )


if __name__ == "__main__":
    cli()
