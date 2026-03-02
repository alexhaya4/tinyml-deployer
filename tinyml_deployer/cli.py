"""CLI entry point for tinyml-deployer."""

from __future__ import annotations

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
    """Deploy TensorFlow Lite models to ESP32 and STM32 microcontrollers."""


@cli.command()
@click.argument("model", type=click.Path(exists=True))
@click.option("--target", "-t", type=str, default="esp32", help="Target MCU (e.g. esp32, stm32f4).")
def analyze(model: str, target: str) -> None:
    """Analyze a TFLite model for target MCU compatibility."""
    from tinyml_deployer.analyzer import analyze_model
    from tinyml_deployer.targets import get_target

    try:
        mcu = get_target(target)
    except ValueError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise SystemExit(1) from None

    console.print(f"\nAnalyzing [bold]{model}[/bold] for [cyan]{mcu.name}[/cyan]...\n")

    try:
        result = analyze_model(model, target)
    except Exception as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise SystemExit(1) from None

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
    mem_table.add_row("Flash usage", f"{_format_bytes(result.flash_usage_bytes)}  (limit: {mcu.flash_kb} KB)")
    mem_table.add_row("Fits in flash", _yes_no(result.fits_in_flash))
    mem_table.add_row("Tensor arena", f"{_format_bytes(result.tensor_arena_bytes)}  (limit: {mcu.ram_kb} KB)")
    mem_table.add_row("Fits in RAM", _yes_no(result.fits_in_ram))
    if result.total_macs > 0:
        mem_table.add_row("Estimated MACs", f"{result.total_macs:,}")
        mem_table.add_row("Estimated latency", f"{result.estimated_latency_ms:.2f} ms")
    console.print(mem_table)
    console.print()

    # -- Compatibility table --
    compat_table = Table(title="Op Compatibility", show_header=False, title_style="bold")
    compat_table.add_column("Property", style="dim")
    compat_table.add_column("Value")
    compat_table.add_row("Target", f"{mcu.name} ({mcu.framework})")
    compat_table.add_row("Compatible", _yes_no(result.compatible))
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
@click.option("--output", "-o", type=click.Path(), default=None, help="Output path for quantized model.")
@click.option(
    "--type", "qtype",
    type=click.Choice(["int8", "float16", "dynamic"]),
    default="int8",
    help="Quantization type (default: int8).",
)
def quantize(model: str, output: str | None, qtype: str) -> None:
    """Quantize a TFLite model for efficient MCU inference."""
    from tinyml_deployer.quantizer import quantize_model

    console.print(
        f"\nQuantizing [bold]{model}[/bold] with [cyan]{qtype}[/cyan] quantization...\n"
    )

    try:
        result = quantize_model(
            input_path=model,
            output_path=output,
            quantization_type=qtype,
        )
    except Exception as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise SystemExit(1) from None

    # -- Summary table --
    table = Table(title="Quantization Result", show_header=False, title_style="bold")
    table.add_column("Property", style="dim")
    table.add_column("Value")
    table.add_row("Input model", model)
    table.add_row("Output model", result.output_path)
    table.add_row("Quantization type", result.quantization_type)
    table.add_row("Original size", _format_bytes(result.original_size))
    table.add_row("Quantized size", _format_bytes(result.quantized_size))
    table.add_row("Compression ratio", f"{result.compression_ratio:.2f}x")
    console.print(table)
    console.print()

    if result.compression_ratio > 1.0:
        console.print("[green]Model successfully compressed.[/green]\n")
    else:
        console.print("[yellow]Quantized model is not smaller than the original.[/yellow]\n")


@cli.command()
@click.argument("model", type=click.Path(exists=True))
@click.option("--target", "-t", type=str, default="esp32", help="Target MCU (e.g. esp32, stm32f4).")
@click.option("--output", "-o", type=click.Path(), default=None, help="Output project directory.")
def deploy(model: str, target: str, output: str | None) -> None:
    """Generate a deployment project for a TFLite model."""
    from tinyml_deployer.deployer import deploy_model

    if output is None:
        from pathlib import Path
        stem = Path(model).stem
        output = f"{stem}_{target}_project"

    console.print(
        f"\nDeploying [bold]{model}[/bold] for [cyan]{target}[/cyan] "
        f"to [bold]{output}[/bold]...\n"
    )

    try:
        result = deploy_model(
            model_path=model,
            target_name=target,
            output_dir=output,
        )
    except Exception as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise SystemExit(1) from None

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
        info_table.add_row("Model size", _format_bytes(result.analysis.model_size_bytes))
        info_table.add_row("Tensor arena", _format_bytes(result.analysis.tensor_arena_bytes))
        info_table.add_row("Target", result.target_name)
        info_table.add_row("Compatible", _yes_no(result.analysis.compatible))
        console.print(info_table)
        console.print()

    console.print(f"[green]Project generated at:[/green] {result.output_dir}\n")


@cli.command()
@click.argument("model", type=click.Path(exists=True))
@click.option("--target", "-t", type=str, default="esp32", help="Target MCU (e.g. esp32, stm32f4).")
@click.option("--compare", is_flag=True, default=False, help="Compare across all supported targets.")
def benchmark(model: str, target: str, compare: bool) -> None:
    """Estimate inference performance on target MCU(s)."""
    from tinyml_deployer.benchmark import benchmark_all_targets, benchmark_model

    if compare:
        console.print(f"\nBenchmarking [bold]{model}[/bold] across all targets...\n")
        try:
            results = benchmark_all_targets(model)
        except Exception as exc:
            console.print(f"[red]Error:[/red] {exc}")
            raise SystemExit(1) from None

        wide = Console(width=100)
        table = Table(title="Target Comparison", title_style="bold", show_lines=True)
        table.add_column("Target", style="cyan", no_wrap=True)
        table.add_column("Clock", justify="right", no_wrap=True)
        table.add_column("FPU", no_wrap=True)
        table.add_column("Latency (ms)", justify="right", no_wrap=True)
        table.add_column("Throughput (inf/s)", justify="right", no_wrap=True)
        table.add_column("Util %", justify="right", no_wrap=True)
        table.add_column("Flash OK", justify="center", no_wrap=True)
        table.add_column("RAM OK", justify="center", no_wrap=True)

        for r in results:
            latency = f"{r.estimated_latency_ms:.4f}" if r.estimated_latency_ms > 0 else "N/A"
            throughput = f"{r.throughput_ips:,.0f}" if r.throughput_ips > 0 else "N/A"
            util = f"{r.ops_utilization_pct:.4f}" if r.ops_utilization_pct > 0 else "N/A"
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
        result = benchmark_model(model, target)
    except Exception as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise SystemExit(1) from None

    table = Table(title="Benchmark Result", show_header=False, title_style="bold")
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
        table.add_row("Ops utilization", f"{result.ops_utilization_pct:.4f} %")
    else:
        table.add_row("Estimated latency", "N/A (no MACs detected)")
    table.add_row("Model size", _format_bytes(result.model_size_bytes))
    table.add_row("Tensor arena", _format_bytes(result.tensor_arena_bytes))
    table.add_row("Fits in flash", _yes_no(result.fits_in_flash))
    table.add_row("Fits in RAM", _yes_no(result.fits_in_ram))
    console.print(table)
    console.print()


if __name__ == "__main__":
    cli()
