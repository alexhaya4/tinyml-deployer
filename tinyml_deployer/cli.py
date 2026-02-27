"""CLI entry point for tinyml-deployer."""

import click
from rich.console import Console

from tinyml_deployer import __version__

console = Console()


@click.group()
@click.version_option(version=__version__, prog_name="tinyml-deployer")
def cli() -> None:
    """Deploy TensorFlow Lite models to ESP32 and STM32 microcontrollers."""


@cli.command()
@click.argument("model", type=click.Path(exists=True))
@click.option("--target", "-t", type=str, default="esp32", help="Target MCU (e.g. esp32, stm32f4).")
def analyze(model: str, target: str) -> None:
    """Analyze a TFLite model for target MCU compatibility."""
    console.print("[yellow]analyze:[/yellow] Coming soon")


@cli.command()
@click.argument("model", type=click.Path(exists=True))
@click.option("--output", "-o", type=click.Path(), default=None, help="Output path for quantized model.")
@click.option("--dtype", type=click.Choice(["int8", "float16"]), default="int8", help="Quantization type.")
def quantize(model: str, output: str | None, dtype: str) -> None:
    """Quantize a TFLite model for efficient MCU inference."""
    console.print("[yellow]quantize:[/yellow] Coming soon")


@cli.command()
@click.argument("model", type=click.Path(exists=True))
@click.option("--target", "-t", type=str, required=True, help="Target MCU (e.g. esp32, stm32f4).")
@click.option("--port", "-p", type=str, default=None, help="Serial port for flashing.")
def deploy(model: str, target: str, port: str | None) -> None:
    """Deploy a TFLite model to a target microcontroller."""
    console.print("[yellow]deploy:[/yellow] Coming soon")


@cli.command()
@click.argument("model", type=click.Path(exists=True))
@click.option("--target", "-t", type=str, required=True, help="Target MCU (e.g. esp32, stm32f4).")
@click.option("--runs", "-r", type=int, default=100, help="Number of inference runs.")
def benchmark(model: str, target: str, runs: int) -> None:
    """Benchmark model inference on target MCU."""
    console.print("[yellow]benchmark:[/yellow] Coming soon")


if __name__ == "__main__":
    cli()
