"""Convert ONNX models to TFLite format for MCU deployment."""

from __future__ import annotations

import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Generator


@dataclass
class ConversionResult:
    """Result of converting an ONNX model to TFLite."""

    original_path: str
    converted_path: str
    original_format: str
    original_size_bytes: int
    converted_size_bytes: int


def _check_onnx_deps() -> None:
    """Verify that ONNX conversion dependencies are installed."""
    try:
        import onnx  # noqa: F401
    except ImportError:
        raise ImportError(
            "The 'onnx' package is required for ONNX model support. "
            "Install it with: pip install tinyml-deployer[onnx]"
        ) from None

    try:
        import onnx2tf  # noqa: F401
    except ImportError:
        raise ImportError(
            "The 'onnx2tf' package is required for ONNX model support. "
            "Install it with: pip install tinyml-deployer[onnx]"
        ) from None


def convert_onnx_to_tflite(
    onnx_path: str,
    output_dir: str,
) -> ConversionResult:
    """Convert an ONNX model to TFLite format.

    The conversion pipeline is:
      .onnx -> onnx2tf -> SavedModel -> tf.lite.TFLiteConverter -> .tflite

    Args:
        onnx_path: Path to the source .onnx model file.
        output_dir: Directory where the converted .tflite file will be written.

    Returns:
        ConversionResult with paths and size information.

    Raises:
        FileNotFoundError: If the ONNX model file does not exist.
        ImportError: If onnx or onnx2tf packages are not installed.
        RuntimeError: If the conversion fails.
    """
    _check_onnx_deps()

    src = Path(onnx_path)
    if not src.exists():
        raise FileNotFoundError(f"ONNX model file not found: {onnx_path}")

    import onnx2tf

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    tflite_filename = src.stem + ".tflite"
    tflite_path = out / tflite_filename

    try:
        onnx2tf.convert(
            input_onnx_file_path=str(src),
            output_folder_path=str(out),
            non_verbose=True,
        )

        # onnx2tf writes output as <output_dir>/model_float32.tflite by
        # default. Rename to match the original model name.
        default_output = out / "model_float32.tflite"
        if default_output.exists() and default_output != tflite_path:
            default_output.rename(tflite_path)
        elif not tflite_path.exists():
            # Fallback -- look for any .tflite produced in the output dir
            tflite_files = list(out.glob("*.tflite"))
            if tflite_files:
                tflite_files[0].rename(tflite_path)
            else:
                raise RuntimeError(
                    f"ONNX conversion completed but no .tflite file was "
                    f"produced in {output_dir}"
                )
    except Exception as exc:
        if isinstance(exc, (ImportError, FileNotFoundError, RuntimeError)):
            raise
        raise RuntimeError(
            f"Failed to convert ONNX model '{onnx_path}' to TFLite: {exc}"
        ) from exc

    return ConversionResult(
        original_path=str(src),
        converted_path=str(tflite_path),
        original_format="onnx",
        original_size_bytes=src.stat().st_size,
        converted_size_bytes=tflite_path.stat().st_size,
    )


def is_onnx_model(model_path: str) -> bool:
    """Check whether a file path has the .onnx extension."""
    return Path(model_path).suffix.lower() == ".onnx"


@contextmanager
def ensure_tflite(model_path: str) -> Generator[str, None, None]:
    """Context manager that yields a .tflite file path.

    If the input is already a .tflite file, the original path is yielded
    unchanged. If it is an .onnx file, it is converted to .tflite in a
    temporary directory and the temp path is yielded. The temporary files
    are cleaned up when the context manager exits.

    Args:
        model_path: Path to a .tflite or .onnx model file.

    Yields:
        Path to a .tflite file (original or converted).

    Raises:
        ImportError: If the input is .onnx and ONNX deps are not installed.
        ValueError: If the file extension is not .tflite or .onnx.
    """
    path = Path(model_path)
    suffix = path.suffix.lower()

    if suffix == ".tflite":
        yield str(path)
    elif suffix == ".onnx":
        with tempfile.TemporaryDirectory(prefix="tinyml_onnx_") as tmpdir:
            result = convert_onnx_to_tflite(str(path), tmpdir)
            yield result.converted_path
    else:
        raise ValueError(
            f"Unsupported model format '{suffix}'. "
            f"Supported formats: .tflite, .onnx"
        )
