"""Convert ONNX models to TFLite format for MCU deployment."""

from __future__ import annotations

import tempfile
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path


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

    import onnx
    import onnx2tf
    import tensorflow as tf

    from tinyml_deployer.io_utils import write_new_file

    # Validate the actual input graph before invoking the converter.
    onnx.checker.check_model(str(src))
    out = Path(output_dir)
    tflite_path = out / (src.stem + ".tflite")
    if tflite_path.exists():
        raise FileExistsError(f"Output already exists: {tflite_path}")
    with tempfile.TemporaryDirectory(prefix="tinyml_convert_") as directory:
        onnx2tf.convert(
            input_onnx_file_path=str(src),
            output_folder_path=directory,
            non_verbose=True,
            not_use_onnxsim=True,
        )
        # Only accept an unambiguous float32 artifact from this conversion.
        candidates = list(Path(directory).glob("*_float32.tflite"))
        if len(candidates) != 1:
            raise RuntimeError(
                "ONNX conversion did not produce exactly one float32 TFLite model."
            )
        content = candidates[0].read_bytes()
        interpreter = tf.lite.Interpreter(model_content=content)
        interpreter.allocate_tensors()
        write_new_file(tflite_path, content)

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
            f"Unsupported model format '{suffix}'. Supported formats: .tflite, .onnx"
        )
