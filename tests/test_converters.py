"""Tests for tinyml_deployer.converters."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tinyml_deployer.converters import (
    ConversionResult,
    _check_onnx_deps,
    convert_onnx_to_tflite,
    ensure_tflite,
    is_onnx_model,
)
from tests.conftest import SINE_MODEL_PATH


class TestIsOnnxModel:
    """Test the is_onnx_model() helper."""

    def test_tflite_returns_false(self) -> None:
        assert is_onnx_model("model.tflite") is False

    def test_onnx_returns_true(self) -> None:
        assert is_onnx_model("model.onnx") is True

    def test_onnx_uppercase(self) -> None:
        assert is_onnx_model("model.ONNX") is True

    def test_other_extension_returns_false(self) -> None:
        assert is_onnx_model("model.h5") is False

    def test_no_extension_returns_false(self) -> None:
        assert is_onnx_model("model") is False


class TestEnsureTflite:
    """Test the ensure_tflite() context manager."""

    def test_tflite_passthrough(self, sine_model_path: str) -> None:
        """A .tflite path should be yielded unchanged."""
        with ensure_tflite(sine_model_path) as result:
            assert result == sine_model_path

    def test_unsupported_format_raises(self, tmp_path: Path) -> None:
        """A non-.tflite, non-.onnx file should raise ValueError."""
        bad_file = tmp_path / "model.h5"
        bad_file.write_bytes(b"fake")
        with pytest.raises(ValueError, match="Unsupported model format"):
            with ensure_tflite(str(bad_file)):
                pass

    @patch("tinyml_deployer.converters.convert_onnx_to_tflite")
    def test_onnx_calls_convert(
        self,
        mock_convert: MagicMock,
        fake_onnx_path: str,
    ) -> None:
        """An .onnx path should trigger conversion."""
        mock_convert.return_value = ConversionResult(
            original_path=fake_onnx_path,
            converted_path=SINE_MODEL_PATH,
            original_format="onnx",
            original_size_bytes=100,
            converted_size_bytes=200,
        )

        with ensure_tflite(fake_onnx_path) as result:
            assert result == SINE_MODEL_PATH
        mock_convert.assert_called_once()


class TestCheckOnnxDeps:
    """Test _check_onnx_deps() import validation."""

    def test_missing_onnx_raises(self) -> None:
        """When onnx is not importable, a clear ImportError is raised."""
        with patch.dict(sys.modules, {"onnx": None}):
            with pytest.raises(ImportError, match="pip install tinyml-deployer"):
                _check_onnx_deps()

    def test_missing_onnx2tf_raises(self) -> None:
        """When onnx2tf is not importable, a clear ImportError is raised."""
        fake_onnx = MagicMock()
        with patch.dict(sys.modules, {"onnx": fake_onnx, "onnx2tf": None}):
            with pytest.raises(ImportError, match="pip install tinyml-deployer"):
                _check_onnx_deps()


class TestConvertOnnxToTflite:
    """Test convert_onnx_to_tflite() with mocked onnx2tf."""

    def test_file_not_found(self, tmp_path: Path) -> None:
        with patch("tinyml_deployer.converters._check_onnx_deps"):
            with pytest.raises(FileNotFoundError):
                convert_onnx_to_tflite(
                    str(tmp_path / "nonexistent.onnx"),
                    str(tmp_path / "out"),
                )

    @patch("tinyml_deployer.converters._check_onnx_deps")
    def test_successful_conversion(
        self,
        mock_check: MagicMock,
        fake_onnx_path: str,
        tmp_path: Path,
    ) -> None:
        """Mock onnx2tf.convert() and verify the result."""
        out_dir = tmp_path / "converted"
        sine_path = SINE_MODEL_PATH

        # Create a mock onnx2tf module
        mock_onnx2tf = MagicMock()

        def fake_convert(**kwargs: object) -> None:
            out = Path(str(kwargs["output_folder_path"]))
            out.mkdir(parents=True, exist_ok=True)
            shutil.copy(sine_path, str(out / "model_float32.tflite"))

        mock_onnx2tf.convert.side_effect = fake_convert

        with patch.dict(sys.modules, {"onnx2tf": mock_onnx2tf}):
            result = convert_onnx_to_tflite(fake_onnx_path, str(out_dir))

        assert result.original_format == "onnx"
        assert result.converted_path.endswith(".tflite")
        assert Path(result.converted_path).exists()
        assert result.original_size_bytes > 0
        assert result.converted_size_bytes > 0

    @patch("tinyml_deployer.converters._check_onnx_deps")
    def test_conversion_no_output_raises(
        self,
        mock_check: MagicMock,
        fake_onnx_path: str,
        tmp_path: Path,
    ) -> None:
        """If onnx2tf produces no .tflite, a RuntimeError is raised."""
        out_dir = tmp_path / "empty_out"
        mock_onnx2tf = MagicMock()

        def fake_convert(**kwargs: object) -> None:
            out = Path(str(kwargs["output_folder_path"]))
            out.mkdir(parents=True, exist_ok=True)
            # Produce no .tflite file

        mock_onnx2tf.convert.side_effect = fake_convert

        with patch.dict(sys.modules, {"onnx2tf": mock_onnx2tf}):
            with pytest.raises(RuntimeError, match="no .tflite file"):
                convert_onnx_to_tflite(fake_onnx_path, str(out_dir))
