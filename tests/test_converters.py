"""Optional real ONNX conversion tests. CI installs the ONNX extra."""

from pathlib import Path

import numpy as np
import pytest

from tinyml_deployer.converters import convert_onnx_to_tflite, ensure_tflite
from tinyml_deployer.evaluation import interpreter_for, predict
from tinyml_deployer.quantizer import quantize_model


@pytest.fixture
def onnx_source(tmp_path):
    onnx = pytest.importorskip("onnx")
    # A missing optional extra may skip; a broken installed extra must fail.
    import onnx2tf  # noqa: F401
    from onnx import TensorProto, helper, numpy_helper

    weights = np.array([[0.8], [-0.3]], np.float32)
    graph = helper.make_graph(
        [
            helper.make_node("MatMul", ["input", "weight"], ["product"]),
            helper.make_node("Sigmoid", ["product"], ["output"]),
        ],
        "conversion",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 2])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 1])],
        [numpy_helper.from_array(weights, "weight")],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=9
    )
    path = tmp_path / "network.onnx"
    onnx.save(model, path)
    return str(path)


def test_real_onnx_predictions(onnx_source, tmp_path):
    ort = pytest.importorskip("onnxruntime")
    result = convert_onnx_to_tflite(onnx_source, str(tmp_path / "converted"))
    source = ort.InferenceSession(onnx_source, providers=["CPUExecutionProvider"])
    target = interpreter_for(Path(result.converted_path).read_bytes())
    for sample in np.linspace(-1, 1, 20, dtype=np.float32).reshape(10, 1, 2):
        expected = source.run(None, {"input": sample})[0]
        np.testing.assert_allclose(predict(target, sample), expected, atol=1e-6, rtol=0)


def test_onnx_quantized_default_output_survives_cleanup(onnx_source):
    data = np.linspace(-0.9, 0.9, 20, dtype=np.float32).reshape(10, 1, 2)
    result = quantize_model(
        onnx_source,
        quantization_type="float16",
        validation_samples=data,
        max_abs_error=0.001,
    )
    assert Path(result.output_path) == Path(onnx_source).with_name(
        "network_quant.tflite"
    )
    assert Path(result.output_path).is_file()
    assert result.evaluation.passed


def test_context_cleans_intermediate_only(onnx_source):
    with ensure_tflite(onnx_source) as path:
        assert Path(path).is_file()
    assert not Path(path).exists()
    assert Path(onnx_source).is_file()


def test_existing_output_untouched(onnx_source, tmp_path):
    out = tmp_path / "converted"
    out.mkdir()
    existing = out / "network.tflite"
    existing.write_bytes(b"user artifact")
    with pytest.raises(FileExistsError):
        convert_onnx_to_tflite(onnx_source, str(out))
    assert existing.read_bytes() == b"user artifact"
