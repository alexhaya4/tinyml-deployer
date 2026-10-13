"""TFLite Micro resolver registrations shared by analysis and generation."""

from __future__ import annotations

# These names are API methods in the pinned esp-tflite-micro dependency.
# Availability does not prove a particular operator version or dtype can run.
RESOLVER_METHODS = {
    "ADD": "AddAdd",
    "AVERAGE_POOL_2D": "AddAveragePool2D",
    "CONCATENATION": "AddConcatenation",
    "CONV_2D": "AddConv2D",
    "DEPTHWISE_CONV_2D": "AddDepthwiseConv2D",
    "DEQUANTIZE": "AddDequantize",
    "FULLY_CONNECTED": "AddFullyConnected",
    "LOGISTIC": "AddLogistic",
    "MAX_POOL_2D": "AddMaxPool2D",
    "MEAN": "AddMean",
    "MUL": "AddMul",
    "PAD": "AddPad",
    "QUANTIZE": "AddQuantize",
    "RELU": "AddRelu",
    "RELU6": "AddRelu6",
    "RESHAPE": "AddReshape",
    "SOFTMAX": "AddSoftmax",
    "SPLIT": "AddSplit",
    "TANH": "AddTanh",
}
