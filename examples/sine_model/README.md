# Sine function integration example

This 337-parameter dense network approximates sin(x) on [0, 2*pi]. It provides a small, inspectable model for testing conversion and firmware generation. It does not represent motor monitoring, sensor accuracy or measured MCU performance.

```bash
python examples/sine_model/train.py --output sine_artifacts
```

Training writes a Keras source model, a float TFLite model, 256 calibration inputs and 65 held-out validation inputs to a new directory. The input arrays have shape `[N, 1, 1]`. Training prints a held-out function MSE; quantization separately checks numerical agreement with the float TFLite baseline.

```bash
tinyml-deployer quantize sine_artifacts/sine.keras --type int8 \
  --calibration sine_artifacts/calibration.npy \
  --validation sine_artifacts/validation.npy --max-abs-error 0.05 \
  --output sine_artifacts/sine_int8.tflite --report sine_quantization.json

tinyml-deployer deploy sine_artifacts/sine_int8.tflite --target esp32s3 \
  --validation sine_artifacts/validation.npy --max-abs-error 0.01 \
  --arena-size 65536 --output sine_project
```

Follow the root README for building, flashing and collecting real device measurements. Adjust the explicit error limits for your application. The checked-in `sine_model.tflite` is a legacy analysis fixture; train again to obtain an original Keras source for quantization.
