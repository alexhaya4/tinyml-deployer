# Complete command transcript

Actual commands, stdout, stderr and exit status captured for this walkthrough. Machine-specific paths are retained here; the video abbreviates them. Durations measure host commands and are not device benchmarks.

## version

Started: 2026-10-13T20:03:20.504035+00:00. Exit: 0. Host elapsed: 0.066 seconds.

```bash
/workspace/scratch/c078454c1106/tinyml-video-venv/bin/tinyml-deployer --version
```

Standard output:

```text
tinyml-deployer, version 0.2.0b1
```

Standard error:

```text

```

## train

Started: 2026-10-13T20:03:20.570245+00:00. Exit: 0. Host elapsed: 5.916 seconds.

```bash
/workspace/scratch/c078454c1106/tinyml-video-venv/bin/python /workspace/scratch/c078454c1106/tinyml-deployer/examples/sine_model/train.py --output sine --epochs 50
```

Standard output:

```text
Epoch 1/50: training loss 0.416543
Epoch 2/50: training loss 0.696634
Epoch 3/50: training loss 0.450040
Epoch 4/50: training loss 0.323893
Epoch 5/50: training loss 0.326220
Epoch 6/50: training loss 0.362253
Epoch 7/50: training loss 0.216956
Epoch 8/50: training loss 0.249376
Epoch 9/50: training loss 0.188018
Epoch 10/50: training loss 0.383841
Epoch 11/50: training loss 0.310816
Epoch 12/50: training loss 0.239201
Epoch 13/50: training loss 0.190858
Epoch 14/50: training loss 0.151573
Epoch 15/50: training loss 0.209351
Epoch 16/50: training loss 0.297568
Epoch 17/50: training loss 0.233429
Epoch 18/50: training loss 0.239541
Epoch 19/50: training loss 0.122314
Epoch 20/50: training loss 0.122577
Epoch 21/50: training loss 0.063010
Epoch 22/50: training loss 0.125805
Epoch 23/50: training loss 0.064979
Epoch 24/50: training loss 0.143375
Epoch 25/50: training loss 0.029775
Epoch 26/50: training loss 0.097912
Epoch 27/50: training loss 0.102505
Epoch 28/50: training loss 0.063567
Epoch 29/50: training loss 0.042651
Epoch 30/50: training loss 0.245316
Epoch 31/50: training loss 0.045872
Epoch 32/50: training loss 0.257340
Epoch 33/50: training loss 0.095812
Epoch 34/50: training loss 0.229845
Epoch 35/50: training loss 0.317198
Epoch 36/50: training loss 0.274391
Epoch 37/50: training loss 0.120741
Epoch 38/50: training loss 0.099857
Epoch 39/50: training loss 0.052949
Epoch 40/50: training loss 0.204973
Epoch 41/50: training loss 0.097572
Epoch 42/50: training loss 0.147954
Epoch 43/50: training loss 0.245699
Epoch 44/50: training loss 0.215780
Epoch 45/50: training loss 0.221422
Epoch 46/50: training loss 0.026101
Epoch 47/50: training loss 0.134753
Epoch 48/50: training loss 0.120263
Epoch 49/50: training loss 0.041967
Epoch 50/50: training loss 0.055183
Saved artifact at '/tmp/tmptghwfiqh'. The following endpoints are available:

* Endpoint 'serve'
  args_0 (POSITIONAL_ONLY): TensorSpec(shape=(None, 1), dtype=tf.float32, name='keras_tensor')
Output Type:
  TensorSpec(shape=(None, 1), dtype=tf.float32, name=None)
Captures:
  139805235806288: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139805235808208: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139805235809744: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139805235806672: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139805235809936: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139805235807824: TensorSpec(shape=(), dtype=tf.resource, name=None)
Held-out function MSE: 0.146188; artifacts saved to sine
```

Standard error:

```text
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
E0000 00:00:1791921800.922968       6 cuda_dnn.cc:8579] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
E0000 00:00:1791921800.926131       6 cuda_blas.cc:1407] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
W0000 00:00:1791921800.934221       6 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1791921800.934253       6 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1791921800.934255       6 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1791921800.934256       6 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
W0000 00:00:1791921805.540427       6 tf_tfl_flatbuffer_helpers.cc:365] Ignored output_format.
W0000 00:00:1791921805.540456       6 tf_tfl_flatbuffer_helpers.cc:368] Ignored drop_control_dependency.
I0000 00:00:1791921805.543240       6 mlir_graph_optimization_pass.cc:425] MLIR V1 optimization pass is not enabled
```

## quantize

Started: 2026-10-13T20:03:26.486823+00:00. Exit: 0. Host elapsed: 4.134 seconds.

```bash
/workspace/scratch/c078454c1106/tinyml-video-venv/bin/tinyml-deployer quantize sine/sine.keras --type int8 --calibration sine/calibration.npy --validation sine/validation.npy --max-abs-error 0.05 --output sine/sine_int8.tflite --report quantization.json
```

Standard output:

```text
Saved artifact at '/tmp/tinyml_source_qp_hks5e/saved_model'. The following endpoints are available:

* Endpoint 'serve'
  args_0 (POSITIONAL_ONLY): TensorSpec(shape=(None, 1), dtype=tf.float32, name='input_layer')
Output Type:
  TensorSpec(shape=(None, 1), dtype=tf.float32, name=None)
Captures:
  140521181275792: TensorSpec(shape=(), dtype=tf.resource, name=None)
  140521181274640: TensorSpec(shape=(), dtype=tf.resource, name=None)
  140521181278672: TensorSpec(shape=(), dtype=tf.resource, name=None)
  140521181277520: TensorSpec(shape=(), dtype=tf.resource, name=None)
  140521181279248: TensorSpec(shape=(), dtype=tf.resource, name=None)
  140521181279440: TensorSpec(shape=(), dtype=tf.resource, name=None)
                 Quantization Result                  
┌────────────────────────────┬───────────────────────┐
│ Input model                │ sine/sine.keras       │
│ Output model               │ sine/sine_int8.tflite │
│ Quantization type          │ int8                  │
│ Float TFLite baseline size │ 3.1 KB                │
│ Quantized size             │ 3.3 KB                │
│ Compression ratio          │ 0.93x                 │
│ Maximum prediction error   │ 0.019417894           │
│ Validation samples         │ 65                    │
└────────────────────────────┴───────────────────────┘

Quantized model is not smaller than the original.
```

Standard error:

```text
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
E0000 00:00:1791921806.878424      42 cuda_dnn.cc:8579] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
E0000 00:00:1791921806.881732      42 cuda_blas.cc:1407] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
W0000 00:00:1791921806.891395      42 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1791921806.891429      42 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1791921806.891430      42 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1791921806.891432      42 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
W0000 00:00:1791921809.459034      42 tf_tfl_flatbuffer_helpers.cc:365] Ignored output_format.
W0000 00:00:1791921809.459075      42 tf_tfl_flatbuffer_helpers.cc:368] Ignored drop_control_dependency.
I0000 00:00:1791921809.462987      42 mlir_graph_optimization_pass.cc:425] MLIR V1 optimization pass is not enabled
/workspace/scratch/c078454c1106/tinyml-video-venv/lib/python3.12/site-packages/tensorflow/lite/python/interpreter.py:457: UserWarning:     Warning: tf.lite.Interpreter is deprecated and is scheduled for deletion in
    TF 2.20. Please use the LiteRT interpreter from the ai_edge_litert package.
    See the [migration guide](https://ai.google.dev/edge/litert/migration)
    for details.
    
  warnings.warn(_INTERPRETER_DELETION_WARNING)
W0000 00:00:1791921809.771563      42 tf_tfl_flatbuffer_helpers.cc:365] Ignored output_format.
W0000 00:00:1791921809.771597      42 tf_tfl_flatbuffer_helpers.cc:368] Ignored drop_control_dependency.
fully_quantize: 0, inference_type: 6, input_inference_type: INT8, output_inference_type: INT8
```

## analyze

Started: 2026-10-13T20:03:30.620842+00:00. Exit: 0. Host elapsed: 3.225 seconds.

```bash
/workspace/scratch/c078454c1106/tinyml-video-venv/bin/tinyml-deployer analyze sine/sine_int8.tflite --target esp32s3
```

Standard output:

```text

Analyzing sine/sine_int8.tflite for esp32s3...

                         Model Info                          
┌───────────┬───────────────────────────────────────────────┐
│ Path      │ sine/sine_int8.tflite                         │
│ Size      │ 3.3 KB                                        │
│ Input     │ serving_default_input_layer:0  [1, 1]  (int8) │
│ Output    │ StatefulPartitionedCall_1:0  [1, 1]  (int8)   │
│ Operators │ FULLY_CONNECTED                               │
│ Total ops │ 3                                             │
└───────────┴───────────────────────────────────────────────┘

                    Memory Estimates                     
┌────────────────────────────┬──────────────────────────┐
│ Model bytes only           │ 3.3 KB  (limit: 8192 KB) │
│ Model below nominal flash  │ Yes                      │
│ Tensor arena               │ 681 B  (limit: 512 KB)   │
│ Estimate below nominal RAM │ Yes                      │
│ Estimated MACs             │ 288                      │
│ Estimated latency          │ 0.00 ms                  │
└────────────────────────────┴──────────────────────────┘

            Op Compatibility            
┌──────────────────┬───────────────────┐
│ Target           │ esp32s3 (ESP-IDF) │
│ Static preflight │ Yes               │
└──────────────────┴───────────────────┘

╭──────────────────────────────────────────── Warnings ────────────────────────────────────────────╮
│   * Static preflight only: Micro kernel version/type support and arena allocation require        │
│ runtime validation.                                                                              │
│   * Flash counts model bytes only; arena is a heuristic. Linker output and device measurements   │
│ determine actual fit.                                                                            │
╰──────────────────────────────────────────────────────────────────────────────────────────────────╯
```

Standard error:

```text
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
E0000 00:00:1791921810.963763      92 cuda_dnn.cc:8579] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
E0000 00:00:1791921810.966983      92 cuda_blas.cc:1407] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
W0000 00:00:1791921810.975049      92 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1791921810.975073      92 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1791921810.975075      92 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1791921810.975076      92 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
/workspace/scratch/c078454c1106/tinyml-video-venv/lib/python3.12/site-packages/tensorflow/lite/python/interpreter.py:457: UserWarning:     Warning: tf.lite.Interpreter is deprecated and is scheduled for deletion in
    TF 2.20. Please use the LiteRT interpreter from the ai_edge_litert package.
    See the [migration guide](https://ai.google.dev/edge/litert/migration)
    for details.
    
  warnings.warn(_INTERPRETER_DELETION_WARNING)
```

## deploy

Started: 2026-10-13T20:03:33.845992+00:00. Exit: 0. Host elapsed: 3.476 seconds.

```bash
/workspace/scratch/c078454c1106/tinyml-video-venv/bin/tinyml-deployer deploy sine/sine_int8.tflite --target esp32s3 --arena-size 65536 --validation sine/validation.npy --max-abs-error 0.01 --repetitions 10 --output sine_project
```

Standard output:

```text

Deploying sine/sine_int8.tflite for esp32s3 to sine_project...

                                          Generated Files                                           
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ File                                                             ┃ Description                   ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ /workspace/scratch/c078454c1106/tinyml-video-work/recorded/sine… │ Generated deployment artifact │
│ /workspace/scratch/c078454c1106/tinyml-video-work/recorded/sine… │ Generated deployment artifact │
│ /workspace/scratch/c078454c1106/tinyml-video-work/recorded/sine… │ Generated deployment artifact │
│ /workspace/scratch/c078454c1106/tinyml-video-work/recorded/sine… │ Generated deployment artifact │
│ /workspace/scratch/c078454c1106/tinyml-video-work/recorded/sine… │ Generated deployment artifact │
│ /workspace/scratch/c078454c1106/tinyml-video-work/recorded/sine… │ Generated deployment artifact │
│ /workspace/scratch/c078454c1106/tinyml-video-work/recorded/sine… │ Generated deployment artifact │
│ /workspace/scratch/c078454c1106/tinyml-video-work/recorded/sine… │ Generated deployment artifact │
│ /workspace/scratch/c078454c1106/tinyml-video-work/recorded/sine… │ Generated deployment artifact │
│ /workspace/scratch/c078454c1106/tinyml-video-work/recorded/sine… │ Generated deployment artifact │
│ /workspace/scratch/c078454c1106/tinyml-video-work/recorded/sine… │ Generated deployment artifact │
│ /workspace/scratch/c078454c1106/tinyml-video-work/recorded/sine… │ Generated deployment artifact │
│ /workspace/scratch/c078454c1106/tinyml-video-work/recorded/sine… │ Generated deployment artifact │
└──────────────────────────────────────────────────────────────────┴───────────────────────────────┘

        Model Summary         
┌──────────────────┬─────────┐
│ Model size       │ 3.3 KB  │
│ Tensor arena     │ 681 B   │
│ Target           │ esp32s3 │
│ Static preflight │ Yes     │
└──────────────────┴─────────┘

Project generated at: /workspace/scratch/c078454c1106/tinyml-video-work/recorded/sine_project
```

Standard error:

```text
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
E0000 00:00:1791921814.215578     110 cuda_dnn.cc:8579] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
E0000 00:00:1791921814.219509     110 cuda_blas.cc:1407] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
W0000 00:00:1791921814.227711     110 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1791921814.227742     110 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1791921814.227743     110 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1791921814.227744     110 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
/workspace/scratch/c078454c1106/tinyml-video-venv/lib/python3.12/site-packages/tensorflow/lite/python/interpreter.py:457: UserWarning:     Warning: tf.lite.Interpreter is deprecated and is scheduled for deletion in
    TF 2.20. Please use the LiteRT interpreter from the ai_edge_litert package.
    See the [migration guide](https://ai.google.dev/edge/litert/migration)
    for details.
    
  warnings.warn(_INTERPRETER_DELETION_WARNING)
```

## emulate-float32

Started: 2026-10-13T20:03:37.322363+00:00. Exit: 0. Host elapsed: 1.07 seconds.

```bash
/workspace/scratch/c078454c1106/tinyml-video-venv/bin/tinyml-deployer emulate /workspace/scratch/c078454c1106/tinyml-video-work/firmware-float32-esp32s3 --qemu /workspace/scratch/c078454c1106/tinyml-video-work/qemu/qemu/bin/qemu-system-xtensa --report float32.json
```

Standard output:

```text
Emulator validation: 24 observations; maximum prediction error 8.9406967e-08.
Physical board validation remains pending; emulator timing is not hardware performance.
```

Standard error:

```text

```

## emulate-int8

Started: 2026-10-13T20:03:38.393108+00:00. Exit: 0. Host elapsed: 1.112 seconds.

```bash
/workspace/scratch/c078454c1106/tinyml-video-venv/bin/tinyml-deployer emulate /workspace/scratch/c078454c1106/tinyml-video-work/firmware-int8-esp32s3 --qemu /workspace/scratch/c078454c1106/tinyml-video-work/qemu/qemu/bin/qemu-system-xtensa --report int8.json
```

Standard output:

```text
Emulator validation: 24 observations; maximum prediction error 0.
Physical board validation remains pending; emulator timing is not hardware performance.
```

Standard error:

```text

```
