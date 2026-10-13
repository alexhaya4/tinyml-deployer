# Migrating from 0.1 to 0.2 development

This is a breaking development update. The published 0.1 package is unchanged.

| Previous behavior | New requirement |
| --- | --- |
| Rebuild dense networks from TFLite tensors with assumed activations | Supply an original Keras, SavedModel or ONNX source for quantization |
| Random INT8 calibration | Supply representative float NPY calibration samples |
| Quantize without comparing predictions | Supply validation samples and an explicit maximum absolute error |
| C filenames containing C++ | Generated inference and model sources use `.cpp` |
| Fixed resolver unrelated to the graph | Register exactly the serialized built-in operators in the supported mapping |
| Automatically choose a heuristic arena | Supply `--arena-size`; validate actual allocation and linker memory use |
| Generate incomplete STM32 HAL stubs | STM32 generation now fails explicitly; analysis and estimates remain available |
| Report desktop delegate nodes as model operators | Read serialized operators and versions from the FlatBuffer |
| Claim benchmark estimates as performance | Use `profile` for actual serial measurements; `benchmark` stays labeled analytical |
| Python versions with unverified dependency combinations | Python 3.11/3.12 with TensorFlow 2.19.1 and pinned ONNX converters |

Python callers of `quantize_model` must provide `validation_samples` and `max_abs_error`; INT8 also needs a representative-dataset factory. `deploy_model` now requires keyword arguments `arena_size`, `validation_data` and `max_abs_error`. `QuantizationResult.original_size` refers to the float TFLite baseline, not the source archive's size.

Generated `model_run` takes input and output byte counts and returns a status. Call `model_init` first. Inputs and outputs use the native tensor dtype; obtain quantization parameters from `deployment.json`. The wrapper is single-instance and requires external serialization. `model_free` disables inference without reclaiming static storage; `model_init` can enable it again.

Existing files are never intentionally overwritten. Choose a new model/report path and a new project directory for every generation. Quantization failures leave no output model. Device reports that exceed the numerical error gate remain available for inspection and cause a nonzero CLI exit.

Historical launch material in `docs/blog_post.md` describes version 0.1 and is not an installation or validation reference for this branch.
