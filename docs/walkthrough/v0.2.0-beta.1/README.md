# TinyML Deployer CLI walkthrough

[Watch the 54-second walkthrough](walkthrough.mp4) · [Preview](preview.gif) · [Poster](poster.png) · [Command transcript](transcript.md)

The recorded workflow covers INT8 quantization, ESP-IDF project generation and execution of two compiled firmware fixtures under QEMU.

## Results and scope

| Workflow | Preserved result | Evidence |
| --- | --- | --- |
| Sine-model quantization | 65 validation samples; maximum absolute error 0.019417894 against a 0.05 limit | [Quantization report](evidence/quantization.json) |
| ESP-IDF project generation | Explicit 65,536-byte arena reservation; static preflight passed | [Generated project](evidence/sine_project/README.md), [deployment manifest](evidence/sine_project/deployment.json) |
| Float32 dense fixture | 24 QEMU observations; maximum absolute error 8.94069672e-08 | [Float32 report](evidence/float32.json) |
| INT8 convolutional fixture | 24 QEMU observations; maximum absolute error 0 | [INT8 report](evidence/int8.json) |

The sine model grew from 3,180 to 3,408 bytes after quantization. This small example passed its numerical gate without reducing model size. Its generated project was not compiled or emulated in this run.

The QEMU results use separate integration fixtures from [CI run 38454808140](https://github.com/alexhaya4/tinyml-deployer/actions/runs/38454808140), commit [`4de2fe6`](https://github.com/alexhaya4/tinyml-deployer/commit/4de2fe6cf631438b43dafdff065a2153b95461cc). Each fixture is compared with desktop execution of its own deployed model. These results do not establish comparative task accuracy or physical-device performance. Physical-board validation remains pending.

## Run the workflow

Follow the current [sine example](../../../examples/sine_model/README.md) for model training, quantization and project generation, and the [emulator guide](../../emulation.md) for QEMU setup and firmware execution.

To repeat the compiled-fixture checks, generate and build the projects using the [firmware validation instructions](../../../scripts/README.md), then run each through `tinyml-deployer emulate`. Fresh training and execution results may differ from this snapshot.

## Reference records

- [Original commands and output](evidence/commands.json)
- [Model and input artifacts](evidence/sine/)
- [Artifact checksums](evidence/capture-sha256.json)
- [Source commit, CI artifacts and QEMU identity](provenance.json)
