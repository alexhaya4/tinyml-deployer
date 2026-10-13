# TinyML Deployer beta walkthrough

[Watch the 54-second MP4](walkthrough.mp4) · [16-second GIF](preview.gif) · [Static poster](poster.png) · [Complete command transcript](transcript.md)

This is a captioned, silent **edited replay of recorded CLI output**, rendered with Pillow and ffmpeg. It is not a live screen recording. Commands ran successfully; no predictions, terminal results or hardware measurements were invented. The movie selects output excerpts, abbreviates command paths and changes presentation timing. Full commands, stdout, stderr, timestamps and exit codes are in [commands.json](evidence/commands.json).

## What the video shows

| Time | Chapter | Evidence |
| --- | --- | --- |
| 00:00 | Overview | Original graphs, checked C++, compiled ESP32-S3 execution |
| 00:06 | INT8 conversion | [Quantization report](evidence/quantization.json), 65 held-out sine inputs, maximum absolute error 0.019417894 against a 0.05 limit |
| 00:15 | Project generation | [Generated sine project](evidence/sine_project), explicit 65,536-byte arena reservation, static preflight only |
| 00:23 | Float32 firmware | [Fresh local QEMU report](evidence/float32.json), 24 observations, maximum error 8.94069672e-08 |
| 00:32 | INT8 firmware | [Fresh local QEMU report](evidence/int8.json), 24 observations, maximum error 0 |
| 00:41 | Traceability | Model identities, complete records and explicit hardware status |
| 00:48 | Next steps | Repeat the workflow, then collect physical-board results |

The sine model grew from 3,180 to 3,408 bytes after quantization. Its numerical gate passed, but this small example did **not** achieve size savings. The generated sine project was not compiled or emulated during this recording.

The QEMU chapters execute two separate integration fixtures compiled by [main CI run 38454808140](https://github.com/alexhaya4/tinyml-deployer/actions/runs/38454808140), at commit [`4de2fe6`](https://github.com/alexhaya4/tinyml-deployer/commit/4de2fe6cf631438b43dafdff065a2153b95461cc). Float32 uses a dense graph; INT8 uses a convolutional graph. Each is compared with desktop execution of its own deployed model. These observations establish neither comparative task accuracy nor hardware performance. Physical-board validation remains pending.

## Reproduce the capture

1. Install this version from source with Python 3.11 or 3.12 and `python -m pip install -e '.[emulation]'`.
2. Follow the [pinned QEMU setup](../../emulation.md). The archive checksum is recorded in [provenance.json](provenance.json).
3. Download `firmware-float32-esp32s3` and `firmware-int8-esp32s3` from the source CI run. Extract each under `firmware/<artifact-name>/`, with `build/` and `deployment.json` immediately inside each directory. Verify the downloaded ZIP checksums against provenance. If Actions artifacts expire, rerun CI at the recorded commit or follow its workflow to rebuild; byte-identical binaries are not assumed.
4. From the repository root, capture into a **new directory**:

```bash
python scripts/capture_walkthrough.py \
  --output fresh-capture \
  --firmware-root firmware \
  --qemu /absolute/path/to/qemu-system-xtensa
```

Fresh model training and emulator timings may differ. Evaluate the actual reports rather than assuming identical measurements. The archived [sine inputs and models](evidence/sine) permit inspection of this capture, and [capture-sha256.json](evidence/capture-sha256.json) records hashes of the complete captured output.

## Reproduce the visuals

The renderer reads the archived command output and JSON reports. Its terminal panels are a presentation of those records, not a separate application UI. The selected file list is drawn from the archived generated project. No network calls are made by the renderer.

Install Pillow 11.3.0, DejaVu Sans fonts and ffmpeg 5.1 or newer. In a fresh checkout, move the existing `walkthrough.mp4` aside before rendering; the script refuses to replace it.

```bash
python docs/walkthrough/render_walkthrough.py
```

The output is 1280 x 720, 12 fps, H.264 video with no audio. The GIF is a 960 x 540, 16-second preview. All essential information also appears in this document and the text transcript. The visual style follows the portfolio's charcoal cards and restrained green accents, inspired by [Beautiful UI](https://www.beautifului.dev/).
