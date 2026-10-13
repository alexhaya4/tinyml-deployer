# ESP32-S3 firmware validation with QEMU

This workflow executes the same compiled binaries used for physical deployment. It validates startup, model initialization, completed inference runs and numerical agreement. It does not establish hardware latency, power consumption, electrical behavior or sensor accuracy.

## Use GitHub CI

Open the PR's **Checks** tab and inspect both `emulate-firmware` jobs. Each job downloads its compiled firmware artifact, executes it under a pinned emulator, validates all expected observations, and exercises incorrect model identity and timeout failures. Download the corresponding `emulation-float32-esp32s3` or `emulation-int8-esp32s3` report from the workflow's artifacts section. No board, Wokwi account or local installation is required.

## Local Linux setup

The validated environment is Ubuntu 24.04, Python 3.12, ESP-IDF 5.5.1, ESP-NN 1.4.1 and Espressif QEMU `esp_develop_9.2.2_20260417`. The runner supports Linux; on Windows use GitHub CI or a Linux environment through WSL2. These download commands are for x86-64 Linux.

From the repository checkout:

```bash
python -m pip install -e '.[dev,emulation]'
sudo apt-get update
sudo apt-get install -y libgcrypt20 libglib2.0-0 libpixman-1-0 libsdl2-2.0-0 libslirp0
mkdir qemu-20260417
cd qemu-20260417
curl --fail --location --retry 3 --max-time 180 --output qemu.tar.xz \
  https://github.com/espressif/qemu/releases/download/esp-develop-9.2.2-20260417/qemu-xtensa-softmmu-esp_develop_9.2.2_20260417-x86_64-linux-gnu.tar.xz
echo '0eecb2a34a5586c0e59110f77b9343b7b336e82fdb0e1a30e1dc1bab8a547e35  qemu.tar.xz' | sha256sum --check
# Extract only after the checksum check succeeds.
tar -xJf qemu.tar.xz
export PATH="$PWD/qemu/bin:$PATH"
cd ..
qemu-system-xtensa --version
```

The pinned [release](https://github.com/espressif/qemu/releases/tag/esp-develop-9.2.2-20260417) includes fixes and additions for ESP32-S3 vector instructions. Generic upstream QEMU and the older emulator chosen by ESP-IDF 5.5.1's installer are not substitutes for this validated release.

## Run a built project

Generate and build the project as documented in the root README. Keep its `deployment.json`, `sdkconfig`, `build/flasher_args.json`, application, bootloader and partition-table binaries together. Then run:

```bash
tinyml-deployer emulate sine_project --report emulation.json
# For an executable installed outside PATH:
tinyml-deployer emulate sine_project --qemu /absolute/path/to/qemu-system-xtensa \
  --timeout 120 --report emulation.html
```

The command builds a temporary full flash image from checked, non-overlapping input regions using esptool 4.12.0. It snapshots the input bytes, starts QEMU with network interfaces disabled and a disposable flash snapshot, requests inference with a new run ID, and validates the complete response. It terminates and reaps QEMU after success, failure or timeout. It preserves the original project, build files and deployment manifest.

`--timeout` controls firmware execution time. Emulator version checks, flash preparation and bounded process cleanup have separate limits. A complete run that exceeds the numerical error budget writes a report and exits nonzero. Incomplete, crashed or mismatched runs exit nonzero with a bounded console tail and do not produce a success report. Existing report files are preserved.

## Read the evidence correctly

- `measurement_kind` is `emulator` at the report root and in the nested validation result.
- `hardware_validation` stays `not_run`; `hardware_timing_valid` is false.
- Firmware file offsets, sizes and hashes, the combined flash-image hash, emulator binary hash and manifest hash identify the tested artifacts. These provide traceability, not cryptographic attestation.
- `esp_nn_optimized` records the build configuration. The runner never switches off optimized kernels to make a test pass.
- Raw latency values are emulated timer observations. Arena and heap values describe the virtual execution. Physical performance requires a real board.
- Passing prediction comparisons establishes agreement for the supplied samples, not task accuracy or coverage of every model/operator configuration.

To exercise the real-process tests against either compiled CI fixture:

```bash
TINYML_QEMU_PROJECT=firmware_checks/float32_project pytest tests/test_emulator.py -q
TINYML_QEMU_PROJECT=firmware_checks/int8_project pytest tests/test_emulator.py -q
```

The normal test suite skips these execution tests when no compiled project is supplied. The dedicated CI jobs always supply one and require QEMU to be present.
