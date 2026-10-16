# Firmware validation fixtures

`prepare_firmware_checks.py --output DIRECTORY` creates two actual ESP-IDF projects for build validation: a float32 dense graph with ADD/MUL and nonlinear activations, and an INT8 CNN. Inputs are deterministic integration fixtures. No device results are generated.

```bash
python scripts/prepare_firmware_checks.py --output firmware_checks
# In a shell with ESP-IDF 5.5.1 exported:
idf.py -C firmware_checks/float32_project set-target esp32s3
idf.py -C firmware_checks/float32_project build
idf.py -C firmware_checks/int8_project set-target esp32s3
idf.py -C firmware_checks/int8_project build
```

The CI workflow runs these builds and preserves binaries, maps, manifests and dependency locks. Run `tinyml-deployer emulate firmware_checks/float32_project --report float32-emulation.json` and the equivalent INT8 command to execute the binaries under pinned QEMU. Hardware measurements require flashing and the real `profile` command.
