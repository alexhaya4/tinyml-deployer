# README figures and evidence

The README visuals are documentation assets. They do not introduce a new application UI or change the CLI's report format.

| Asset | Source |
| --- | --- |
| `assets/pipeline.svg` | Editable vector diagram of the implemented conversion, build and execution boundaries |
| `assets/prediction-agreement.svg` | Matplotlib plot of actual sample errors in the archived QEMU reports |
| `assets/evidence-screenshot.png` | Chromium screenshot of the generated documentation evidence page |
| `assets/capture.json` | Browser version, viewport and capture settings |

## Source evidence

The byte-for-byte JSON reports in [evidence/qemu-2026-10-13](evidence/qemu-2026-10-13/) were downloaded from [CI run 38444732537](https://github.com/alexhaya4/tinyml-deployer/actions/runs/38444732537), testing commit `71612ef0016fdcc61e271b842777dcd57696a3fd`. Both downloaded ZIP SHA-256 values matched the artifact digests returned by GitHub. [provenance.json](evidence/qemu-2026-10-13/provenance.json) records those digests, the extracted report hashes and source links.

The snapshot is deliberately historical. It does not automatically claim to describe a later commit. Retain its original identity if new evidence is added.

These are deterministic integration fixtures, not application datasets. Each graph uses eight supplied validation samples and three repetitions. The float32 dense graph and INT8 convolutional graph differ in architecture; the results are not a matched quantization comparison. All timing and memory observations belong to emulator execution. No physical performance or task-accuracy claims are made.

## Regenerate

From the repository root, install the documentation-only plotting dependency and run:

```bash
python -m pip install matplotlib==3.10.8
python docs/build_visuals.py
```

The generator verifies the archived report checksums and checks observation counts, record uniqueness, maximum error and the reported prediction gate. It then recreates the SVG plot and `docs/evidence/qemu-2026-10-13/index.html` from those reports. No measurements are embedded in the page template.

To recapture the screenshot with Node.js and Playwright available:

```bash
npm install --no-save --package-lock=false playwright
npx playwright install chromium
node docs/capture_visuals.cjs
```

Use a disposable tooling environment if you do not want `node_modules` in your checkout. `NODE_PATH` may point to an existing installation. `PLAYWRIGHT_CHROMIUM_EXECUTABLE` optionally selects a local Chromium executable. Capture versions are recorded in `assets/capture.json`; browser or font changes can alter raster output.

The capture opens the real generated HTML, waits for fonts, checks both report cards and verifies no horizontal overflow at 390, 768 and 1280 pixels before taking a full-page screenshot. No image-generation model, composited browser frame or fabricated terminal session is used.

Open the HTML locally to inspect the original report links and expandable model identities. The page has no network dependencies, analytics or client-side scripts. Its layout uses restrained cards, clear metadata and explicit evidence boundaries, informed by the [Beautiful UI reference](https://www.beautifului.dev/) requested for the project; no third-party component code or imagery is copied.
