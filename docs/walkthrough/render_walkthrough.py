"""Render a captioned replay from archived command output and real JSON reports.

Requires Pillow, DejaVu fonts and ffmpeg. Run from the repository root:
    python docs/walkthrough/render_walkthrough.py
The output is an edited replay, not a screen recording or real-time telemetry.
"""

from __future__ import annotations

import json
import subprocess
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent / "v0.2.0-beta.1"
EVIDENCE = ROOT / "evidence"
W, H, FPS = 1280, 720, 12
BG, CARD, BORDER = "#111518", "#1b2227", "#344147"
FG, MUTED, GREEN, BLUE = "#f0f4f5", "#a4b3bc", "#83dbbc", "#91b7ff"
DURATIONS = [6, 9, 8, 9, 9, 7, 6]
TITLES = ["TinyML, with evidence.", "Quantize with an error budget.",
          "Generate a checked C++ project.", "Boot real float32 firmware.",
          "Validate the INT8 execution path.", "Keep the results inspectable.",
          "Ready to explore. Hardware comes next."]
SUBTITLES = ["Alex Haya  /  v0.2.0-beta.1  /  ESP32-S3",
             "Sine example: original Keras graph, supplied calibration and held-out inputs.",
             "Sine example: project generation and static preflight, before compilation.",
             "CI reference fixture: dense graph, compiled with ESP-IDF 5.5.1.",
             "CI reference fixture: convolutional graph, with ESP-NN optimizations enabled.",
             "Raw command output, model identities and complete inference records.",
             "Source installation  /  Python 3.11 or 3.12  /  Beta release"]
COMMANDS = {r["label"]: r for r in json.loads((EVIDENCE / "commands.json").read_text())}
Q = json.loads((EVIDENCE / "quantization.json").read_text())
REPORTS = {k: json.loads((EVIDENCE / f"{k}.json").read_text()) for k in ("float32", "int8")}
MANIFEST = json.loads((EVIDENCE / "sine_project/deployment.json").read_text())


def font(size: int, bold: bool = False, mono: bool = False) -> ImageFont.FreeTypeFont:
    name = "DejaVuSansMono" if mono else "DejaVuSans"
    return ImageFont.truetype(f"/usr/share/fonts/truetype/dejavu/{name}{'-Bold' if bold else ''}.ttf", size)


def label(d: ImageDraw.ImageDraw, xy: tuple[int, int], value: str, size: int = 20,
          fill: str = FG, bold: bool = False, mono: bool = False) -> None:
    d.text(xy, value, font=font(size, bold, mono), fill=fill)


def card(d: ImageDraw.ImageDraw, box: tuple[int, int, int, int]) -> None:
    d.rounded_rectangle(box, radius=18, fill=CARD, outline=BORDER, width=1)


def metric(d: ImageDraw.ImageDraw, y: int, title: str, value: str, detail: str = "") -> None:
    label(d, (920, y), title.upper(), 13, MUTED, True)
    label(d, (920, y + 25), value, 30, GREEN, True)
    if detail:
        label(d, (920, y + 66), detail, 15, MUTED)


def terminal(d: ImageDraw.ImageDraw, lines: list[str], progress: float, size: int = 19) -> None:
    card(d, (48, 204, 880, 610))
    label(d, (72, 220), "CAPTURED OUTPUT  /  EXCERPTS", 12, MUTED, True)
    count = min(len(lines), max(1, int(len(lines) * min(1, progress * 3))))
    for i, line in enumerate(lines[:count]):
        label(d, (72, 258 + i * 25), line, size, GREEN if line.startswith("$") else FG, mono=True)


def frame(scene: int, progress: float, elapsed: float, preview: bool = False) -> Image.Image:
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    for x in range(24, W, 32):
        for y in range(24, H, 32):
            d.ellipse((x, y, x+1, y+1), fill="#263138")
    label(d, (48, 28), "AH  /  TINYML DEPLOYER", 15, GREEN, True)
    label(d, (945, 28), "v0.2.0-beta.1   /   QEMU", 15, MUTED)
    label(d, (48, 91), TITLES[scene], 38, FG, True)
    label(d, (48, 151), SUBTITLES[scene], 19, MUTED)
    if scene == 0:
        for i, (title, text, detail) in enumerate([
            ("01 / CONVERT", "Original graphs", "Measured prediction error"),
            ("02 / GENERATE", "Checked C++", "Explicit resource budgets"),
            ("03 / EXECUTE", "ESP32-S3 in QEMU", "Compiled firmware, real output"),
        ]):
            x = 48 + i * 402
            card(d, (x, 259, x + 380, 506))
            label(d, (x+24, 286), title, 15, GREEN, True)
            label(d, (x+24, 340), text, 27, FG, True)
            label(d, (x+24, 397), detail, 18, MUTED)
            width = int(330 * min(1, max(0, progress * 3 - i)))
            if width:
                d.rounded_rectangle((x+24, 465, x+24+width, 469), 2, fill=GREEN)
        label(d, (48, 552), "A captioned walkthrough of actual CLI runs and archived reports.", 22)
        label(d, (48, 587), "No physical board is required for this execution path.", 20, MUTED)
    elif scene == 1:
        out = COMMANDS["quantize"]["stdout"]
        start = out.index("Quantization Result")
        lines = ["$ tinyml-deployer quantize sine/sine.keras ...", ""]
        lines += [line.strip() for line in out[start:].splitlines() if line.strip()]
        terminal(d, lines, progress, 18)
        card(d, (900, 204, 1232, 610))
        metric(d, 232, "Maximum absolute error", f"{Q['evaluation']['max_absolute_error']:.5f}", "Limit: 0.05  /  Gate passed")
        metric(d, 350, "Validation samples", str(Q["evaluation"]["samples"]), "Separate from calibration")
        metric(d, 466, "Model bytes", f"{Q['original_size']} → {Q['quantized_size']}", "INT8 grew in this small example")
    elif scene == 2:
        files = ["main/inference.cpp", "main/inference.h", "main/model_data.cpp", "main/validation_data.h", "main/main.cpp", "main/idf_component.yml", "CMakeLists.txt", "deployment.json"]
        terminal(d, ["$ tinyml-deployer deploy sine/sine_int8.tflite ...", "", "Generated files (selected):", *["  " + f for f in files], "", "Static preflight: passed"], progress, 18)
        card(d, (900, 204, 1232, 610))
        metric(d, 232, "Arena reservation", f"{MANIFEST['arena_reserved_bytes']:,} B", "Explicit budget, not usage")
        metric(d, 350, "Generated API", "Checked C++", "Operators + buffer validation")
        metric(d, 466, "Sine firmware build", "Not run here", "Next: compiled CI fixtures")
    elif scene in (3, 4):
        kind = "float32" if scene == 3 else "int8"
        r = REPORTS[kind]
        v = r["validation"]
        lines = [f"$ tinyml-deployer emulate firmware-{kind}-esp32s3", f"  --qemu <pinned-qemu> --report {kind}.json", "", "Boot log (captured report excerpt):"]
        lines += [line for line in r["console_tail"] if "ESP-IDF v" in line or "Project name:" in line or "Calling app_main()" in line]
        lines += ["", "CLI result:"]
        for line in COMMANDS[f"emulate-{kind}"]["stdout"].splitlines():
            lines.extend(textwrap.wrap(line, 67))
        terminal(d, lines, progress, 18)
        card(d, (900, 204, 1232, 610))
        metric(d, 232, "Observations", str(v["observations"]), "8 inputs x 3 repetitions")
        metric(d, 350, "Maximum prediction error", f"{v['max_absolute_error']:.7g}", f"Limit: {v['max_allowed_error']:g}")
        metric(d, 466, "Prediction agreement", "Passed", "Versus this model on desktop")
    elif scene == 5:
        for i, kind in enumerate(("float32", "int8")):
            x = 48 + i * 604
            r = REPORTS[kind]
            v = r["validation"]
            card(d, (x, 226, x + 580, 510))
            label(d, (x+24, 250), f"{kind.upper()}.JSON  /  ACTUAL REPORT", 16, GREEN, True)
            fields = [("measurement_kind", r["measurement_kind"]), ("prediction_agreement_passed", str(v["prediction_agreement_passed"]).lower()), ("hardware_validation", r["hardware_validation"]), ("hardware_timing_valid", str(r["hardware_timing_valid"]).lower())]
            for j, (key, value) in enumerate(fields):
                label(d, (x+24, 300+j*35), f"{key}: {value}", 17, FG, mono=True)
            label(d, (x+24, 459), f"model SHA256: {v['model_sha256'][:24]}...", 16, MUTED, mono=True)
        label(d, (48, 545), "These are different test graphs, each checked against its own model.", 22)
        label(d, (48, 581), "They do not establish quantization savings, task accuracy or board speed.", 20, MUTED)
    else:
        card(d, (48, 228, 1232, 590))
        label(d, (78, 259), "SOURCE + COMMANDS + REPORTS + VIDEO", 16, GREEN, True)
        label(d, (78, 310), "github.com/alexhaya4/tinyml-deployer", 33, FG, True)
        label(d, (78, 378), "Try the sine example and rerun the QEMU fixtures.", 25)
        label(d, (78, 425), "Next validation step: flash a board and collect serial results.", 23, MUTED)
        label(d, (78, 519), "Alex Haya  /  Embedded systems + applied ML", 20, BLUE)
    footer = "Edited replay of recorded CLI output  |  Hardware validation pending"
    if preview:
        footer = "Animated preview  |  Watch the full 54-second walkthrough"
    label(d, (48, 645), footer, 15, MUTED)
    label(d, (1132, 645), f"{scene+1:02d} / 07", 15, GREEN, mono=True)
    for i in range(7):
        x = 48 + i * 171
        d.rounded_rectangle((x, 690, x+158, 694), 2, fill=BORDER)
        fraction = 1 if i < scene else progress if i == scene else 0
        if fraction > 0:
            d.rounded_rectangle((x, 690, x+max(1, int(158*fraction)), 694), 2, fill=GREEN)
    return im


def main() -> None:
    out = ROOT / "walkthrough.mp4"
    if out.exists():
        raise FileExistsError(out)
    command = ["ffmpeg", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-an", "-c:v", "libx264", "-preset", "medium", "-crf", "21", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)]
    elapsed = 0
    with subprocess.Popen(command, stdin=subprocess.PIPE) as process:
        assert process.stdin
        for scene, duration in enumerate(DURATIONS):
            for n in range(duration * FPS):
                process.stdin.write(frame(scene, n / (duration * FPS - 1), elapsed).tobytes())
                elapsed += 1 / FPS
            print(f"Rendered chapter {scene+1}/7", flush=True)
        process.stdin.close()
        if process.wait():
            raise RuntimeError("ffmpeg failed")
    frame(0, 1, 0).save(ROOT / "poster.png")
    frames = []
    for scene in (0, 1, 3, 4):
        for n in range(16):
            frames.append(frame(scene, n / 15, 0, preview=True).resize((960, 540), Image.Resampling.LANCZOS))
    frames[0].save(ROOT / "preview.gif", save_all=True, append_images=frames[1:], duration=250, loop=0, optimize=True)
    for scene in range(7):
        frame(scene, 1, 0).save(ROOT / f"chapter-{scene+1}.png")


if __name__ == "__main__":
    main()
