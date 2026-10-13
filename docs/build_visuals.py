"""Regenerate documentation figures and an HTML evidence view from archived CI data."""

from __future__ import annotations

import hashlib
import html
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
EVIDENCE = ROOT / "evidence" / "qemu-2026-10-13"
ASSETS = ROOT / "assets"


def load_evidence() -> tuple[dict, list[tuple[dict, dict]]]:
    """Reject changed or inconsistent evidence before generating any visuals."""
    provenance = json.loads((EVIDENCE / "provenance.json").read_text())
    reports = []
    for entry in provenance["reports"]:
        raw = (EVIDENCE / entry["file"]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != entry["sha256"]:
            raise ValueError(f"Evidence checksum mismatch: {entry['file']}")
        report = json.loads(raw)
        validation = report["validation"]
        records = [r for r in validation["records"] if r["event"] == "sample"]
        checks = (
            report["measurement_kind"] == "emulator",
            validation["measurement_kind"] == "emulator",
            report["hardware_validation"] == "not_run",
            report["hardware_timing_valid"] is False,
            len(records) == validation["observations"],
            len({(r["sample"], r["repetition"]) for r in records}) == len(records),
            max(r["max_abs_error"] for r in records) == validation["max_absolute_error"],
            validation["prediction_agreement_passed"]
            == (validation["max_absolute_error"] <= validation["max_allowed_error"]),
        )
        if not all(checks):
            raise ValueError(f"Inconsistent evidence: {entry['file']}")
        reports.append((entry, report))
    return provenance, reports


def error_figure(reports: list[tuple[dict, dict]]) -> None:
    """Plot observed prediction error; do not treat fixture graphs as matched models."""
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "svg.fonttype": "none", "svg.hashsalt": "tinyml-evidence"})
    figure, axes = plt.subplots(1, len(reports), figsize=(12, 4.4), squeeze=False)
    figure.set_facecolor("#fafbf9")
    for ax, (entry, report) in zip(axes[0], reports, strict=True):
        validation = report["validation"]
        grouped: dict[int, list[float]] = {}
        for record in validation["records"]:
            if record["event"] == "sample":
                grouped.setdefault(record["sample"], []).append(record["max_abs_error"])
        samples = sorted(grouped)
        errors = [max(grouped[index]) for index in samples]
        ax.set_facecolor("#fafbf9")
        ax.bar(samples, errors, color="#087f70", width=0.5, zorder=3)
        ax.scatter(samples, errors, color="#087f70", s=24, zorder=4)
        ax.set_ylim(-5e-9, 1.15e-7)
        ax.set_xticks(samples)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        ax.set_xlabel("Validation sample index")
        ax.set_ylabel("Maximum absolute prediction error")
        ax.set_title(entry["name"].upper(), loc="left", weight="bold", pad=24)
        ax.grid(axis="y", color="#dce4df", zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["bottom", "left"]].set_color("#a4b5ad")
        ax.text(0.98, 0.94,
                f"Observed max: {validation['max_absolute_error']:.3g}\n"
                f"Allowed max: {validation['max_allowed_error']:.3g}",
                transform=ax.transAxes, va="top", ha="right", fontsize=10,
                bbox={"facecolor": "#edf3ef", "edgecolor": "none", "pad": 8})
    figure.suptitle("Prediction agreement in QEMU", x=0.055, ha="left",
                   fontsize=19, weight="bold", y=0.96)
    figure.text(0.055, 0.87,
                "Separate test graphs, each compared with its own desktop TFLite baseline.",
                color="#52665d", fontsize=11)
    figure.text(0.055, 0.015,
                "Each point is the maximum over the recorded repetitions. Error limits lie above the plotted range.",
                color="#52665d", fontsize=9)
    figure.subplots_adjust(top=0.69, bottom=0.18, left=0.08, right=0.97, wspace=0.28)
    figure.savefig(ASSETS / "prediction-agreement.svg", metadata={"Date": None})
    plt.close(figure)


def evidence_page(provenance: dict, reports: list[tuple[dict, dict]]) -> None:
    """Render a documentation view without changing the runtime report format."""
    esc = html.escape
    cards, rows = [], []
    for entry, report in reports:
        val = report["validation"]
        name = esc(entry["name"].upper())
        samples = {r["sample"] for r in val["records"] if r["event"] == "sample"}
        repetitions = {r["repetition"] for r in val["records"] if r["event"] == "sample"}
        status = "PASS" if val["prediction_agreement_passed"] else "FAIL"
        app_files = [f for f in report["firmware_files"] if f["path"].endswith("_project.bin")]
        if len(app_files) != 1:
            raise ValueError("Expected one application binary in this evidence snapshot")
        app = app_files[0]
        cards.append(f'''<article class="card"><div class="card-head"><h2>{name}</h2>
<span class="status">{status}</span></div><div class="observations">{val['observations']}
<span>validated observations</span></div><p class="muted">{len(samples)} samples, {len(repetitions)} repetitions each</p>
<dl><div><dt>Maximum prediction error</dt><dd>{val['max_absolute_error']:.8g}</dd></div>
<div><dt>Configured error limit</dt><dd>{val['max_allowed_error']:.8g}</dd></div>
<div><dt>Application binary</dt><dd>{app['size_bytes']:,} bytes</dd></div>
<div><dt>ESP-NN optimizations</dt><dd>{'Enabled' if report['esp_nn_optimized'] else 'Disabled'}</dd></div></dl>
<a href="{esc(entry['file'])}">Inspect original JSON report</a></article>''')
        rows.append(f"<tr><th>{name}</th><td><code>{esc(val['model_sha256'])}</code></td></tr>")
    page = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>TinyML Deployer | QEMU validation evidence</title><style>
*{{box-sizing:border-box}}body{{margin:0;background:#f5f7f3;color:#19352c;font:16px/1.55 system-ui,sans-serif}}
main{{max-width:1160px;margin:auto;padding:40px 36px}}header{{border-bottom:1px solid #ccd8d0;padding-bottom:24px}}
.eyebrow{{font-size:12px;font-weight:650;letter-spacing:2px;color:#506c61;text-transform:uppercase}}
h1{{font-size:40px;line-height:1.15;letter-spacing:-1.3px;margin:18px 0 12px}}p{{margin:10px 0}}
.lede{{max-width:850px;font-size:17px;color:#52675f}}.meta{{display:flex;gap:10px;flex-wrap:wrap;margin-top:20px}}
.tag{{background:#e6eee8;border:1px solid #cad8ce;border-radius:5px;padding:5px 10px;font-size:12px}}
.notice{{margin:22px 0;background:#fff8e7;border-left:3px solid #b28a31;padding:14px 18px;font-size:14px;color:#675020}}
.grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:20px}}.card{{border:1px solid #ccd9d0;border-radius:12px;background:white;padding:24px}}
.card-head{{display:flex;align-items:center;justify-content:space-between}}h2{{font-size:17px;margin:0;letter-spacing:.5px}}
.status{{background:#def4e8;color:#126544;font-size:11px;font-weight:750;border-radius:4px;padding:4px 9px;letter-spacing:1px}}
.observations{{font-size:52px;line-height:1.2;font-weight:650;margin-top:18px;letter-spacing:-2px}}
.observations span{{font-size:14px;font-weight:400;letter-spacing:0;margin-left:8px;color:#52675f}}.muted{{font-size:13px;color:#52675f}}
dl{{margin:22px 0 18px}}dl div{{display:flex;justify-content:space-between;gap:12px;border-top:1px solid #e7ede8;padding:9px 0;font-size:13px}}
dd{{margin:0;font-variant-numeric:tabular-nums;font-weight:600}}a{{color:#066c59;text-underline-offset:3px;font-size:13px}}
.scope{{margin-top:22px;display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:24px;background:#eaf0eb;border-radius:10px;padding:20px 24px}}
.scope h2{{font-size:14px}}.scope p{{font-size:13px;margin-bottom:0;color:#52675f}}
footer{{margin-top:24px;border-top:1px solid #ccd8d0;padding-top:18px;color:#52675f;font-size:12px;overflow-wrap:anywhere}}
details{{margin-top:14px}}summary{{cursor:pointer}}table{{border-collapse:collapse;width:100%;font-size:12px;margin:12px 0}}th,td{{text-align:left;padding:10px;border-bottom:1px solid #ccd8d0}}code{{overflow-wrap:anywhere}}
@media(max-width:700px){{main{{padding:24px 18px}}h1{{font-size:30px}}.grid,.scope{{grid-template-columns:1fr}}.observations span{{display:block;margin:4px 0}}}}
</style></head><body><main><header>
<div class="eyebrow">tinyml-deployer / execution evidence</div>
<h1>Compiled firmware. Checked predictions.</h1>
<p class="lede">ESP32-S3 firmware executed under Espressif QEMU. These values come directly from archived GitHub Actions reports.</p>
<div class="meta"><span class="tag">ESP32-S3</span><span class="tag">ESP-IDF {esc(reports[0][1]['validation']['idf_version'])}</span>
<span class="tag">{esc(reports[0][1]['emulator_version'])}</span><span class="tag">{esc(provenance['run_created_at'][:10])}</span></div></header>
<div class="notice"><strong>Emulator evidence.</strong> Physical board validation has not run. These results do not establish device latency, power consumption or task accuracy.</div>
<section class="grid">{''.join(cards)}</section>
<section class="scope"><div><h2>What passed</h2><p>Startup, model initialization, complete inference records and numerical agreement with each graph's desktop TFLite output.</p></div>
<div><h2>How to interpret the two results</h2><p>The float32 dense graph and INT8 convolutional graph are different fixtures. Their sizes and errors are not a quantization comparison.</p></div></section>
<footer>Source: <a href="{esc(provenance['run_url'])}">GitHub Actions run {provenance['run_id']}</a> | 
commit <code>{esc(provenance['commit'])}</code><br>Documentation view of archived reports. The original JSON files retain all records, hashes and emulator timing fields.
<details><summary>Model identities and provenance</summary><table><tbody>{''.join(rows)}</tbody></table>
<a href="provenance.json">Artifact IDs and SHA-256 checksums</a></details></footer>
</main></body></html>
'''
    (EVIDENCE / "index.html").write_text(page, encoding="utf-8")


def main() -> None:
    provenance, reports = load_evidence()
    ASSETS.mkdir(exist_ok=True)
    error_figure(reports)
    evidence_page(provenance, reports)
    print("Generated prediction-agreement.svg and the HTML evidence view.")


if __name__ == "__main__":
    main()
