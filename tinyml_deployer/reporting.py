"""Portable JSON and HTML reports with explicit measurement provenance."""

from __future__ import annotations

import html
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from tinyml_deployer.io_utils import write_new_file


def report_dict(value: Any) -> dict:
    return asdict(value) if is_dataclass(value) else dict(value)


def write_report(value: Any, path: str) -> None:
    data = json.dumps(report_dict(value), indent=2, allow_nan=False, sort_keys=True)
    output = Path(path)
    if output.suffix.lower() == ".html":
        content = (
            '<!doctype html><html lang="en"><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            "<title>TinyML deployment report</title><style>"
            "body{font:16px system-ui;background:#171819;color:#eee;max-width:1000px;"
            "margin:48px auto;padding:24px}pre{white-space:pre-wrap;overflow-wrap:anywhere;"
            "padding:24px;border:1px solid #45484c;border-radius:12px;background:#222426}"
            "h1{color:#91a7ff}</style><h1>TinyML deployment report</h1>"
            "<p>Read the validation and measurement fields before interpreting results.</p>"
            f"<pre>{html.escape(data)}</pre></html>"
        )
    elif output.suffix.lower() == ".json":
        content = data + "\n"
    else:
        raise ValueError("Report path must end in .json or .html.")
    write_new_file(output, content.encode("utf-8"))
