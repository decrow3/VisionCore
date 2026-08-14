#!/usr/bin/env python3
"""Wait for the native TF extension, then atomically rebuild the figure series."""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PRODUCTION = ROOT / "outputs/redundancy_resolved_v1_twin/rr100_zero_gaze_native_extended_tf_32_60_v1"
SERIES = ROOT / "outputs/fig4_active_sensing/rr100_power_routing_figure_series_v1"
LOG = SERIES / "finalizer_status.json"


def status(payload: dict[str, object]) -> None:
    payload = {"updated_utc": datetime.now(timezone.utc).isoformat(), **payload}
    temporary = LOG.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(LOG)


def run(script: str) -> None:
    status({"status": "running", "script": script})
    try:
        subprocess.run([str(ROOT / ".venv/bin/python"), str(ROOT / script)], cwd=ROOT, check=True)
    except Exception as error:
        status({"status": "failed", "script": script, "error": repr(error)})
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--no-wait", action="store_true")
    args = parser.parse_args()
    SERIES.mkdir(parents=True, exist_ok=True)
    manifest_path = PRODUCTION / "analysis_manifest.json"
    while True:
        production_status = "missing"
        if manifest_path.exists():
            production_status = json.loads(manifest_path.read_text()).get("status", "unknown")
        if production_status == "production_complete":
            break
        status({"status": "waiting_for_native_tf", "production_status": production_status})
        if args.no_wait:
            raise SystemExit("Native TF production is not complete")
        time.sleep(max(args.poll_seconds, 5.0))

    scripts = [
        "declan/fig4_active_sensing/analyze_rr100_native_extended_tf_f0.py",
        "declan/fig4_active_sensing/prepare_rr100_power_routing_data.py",
        "declan/fig4_active_sensing/make_rr100_power_routing_figure02_filters.py",
        "declan/fig4_active_sensing/analyze_rr100_power_routing_models.py",
        "declan/fig4_active_sensing/make_rr100_power_routing_figure03_examples.py",
        "declan/fig4_active_sensing/make_rr100_power_routing_figure04_population.py",
        "declan/fig4_active_sensing/make_rr100_power_routing_figure05_gain_form.py",
        "declan/fig4_active_sensing/make_rr100_power_routing_figure06_population_channels.py",
        "declan/fig4_active_sensing/write_rr100_power_routing_series_readme.py",
        "declan/fig4_active_sensing/assemble_rr100_power_routing_multipage_pdf.py",
    ]
    for script in scripts:
        run(script)
    status({"status": "complete", "scripts": scripts})


if __name__ == "__main__":
    main()
