#!/usr/bin/env python3
"""Combine Figures 01--06 into one ordered multipage PDF."""
from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "outputs/fig4_active_sensing/rr100_power_routing_figure_series_v1"
OUTPUT = BASE / "rr100_fem_power_routing_figures_01_06.pdf"
INPUTS = [
    BASE / "01_input_redistribution/figure01_retinal_power_redistribution.pdf",
    BASE / "02_unit_filtering/figure02_unit_specific_spectral_routing.pdf",
    BASE / "03_response_examples/figure03_routing_response_examples.pdf",
    BASE / "04_population_prediction/figure04_global_routing_hybrid_population.pdf",
    BASE / "05_additive_multiplicative/figure05_additive_multiplicative_rate_test.pdf",
    BASE / "06_population_channels/figure06_low_high_sf_routing_and_coverage.pdf",
]


def main() -> None:
    missing = [str(path) for path in INPUTS if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing figure PDFs: {missing}")
    temporary = OUTPUT.with_suffix(".pdf.tmp")
    subprocess.run(["pdfunite", *map(str, INPUTS), str(temporary)], check=True)
    temporary.replace(OUTPUT)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "multipage_pdf_complete",
        "page_order": [str(path.relative_to(BASE)) for path in INPUTS],
        "output": str(OUTPUT.relative_to(BASE)),
        "expected_pages": 6,
        "assembly": "lossless PDF concatenation with pdfunite; no page rasterization",
    }
    (BASE / "multipage_pdf_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
