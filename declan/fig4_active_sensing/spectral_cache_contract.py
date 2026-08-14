"""Validation and quarantine contract for RR100 spectral caches."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class SpectralCacheContractError(RuntimeError):
    """Raised when a spectral cache is unsafe for scientific analysis."""


class SupersededSpectralCacheError(SpectralCacheContractError):
    """Raised when an analysis attempts to consume a quarantined cache."""


def validate_artifact_not_superseded(path: Path | str, *, label: str) -> Path:
    """Reject an input if it or any containing analysis directory is marked."""
    resolved = Path(path).resolve()
    current = resolved if resolved.is_dir() else resolved.parent
    for directory in (current, *current.parents):
        marker_path = directory / "SUPERSEDED.json"
        if marker_path.is_file():
            marker = _load_json(marker_path)
            raise SpectralCacheContractError(
                f"Refusing superseded {label} {resolved}. Marker: {marker_path}. "
                f"Reason: {marker.get('reason', 'unspecified defect')}."
            )
    return resolved


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SpectralCacheContractError(f"Cannot read valid JSON from {path}: {error}") from error


def _source_entries(value: Any):
    if isinstance(value, dict):
        if "path" in value and "sha256" in value:
            yield value
        for child in value.values():
            yield from _source_entries(child)
    elif isinstance(value, list):
        for child in value:
            yield from _source_entries(child)


def validate_spectral_cache(
    cache_dir: Path | str,
    *,
    allow_superseded_for_audit: bool = False,
    require_rounds: int | None = None,
) -> dict[str, Any]:
    """Validate identity alignment, completeness, and declared provenance."""
    cache_dir = Path(cache_dir).resolve()
    marker_path = cache_dir / "SUPERSEDED.json"
    if marker_path.exists() and allow_superseded_for_audit:
        marker = _load_json(marker_path)
        return {
            "cache_dir": str(cache_dir),
            "superseded_historical_access": True,
            "permitted_use": marker.get("permitted_use"),
            "reason": marker.get("reason"),
        }
    if marker_path.exists():
        marker = _load_json(marker_path)
        raise SupersededSpectralCacheError(
            f"Refusing superseded spectral cache {cache_dir}. Reason: "
            f"{marker.get('reason', 'unspecified defect')}. Replacement: "
            f"{marker.get('replacement', 'no replacement cache has been frozen')}."
        )

    manifest_path = cache_dir / "manifest.json"
    arrays_path = cache_dir / "condition_spectra.npz"
    if not manifest_path.is_file() or not arrays_path.is_file():
        raise SpectralCacheContractError(
            f"Spectral cache must contain manifest.json and condition_spectra.npz: {cache_dir}"
        )
    manifest = _load_json(manifest_path)
    scope = manifest.get("scope", {})
    expected_conditions = scope.get("conditions")
    expected_rounds = scope.get("rounds")
    if not isinstance(expected_conditions, int) or expected_conditions <= 0:
        raise SpectralCacheContractError("Manifest must declare a positive scope.conditions")
    if not isinstance(expected_rounds, int) or expected_rounds <= 0:
        raise SpectralCacheContractError("Manifest must declare a positive scope.rounds")
    if require_rounds is not None and expected_rounds != require_rounds:
        raise SpectralCacheContractError(
            f"Expected {require_rounds} complete rounds, manifest declares {expected_rounds}"
        )

    required_identity = ("matrix_row_index", "image_index", "trace_index", "round_index")
    with np.load(arrays_path, allow_pickle=False) as archive:
        missing = [key for key in required_identity if key not in archive.files]
        if missing:
            raise SpectralCacheContractError(f"Missing spectral identity arrays: {missing}")
        identities = {key: np.asarray(archive[key], dtype=np.int64) for key in required_identity}
        if "spatial_frequency_mode_count" in archive.files:
            mode_count = np.asarray(archive["spatial_frequency_mode_count"], dtype=np.int64)
            declared_support = np.asarray(archive["spatial_frequency_has_support"], dtype=bool)
            if mode_count.ndim != 1 or not np.array_equal(mode_count > 0, declared_support):
                raise SpectralCacheContractError("Spatial-frequency mode-count and support declarations disagree")
            unsupported_spatial_frequency_bins = np.flatnonzero(mode_count == 0).astype(int).tolist()
        else:
            unsupported_spatial_frequency_bins = []
    for key, values in identities.items():
        if len(values) != expected_conditions:
            raise SpectralCacheContractError(
                f"{key} has {len(values)} rows; manifest declares {expected_conditions} conditions"
            )
    if not np.array_equal(identities["matrix_row_index"], np.arange(expected_conditions)):
        raise SpectralCacheContractError("matrix_row_index is not the unique contiguous storage-row axis")
    actual_rounds = np.unique(identities["round_index"])
    if len(actual_rounds) != expected_rounds:
        raise SpectralCacheContractError(
            f"Found {len(actual_rounds)} rounds in arrays; manifest declares {expected_rounds}"
        )
    counts = np.bincount(identities["round_index"] - identities["round_index"].min())
    if len(np.unique(counts)) != 1:
        raise SpectralCacheContractError(f"Incomplete or unbalanced rounds: row counts are {counts.tolist()}")

    checked_sources: list[str] = []
    for entry in _source_entries(manifest.get("sources", {})):
        source = Path(str(entry["path"]))
        if source.is_file():
            actual = sha256(source)
            if actual != str(entry["sha256"]):
                raise SpectralCacheContractError(
                    f"Source identity mismatch for {source}: expected {entry['sha256']}, got {actual}"
                )
            checked_sources.append(str(source))

    condition_entries = [
        entry for entry in _source_entries(manifest.get("sources", {}))
        if Path(str(entry["path"])).name == "condition_index.csv"
    ]
    if len(condition_entries) != 1:
        raise SpectralCacheContractError(
            "Manifest must identify exactly one condition_index.csv source to prevent mixed provenance"
        )
    condition_path = Path(str(condition_entries[0]["path"]))
    if not condition_path.is_file():
        raise SpectralCacheContractError(f"Condition identity source is missing: {condition_path}")
    condition = pd.read_csv(condition_path)
    for key in required_identity:
        if key not in condition:
            raise SpectralCacheContractError(f"Condition table lacks required identity column {key}")
        observed = condition[key].to_numpy(dtype=np.int64)
        expected = identities[key]
        if not np.array_equal(observed, expected):
            mismatch = int(np.count_nonzero(observed != expected)) if len(observed) == len(expected) else -1
            raise SpectralCacheContractError(
                f"Spectral identities disagree with condition_index.csv for {key} "
                f"({mismatch if mismatch >= 0 else 'different-length'} mismatches)"
            )

    return {
        "cache_dir": str(cache_dir),
        "manifest_status": manifest.get("status"),
        "conditions": expected_conditions,
        "rounds": expected_rounds,
        "round_row_count": int(counts[0]),
        "checked_source_hashes": checked_sources,
        "unsupported_spatial_frequency_bins": unsupported_spatial_frequency_bins,
        "superseded_historical_access": marker_path.exists(),
    }


def validated_spectral_cache_from_environment(
    variable: str = "RR100_CORRECTED_SPECTRAL_CACHE",
) -> Path:
    """Resolve an explicit cache path for legacy scripts without a CLI."""
    raw = os.environ.get(variable)
    if not raw:
        raise SpectralCacheContractError(
            f"Set {variable} to an explicitly selected, frozen corrected spectral cache. "
            "The superseded three-round cache is not a default."
        )
    path = Path(raw).resolve()
    validate_spectral_cache(path)
    return path


def validate_grating_only_tuning(path: Path | str) -> dict[str, Any]:
    """Reject natural-movie fields from the clean tuning artifact."""
    path = Path(path).resolve()
    forbidden = ("movie", "power_map", "accepted_power", "image", "trace", "condition")
    required = {
        "rr100_index", "measured_sf_cpd", "measured_tf_hz",
        "measured_grating_orientation_deg", "measured_signed_f0_hz",
        "measured_positive_f0_hz", "heldout_harmonic_prediction_f0_hz",
        "heldout_separable_prediction_f0_hz", "heldout_chosen_prediction_f0_hz",
    }
    with np.load(path, allow_pickle=False) as archive:
        keys = tuple(archive.files)
        contaminated = [key for key in keys if any(fragment in key.lower() for fragment in forbidden)]
        missing = sorted(required.difference(keys))
        if contaminated or missing:
            raise SpectralCacheContractError(
                f"Grating-only tuning contract failed; forbidden={contaminated}, missing={missing}"
            )
        units = np.asarray(archive["rr100_index"], dtype=int)
        signed = np.asarray(archive["measured_signed_f0_hz"])
    if not np.array_equal(units, np.arange(100)) or signed.shape[0] != 100:
        raise SpectralCacheContractError("Grating-only artifact must contain the complete RR100 unit axis")
    return {"path": str(path), "keys": list(keys), "units": len(units), "tensor_shape": list(signed.shape)}
