"""Read-only provenance verification before real training or evaluation."""
from __future__ import annotations

import hashlib
from pathlib import Path

from src.utils import resolve_path


def verify_manifest_integrity(config: dict) -> dict:
    """Verify audited input bytes once and return their count and manifest hash.

    Matching bytes retain the validity of the original decoded-pixel/perceptual
    audit, so image decoding is not repeated. Synthetic smoke fixtures may use
    the loader directly, but cannot pass this check without populated audit
    fields. Files must remain read-only during a run.
    """
    # Import lazily: this utility can be loaded safely by evaluation after a
    # long-running training process has already imported its data loader.
    from src.data_loader import load_manifest

    frame = load_manifest(config)
    audit_fields = {"byte_sha256", "pixel_sha256", "group_id"}
    missing = sorted(audit_fields - set(frame.columns))
    if missing:
        raise ValueError(
            f"Real runs require an audited manifest; missing fields: {', '.join(missing)}. "
            "Run python src/prepare_data.py first."
        )
    for field in sorted(audit_fields):
        values = frame[field].astype(str).str.strip()
        if values.eq("").any():
            raise ValueError(f"Audited manifest has an empty {field}; regenerate dataset preparation.")
        if field.endswith("sha256"):
            if not values.str.fullmatch(r"[0-9a-fA-F]{64}").all():
                raise ValueError(f"Audited manifest has an invalid {field}; expected 64 hexadecimal characters.")
            values = values.str.lower()
        frame[field] = values
        if (frame.groupby(field)["split"].nunique() > 1).any():
            raise ValueError(f"Data leakage detected: {field} crosses dataset splits.")

    def stream_sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    for row in frame.itertuples(index=False):
        path = resolve_path(row.path, config)
        if not path.is_file():
            raise FileNotFoundError(f"An audited image is missing: {path}. Restore it or rerun dataset preparation.")
        actual_hash = stream_sha256(path)
        if actual_hash != row.byte_sha256:
            raise ValueError(
                f"Image integrity check failed: {path} differs from its audited byte_sha256. "
                "Restore the original image or prepare a new audited dataset before running the experiment."
            )
    manifest_path = resolve_path(config.get("manifest_path", "data/processed/manifest.csv"), config)
    return {"checked_images": len(frame), "dataset_fingerprint": stream_sha256(manifest_path)}
