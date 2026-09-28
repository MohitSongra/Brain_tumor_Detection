"""Download, audit, group, and prepare MRI images without modifying raw sources.

The included manifest is the source of truth for all downstream experiments.
Exact duplicates are removed, possible perceptual relatives stay together, and
development groups overlapping the supplied test set are excluded.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import shutil
import sys
import urllib.request
import zipfile

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from src.preprocessing import SUPPORTED_EXTENSIONS, load_rgb_image
from src.utils import ROOT, get_class_names, load_config, resolve_path, setup_logging, write_json

LOGGER = logging.getLogger(__name__)
DATASET_URL = "https://www.kaggle.com/api/v1/datasets/download/masoudnickparvar/brain-tumor-mri-dataset?datasetVersionNumber=2"
DATASET_PAGE = "https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset"
SPLIT_ALIASES = {"training": "train", "train": "train", "testing": "test", "test": "test", "validation": "validation", "val": "validation", "valid": "validation"}


def sha256_file(path: Path) -> str:
    """Hash a file with bounded memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative_path(path: Path) -> str:
    """Store portable root-relative paths; external source provenance is absolute."""
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def download_dataset(config: dict) -> Path:
    """Retrieve the public version-pinned archive and extract only safe files."""
    raw_dir = resolve_path(config.get("raw_dir", "data/raw"), config)
    raw_dir.mkdir(parents=True, exist_ok=True)
    archive = raw_dir / "nickparvar_v2.zip"
    if not archive.is_file():
        temporary = archive.with_suffix(".zip.partial")
        LOGGER.info("Downloading the public dataset version 2; this may take several minutes.")
        try:
            request = urllib.request.Request(DATASET_URL, headers={"User-Agent": "BrainMRI-Academic-Project/1.0"})
            with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as output:
                shutil.copyfileobj(response, output, length=1024 * 1024)
            if not zipfile.is_zipfile(temporary):
                raise ValueError("The public download did not return a ZIP archive. Follow DATASET.md for manual acquisition.")
            temporary.replace(archive)
        except Exception:
            if temporary.exists():
                temporary.unlink()
            raise
    if not zipfile.is_zipfile(archive):
        raise ValueError(f"Dataset archive is not a valid ZIP: {archive}")
    checksum = sha256_file(archive)
    destination = raw_dir / "nickparvar_v2"
    marker = destination / ".extraction.json"
    if marker.is_file():
        metadata = json.loads(marker.read_text(encoding="utf-8"))
        if metadata.get("archive_sha256") != checksum:
            raise ValueError("The downloaded archive differs from the extracted source. Select a fresh raw_dir.")
        LOGGER.info("Using previously extracted dataset, archive SHA-256 %s", checksum)
        return destination
    destination.mkdir(parents=True, exist_ok=True)
    destination_resolved = destination.resolve()
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            target = (destination / member.filename).resolve()
            if not target.is_relative_to(destination_resolved):
                raise ValueError("Dataset archive contains an unsafe path.")
            # ZIP symlinks are not needed for image datasets.
            if (member.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("Dataset archive contains an unsupported symbolic link.")
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                with bundle.open(member) as stream:
                    existing = hashlib.sha256(stream.read()).hexdigest()
                if sha256_file(target) != existing:
                    raise ValueError(f"Refusing to overwrite a different raw file: {target}")
            else:
                with bundle.open(member) as stream, target.open("wb") as output:
                    shutil.copyfileobj(stream, output)
    write_json(marker, {"archive_sha256": checksum, "archive": relative_path(archive), "source_url": DATASET_PAGE,
                        "download_url": DATASET_URL, "dataset_version": 2, "license": "CC BY 4.0 (publisher metadata; see DATASET.md)"})
    LOGGER.info("Dataset extracted, archive SHA-256 %s", checksum)
    return destination


def detect_layout(source: Path, class_names: set[str]) -> tuple[Path, str]:
    """Locate one official, prepared, or unsplit layout, including archive wrappers."""
    if not source.is_dir():
        raise FileNotFoundError(f"Dataset source directory does not exist: {source}. Download or place images as described in DATASET.md.")
    candidates = []
    directories = [source] + sorted((path for path in source.rglob("*") if path.is_dir()), key=lambda p: (len(p.parts), str(p)))
    for directory in directories:
        children = {path.name.lower(): path for path in directory.iterdir() if path.is_dir()}
        if "training" in children and "testing" in children:
            candidates.append((directory, "official"))
        elif "train" in children and "test" in children:
            candidates.append((directory, "prepared"))
        elif set(children) & class_names:
            # Class folders under a split are not independent unsplit datasets.
            if directory.name.lower() not in SPLIT_ALIASES:
                candidates.append((directory, "unsplit"))
    if not candidates:
        raise ValueError("Incorrect dataset structure. Expected Training/Testing, train/validation/test, or class-name folders.")
    shallow = min(len(path.parts) for path, _ in candidates)
    candidates = [(path, layout) for path, layout in candidates if len(path.parts) == shallow]
    if len(candidates) != 1:
        raise ValueError("Multiple dataset roots found. Pass --source with the desired dataset directory.")
    return candidates[0]


def perceptual_hash(pixels: np.ndarray) -> str:
    """Compute a 64-bit DCT perceptual hash for candidate grouping, not proof of identity."""
    gray = cv2.cvtColor(pixels, cv2.COLOR_RGB2GRAY)
    reduced = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    frequencies = cv2.dct(reduced)[:8, :8]
    median = np.median(frequencies.flatten()[1:])
    bits = (frequencies > median).flatten()
    number = 0
    for bit in bits:
        number = (number << 1) | int(bit)
    return f"{number:016x}"


class UnionFind:
    """Deterministic connected components for duplicate, perceptual, and patient groups."""
    def __init__(self, count: int):
        self.parent = list(range(count))

    def find(self, index: int) -> int:
        while self.parent[index] != index:
            self.parent[index] = self.parent[self.parent[index]]
            index = self.parent[index]
        return index

    def union(self, left: int, right: int) -> None:
        a, b = self.find(left), self.find(right)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


def patient_lookup(config: dict, source: Path) -> tuple[dict[str, str], re.Pattern | None]:
    """Read optional path,patient_id CSV or an explicit filename extraction regex."""
    lookup: dict[str, str] = {}
    metadata = config.get("patient_metadata")
    if metadata:
        frame = pd.read_csv(resolve_path(metadata, config), dtype=str, keep_default_na=False)
        if not {"path", "patient_id"}.issubset(frame.columns):
            raise ValueError("patient_metadata CSV requires path and patient_id columns.")
        if frame.empty:
            raise ValueError("patient_metadata CSV is empty; supply patient IDs for every eligible image.")
        for row in frame.itertuples(index=False):
            path = Path(row.path)
            path = path if path.is_absolute() else source / path
            key = str(path.resolve())
            if not row.patient_id:
                raise ValueError("Patient metadata contains an empty patient_id.")
            if key in lookup and lookup[key] != row.patient_id:
                raise ValueError(f"Conflicting patient IDs for {row.path}.")
            lookup[key] = row.patient_id
    expression = config.get("patient_id_regex")
    try:
        pattern = re.compile(expression) if expression else None
    except re.error as exc:
        raise ValueError(f"Invalid patient_id_regex: {exc}") from exc
    return lookup, pattern


def inspect_source(config: dict, source: Path) -> tuple[pd.DataFrame, str, Path]:
    """Inspect each image and retain explicit records for excluded inputs."""
    mapping = config.get("binary_mapping", {})
    known_classes = set(mapping) | set(config.get("source_classes", []))
    root, layout = detect_layout(source, {name.lower() for name in known_classes})
    lookup, patient_pattern = patient_lookup(config, root)
    class_directories = []
    if layout == "unsplit":
        class_directories = [(directory, "unsplit") for directory in root.iterdir() if directory.is_dir()]
    else:
        for split_dir in sorted(root.iterdir()):
            if split_dir.is_dir() and split_dir.name.lower() in SPLIT_ALIASES:
                class_directories.extend((directory, SPLIT_ALIASES[split_dir.name.lower()]) for directory in split_dir.iterdir() if directory.is_dir())
    # When the caller supplied data/train/... in place, binary remapping may
    # have created managed copies next to the original class folders. Ignore
    # only those known copies on reruns, retaining the unchanged original inputs.
    managed_copies: set[Path] = set()
    existing_manifest = resolve_path(config.get("manifest_path", "data/processed/manifest.csv"), config)
    if existing_manifest.is_file():
        previous = pd.read_csv(existing_manifest, keep_default_na=False)
        if {"path", "source_path"}.issubset(previous.columns):
            input_folders = [directory.resolve() for directory, _ in class_directories]
            original_paths = {resolve_path(value, config) for value in previous["source_path"]}
            if any(any(path.is_relative_to(folder) for folder in input_folders) for path in original_paths):
                managed_copies = {resolve_path(value, config) for value in previous["path"]} - original_paths
    records = []
    for directory, original_split in sorted(class_directories, key=lambda value: str(value[0])):
        source_class = directory.name.lower()
        image_paths = sorted(path for path in directory.rglob("*")
                             if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
                             and path.resolve() not in managed_copies)
        if not image_paths:
            continue
        if source_class not in known_classes:
            raise ValueError(f"Unknown class directory {directory.name!r}; update source_classes and binary_mapping.")
        for path in image_paths:
            row = {"source_path": relative_path(path), "source_class": source_class, "binary_label": mapping.get(source_class, ""),
                   "original_split": original_split, "width": 0, "height": 0, "byte_sha256": sha256_file(path),
                   "pixel_sha256": "", "phash": "", "patient_id": "", "group_id": "", "split": "", "path": "", "label": "",
                   "excluded": False, "exclusion_reason": ""}
            # Broad marker intentionally excludes all explicitly augmented filenames.
            if re.search(r"aug", path.stem, flags=re.IGNORECASE):
                row.update(excluded=True, exclusion_reason="explicit_augmentation_marker")
            try:
                image = load_rgb_image(path)
                pixels = np.asarray(image)
                row["width"], row["height"] = image.size
                row["pixel_sha256"] = hashlib.sha256(str(pixels.shape).encode() + pixels.tobytes()).hexdigest()
                row["phash"] = perceptual_hash(pixels)
            except (ValueError, OSError) as exc:
                row.update(excluded=True, exclusion_reason=f"invalid_image: {exc}")
            patient_id = lookup.get(str(path.resolve()), "")
            if patient_pattern:
                match = patient_pattern.search(path.stem)
                if not match and not row["excluded"]:
                    raise ValueError(f"patient_id_regex does not match {path.name}; explicit grouping must cover all eligible images.")
                if match:
                    extracted = match.groupdict().get("patient_id") or (match.group(1) if match.lastindex else match.group(0))
                    if patient_id and patient_id != extracted:
                        raise ValueError(f"CSV and filename patient IDs disagree for {path.name}.")
                    patient_id = extracted
            if lookup and not patient_id and not row["excluded"]:
                raise ValueError(f"Patient metadata does not identify {path}. Supply complete metadata for eligible images.")
            row["patient_id"] = patient_id
            records.append(row)
    if not records:
        raise ValueError("The dataset contains no supported images. JPEG, PNG, and BMP are supported.")
    LOGGER.info("Inspected %d images from %s (%s layout).", len(records), root, layout)
    return pd.DataFrame(records), layout, root


def audit_groups(frame: pd.DataFrame, config: dict, layout: str) -> tuple[pd.DataFrame, dict]:
    """Audit duplicates and conservative relationships before assigning new splits."""
    frame = frame.copy().reset_index(drop=True)
    active = list(frame.index[~frame["excluded"]])
    if not active:
        raise ValueError("No eligible images remain after decoding and augmentation exclusions.")
    groups = UnionFind(len(frame))
    priority = {"test": 0, "validation": 1, "train": 2, "unsplit": 2}
    duplicate_count = 0
    for hash_field in ("byte_sha256", "pixel_sha256"):
        for _, subset in frame.loc[active].groupby(hash_field):
            if len(subset) <= 1:
                continue
            if subset["source_class"].nunique() > 1:
                paths = subset["source_path"].tolist()
                raise ValueError(f"Exact-image label conflict ({hash_field}) requires review: {paths}")
            indices = sorted(subset.index, key=lambda index: (priority[frame.at[index, "original_split"]], frame.at[index, "source_path"]))
            keeper = indices[0]
            for index in indices[1:]:
                groups.union(keeper, index)
                if not frame.at[index, "excluded"]:
                    frame.loc[index, ["excluded", "exclusion_reason"]] = [True, f"exact_duplicate_of:{frame.at[keeper, 'source_path']}"]
                    duplicate_count += 1
    # Include exact-duplicate nodes in grouping: an excluded image may link a
    # retained patient or perceptual relative across original partitions.
    for _, subset in frame.loc[active].groupby("patient_id"):
        if not subset.iloc[0]["patient_id"]:
            continue
        indices = list(subset.index)
        for index in indices[1:]:
            groups.union(indices[0], index)
    distance = int(config.get("phash_distance", 4))
    if not 0 <= distance <= 16:
        raise ValueError("phash_distance must be between 0 and 16.")
    hashes = [(index, int(frame.at[index, "phash"], 16)) for index in active]
    candidate_pairs = 0
    # 7,000 images require ~25M very cheap 64-bit XOR/popcount comparisons;
    # no O(N^2) matrix is allocated.
    for offset, (index, value) in enumerate(hashes):
        for other, other_value in hashes[:offset]:
            if (value ^ other_value).bit_count() <= distance:
                groups.union(index, other)
                candidate_pairs += 1
    components: dict[int, list[int]] = defaultdict(list)
    for index in active:
        components[groups.find(index)].append(index)
    overlap_count = 0
    for indices in components.values():
        group_key = hashlib.sha256("\n".join(sorted(frame.loc[indices, "source_path"])).encode()).hexdigest()[:16]
        frame.loc[indices, "group_id"] = group_key
        splits = set(frame.loc[indices, "original_split"])
        if "test" in splits and len(splits) > 1:
            for index in indices:
                if frame.at[index, "original_split"] != "test" and not frame.at[index, "excluded"]:
                    frame.loc[index, ["excluded", "exclusion_reason"]] = [True, "development_group_overlaps_official_test"]
                    overlap_count += 1
        elif layout == "prepared" and "validation" in splits and "train" in splits:
            for index in indices:
                if frame.at[index, "original_split"] == "train" and not frame.at[index, "excluded"]:
                    frame.loc[index, ["excluded", "exclusion_reason"]] = [True, "training_group_overlaps_supplied_validation"]
    summary = {"perceptual_hash": "64-bit DCT", "phash_distance": distance, "candidate_pairs": candidate_pairs,
               "candidate_group_count": len(components), "largest_candidate_group": max(map(len, components.values())),
               "exact_duplicates_removed": duplicate_count, "development_overlapping_test_removed": overlap_count,
               "patient_mapping_available": bool(config.get("patient_metadata") or config.get("patient_id_regex")),
               "limitations": ["Perceptual similarity identifies candidates, not verified shared patients.",
                               "Without patient metadata, patient-level independence cannot be established.",
                               "Explicit augmentation markers are excluded; unmarked transformations may remain."]}
    return frame, summary


def grouped_holdout(frame: pd.DataFrame, fraction: float, seed: int) -> tuple[pd.Index, pd.Index]:
    """Choose a reproducible group-disjoint holdout balanced by original class.

    Singleton groups use ordinary stratified splitting. Related-image groups use
    repeated seeded group splits, choosing the closest feasible class counts.
    """
    if not 0 < fraction < 1:
        raise ValueError("Split fractions must be between 0 and 1.")
    counts = frame["source_class"].value_counts()
    if (counts < 2).any():
        raise ValueError("At least two eligible images per source class are required to create this holdout.")
    if frame["group_id"].nunique() == len(frame):
        train, heldout = train_test_split(frame.index, test_size=fraction, random_state=seed, stratify=frame["source_class"])
        return pd.Index(train), pd.Index(heldout)
    class_names = sorted(counts.index)
    target = counts.reindex(class_names).to_numpy(dtype=float) * fraction
    best = None
    splitter = GroupShuffleSplit(n_splits=256, test_size=fraction, random_state=seed)
    for train_position, hold_position in splitter.split(frame, frame["source_class"], groups=frame["group_id"]):
        train = frame.iloc[train_position]
        heldout = frame.iloc[hold_position]
        if set(train["source_class"]) != set(class_names) or set(heldout["source_class"]) != set(class_names):
            continue
        actual = heldout["source_class"].value_counts().reindex(class_names, fill_value=0).to_numpy()
        score = float(np.mean(np.abs(actual - target) / np.maximum(target, 1)))
        if best is None or score < best[0]:
            best = (score, train.index, heldout.index)
    if best is None:
        raise ValueError("Cannot form a group-disjoint split containing every source class; inspect grouping or provide more data.")
    return best[1], best[2]


def assign_splits(frame: pd.DataFrame, config: dict, layout: str) -> pd.DataFrame:
    """Preserve eligible supplied splits, deriving only missing partitions."""
    frame = frame.copy()
    eligible = frame.loc[~frame["excluded"]]
    seed = int(config.get("random_seed", 42))
    if layout == "unsplit":
        development, test = grouped_holdout(eligible, float(config.get("test_split", 0.15)), seed)
        validation_fraction = float(config.get("unsplit_validation_split", 0.15)) / (1 - float(config.get("test_split", 0.15)))
        train, validation = grouped_holdout(eligible.loc[development], validation_fraction, seed + 1)
        frame.loc[test, "split"] = "test"
    else:
        frame.loc[eligible.index, "split"] = eligible["original_split"]
        development = eligible.loc[eligible["original_split"] == "train"]
        if (eligible["original_split"] == "validation").any():
            train = development.index
            validation = eligible.index[eligible["original_split"] == "validation"]
        else:
            train, validation = grouped_holdout(development, float(config.get("validation_split", 0.2)), seed)
    frame.loc[train, "split"] = "train"
    frame.loc[validation, "split"] = "validation"
    included = frame.loc[~frame["excluded"]]
    for split in ("train", "validation", "test"):
        if not (included["split"] == split).any():
            raise ValueError(f"No eligible images remain in {split}.")
    for field in ("byte_sha256", "pixel_sha256", "group_id"):
        if (included.groupby(field)["split"].nunique() > 1).any():
            raise AssertionError(f"Preparation would leak {field} across splits.")
    return frame


def prepare_dataset(config: dict, source: str | Path | None = None, download: bool = False, force: bool = False) -> dict:
    """Create audited prepared files and manifests; raw files are never edited."""
    source_path = download_dataset(config) if download else resolve_path(source or config.get("raw_dir", "data/raw"), config)
    processed = resolve_path(config.get("processed_dir", "data/processed"), config)
    manifest_path = resolve_path(config.get("manifest_path", "data/processed/manifest.csv"), config)
    processed.mkdir(parents=True, exist_ok=True)
    frame, layout, source_root = inspect_source(config, source_path)
    try:
        frame, audit = audit_groups(frame, config, layout)
        frame = assign_splits(frame, config, layout)
    except ValueError as exc:
        frame.to_csv(processed / "audit_failed.csv", index=False)
        write_json(processed / "audit_failure.json", {"status": "failed", "reason": str(exc), "source": relative_path(source_root)})
        raise
    classes = get_class_names(config)
    binary = config.get("task", "binary") == "binary"
    for index, row in frame.loc[~frame["excluded"]].iterrows():
        if binary:
            if row["binary_label"] == "":
                raise ValueError(f"No binary mapping for {row['source_class']}.")
            label = int(row["binary_label"])
        else:
            label = classes.index(row["source_class"])
        class_name = classes[label]
        original = resolve_path(row["source_path"], config)
        filename = f"{row['source_class']}__{original.stem}__{row['byte_sha256'][:12]}{original.suffix.lower()}"
        class_directory = resolve_path(config.get(f"{row['split']}_dir", f"data/{row['split']}"), config) / class_name
        # Already prepared binary/class-aligned inputs need no extra copy.
        destination = original if original.parent == class_directory else class_directory / filename
        frame.at[index, "path"] = relative_path(destination)
        frame.at[index, "label"] = label
    included = frame.loc[~frame["excluded"]].sort_values(["split", "path"]).reset_index(drop=True)
    signature_fields = ["source_path", "source_class", "byte_sha256", "group_id", "split", "path", "label"]
    fingerprint = hashlib.sha256(included[signature_fields].to_csv(index=False).encode()).hexdigest()
    previous = None
    if manifest_path.is_file():
        previous = pd.read_csv(manifest_path, keep_default_na=False)
        previous_signature = hashlib.sha256(previous[signature_fields].to_csv(index=False).encode()).hexdigest()
        if previous_signature != fingerprint and not force:
            raise FileExistsError("A different prepared manifest exists. Use --force to replace managed prepared artifacts, or choose new output paths.")
    LOGGER.info("Materializing %d audited images using hardlinks when possible.", len(included))
    for row in included.itertuples(index=False):
        original = resolve_path(row.source_path, config)
        destination = resolve_path(row.path, config)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if sha256_file(destination) != row.byte_sha256:
                raise FileExistsError(f"Refusing to overwrite a modified prepared image: {destination}")
        else:
            try:
                os.link(original, destination)
            except OSError:
                shutil.copy2(original, destination)
    # Do not delete arbitrary pre-existing files. Only obsolete files explicitly
    # listed in our previous manifest can be removed on a forced replacement.
    if previous is not None and force:
        managed_roots = [resolve_path(config.get(f"{split}_dir", f"data/{split}"), config).resolve() for split in ("train", "validation", "test")]
        new_paths = set(included["path"])
        for old in previous.itertuples(index=False):
            if old.path not in new_paths and old.path != old.source_path:
                obsolete = resolve_path(old.path, config).resolve()
                if any(obsolete.is_relative_to(root) for root in managed_roots) and obsolete.is_file() and sha256_file(obsolete) == old.byte_sha256:
                    obsolete.unlink()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    included.to_csv(manifest_path, index=False)
    frame.to_csv(processed / "audit_manifest.csv", index=False)
    frame.loc[frame["excluded"]].to_csv(processed / "exclusions.csv", index=False)
    audit.update({"status": "passed", "source": relative_path(source_root), "layout": layout, "task": config.get("task", "binary"),
                  "total_inspected": len(frame), "included": len(included), "excluded": int(frame["excluded"].sum()),
                  "exclusion_counts": dict(Counter(frame.loc[frame["excluded"], "exclusion_reason"].map(lambda value: value.split(":")[0]))),
                  "split_class_counts": included.groupby(["split", "source_class"]).size().unstack(fill_value=0).to_dict(orient="index"),
                  "class_names": classes, "dataset_fingerprint": sha256_file(manifest_path),
                  "split_signature": fingerprint, "random_seed": int(config.get("random_seed", 42)),
                  "cross_split_byte_duplicates": 0, "cross_split_pixel_duplicates": 0, "cross_split_candidate_groups": 0})
    extraction_marker = source_root / ".extraction.json"
    if extraction_marker.is_file():
        audit["download_provenance"] = json.loads(extraction_marker.read_text(encoding="utf-8"))
    write_json(processed / "audit.json", audit)
    write_json(processed / "class_names.json", classes)
    LOGGER.info("Dataset audit passed: %d included, %d excluded. Counts: %s", len(included), audit["excluded"], audit["split_class_counts"])
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="YAML configuration path.")
    parser.add_argument("--source", default=None, help="Dataset root; defaults to raw_dir.")
    parser.add_argument("--download", action="store_true", help="Download the public Nickparvar version 2 archive.")
    parser.add_argument("--force", action="store_true", help="Replace a previous managed prepared dataset after auditing.")
    args = parser.parse_args()
    setup_logging()
    try:
        prepare_dataset(load_config(args.config), source=args.source, download=args.download, force=args.force)
    except (ValueError, FileNotFoundError, FileExistsError, OSError, zipfile.BadZipFile) as exc:
        LOGGER.error("Dataset preparation failed: %s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
