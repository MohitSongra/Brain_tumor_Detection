# Dataset guide

## Selected dataset and provenance

- **Name:** Brain Tumor MRI Dataset, by Masoud Nickparvar.
- **Pinned version:** 2, published February 13, 2026.
- **Source:** https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset
- **Public archive:** https://www.kaggle.com/api/v1/datasets/download/masoudnickparvar/brain-tumor-mri-dataset?datasetVersionNumber=2
- **Metadata:** https://www.kaggle.com/api/v1/datasets/view/masoudnickparvar/brain-tumor-mri-dataset
- **Advertised count:** 7,200 JPEG images; glioma, meningioma, pituitary, and notumor.
- **Current metadata license:** Creative Commons Attribution 4.0 International (CC BY 4.0). Attribute the curator, dataset/version/link, and describe modifications. The collection combines Figshare, SARTAJ, and Br35H; review those original sources' terms and provenance before redistribution. A public download does not establish consent for every use.

Suggested attribution: “Masoud Nickparvar, Brain Tumor MRI Dataset, version 2, Kaggle, 2026, CC BY 4.0; dataset filtered for identifiable augmentation and duplicate leakage and remapped for binary classification.”

## Download

After installing requirements:

```powershell
python src/prepare_data.py --download
```

The command uses the verified public version-pinned URL, checks the ZIP, extracts safely, and records the archive SHA256 and source metadata. It does not need Kaggle credentials. Raw files remain unchanged. If the source becomes unavailable, download version 2 manually from Kaggle, extract it beneath `data/raw`, and run:

```powershell
python src/prepare_data.py --source data/raw
```

No private keys, tokens, or credentials should be committed. Do not combine different dataset versions in one raw directory.

## Original layout

```text
Training/
  glioma/       1400
  meningioma/   1400
  notumor/      1400
  pituitary/    1400
Testing/
  glioma/        400
  meningioma/    400
  notumor/       400
  pituitary/     400
```

Version 2 includes 100 `Tr-aug-me_*` and 103 `Te-aug-me_*` meningioma images. The pipeline excludes all augmentation-marked filenames from both splits. This leaves 5,500 development and 1,497 test candidates before auditing. Filename suffixes do not establish source-image or patient relationships. The retained files are **non-aug-marked**, not certified originals.

## Prepared layout and class mapping

```text
data/
  raw/                       untouched download/extraction
  processed/
    manifest.csv             included images and fixed split assignments
    audit_manifest.csv       full inventory and exclusions
    exclusions.csv           exclusion reasons
    audit.json               counts, checks, provenance, limitations
  train/
    no_tumor/
    tumor/
  validation/
    no_tumor/
    tumor/
  test/
    no_tumor/
    tumor/
```

Binary class 0 is `no_tumor` (`notumor` in the source); class 1 is `tumor` (glioma, meningioma, pituitary). Multiclass preserves the configured source labels/order. Source classes remain in the manifest even for binary training, and stratification uses these source categories. The binary task is imbalanced because three source classes collapse into tumor; weights use only retained training labels.

## Split and integrity policy

1. Validate supported image decoding and inventory dimensions, byte/pixel hashes, and a perceptual hash.
2. Exclude augmentation-marked/corrupt files and deduplicate identical images; conflicting exact-image labels stop preparation.
3. Form conservative perceptual-candidate and optional patient groups. A 64-bit hash distance up to four is the configurable candidate threshold. Similarity is a heuristic, not a diagnosis or proof of common patient identity.
4. Preserve eligible official test assignments. Exclude development groups overlapping official test; never move held-out images into training.
5. Split development approximately 80/20 into train/validation with group separation and source-class stratification. Exact ratios can vary to keep groups intact.
6. Confirm no included byte/pixel/candidate group crosses splits. Save all exclusions and final counts.

Actual files and audit output are the source of truth. Preparation can legitimately retain fewer than the advertised images. It never randomly divides pre-generated augmented examples across train and test. Training augmentation is online only.

## Other supported datasets and patient identities

Preparation also accepts `train/validation/test/<class>` directories or unsplit `<class>` directories, including binary `yes/no` or `tumor/no_tumor` via configurable mappings. Unsplit input defaults to approximately 70/15/15. Unsupported labels and empty/infeasible splits produce actionable errors.

For already divided folders located directly under `data`, run `python src/prepare_data.py --source data`. Supplied validation assignments are preserved, and overlapping training groups are excluded. Unchanged preparation reruns are idempotent; a changed assignment requires the explicit `--force` flag. Exact-image label conflicts compare original source categories even when both would map to binary tumor.

Optional `patient_metadata` points to a project-root-relative CSV with `path,patient_id`. Each CSV image path is relative to the detected dataset root, for example `Training/glioma/Tr-gl_0001.jpg`. Every eligible image must have an ID; incomplete or conflicting metadata fails preparation. An explicit `patient_id_regex` searches each filename stem, using the named `patient_id` group, otherwise the first capture group, otherwise the full match. Every eligible filename must match; CSV and regex IDs must agree when both are supplied.

Do not derive patient identities from ordinary image sequence numbers. The selected public archive includes no patient metadata file, so this project cannot claim patient-level independence. Perceptual auditing cannot detect every unmarked augmented or correlated slice.

See `data/processed/audit.json` after preparation for the actual final counts and provenance, and `docs/limitations.md` for interpretation limits.

## Verified local audit (September 21, 2026)

The downloaded archive SHA-256 is `882817250048c78ef7a759cf23e540d7b581f2327b16663c9d3db12f5d2ffdb4`. All 7,200 files were inspected. The audit excluded 203 augmentation-marked images, 314 exact duplicates, and 1,075 development images in candidate groups overlapping the official test split. The resulting experiment contains **5,608 images**:

| Split | Glioma | Meningioma | No tumor | Pituitary | Total |
|---|---:|---:|---:|---:|---:|
| Train | 1,029 | 898 | 427 | 966 | 3,320 |
| Validation | 254 | 220 | 107 | 231 | 812 |
| Test | 383 | 293 | 400 | 400 | 1,476 |

The manifest SHA-256 is `7cbfdfaf4cb792140f3144366a6f56b31d0d1ecd8a866e6d4749dd874927715a`. No retained byte hash, decoded-pixel hash, or candidate group crosses splits. Conservative filtering substantially reduced the development no-tumor population; the resulting imbalance and difference in class proportions are disclosed rather than repaired by modifying test data. This is still not verified patient-level separation.
