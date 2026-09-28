# Deliverable file inventory

This inventory lists the project source, configuration, documentation, notebooks, verification code, and Word manuscript. Consult [results](results.md) and the [verification record](verification.md) for measured training outcomes and completed checks. All three models—Custom CNN, EfficientNetB0, and MobileNetV2—have completed real-data binary training and evaluation. EfficientNetB0 remains the predetermined application model.

Downloaded/prepared images, archives, the virtual environment, caches, saved weights, and generated runtime plots/reports/notebook copies are excluded, apart from the expressly listed Word manuscript deliverable. Empty directory placeholders, temporary paper render/build files, and internal interface-audit working files are also excluded. Paths below are relative to the project root.

## Application, dependencies, and configuration

| File | Purpose |
|---|---|
| `app.py` | Streamlit upload, inference, probabilities, Grad-CAM, and disclaimer |
| `requirements.txt` | Tested direct dependency versions |
| `requirements-lock.txt` | Exact resolved verification environment |
| `.gitignore` | Keep local data, environments, models, and generated artifacts out of source control |
| `.streamlit/config.toml` | Interface theme, upload-size setting, and telemetry preference |
| `config/config.yaml` | Configurable dataset, model, optimization, augmentation, runtime, and output settings |

## Models

| File | Purpose |
|---|---|
| `models/__init__.py` | Model package marker |
| `models/custom_cnn.py` | Custom CNN construction and compilation |
| `models/efficientnet_model.py` | EfficientNetB0 ImageNet transfer learning and limited fine-tuning; predetermined application model |
| `models/mobilenet_model.py` | MobileNetV2 ImageNet transfer learning with embedded `Rescaling(1/127.5, offset=-1)` to `[-1,1]` |
| `models/registry.py` | Shared model identifiers/display names and lazy architecture imports |

## Data, training, and inference source

| File | Purpose |
|---|---|
| `src/__init__.py` | Source package marker |
| `src/utils.py` | Configuration, project-relative paths, seeds, environment, JSON, and checkpoint selection helpers |
| `src/preprocessing.py` | Validated RGB decoding, resize, and training augmentation |
| `src/prepare_data.py` | Public download, source inspection, exclusions, duplicate/group audit, and split preparation |
| `src/data_loader.py` | Validated manifest loading, bounded TensorFlow batches, and training-only class weights |
| `src/integrity.py` | Confirm audited prepared files have not changed before real training/evaluation |
| `src/eda.py` | Dataset distributions, MRI examples, dimensions, intensities, and augmentation plots |
| `src/train.py` | Development benchmark, training callbacks, staged fine-tuning, checkpoint selection, and evaluation |
| `src/evaluate.py` | Measured binary/multiclass metrics, curves, predictions, and provenance |
| `src/compare.py` | Three-model comparison with provenance checks for every available pair and explicit unavailable rows |
| `src/predict.py` | Metadata-validated saved-model loading and shared single-image inference |
| `src/gradcam.py` | Class-specific gradient heatmaps and saved interpretation images |

## Notebooks and verification/report scripts

| File | Purpose |
|---|---|
| `notebooks/01_EDA.ipynb` | Inspect actual manifest data and display EDA artifacts |
| `notebooks/02_Model_Analysis.ipynb` | Inspect all three registered models' measured comparisons, curves, and prediction errors |
| `scripts/verify_notebooks.py` | Execute notebooks with the project environment into separate output copies |
| `scripts/verify_trained_models.py` | Verify all three real saved models and Grad-CAM; verify CLI/application agreement and publication alias for EfficientNet |
| `scripts/build_results_report.py` | Build the results document from matching audited, trained, and evaluated artifacts |
| `scripts/build_paper_figures.py` | Recompute metrics from saved predictions, validate matching experiments, and produce publication plots with hashes |

## Documentation

| File | Purpose |
|---|---|
| `README.md` | Installation, architecture, commands, metrics, reproducibility, limitations, and disclaimer |
| `DATASET.md` | Dataset version/source/license, download/layout instructions, exclusions, and leakage limitations |
| `PRODUCT.md` | Interface purpose, audience, behavior, and constraints |
| `DESIGN.md` | Interface layout, hierarchy, visual choices, and accessibility guidance |
| `docs/methodology.md` | Academic methodology and interpretation of the training/evaluation process |
| `docs/limitations.md` | Dataset/model limitations, intended use, and future research directions |
| `docs/results.md` | Measured experiment results when available; otherwise explicit unavailable values |
| `docs/verification.md` | Actual environment, completed software checks, and outstanding verification |
| `docs/file_inventory.md` | This deliverable inventory |

## Word manuscript

| File | Purpose |
|---|---|
| `paper/Brain_Tumor_Detection_IEEE_Paper.docx` | IEEE-style academic manuscript in Word format, with author details confirmed by the user on September 25, 2026; review status is recorded in `docs/verification.md` |
| `paper/README.md` | Manuscript editing, evidence, rebuilding, and confirmed author details including user-supplied student IDs |
| `paper/manuscript.md` | Editable manuscript source with tokens populated from measured MobileNetV2 results |
| `paper/build_manuscript.py` | Build native editable Word text/tables using the retained IEEE styles and measured results |
| `paper/experiment_protocol.md` | Fixed three-model extension protocol and disclosure of prior test-set inspection |
| `paper/references_notes.md` | Verified primary references and IEEE-hosted template provenance |
| `paper/templates/` | Unmodified official-host template references used for formatting |

The requested manuscript deliverable is Word only. Author names, student IDs, email addresses, and the shared Vidyalankar Institute of Technology affiliation were supplied by the user on September 25, 2026 and are recorded in `paper/README.md`. No IEEE acceptance, endorsement, or publication is claimed. The manuscript discloses that MobileNetV2 was added after the original two-model test results and sample errors had been inspected. Reusing that holdout is an exploratory three-way extension on one dataset without verified patient identities and one run per architecture, with no statistical-significance claim. Updating author metadata does not change those experiments or qualifications.

## Tests

| File | Purpose |
|---|---|
| `tests/conftest.py` | Import setup and bounded TensorFlow thread settings |
| `tests/test_data.py` | Image decoding, splits, exclusions, leakage controls, and dataset behavior |
| `tests/test_models.py` | Model construction, saving/loading, predictions, probability bounds, and Grad-CAM |
| `tests/test_mobilenet.py` | MobileNetV2 preprocessing, model construction, serialization, fine-tuning, and inference contracts |
| `tests/test_three_model_comparison.py` | Third-model provenance mismatches, unavailable results, unchanged existing metrics, and lazy imports |
| `tests/test_evaluation.py` | Metric arithmetic, undefined cases, threshold limits, comparison provenance, and evaluation integration |
| `tests/test_app.py` | Streamlit states, invalid inputs, loaded-model behavior, and disclaimer |
| `tests/test_training.py` | Training integration, cross-phase checkpoint selection, best-head restoration, and preservation of the EfficientNet application bundle during MobileNet training |
| `tests/test_utils.py` | Configuration and checkpoint-selection behavior |
| `tests/test_reporting.py` | Reporting provenance, configured links, missing results, and notebook source protection |
| `tests/test_verification.py` | Reject stale or incompatible evaluations during final trained-model verification |

Generated artifacts remain reproducible through the documented commands. Real experiment metrics must come from the audited MRI test evaluation; temporary test fixtures are software verification only.
