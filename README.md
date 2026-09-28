# Brain Tumor Detection Using Deep Learning

## 1. Overview

**Brain Tumor Detection from MRI Images Using Deep Learning** is a modular B.Tech academic project for classifying 2D brain MRI images. It includes an audited data pipeline, a custom CNN baseline, ImageNet transfer learning with EfficientNetB0 and MobileNetV2, measured evaluation, Grad-CAM, notebooks, and a local Streamlit interface. EfficientNetB0 remains the predetermined application model.

This is an academic/research prototype, not a diagnostic system. It classifies images into dataset categories; it does not locate, stage, or diagnose a tumor.

## 2. Problem Statement

Learn a reproducible image classifier that distinguishes images labeled tumor from images labeled no tumor, while evaluating false positives and false negatives rather than relying only on accuracy. Dataset labels and research test performance do not establish clinical correctness.

## 3. Objectives

- Train and compare an independently initialized CNN, ImageNet EfficientNetB0, and ImageNet MobileNetV2.
- Prevent identifiable image/group overlap between development and test data.
- Report sensitivity, specificity, precision, F1, ROC-AUC, and accuracy honestly.
- Provide a shared inference pipeline and class-specific Grad-CAM explanations.
- Support both binary classification and configurable multiclass experiments.

## 4. Dataset

The default is [Masoud Nickparvar's Brain Tumor MRI Dataset, version 2](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset). See [DATASET.md](DATASET.md) for download instructions, attribution, directory structures, exclusions, and limitations.

Binary mapping is explicit: **0 = no_tumor**, **1 = tumor**. Glioma, meningioma, and pituitary labels map to tumor. Multiclass order is configured as `glioma, meningioma, notumor, pituitary`.

The public archive advertises 7,200 images. Preparation excludes 203 filenames marked as pre-augmented from both official splits, then performs independent duplicate/group audits. Actual retained counts are written to `data/processed/audit.json`; never substitute advertised counts for measured counts.

The eligible official Testing assignment is held out. Training supplies train/validation using approximately 80/20 group-disjoint splitting, choosing the closest source-category balance among 256 seeded candidate splits. Byte hashes, decoded-pixel hashes, perceptual candidate groups, and supplied patient IDs must not cross splits. Exact matches with conflicting labels stop preparation. Perceptual groups are connected components of similarity matches; similarity is a conservative grouping heuristic, not proof of shared patient identity. No patient IDs are supplied by the default dataset, so patient-level independence cannot be guaranteed. Unmarked augmented variants may escape detection. Augmentation is performed online only on training data.

## 5. Technologies Used

Python 3.11; TensorFlow 2.20 and Keras 3.12.4; NumPy, Pandas, Matplotlib, Seaborn, OpenCV, scikit-learn, Pillow, PyYAML, Streamlit, pytest, and notebook execution tools. OpenCV's headless package supplies image operations without an unnecessary desktop GUI dependency. Keras 3.12.4 fixes the per-channel rescaling shape bug found while verifying ImageNet EfficientNet construction with the initially planned 3.11.3.

## 6. System Architecture

```text
Public MRI archive → integrity audit → grouped split manifest
                                          ↓
RGB conversion → resize → train-only augmentation → model normalization
                                          ↓
               Custom CNN / EfficientNetB0 / MobileNetV2
                                          ↓
                validation-based checkpoint selection
                                          ↓
             fixed test evaluation + exploratory comparison
                                          ↓
                    CLI / Streamlit → prediction + Grad-CAM
```

Model metadata carries image size, class order, input range, task, threshold, and dataset fingerprint. CLI and UI use the same preprocessing and prediction functions. The lightweight `models/registry.py` supplies model identifiers and display names without importing TensorFlow until an architecture is requested.

## 7. Data Preprocessing

JPEG, PNG, and BMP inputs are decoded and validated, converted to RGB, and resized to 224×224. Grayscale and RGBA images are converted consistently. All three saved models accept float inputs in `[0,255]`. EfficientNetB0 contains its own normalization; the custom CNN embeds `Rescaling(1/255)` to `[0,1]`; MobileNetV2 embeds `Rescaling(1.0/127.5, offset=-1.0)` to `[-1,1]`. Do not apply an additional division or MobileNet preprocessing outside the saved model.

Training augmentation uses horizontal flips, rotations up to 10 degrees, zoom/translation up to 5%, and contrast variation up to 10%. Validation/test and inference receive no augmentation. Batches and prefetching are bounded; the entire decoded dataset is not cached in memory. Class weights are computed from training labels only.

## 8. Model Architecture

**Custom CNN:** three convolution/batch-normalization/ReLU/pooling blocks (32, 64, 128 filters), global average pooling, Dense(128), dropout, and a sigmoid output.

**EfficientNetB0:** an ImageNet backbone, global average pooling, dropout, Dense(128, ReLU), dropout, and sigmoid output. The backbone is initially frozen. Fine-tuning restores the best head checkpoint, unfreezes only the last 20 layers except batch normalization, and uses `1e-5`. Batch-normalization statistics remain fixed. Both phases compete on validation loss; a worse fine-tuning result does not replace the better head model.

**MobileNetV2:** embedded `[-1,1]` input rescaling, an ImageNet MobileNetV2 backbone, and the same global-average-pooling, dropout, Dense(128, ReLU), dropout, and sigmoid head. It uses the same frozen-head and limited fine-tuning procedure. Its model, metadata, and results are saved separately under `mobilenet`; training it does not replace the EfficientNet application bundle.

Binary training uses Adam and binary crossentropy. Multiclass mode uses softmax and sparse categorical crossentropy. Normalization is saved with the model; training augmentation is not part of inference.

## 9. Training

Activate the environment first, then run from the project root:

```powershell
python src/prepare_data.py --download
python src/eda.py
python src/train.py --model custom_cnn
python src/train.py --model efficientnet
python src/train.py --model mobilenet
python src/compare.py
python scripts/build_results_report.py
```

The equivalent module form, such as `python -m src.train`, is also supported. `python src/train.py` defaults to EfficientNet. ImageNet weights download from the official Keras source on first use. A download failure is reported; random weights are never silently substituted.

Default limits are 20 epochs for CNN training and each transfer-learning head, plus 10 fine-tuning epochs for each transfer model, with early stopping patience 5 and ReduceLROnPlateau patience 2. The recorded experiments use seed 42 and fixed binary threshold 0.5; checkpoints are selected only by validation loss. CPU batch size is 8; a usable GPU uses 32. Training first benchmarks development batches and restores initial weights/optimizer state before fitting. CPU runs can take substantial time; progress is saved every epoch in the model folder's `epochs.jsonl` and training log.

Each architecture keeps its own `.keras` checkpoints, metadata, class names, history, and evaluation. The main EfficientNet bundle is also copied to `models/saved/best_model.keras`. Existing final models are protected; `--overwrite` explicitly retrains them. `--smoke` uses small isolated runs under `results/smoke`, never publishes academic results, and never replaces the application model. `--skip-evaluation` postpones test evaluation.

All experiment settings live in `config/config.yaml`. `--config path/to/override.yaml` merges an override with defaults. Paths are relative to the project root. For multiclass experiments, set `task: multiclass` and separate saved-model/result paths to preserve the binary run.

## 10. Evaluation Metrics

```powershell
python src/evaluate.py
python src/evaluate.py --model models/saved/custom_cnn/best_model.keras --output-dir results/evaluation/custom_cnn
python src/evaluate.py --model models/saved/mobilenet/best_model.keras --output-dir results/evaluation/mobilenet
python src/compare.py
```

For binary classification, tumor is the positive class:

| Metric | Meaning |
|---|---|
| Accuracy | `(TP + TN) / all images` |
| Precision | `TP / (TP + FP)`; how often positive predictions match dataset labels |
| Sensitivity / recall | `TP / (TP + FN)`; captures labeled tumor cases |
| Specificity | `TN / (TN + FP)`; captures labeled no-tumor cases |
| F1 | Harmonic mean of precision and recall |
| ROC-AUC | Discrimination across thresholds using continuous probabilities |

Accuracy alone can hide poor minority-class behavior. Sensitivity exposes false negatives; specificity exposes false positives. These are research metrics against dataset labels, not evidence of clinical utility. Undefined metrics are recorded as null and explained. Multiclass reporting includes per-class and macro/weighted summaries.

Evaluation writes metrics JSON/CSV, a classification report, predictions, confusion matrix, ROC and precision-recall curves. Histories contain measured training/validation accuracy and loss. Every available model pair must share the same split fingerprint, task, class order, threshold, and test size. A missing model is explicitly `Not available`. Test results must not guide hyperparameter or checkpoint selection.

## 11. Results

Measured results and overfitting analysis are maintained in [docs/results.md](docs/results.md). The generated comparison is in `results/evaluation/comparison.md`. Values are populated only after actual MRI training and evaluation. Before that: **Training required — no measured result available.** No minimum accuracy is promised.

The original binary experiments on September 21, 2026 and the MobileNetV2 extension on September 24 used the same 3,320 training, 812 validation, and 1,476 held-out test images after auditing. At the fixed threshold of 0.5:

| Model | Accuracy | Precision | Sensitivity / recall | F1 | Specificity | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Custom CNN | 87.20% | 93.44% | 88.66% | 90.99% | 83.25% | 0.9332 |
| EfficientNetB0 | 93.36% | 99.59% | 91.26% | 95.25% | 99.00% | 0.9850 |
| MobileNetV2 | 92.89% | 98.89% | 91.26% | 94.93% | 97.25% | 0.9820 |

CNN training completed 11 epochs in 34.61 minutes; EfficientNet completed 12 head and 10 fine-tuning epochs in 59.99 minutes on CPU. MobileNetV2 completed 14 head and 7 fine-tuning epochs in 47.00 minutes; both phases stopped early, and fine-tuning epoch 2 was selected by validation loss. These wall times are not controlled architecture-speed comparisons. The EfficientNet test confusion matrix includes 94 false negatives and 4 false positives; MobileNetV2 includes 94 false negatives and 11 false positives. These are measured results from one public image dataset, not clinical validation. Patient-level independence remains unverified.

**Sequential extension disclosure:** the original Custom CNN and EfficientNetB0 test results and sample prediction errors had already been inspected before MobileNetV2 was added. The MobileNet extension retained the same audited split, seed 42, fixed threshold 0.5, maximum 20 head and 10 fine-tuning epochs, and validation-only checkpoint selection. Reusing this holdout makes the three-way comparison exploratory; it is not fresh, untouched external confirmation. The study uses one dataset without verified patient identities and one run per architecture; no statistical-significance claim is made.

The IEEE-style academic manuscript is provided as the Word document `paper/Brain_Tumor_Detection_IEEE_Paper.docx`. On September 25, 2026, the user confirmed the authors as Mohit Singh, Rudra Dalvi, Sameed Mulla, and Adarsh Yadav, from the Department of Information Technology, Vidyalankar Institute of Technology, Wadala, Mumbai, India. The user-supplied student IDs and email addresses are recorded in [the manuscript guide](paper/README.md). This is a manuscript draft, with no claim of IEEE acceptance or publication; the requested paper deliverable is Word only. Consult the verification record for the current document review status.

## 12. Grad-CAM Explainability

```powershell
python src/gradcam.py --image path/to/mri.jpg
```

Grad-CAM differentiates the selected class's pre-activation score with respect to the final spatial feature maps. Binary tumor uses logit `z`; no-tumor uses `-z`. Gradient-weighted channels form a resized heatmap and overlay. The implementation does not alter model weights or output activation and supports saved/reloaded models.

It is an interpretability aid, not a segmentation mask or proof that highlighted regions are medically correct. A zero heatmap means no positive attribution was produced by this method, not that a tumor is absent.

## 13. Streamlit Application

```powershell
streamlit run app.py
```

Open the local address printed by Streamlit. Upload a supported image and select **Predict**. The application shows class, confidence, each class probability, and original/heatmap/overlay images. Confidence is an **uncalibrated model probability**. Images are processed locally and uploaded images are not saved by the app. CLI prediction explicitly saves its requested output files.

The app rejects missing or incompatible model bundles and software-only test models. It cannot reliably determine whether arbitrary uploaded content is a suitable MRI. It supports 2D raster images, not DICOM/NIfTI volumes.

## 14. Installation

Use **Python 3.11**, not this machine's default Python 3.14. With `uv` installed, these commands select the available 3.11 runtime without changing the global Python installation:

```powershell
uv venv .venv --python 3.11
.venv\Scripts\Activate.ps1
uv pip install -r requirements.txt
```

If the selected Python interpreter is already 3.11, the standard alternative is:

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

In a uv-created environment without pip, use `uv pip install -r requirements.txt` instead, or create it with `uv venv .venv --python 3.11 --seed`. If PowerShell activation is restricted, run `.venv\Scripts\python.exe` and `.venv\Scripts\streamlit.exe` directly. Linux activation is `source .venv/bin/activate`.

Native Windows uses CPU with modern TensorFlow. For optional GPU execution, use a compatible Linux/WSL2 environment, NVIDIA driver, and the matching TensorFlow CUDA extra. Consult [official installation guidance](https://www.tensorflow.org/install/pip), install `tensorflow[and-cuda]==2.20.0` in that environment, and verify a real tensor operation on the GPU. GPU detection alone does not establish RTX 50-series kernel compatibility. This project's verified execution target is Windows CPU; it does not install WSL or modify system drivers.

## 15. Usage

```powershell
# Inspect command options
python src/prepare_data.py --help
python src/train.py --help

# Use a manually extracted dataset instead of downloading
python src/prepare_data.py --source data/raw

# Single-image prediction, JSON, and Grad-CAM
python src/predict.py --image path/to/mri.jpg
python src/predict.py --image path/to/mri.jpg --threshold 0.5 --output-dir results/prediction
python src/predict.py --image path/to/mri.jpg --model models/saved/mobilenet/best_model.keras --output-dir results/prediction/mobilenet

# Tests and isolated integration runs
python -m pytest -q
python src/train.py --model custom_cnn --smoke

# Web interface
streamlit run app.py

# Execute both notebooks into results/notebooks without changing their sources
python scripts/verify_notebooks.py

# Verify all three trained bundles, CLI/app agreement, and Grad-CAM
python scripts/verify_trained_models.py
```

Never optimize the threshold against the test set. Notebook 01 calls the same EDA code; notebook 02 reads measured histories/evaluations rather than retraining or inventing results. Open notebooks with a Python environment containing the requirements. They detect the project root from either root or notebook working directories.

## 16. Project Structure

```text
app.py                       Streamlit application
requirements.txt             Tested direct dependencies
requirements-lock.txt        Exact resolved verification environment
README.md / DATASET.md       User and dataset guides
config/config.yaml           Experiment defaults
data/
  raw/                       Archive and untouched extraction
  processed/                 Manifests, exclusions, audit, provenance
  train/ validation/ test/    Audited prepared image folders
models/
  custom_cnn.py              Baseline model
  efficientnet_model.py      EfficientNetB0 transfer learning and fine-tuning
  mobilenet_model.py         MobileNetV2 with embedded [-1,1] preprocessing
  registry.py                Shared model names and lazy architecture imports
  saved/                    Separate model bundles and main EfficientNet bundle
src/
  __init__.py
  preprocessing.py          Shared decoding, resize, augmentation
  data_loader.py            Manifest-backed TensorFlow datasets
  integrity.py              Verify audited files before real training/evaluation
  prepare_data.py           Download, integrity audit, grouping, split preparation
  eda.py                    Dataset and augmentation plots
  train.py                  Benchmark, fit, callbacks, checkpoints, evaluation
  evaluate.py               Metrics, predictions, reports, curves
  compare.py                Provenance-checked experiment comparison
  predict.py                Shared inference and command-line entry
  gradcam.py                Class-specific interpretation
  utils.py                  Configuration, paths, seeds, artifact utilities
notebooks/
  01_EDA.ipynb
  02_Model_Analysis.ipynb
scripts/
  verify_notebooks.py        Execute notebooks with the project environment
  verify_trained_models.py   Verify actual trained bundles and CLI/app agreement
  build_results_report.py    Build the report from measured artifacts
results/
  eda/ training/ evaluation/ gradcam/
docs/
  methodology.md
  results.md
  limitations.md
  verification.md
  file_inventory.md
tests/                       Preprocessing, splits, models, metrics, app, training
paper/
  Brain_Tumor_Detection_IEEE_Paper.docx  Word manuscript with confirmed author information
```

The [complete file inventory](docs/file_inventory.md) explains each deliverable. The [verification record](docs/verification.md) records checks actually performed in this environment.

Generated data and large artifacts remain local and are ignored by Git. Raw images are never intentionally modified; prepared images may be hardlinks, so treat both raw and prepared images as read-only. Real training and evaluation verify each prepared file against its audited byte hash before use; modified files require a fresh audit. Reproducibility records include seeds for Python/NumPy/TensorFlow, deterministic TensorFlow operations, configuration snapshots, runtime versions, and manifest fingerprints. Different hardware/library versions may still produce numerical differences.

## 17. Limitations

See [docs/limitations.md](docs/limitations.md): source diversity, missing patient identities, mixed acquisition protocols, label uncertainty, class imbalance after filtering, possible unmarked transformations, domain shift, bias, false positives/negatives, lack of calibration, and lack of external or clinical validation. MobileNetV2 was added after the original test results and sample errors were inspected, so the three-way extension is exploratory. A single run per model on this one dataset does not establish statistical significance. Grad-CAM is not a medical explanation. No clinical approval or diagnostic claim is made.

## 18. Future Scope

Larger multi-center datasets; verified patient-level separation; external validation; richer multiclass evaluation; tumor segmentation and U-Net; Vision Transformers; ensembles; federated learning; calibration; uncertainty estimation; more rigorous explainability; and deployment optimization. Each requires fresh validation rather than assuming this dataset's results transfer.

## 19. Disclaimer

This application is an academic/research prototype for brain MRI image classification. It is not a medical diagnostic device and should not be used to make medical decisions. Predictions should be reviewed by qualified medical professionals.
