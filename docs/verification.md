# Verification record

## Execution environment

Verification uses an isolated project `.venv` with Python 3.11.15, TensorFlow 2.20.0, Keras 3.12.4, and NumPy 2.2.6 on Windows CPU. The laptop has an AMD Ryzen 7 260 (8 cores, 16 logical processors), about 23 GiB RAM, and an NVIDIA RTX 5050; native-Windows TensorFlow does not expose that GPU. Configured CPU batch size is 8 with bounded input loading. Exact resolved dependencies are in `requirements-lock.txt`.

The initially planned Keras 3.11.3 failed real ImageNet construction due to a per-channel Rescaling symbolic-shape bug. Keras 3.12.4 resolved the issue; a regression test and real pretrained inference verified the correction.

## Original verification: September 21, 2026

- Dependency consistency check passed for all 105 installed packages.
- All project modules imported; syntax compilation passed.
- The original full test suite passed: **141 tests** in 58.93 seconds, including preprocessing, leakage rules, binary/multiclass outputs, model save/reload, probability bounds, Grad-CAM, metrics arithmetic, checkpoint selection, configuration/metadata validation, artifact provenance, and Streamlit states. The historical machine-readable record is `results/verification/pytest.xml`.
- Both CNN and EfficientNet command-line smoke training runs completed in isolated output folders. EfficientNet exercised head training, checkpoint reload, fine-tuning, and phase selection.
- Real ImageNet EfficientNetB0 constructed and inferred at 224×224 RGB; random weights were not substituted for production.
- Both notebooks executed against the original completed real-data results using a temporary kernel pointing at the project environment. EDA took 18.691 seconds and model analysis 3.495 seconds. Source notebook hashes stayed unchanged. The copies and execution record under `results/notebooks` have since been refreshed for the three-model extension described below.
- Live Streamlit loading, actual trained-model prediction, probabilities, and all three Grad-CAM images were inspected in the browser on September 21. The default desktop viewport was 875 × 705 CSS pixels. A mobile override requested 390 × 844; the browser reported 394 × 852 CSS pixels. Neither layout had horizontal overflow, and the mobile columns/images stacked vertically. The override was reset afterward. These are historical browser observations, not a claim that a server is currently running.
- Replacing a valid image with a corrupt upload removed the previous prediction and interpretation, displayed a useful error, and disabled Predict. A valid upload restored the workflow. The exact disclaimer remained present in all tested states.
- The UI detector reported no findings. Native Streamlit controls retain accessible labels and keyboard behavior.

Upstream Keras/NumPy deprecation warnings were emitted during serialization tests. TensorFlow also logged an ignored graph-attribute compatibility warning during training. Neither prevented the verified operations; these are not reported as test failures.

## Dataset verification

The public version-2 archive was downloaded and inspected. SHA-256:

`882817250048c78ef7a759cf23e540d7b581f2327b16663c9d3db12f5d2ffdb4`

The audit retained 5,608 of 7,200 images: 3,320 train, 812 validation, 1,476 test. Exclusions were 203 augmentation-marked filenames, 314 exact duplicates, and 1,075 development images in candidate groups overlapping official test. No included byte-hash, decoded-pixel-hash, or conservative candidate group crosses a split. Patient-level independence is unverified because the source does not supply patient IDs.

## Completed real-data experiments

| Model | Actual epochs | Selected checkpoint | Training wall time | Test images |
|---|---:|---|---:|---:|
| Custom CNN | 11 | Baseline epoch 6 | 34.61 minutes | 1,476 |
| EfficientNetB0 | 12 head + 10 fine-tuning | Fine-tuning epoch 7 | 59.99 minutes | 1,476 |
| MobileNetV2 | 14 head + 7 fine-tuning | Fine-tuning epoch 2 | 47.00 minutes | 1,476 |

The CNN and EfficientNet head stopped early; EfficientNet fine-tuning reached its configured ten-epoch cap. Both MobileNet phases stopped early. All three final models were selected using validation loss and evaluated at threshold 0.5 on the same audited test manifest. The original runs partly overlapped, so wall times are not controlled architecture-speed comparisons. Metrics and confusion matrices were independently recomputed from saved predictions and agreed with the reports. See [measured results](results.md).

The original two-model verifier passed on September 21. It rechecked all 5,608 prepared image hashes, matched each evaluation to its model SHA-256 and manifest, verified the published EfficientNet bundle, loaded both saved models, generated bounded probabilities and Grad-CAM for fixed test examples, and compared CLI probabilities with shared inference. A real-model Streamlit AppTest matched the CLI/shared inference probabilities within `1e-6`; only its upload interaction was mocked. The report at `results/verification/trained_models.json` has since been replaced by the successful three-model rerun described below.

The live browser used `notumor__Te-no_1__dc8a25950f07.jpg`. The documented default prediction command classified it as No Tumor, with class probability `0.9999592371823383`; both CLI and browser displayed 100.00% after rounding. This is an uncalibrated model probability. Its prediction JSON and Grad-CAM files are in `results/verification/browser_example/`. The separate Grad-CAM CLI was also tested with the trained CNN. Training curves, confusion matrices, and example Grad-CAM figures were visually inspected.

The standalone `python src/evaluate.py` command was run against the published default bundle and completed successfully, reproducing the same EfficientNet metrics. No threshold or checkpoint was changed for this command check. Browser observations are recorded in `results/verification/browser.json`.

The original final cross-check exposed a helper issue: JSON scientific notation was interpreted as a string by the YAML loader. The verifier now exports actual YAML, preserving numeric types. A regression test and the complete successful rerun verified the correction.

## Three-model extension verification: September 24, 2026

- The expanded full suite passed **156 tests** in 115.37 seconds. Its JUnit record, `results/verification/pytest_three_models.xml`, reports 156 tests, zero failures/errors/skips, and 115.244 seconds of recorded suite time. These tests include MobileNet preprocessing, binary/multiclass construction, serialization, fine-tuning, inference, model-specific evaluation routing, and three-model comparison provenance.
- Three subsequently added focused training-orchestration regression cases passed separately in 15.35 seconds. They verify that worse fine-tuning retains the head checkpoint, better fine-tuning is selected, ties retain the earlier phase, and fine-tuning reloads the saved best head. Every case also verifies that a MobileNet run preserves the existing EfficientNet application model, sidecars, and published history byte for byte. These temporary software fixtures perform no research training. The full suite has not been rerun as a combined 159-test run.
- MobileNetV2 completed real ImageNet training and evaluation on the existing audited binary split: 21 epochs in 2,819.79 seconds, selected fine-tuning epoch 2, validation loss 0.036689. Measured test accuracy is 92.89%, precision 98.89%, sensitivity 91.26%, specificity 97.25%, F1 94.93%, and ROC-AUC 0.9820. Its confusion matrix is TN=389, FP=11, FN=94, TP=982. No result was generated from a software fixture.
- The updated real-model verifier passed for **Custom CNN, EfficientNetB0, and MobileNetV2**, and for the application. It rechecked all 5,608 image hashes, validated each evaluation's model and manifest provenance, checked bounded probabilities and saved Grad-CAM, and matched each model's CLI output to shared inference. It also confirmed that the published application model and sidecars still match EfficientNet. The refreshed `results/verification/trained_models.json` was written on September 24 at 11:42 local time.
- The September 24 real-model Streamlit AppTest passed with EfficientNet, matching CLI/shared-inference probabilities within `1e-6` and producing Grad-CAM. Only the upload interaction was mocked. This rerun did not repeat the September 21 desktop/mobile browser-layout inspection.
- Both notebooks were rerun successfully against all three completed models. EDA took **18.186 seconds** and model analysis **2.749 seconds**. Their source hashes stayed unchanged; executed copies and timestamped records are in `results/notebooks/verification.json` and the same directory.
- The September 24 four-page Word manuscript passed final visual review of every rendered page at 144 dpi. Its five native tables, two measured-data figures, eight references, headings, and then-intentional author placeholders rendered without clipping or overlap. An independent audit recomputed all three models' metrics from saved predictions, checked parameter counts against serialized weights, and verified model, figure, and source hashes. Twelve protected IEEE template parts remained byte-identical. The body preserved the reference's section geometry; the local renderer's continuous-section margin behavior also occurred in the unmodified reference. Historical rendering and QA records are in `paper/build/final-render-v1/` and `paper/build/final_verification.json`. The author placeholders were replaced in the verified September 25 revision below.

## Author and IEEE layout revision: September 25, 2026

The user supplied the four authors' names, IDs, emails, and affiliation. The current Word manuscript includes Mohit Singh, Rudra Dalvi, Sameed Mulla, and Adarsh Yadav in that order, arranged in two rows of two native Word author columns. Each block includes Department of Information Technology, Vidyalankar Institute of Technology, Wadala, Mumbai, India, followed by the exact supplied email and ID. All author placeholders were removed, including the document's creator metadata.

The revised document remains four pages. Every page was rendered at 144 dpi and visually inspected at original resolution; the author blocks, two-column body, five editable tables, two figures, and references are readable without clipping or overlap. An independent review confirmed the first-page alignment and author order. The structural audit confirmed four sections and two explicit author column breaks. Research content and body formatting from the abstract onward remain XML-identical to the prior Word version, and the scientific figures are byte-identical. All 12 protected template parts remain unchanged. Current review evidence is in `paper/build/authors_verification_20260925.json` and `paper/build/authors-render-v1/`. The prior version is retained internally under `paper/build/before-author-update-20260925/`. Only the updated Word file is the paper deliverable.

The MobileNet extension was added after the original CNN/EfficientNet test results and sample errors had been inspected. It retained the same split, seed 42, threshold 0.5, maximum 20 head plus 10 fine-tuning epochs, and validation-only checkpoint selection. Reusing this holdout is an exploratory three-way comparison, not fresh external confirmation. This is one run per architecture on one dataset without verified patient identities; no statistical-significance or clinical-validation claim is made.

## Artifact locations and remaining work

- Main application bundle: `models/saved/best_model.keras`, `metadata.json`, and `class_names.json` in the same folder.
- Model-specific bundles and phase checkpoints: `models/saved/custom_cnn/`, `models/saved/efficientnet/`, and `models/saved/mobilenet/`.
- Measured histories, configurations, runtimes, and epoch logs: `results/training/` and the model folders.
- Metrics, reports, per-image predictions, and curves: `results/evaluation/`.
- Dataset exploration and interpretation figures: `results/eda/` and `results/gradcam/`.
- Executed notebook copies: `results/notebooks/`; software and real-model checks: `results/verification/`.
- Word manuscript: `paper/Brain_Tumor_Detection_IEEE_Paper.docx`; editable source and rebuilding instructions are in `paper/`.

All three required binary training/evaluation runs, the recorded software checks, the real-model verifier, notebook execution, and updated Word visual review are complete. Author details are populated from the user's September 25 message. Before submission, perform human-author review against the chosen conference's requirements. Start the application with `streamlit run app.py` and use the address it prints. Historical browser verification at `http://127.0.0.1:8501` does not imply a server remains running. A new checkout must install dependencies and download/prepare/train, or receive complete saved-model bundles separately, because generated data, environments, and model artifacts are Git-ignored.

GPU execution, real multiclass training, external validation, probability calibration, and verified patient-level separation were not performed. Binary and multiclass software paths were tested, but the measured experiments are binary only. These are disclosed limitations and future work, not measured capabilities.
