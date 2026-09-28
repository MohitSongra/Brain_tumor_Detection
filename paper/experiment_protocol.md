# Three-model comparison protocol

Recorded on September 24, 2026 while MobileNetV2 training was in progress and before its test evaluation.

## Scope and timing

This is a sequential extension of the existing binary MRI experiment. Custom CNN and EfficientNetB0 were trained and evaluated on September 21, 2026. Their test scores and example mistakes had already been inspected before MobileNetV2 was added. The new comparison reuses that same test population and must be described as exploratory, rather than a fresh blinded confirmation. No model is retrained to improve an already observed test score.

## Fixed settings for the added algorithm

- Algorithm: Keras MobileNetV2, width multiplier 1.0, ImageNet initialization, no original classification head.
- Shared input: RGB float pixels in [0,255], 224 by 224. Saved MobileNet preprocessing maps these pixels to [-1,1].
- Head: global average pooling, dropout 0.3, Dense(128, ReLU), dropout 0.3, one sigmoid output.
- CPU batch size 8, seed 42, Adam, training-only balanced class weights.
- Frozen-head maximum 20 epochs at initial learning rate 0.001; fine-tuning maximum 10 epochs at 0.00001.
- Fine-tune the last 20 backbone layers excluding batch normalization. Backbone batch normalization always uses inference statistics.
- Early stopping patience 5; learning-rate reduction patience 2 and factor 0.2; minimum learning rate 0.0000001.
- Same online augmentation as the existing models: horizontal flip, rotation up to 10 degrees, zoom and translation up to 5%, contrast variation up to 10%.
- Restore the best head checkpoint before fine-tuning. Select the lowest validation-loss checkpoint across phases.
- Binary threshold 0.5, fixed before MobileNetV2 test evaluation. No test-driven threshold optimization.
- Same audited manifest SHA-256: `7cbfdfaf4cb792140f3144366a6f56b31d0d1ecd8a866e6d4749dd874927715a`.
- 3,320 training, 812 validation, and 1,476 test images. No patient identifiers are available.

The full effective configuration and environment are saved in `results/training/mobilenet/`. The smoke test used separate outputs and does not supply research results. EfficientNetB0 remains the predetermined application model regardless of the new test scores.

## Reporting

Report all three models, including unfavorable outcomes, using their actual saved predictions. Compare accuracy, tumor precision, sensitivity, F1, specificity, ROC-AUC, and confusion counts. Training histories describe validation behavior; weighted, augmented training loss and unweighted validation loss are not identical objectives.

One seed per algorithm is evaluated. No claim of statistical significance, independent patient sampling, external validation, calibration, clinical utility, or equal-compute architecture benchmarking is supported. Recorded training wall times may be affected by concurrent machine activity.

## Paper metadata

The requested deliverable is an editable IEEE conference-format Word manuscript. At the time this protocol was recorded on September 24, author names, department, institution, city/country, and email fields were intentionally marked placeholders at the user's request. The manuscript is a draft for author review; it has not been submitted, accepted, or published by IEEE.

### Author metadata update — September 25, 2026

The user subsequently confirmed the author information below, replacing the original placeholders. Student IDs and email addresses are recorded exactly as supplied.

| Author | User-supplied student ID | Email |
|---|---|---|
| Mohit Singh | 23101C0002 | mohit.songra@vit.edu.in |
| Rudra Dalvi | 23101C0005 | rudra.dalvi@vit.edu.in |
| Sameed Mulla | 23101C0030 | sameed.mulla@vit.edu.in |
| Adarsh Yadav | 23101C0012 | adarsh.yadav@vit.edu.in |

Shared affiliation: Department of Information Technology, Vidyalankar Institute of Technology, Wadala, Mumbai, India.

This dated update concerns author metadata only. The September 24 experimental protocol, dataset, trained models, evaluation results, and disclosure of earlier test-set inspection remain unchanged. Confirmation of author details does not imply manuscript approval, submission, or publication.
