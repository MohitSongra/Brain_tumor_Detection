# Methodology

## Research question

This project compares a custom convolutional network, ImageNet-initialized EfficientNetB0, and ImageNet-initialized MobileNetV2 for classifying brain MRI raster images. EfficientNetB0 remains the predetermined application model. The primary task maps `notumor` to class 0 (`no_tumor`) and `glioma`, `meningioma`, and `pituitary` to class 1 (`tumor`). A configuration-controlled multiclass task preserves the four source categories. Predictions concern images in the dataset and do not establish a clinical diagnosis.

## Data and separation

The selected source is [Masoud Nickparvar's Brain Tumor MRI Dataset, version 2](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset). Its February 2026 release contains 7,200 JPEGs. The file listing includes 203 names marked `aug`: 100 training meningioma images and 103 testing meningioma images. Preparation excludes all of these before further processing; the remaining 6,997 files are **non-aug-marked**, not certified original images. Local audit reports, rather than these source counts, determine the final experiment population.

The preparation pipeline records original relative paths, source categories, binary or multiclass labels, hashes, dimensions, and split membership in a manifest. The source `Testing` partition remains the held-out test population after eligibility and leakage filtering. Validation comes from the source `Training` partition, using a fixed random seed and stratification of original categories. Augmentation is generated only after partitioning and only during training.

Byte and decoded-pixel hashes identify exact duplicates. Perceptual hashes flag possible visually similar or transformed images for conservative leakage handling. Conflicting labels and cross-split duplicates require an auditable resolution; labels must not be inferred from model predictions. An image number such as `Te-me_12` is not a patient identifier. Where genuine patient metadata is provided, images from a patient must share a split. This collection provides no documented patient mapping, so image-level separation cannot be represented as patient-level validation.

## Image processing and augmentation

Each accepted raster image is decoded, converted to RGB, and resized to the configured size (224 × 224 by default). A common input contract supplies float RGB intensities in `[0, 255]` to all three models. Normalization is embedded in each saved model: the custom CNN uses `Rescaling(1.0 / 255.0)` to `[0, 1]`, EfficientNetB0 retains its built-in preprocessing, and MobileNetV2 uses `Rescaling(1.0 / 127.5, offset=-1.0)` to `[-1, 1]`. The MobileNet transformation is exactly `x / 127.5 - 1`; neither training nor inference applies a second preprocessing transform outside the model.

Training augmentation uses mild horizontal flips, rotation, zoom, translation, and contrast variation. Validation, test, and inference use identical deterministic resizing and normalization without augmentation. The project does not infer MRI modality, reconstruct a volume, apply clinical windowing, or remove the skull.

## Models and optimization

The baseline stacks convolution, batch normalization, ReLU, and pooling blocks, followed by global average pooling and a regularized classifier. Each transfer model uses its ImageNet backbone, global average pooling, dropout, a 128-unit ReLU dense layer, and dropout before its output. MobileNetV2's explicit input rescaling precedes its backbone. Binary models produce one sigmoid probability; multiclass models produce softmax probabilities in the saved class order. `models/registry.py` defines the stable identifiers `custom_cnn`, `efficientnet`, and `mobilenet`; model implementations are imported only when requested.

EfficientNetB0 comes from the EfficientNet family introduced by Mingxing Tan and Quoc V. Le in [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks (ICML 2019)](https://arxiv.org/abs/1905.11946). This project's classification head and MRI experiments are separate from that paper's original benchmarks.

For both EfficientNetB0 and MobileNetV2, train the head with the backbone frozen using Adam and binary cross-entropy (or sparse categorical cross-entropy for multiclass). The initial learning rate is `1e-3`. Fine-tuning restores the best head checkpoint, unfreezes the last 20 backbone layers except batch-normalization layers, keeps the backbone in inference mode for batch-normalization behavior, recompiles, and uses `1e-5`. Both transfer models have the same configured limits of 20 head epochs and 10 fine-tuning epochs; the baseline has a 20-epoch limit. Early stopping with patience 5, best-model checkpointing, and learning-rate reduction with patience 2 monitor validation loss. Select each model's final checkpoint using validation loss across completed stages, never test performance. Actual epochs can differ because stopping depends on validation behavior.

The staged freezing, recompilation, and batch-normalization handling follow the [official Keras transfer-learning guidance](https://keras.io/guides/transfer_learning/).

Training-only class weights compensate for unequal class frequencies. Even source data balanced across four categories is imbalanced after three tumor categories are merged into one class. Neither validation nor test examples are oversampled. CPU runs use a reduced configurable batch size and the same data/model contracts; training duration can be substantial.

## Evaluation and comparison

Inference on the test manifest is deterministic and preserves manifest row order. A fixed configurable threshold, default 0.5, turns tumor probability into a binary label. Threshold changes must be decided using training/validation information before accessing test performance. Model metadata and the manifest fingerprint must match the evaluation configuration.

For the positive tumor class, sensitivity/recall is `TP / (TP + FN)`, specificity is `TN / (TN + FP)`, precision is `TP / (TP + FP)`, and F1 is `2TP / (2TP + FP + FN)`. Accuracy measures correct predictions over all images. ROC-AUC summarizes ranking across thresholds; precision–recall curves help interpret performance under imbalance. Accuracy alone may conceal missed minority-class images. Sensitivity describes missed positive cases and specificity describes false-positive behavior; neither metric alone demonstrates clinical usefulness.

Binary summaries report tumor as the positive class. Multiclass summaries report macro values, per-class one-versus-rest statistics, and support-weighted averages. A mathematically undefined metric is stored as JSON `null` and displayed as unavailable. Macro averages remain unavailable if any constituent is undefined; weighted averages exclude only classes with zero support.

Each actual evaluation writes metrics, a classification report, per-image probabilities, confusion matrix, ROC and precision–recall plots, a timestamp, model hash, and dataset fingerprint. Every available pair among the three registered models must have matching task, class mapping, threshold, sample count, and manifest fingerprint. Missing experiments remain `Not available`. Model selection and hyperparameter tuning must not use this comparison's held-out scores. EfficientNetB0 is retained for the application regardless of the three-way score ordering.

## Sequential extension and strength of evidence

The Custom CNN and EfficientNetB0 test metrics and sample prediction errors were inspected before MobileNetV2 was added. MobileNetV2 is therefore a sequential extension of the original two-model experiment. It reuses the same audited train/validation/test manifest, seed 42, fixed threshold 0.5, and the transfer-learning training defaults described above. Its checkpoint is selected only on validation loss; neither the original models' measured metrics nor their selected checkpoints are replaced to accommodate the added model.

Reusing a previously inspected test population makes the three-way comparison exploratory. It is not a fresh, untouched external confirmation, even though test images are excluded from gradient updates and checkpoint selection. This study uses one public dataset without verified patient identities and one training run per architecture. It has no independent external cohort, repeated-seed analysis, or statistical-significance finding. A new locked external dataset with patient-level independence would be needed for confirmatory comparison.

## Interpretation and overfitting

Grad-CAM computes gradients of the selected class score with respect to the chosen convolutional feature map, averages gradients spatially, forms a weighted activation map, and overlays its normalized positive activations on the displayed image. A weak or zero map is possible. The heatmap identifies model-associated regions; it is not a segmentation mask, causal explanation, or verification of tumor location.

The underlying method is described by Selvaraju and colleagues in [Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization](https://arxiv.org/abs/1610.02391), originally presented at ICCV 2017, with an expanded IJCV version in 2019. This implementation explains binary tumor using logit `z` and no-tumor using `-z`, avoiding gradients through a saturated sigmoid.

Inspect training and validation loss and accuracy together. Falling training loss alongside rising validation loss suggests overfitting but does not prove a particular cause. Use conservative augmentation, dropout, validation-driven early stopping, and carefully limited fine-tuning; change one experiment configuration at a time and record it. Test results remain a final evaluation rather than a tuning signal.

## Reproducibility and disclosure

Record the configuration, seed, package versions, dataset version, manifest, excluded files, model metadata, and generated metrics. Seed Python, NumPy, and TensorFlow. Hardware kernels and parallel execution can still produce numerical variation; approximately reproducible runs should not be described as bit-identical. Synthetic fixtures exercise software only and cannot contribute measured MRI performance.

The requested IEEE-style manuscript is forthcoming as `paper/Brain_Tumor_Detection_IEEE_Paper.docx`, a Word-only academic draft. Clearly marked author, affiliation, and contact placeholders are intentional and must be replaced with the student's real details. Formatting does not imply IEEE acceptance, endorsement, or publication. The manuscript must preserve the sequential-extension disclosure and use only completed measured evaluations.

This application is an academic/research prototype for brain MRI image classification. It is not a medical diagnostic device and should not be used to make medical decisions. Predictions should be reviewed by qualified medical professionals.
