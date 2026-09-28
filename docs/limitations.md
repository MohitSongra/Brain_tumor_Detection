# Limitations and future work

## Dataset and experimental validity

- This is a small public collection of 2D raster images, not a representative sample of complete clinical MRI examinations. Dataset size, selection, labels, scanner protocols, MRI acquisition differences, and geographic diversity constrain generalization.
- The collection combines sources. Scanner appearance, background, annotation, compression, and source-specific acquisition differences can become shortcuts correlated with class labels.
- Patient identifiers are not documented in the selected release. Hash separation and exclusion of known augmented images reduce leakage risk but cannot guarantee patient independence or detect every transformed duplicate. Repeated slices, unmarked variants, or shared subjects may remain.
- Excluding filename-marked augmentation and other audit findings changes the population and class distribution. Results must cite final manifest counts, exclusions, and the dataset version.
- Binary merging creates class imbalance. Class weights can help optimization but do not correct selection bias, mislabeled data, or non-representative test data.
- A single held-out image split has sampling uncertainty. Comparing models on that split does not constitute external or patient-level validation. Repeatedly tuning against test scores invalidates their role as a held-out estimate.
- The original Custom CNN and EfficientNetB0 test results and sample errors were inspected before adding MobileNetV2. The third model reuses the same audited split, seed 42, fixed threshold 0.5, maximum 20 head and 10 fine-tuning epochs, and validation-only checkpoint selection. This sequential extension is an exploratory three-way comparison, not fresh, untouched external confirmation. The existing two-model measured results remain unchanged.
- This is one run per architecture on one dataset without verified patient identities. No repeated-seed variability, independent patient-level confirmation, or statistical-significance finding is established. A difference in point estimates alone does not prove that one architecture is superior.

## Model behavior

- A tumor/no-tumor score cannot identify pathology, stage a disease, establish benignity or malignancy, or replace examination of the full MRI study and clinical history.
- Resizing to 224 × 224 may obscure small structures. RGB conversion does not create additional MRI information. The image pipeline accepts supported raster formats, not DICOM volumes or NIfTI studies.
- False positive and false negative predictions are expected. An uploaded file can be valid as an image yet be unrelated to MRI; the application does not have a validated out-of-distribution detector.
- Displayed confidence is the selected class's raw model probability. It is not a calibrated probability of disease, a measure of diagnostic certainty, or a confidence interval.
- A configurable binary threshold trades sensitivity and specificity. A threshold optimized using test labels would bias reported performance.
- ImageNet transfer learning is useful initialization, but its natural-image features are not proof of medical suitability. CPU training is supported but can be slow; constrained training budgets may limit convergence.
- Custom CNN, EfficientNetB0, and MobileNetV2 use different architectures and embedded normalizations. Matching split, seed, and epoch limits does not guarantee equal optimization difficulty, effective capacity, or compute cost. EfficientNetB0 remains the predetermined application model rather than being replaced according to the reused test ranking.
- Grad-CAM is an interpretability aid, not proof that the highlighted region is medically correct. It can highlight irrelevant features or produce a weak/empty heatmap, and it is not validated tumor segmentation.

## Safety and intended use

There is no clinical validation, prospective evaluation, regulatory approval, or deployment-quality monitoring. The system is not approved for clinical diagnosis and must not be used to make treatment, triage, or screening decisions. Do not upload patient-identifiable images to a public deployment without appropriate authorization and privacy controls. Local upload handling does not itself establish compliance with any health-data regulation.

The forthcoming Word manuscript at `paper/Brain_Tumor_Detection_IEEE_Paper.docx` is an IEEE-style academic draft with intentional, clearly marked author placeholders. It is not evidence of acceptance, publication, or endorsement by IEEE. The requested manuscript deliverable is Word only.

This application is an academic/research prototype for brain MRI image classification. It is not a medical diagnostic device and should not be used to make medical decisions. Predictions should be reviewed by qualified medical professionals.

## Future work

1. Obtain larger, licensed, diverse multi-center datasets with verified labels, acquisition metadata, and genuine patient identifiers; perform patient-grouped development and external validation.
2. Compare multiclass tumor classification and explicit tumor segmentation, including U-Net methods with annotated masks and segmentation-specific evaluation.
3. Study Vision Transformers and ensembles under equal data and compute budgets, using a locked external test population.
4. Measure calibration, uncertainty, robustness, subgroup performance, and out-of-distribution behavior; add confidence intervals based on the appropriate independent sampling unit.
5. Evaluate federated learning where institutions have a justified collaboration and governance framework.
6. Test explainability methods against available annotations and clinical expert review without treating heatmaps as ground truth.
7. Optimize inference through profiling, quantization, or export only after checking that prediction and calibration behavior remain acceptable for the research task.
