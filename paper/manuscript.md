# Brain Tumor Detection from MRI Images Using Deep Learning

Mohit Singh
Department of Information Technology
Vidyalankar Institute of Technology
Wadala, Mumbai, India
mohit.songra@vit.edu.in
Student ID: 23101C0002

Rudra Dalvi
Department of Information Technology
Vidyalankar Institute of Technology
Wadala, Mumbai, India
rudra.dalvi@vit.edu.in
Student ID: 23101C0005

Sameed Mulla
Department of Information Technology
Vidyalankar Institute of Technology
Wadala, Mumbai, India
sameed.mulla@vit.edu.in
Student ID: 23101C0030

Adarsh Yadav
Department of Information Technology
Vidyalankar Institute of Technology
Wadala, Mumbai, India
adarsh.yadav@vit.edu.in
Student ID: 23101C0012

## Abstract

This study presents a reproducible academic pipeline for binary brain MRI image classification and an exploratory comparison of a custom convolutional neural network, EfficientNetB0, and MobileNetV2. A public MRI dataset is audited to exclude files explicitly marked as augmented, exact duplicates, and development images in similarity groups overlapping the supplied test split. Of 7,200 inspected images, 5,608 are retained: 3,320 for training, 812 for validation, and 1,476 for testing. Shared preprocessing, training-only augmentation, balanced class weights, validation-based checkpoint selection, and a fixed threshold support comparison. The custom CNN achieves 87.20% test accuracy and 0.9332 ROC-AUC; EfficientNetB0 achieves 93.36% accuracy and 0.9850 ROC-AUC. MobileNetV2 achieves {{mobilenet_accuracy}}% accuracy and {{mobilenet_roc_auc}} ROC-AUC under the recorded extension protocol. Sensitivity, specificity, precision, F1, and confusion counts expose remaining classification errors. The implementation provides single-image inference and class-specific Grad-CAM. Results describe one seed and an image-level population without verified patient identifiers. MobileNetV2 was added after earlier test results were inspected, so the comparison is exploratory. The system is an academic prototype, not a medical diagnostic device.

## Keywords

Brain MRI, image classification, convolutional neural network, transfer learning, EfficientNetB0, MobileNetV2, Grad-CAM, data leakage.

## Introduction

Brain MRI image classification offers an accessible setting for studying deep learning, but a credible academic project requires more than an accuracy figure. Image preparation, partition integrity, model selection, reproducibility, and interpretation determine what an experiment actually demonstrates. A classifier can distinguish dataset categories while relying on acquisition or presentation differences that do not generalize to new patients. This work therefore treats tumor detection as a supervised image-labeling task and avoids equating a predicted label with a clinical diagnosis.

The selected public collection, Brain Tumor MRI Dataset, contains glioma, meningioma, pituitary, and no-tumor images [1]. The primary task combines the three tumor categories into one positive class. This mapping simplifies the comparison while retaining original source labels for auditing and stratification. Duplicate images and related variants require particular attention because leakage can inflate apparent generalization in machine-learning studies [2].

The study implements three established algorithms in one modular TensorFlow/Keras pipeline: a small CNN trained from random initialization and two ImageNet-initialized transfer models. It provides auditable data preparation, preserved configurations and predictions, several complementary evaluation metrics, and a Streamlit interface. The contribution is an implemented and evaluated academic workflow with explicit evidence limits, rather than a new architecture or a claim of state-of-the-art MRI performance. EfficientNetB0 is the predetermined application model; the additional comparison does not change that deployment choice according to test scores.

## Background

Convolutional models learn spatial representations through shared filters. A compact CNN provides a useful baseline for determining what can be learned directly from the retained training images. Transfer learning instead begins with parameters learned from a larger image collection, then adapts the classification head and selected feature layers to the target labels.

EfficientNet introduces compound scaling of network depth, width, and input resolution [3]. Its B0 variant supplies the first pretrained backbone. MobileNetV2 uses inverted residual blocks and linear bottlenecks to support efficient feature extraction [4]. Architectural efficiency motivates including MobileNetV2, but parameter counts alone cannot establish runtime or energy advantages in this experiment.

Grad-CAM combines feature activations with class-score gradients to produce spatial attribution maps [5]. Such maps can help inspect a model's behavior, although they do not establish that highlighted regions contain pathology. Dropout randomly suppresses training activations as a regularization mechanism [6]. Both transfer heads use dropout, while the baseline includes dropout after its dense layer. These methods are established components; their inclusion does not guarantee generalization or eliminate overfitting.

## Data and Methods

### Dataset and Binary Labels

Version 2 of the Nickparvar dataset, published in 2026, supplies 5,600 training and 1,600 testing JPEG images across four categories [1]. Its current metadata specifies CC BY 4.0 and identifies combined source collections. This study uses the version-pinned archive and records its checksum. The no-tumor category maps to class 0; glioma, meningioma, and pituitary map to class 1. Source categories remain available in the manifest even though all reported experiments use binary labels.

Table I summarizes the retained cohort by original category and split. The test set contains 400 no-tumor and 1,076 tumor images. These are image counts, not patient counts: the downloaded archive provides no patient-identity metadata. Consequently, neither the original split nor the project's audit establishes independent patients.

{{TABLE_COHORT}}

### Leakage Audit and Partitioning

Preparation validates image decoding and records byte hashes, decoded-pixel hashes, dimensions, and 64-bit perceptual hashes. All filenames explicitly marked as augmented are excluded from both original partitions. Exact-image duplicates are removed, and conflicting labels for identical images cause preparation to fail. Edges connect images whose perceptual-hash Hamming distance is at most four; connected components form conservative candidate groups. A group's endpoints can therefore be farther apart than four. Similarity indicates a possible relationship rather than confirmed shared identity.

Eligible official test assignments are preserved. Development images in candidate groups overlapping the official test partition are excluded rather than moved into training. Remaining development groups are divided approximately 80/20 into training and validation. The implementation chooses the closest source-category balance among 256 seeded, group-disjoint candidate splits. The audit excludes 203 augmentation-marked images, 314 exact duplicates, and 1,075 development images overlapping test candidate groups, retaining 5,608 images. No retained byte hash, pixel hash, or candidate group crosses splits.

This procedure reduces identifiable contamination without certifying every image as an original or every group as a patient. Unmarked transformations and correlated slices may remain. Conservative filtering also reduces the development no-tumor population substantially, changing prevalence. The audit preserves these exclusions and their consequences rather than modifying test composition to improve balance. File-integrity checks and a manifest fingerprint link training and evaluation to the same prepared data.

### Preprocessing and Augmentation

Each image is decoded, oriented, converted to RGB, and resized to 224 × 224 pixels with antialiased bilinear interpolation. The shared loader returns float32 pixels in [0,255]. Normalization belongs to the saved model: the CNN rescales to [0,1], EfficientNetB0 retains its built-in preprocessing, and MobileNetV2 applies x/127.5 − 1 to obtain [−1,1]. Keeping this transformation inside each model prevents inconsistent or repeated normalization during inference.

Only training batches receive online augmentation: horizontal flips, rotations up to 10 degrees, zoom and translation up to 5%, and contrast variation up to 10%. Outputs are clipped to the valid input range. Validation and test images receive deterministic decoding and resizing only. Augmented batches are never saved and repartitioned. These transformations are modest computational regularizers; their clinical suitability across acquisition protocols is not established by this experiment.

### Model Architectures

The custom CNN contains three convolutional blocks with 32, 64, and 128 filters. Each uses a 3 × 3 convolution, batch normalization, ReLU, and max pooling. Global average pooling feeds a 128-unit ReLU layer, dropout with rate 0.3, and a one-unit sigmoid classifier. Its total parameter count is 110,561.

EfficientNetB0 and MobileNetV2 remove their original classification heads and use ImageNet weights. Each new head comprises global average pooling, dropout at 0.3, a 128-unit ReLU layer, another dropout layer, and one sigmoid output. Their complete models contain 4,213,668 and 2,422,081 parameters, respectively, including frozen and nontrainable parameters. MobileNetV2 uses width multiplier 1.0. Table II contrasts the architectures; parameter counts refer to these implementations, not all possible backbone configurations.

{{TABLE_ARCHITECTURES}}

### Training Objective and Class Imbalance

The objective is binary cross-entropy, which penalizes the negative logarithm of the probability assigned to the observed class. Adam performs optimization [7]. Each training example's loss is multiplied by its class weight. Balanced weights equal the total training count divided by twice the corresponding class count. They are approximately 3.8876 for no tumor and 0.5738 for tumor, reflecting 427 negative and 2,893 positive training images. Validation and test examples are neither oversampled nor class-weighted.

## Experimental Protocol

### Training and Checkpoint Selection

The baseline trains for at most 20 epochs. Each transfer model initially freezes its backbone and trains the new head for at most 20 epochs at an initial learning rate of 0.001. Fine-tuning restores the best head checkpoint, unfreezes trainable layers within the last 20 backbone layers except batch-normalization layers, and recompiles at 0.00001 for at most 10 further epochs. Backbone batch-normalization statistics remain in inference mode throughout.

Early stopping monitors validation loss with patience five. Learning-rate reduction uses factor 0.2, patience two, and minimum rate 10⁻⁷. Checkpoints are selected by minimum validation loss, including comparison across head and fine-tuning phases. The final epoch is therefore not necessarily the evaluated model. A fixed threshold of 0.5 converts probabilities to labels; no test-based threshold search is performed.

All recorded runs use CPU batch size eight, Python 3.11.15, TensorFlow 2.20.0, and Keras 3.12.4 on a Windows AMD64 machine with no TensorFlow GPU device. Python, NumPy, and TensorFlow seeds are set to 42 with deterministic settings. Configurations, environment metadata, per-epoch histories, model metadata, and predictions are saved. Such controls support approximate reproduction, but hardware and library differences can still affect results.

### Sequential Comparison and Timing

The custom CNN and EfficientNetB0 were trained and tested on September 21, 2026. MobileNetV2 was added on September 24 after those test results and example errors had already been inspected. Its fixed settings, common manifest, threshold, and reporting rules were recorded before its own test evaluation. The extension reuses the same 1,476 test images and is therefore an exploratory comparison, not a fresh blinded confirmation. Previously evaluated models are not retrained to pursue better test scores.

One seed is evaluated per model. Table III records actual epochs, selected checkpoints, and wall times. Durations can include effects of concurrent machine activity; they are not controlled efficiency benchmarks. Equal input size and transfer-phase limits improve comparability but do not equalize optimization difficulty, pretrained representations, or computation.

{{TABLE_TRAINING}}

### Evaluation and Explainability

Tumor is the positive class. With true positives TP, true negatives TN, false positives FP, and false negatives FN, accuracy is (TP + TN)/N, precision is TP/(TP + FP), sensitivity or recall is TP/(TP + FN), specificity is TN/(TN + FP), and F1 is 2TP/(2TP + FP + FN). ROC-AUC summarizes ranking across thresholds. Precision–recall curves complement ROC curves because precision depends on class prevalence and imbalance can obscure interpretation [8]. Undefined metric denominators are reported as unavailable rather than fabricated values.

Accuracy alone cannot distinguish missed tumor-labeled images from incorrectly flagged no-tumor images. Sensitivity exposes the first error type, while specificity exposes the second. These quantities describe agreement with dataset labels and do not quantify clinical outcomes. Test ROC and precision–recall curves and confusion counts are calculated from saved test probabilities in stable manifest order.

For Grad-CAM, channel weights average the gradient of a class score over spatial positions. A weighted sum of feature maps passes through ReLU and is normalized when its maximum is positive. The implementation targets the pre-sigmoid logit for tumor and its negative for no tumor, preserving class-specific attribution without modifying the saved classifier. The map is resized and overlaid on the original image; a zero map is not evidence of tumor absence. The Streamlit interface displays the predicted class, class probabilities, and attribution. Its confidence display is a model probability, not a calibrated medical certainty.

## Results and Discussion

### Held-Out Classification

Table IV reports the common test comparison, and Table V provides confusion counts. The custom CNN obtains 87.20% accuracy and 0.9332 ROC-AUC. EfficientNetB0 obtains 93.36% accuracy and 0.9850 ROC-AUC on the same image population. Its 91.26% sensitivity and 99.00% specificity correspond to 94 false negatives and four false positives; the baseline produces 122 false negatives and 67 false positives. Thus, the observed EfficientNetB0 improvement is accompanied by reductions in both error counts, but missed positive images remain.

{{TABLE_RESULTS}}

{{TABLE_CONFUSION}}

MobileNetV2 records {{mobilenet_accuracy}}% accuracy, {{mobilenet_precision}}% precision, {{mobilenet_recall}}% sensitivity, {{mobilenet_specificity}}% specificity, {{mobilenet_f1}}% F1, and {{mobilenet_roc_auc}} ROC-AUC. Its confusion counts include {{mobilenet_fn}} false negatives and {{mobilenet_fp}} false positives. It matches EfficientNetB0's sensitivity but produces seven more false positives, with slightly lower accuracy, F1, and ROC-AUC. Equal false-negative counts do not imply that the same images were misclassified. These measurements do not establish superiority outside this dataset. Fig. 1 displays the ROC comparison; threshold-dependent errors should be considered alongside ranking performance.

{{FIGURE_ROC}}

### Validation Behavior and Computational Context

The CNN ran 11 epochs, selected epoch six by validation loss, and required approximately 34.61 minutes. Validation loss fluctuated sharply, including increases at epochs two and seven. The final validation loss, 0.2233, exceeded the selected minimum of 0.0964 despite an overall decline in training loss. This pattern supports retaining the best checkpoint and indicates unstable validation behavior; it is not a smooth, monotonic overfitting trajectory.

EfficientNetB0 completed 12 head-training epochs and 10 fine-tuning epochs in approximately 59.99 minutes. The selected checkpoint was fine-tuning epoch seven, with validation loss 0.0525 compared with the best head loss of 0.1008. Later validation loss rose modestly, ending at 0.0616. MobileNetV2 completed {{mobilenet_epochs}} epochs in {{mobilenet_minutes}} minutes. Fig. 2 compares measured validation loss and accuracy, marking the start of fine-tuning. The project also retains the complete training curves.

{{mobilenet_validation_analysis}}

{{FIGURE_HISTORY}}

Training loss uses weighted labels and augmented inputs, whereas validation loss uses unweighted, unaugmented inputs. Their absolute levels therefore do not measure identical objectives. Curve differences should not be interpreted as a pure generalization gap. The observed results are compatible with useful pretrained representations for this task, but a single run cannot separate architecture effects from initialization, optimization, and sampling variability. No significance test, repeated-seed interval, or equal-compute superiority claim is supported.

## Limitations

The collection combines sources with incompletely characterized acquisition conditions and demographic coverage. Labels are inherited from the public dataset and have not been independently adjudicated here. The audit excludes detectable duplicates and similarity-group overlaps, but absent patient identities prevent verified patient-level separation. Unmarked transformations, correlated slices, scanner differences, and nonanatomical cues may influence performance.

Conservative filtering changes development prevalence, while the test set remains dominated by tumor-labeled images. Precision and displayed probabilities may behave differently under another prevalence. No external institution, prospective population, calibration study, or out-of-distribution detector is evaluated. A supplied image is not automatically verified to be a valid brain MRI, and the interface should not be used as a triage system.

The comparison uses one seed and sequential reuse of an inspected test set. Subsequent algorithm selection may be informed by previous experience even when settings are recorded before the added model's evaluation. A new independent test population would be needed for stronger confirmation. Grad-CAM is an interpretability aid, not a segmentation or medical explanation. Larger multicenter datasets with reliable patient identifiers, repeated trials, external validation, calibration, and uncertainty estimation are appropriate next steps.

## Conclusion

This project implements an auditable binary MRI image-classification workflow covering preparation, three established neural architectures, training, evaluation, and an interactive prediction interface. Both transfer models exceed the compact CNN's recorded accuracy, sensitivity, specificity, F1, and ROC-AUC. EfficientNetB0 has the highest measured accuracy and specificity, while MobileNetV2 matches its sensitivity with fewer parameters. Neither model eliminates classification errors, and no statistical or clinical superiority is established. The comparison uses a fixed threshold and shared manifest, with the sequential extension explicitly disclosed. This application is an academic/research prototype for brain MRI image classification. It is not a medical diagnostic device and should not be used to make medical decisions. Predictions should be reviewed by qualified medical professionals.

## Acknowledgment

OpenAI Codex assisted with project code, documentation, plot-generation code, and drafting all manuscript sections. This disclosure includes implementation and substantive text generation, not only language editing. The draft requires human-author review of the implementation, measured results, references, and final manuscript before submission. No funding or institutional endorsement is asserted.

## References

[1] M. Nickparvar, “Brain Tumor MRI Dataset,” version 2, Kaggle, 2026. Accessed: Sep. 24, 2026. [Online]. Available: https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

[2] S. Kapoor and A. Narayanan, “Leakage and the reproducibility crisis in machine-learning-based science,” Patterns, vol. 4, no. 9, Art. no. 100804, 2023, doi: 10.1016/j.patter.2023.100804.

[3] M. Tan and Q. V. Le, “EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks,” in Proc. 36th Int. Conf. Machine Learning, PMLR, vol. 97, 2019, pp. 6105–6114. [Online]. Available: https://proceedings.mlr.press/v97/tan19a.html

[4] M. Sandler, A. Howard, M. Zhu, A. Zhmoginov, and L.-C. Chen, “MobileNetV2: Inverted Residuals and Linear Bottlenecks,” in Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition, 2018, pp. 4510–4520, doi: 10.1109/CVPR.2018.00474.

[5] R. R. Selvaraju, M. Cogswell, A. Das, R. Vedantam, D. Parikh, and D. Batra, “Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization,” in Proc. IEEE Int. Conf. Computer Vision, 2017, pp. 618–626, doi: 10.1109/ICCV.2017.74.

[6] N. Srivastava, G. Hinton, A. Krizhevsky, I. Sutskever, and R. Salakhutdinov, “Dropout: A Simple Way to Prevent Neural Networks from Overfitting,” J. Machine Learning Research, vol. 15, no. 56, pp. 1929–1958, 2014.

[7] D. P. Kingma and J. Ba, “Adam: A Method for Stochastic Optimization,” in Proc. 3rd Int. Conf. Learning Representations, 2015. [Online]. Available: https://arxiv.org/abs/1412.6980

[8] T. Saito and M. Rehmsmeier, “The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets,” PLOS ONE, vol. 10, no. 3, e0118432, 2015, doi: 10.1371/journal.pone.0118432.
