# MRI sample images for testing the application

This pack contains 14 unchanged images copied from the project's audited held-out test set.

## Try the application

1. Open http://127.0.0.1:8501 while the local Streamlit server is running.
2. Select Browse files and choose an image from regular_samples.
3. Select Predict and compare the result with the dataset_label in labels.csv.
4. Inspect the probability bars and Grad-CAM, then try another image.

To restart from the project folder, run:

    .venv\Scripts\streamlit.exe run app.py

regular_samples contains 12 images: 6 no-tumor, 2 glioma, 2 meningioma, and 2 pituitary. The model's primary binary task maps all three tumor categories to Tumor. The samples were chosen with seed 42 without consulting prediction correctness.

known_mistakes contains two additional, deliberately selected mistakes from the completed evaluation:

- 13_false_negative.jpg: dataset label Tumor; the saved model predicted No Tumor.
- 14_false_positive.jpg: dataset label No Tumor; the saved model predicted Tumor.

These cases demonstrate limitations. Do not expect every sample to be classified correctly. labels.csv contains dataset labels, recorded predictions, tumor probabilities, original paths, and hashes. Recorded predictions apply to the current EfficientNet checkpoint at threshold 0.5; retraining or changing the threshold may change them. Displayed confidence is an uncalibrated probability.

## What this test establishes

These images can verify upload, preprocessing, inference, probability display, and Grad-CAM. They were already included in the reported held-out evaluation; this pack is not new independent validation. Do not estimate overall model accuracy from this small, partly selected pack. The full 1,476-image evaluation measured 93.36% accuracy, 91.26% sensitivity, and 99.00% specificity. Patient-level independence is unverified.

## Source and attribution

Masoud Nickparvar, Brain Tumor MRI Dataset, version 2, Kaggle, 2026.
https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

Publisher metadata lists CC BY 4.0. The source combines Figshare, SARTAJ, and Br35H; consult DATASET.md in the project and upstream terms before redistribution. This pack selects and renames files without changing their image bytes. Dataset labels are research annotations, not independently verified clinical diagnoses.

This application is an academic/research prototype for brain MRI image classification. It is not a medical diagnostic device and should not be used to make medical decisions. Predictions should be reviewed by qualified medical professionals.
