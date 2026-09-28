# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Stack

User-selected Python, TensorFlow/Keras, and Streamlit. The app runs locally; no additional frontend framework or hosting service is required.

## Users

B.Tech students, project examiners, and academic researchers demonstrating image classification on public 2D brain MRI datasets.

## Product Purpose

Provide a complete, reproducible experiment and a simple interface to inspect a trained classifier's prediction, probabilities, and Grad-CAM attribution.

## Operating Context

An individual uploads a raster MRI image during a local academic demonstration. Windows CPU is the verified runtime target. Training, evaluation, and dataset auditing happen through documented scripts and notebooks.

## Capabilities and Constraints

Binary classification is primary; multiclass experiments are configurable. Results must be measured, traceable, and never fabricated. The UI shares preprocessing and prediction logic with the CLI. Missing models, invalid images, and incompatible metadata need useful error messages. Uploaded app images are not persisted.

## Evidence on Hand

The pinned public dataset has been downloaded and audited. Tests exercise model construction, serialization, inference, class mapping, and explanations. Real training status and final results are recorded in docs/results.md and generated experiment artifacts.

## Product Principles

- Research claims must match measured evidence and its limitations.
- Prediction confidence is an uncalibrated model probability.
- Grad-CAM is an interpretability aid, not medically verified localization.
- Preserve the user's simple upload, predict, inspect workflow.
- Display the exact non-diagnostic disclaimer in every application state.
