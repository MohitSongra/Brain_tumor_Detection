"""Local Streamlit interface for a trained academic MRI image classifier."""
from __future__ import annotations

import hashlib
import json
import logging

import streamlit as st

from src.utils import DISCLAIMER, configure_runtime, load_config, resolve_path


@st.cache_resource(show_spinner=False)
def load_application_model(path: str, modified: tuple[int, ...]):
    """Reload the bundle when a new trained checkpoint replaces it."""
    from src.predict import load_model_bundle
    model, metadata = load_model_bundle(path)
    if not metadata.get("trained_on_real_data", False):
        raise ValueError("This model is a software test artifact. Train on the audited MRI dataset before using the application.")
    return model, metadata


def main() -> None:
    st.set_page_config(page_title="Brain Tumor Detection", page_icon=":material/neurology:", layout="wide")
    st.title("Brain Tumor Detection Using Deep Learning")
    st.write("Explore how a trained image classifier interprets a brain MRI image. Upload an image to see its predicted class, model probabilities, and Grad-CAM visualization.")
    # Render before any early return so the disclaimer also appears in error states.
    st.info(DISCLAIMER, icon=":material/info:")
    try:
        config = load_config()
        configure_runtime(config)
    except (ValueError, FileNotFoundError, RuntimeError, OSError) as error:
        st.error(f"Configuration could not be loaded. {error}")
        return
    model_path = resolve_path(config["model_path"], config)
    ready = model_path.is_file()
    bundle_paths = [model_path, model_path.parent / "metadata.json", model_path.parent / "class_names.json"]
    try:
        timestamps = tuple(path.stat().st_mtime_ns if path.exists() else 0 for path in bundle_paths)
    except OSError as error:
        st.error(f"The model bundle could not be read. {error}")
        return
    bundle_identity = (str(model_path), timestamps)
    if st.session_state.get("model_bundle_identity") != bundle_identity:
        st.session_state.pop("prediction", None)
        st.session_state.pop("visualization", None)
        st.session_state["model_bundle_identity"] = bundle_identity
    if not ready:
        st.warning("Training required — no measured result available. Prepare the dataset and train EfficientNetB0 using the README instructions to enable predictions.")
    left, right = st.columns([1, 1], gap="large")
    with left:
        st.subheader("Upload an MRI image")
        upload = st.file_uploader("Choose a JPEG, PNG, or BMP image", type=["jpg", "jpeg", "png", "bmp"],
                                  help="Use a single 2D image. DICOM and NIfTI volumes are not supported.")
        st.caption("Images are processed locally. Uploaded images are not saved by this application.")
        source = None
        digest = None
        if upload is not None:
            data = upload.getvalue()
            digest = hashlib.sha256(data).hexdigest()
            if st.session_state.get("image_digest") != digest:
                st.session_state.pop("prediction", None)
                st.session_state.pop("visualization", None)
            st.session_state["image_digest"] = digest
            try:
                if len(data) > config.get("max_upload_mb", 20) * 1024 * 1024:
                    raise ValueError("Image exceeds the configured upload limit. Use a smaller image.")
                from src.preprocessing import load_rgb_image
                source = load_rgb_image(data)
                st.image(source, caption="Uploaded image", width="stretch")
            except (ValueError, OSError) as error:
                st.error(f"Cannot read this image. {error}")
        else:
            st.session_state.pop("prediction", None)
            st.session_state.pop("visualization", None)
        pressed = st.button("Predict", type="primary", disabled=source is None or not ready, width="stretch")
    if pressed and source is not None:
        from tensorflow.errors import OpError
        try:
            with st.spinner("Calculating prediction and interpretation…"):
                model, metadata = load_application_model(str(model_path), timestamps)
                from src.predict import predict_image
                from src.gradcam import render_gradcam
                result = predict_image(source, model, metadata)
                visualization = render_gradcam(source, model, metadata, result["class_index"])
                st.session_state["prediction"] = {**result, "class_names": metadata["class_names"],
                                                    "model_name": metadata.get("model_name", model.name)}
                st.session_state["visualization"] = visualization
        except (ValueError, FileNotFoundError, RuntimeError, OSError, OpError) as error:
            st.session_state.pop("prediction", None)
            st.session_state.pop("visualization", None)
            st.error(f"Prediction could not be completed. {error}")
            logging.exception("Application prediction failed")
    result = st.session_state.get("prediction")
    with right:
        st.subheader("Prediction")
        if result is None:
            st.write("Your result will appear here after you upload an image and select Predict.")
            st.caption("The classifier does not verify that an uploaded image is an MRI or determine whether it is suitable for this model.")
        else:
            label = result["class_name"].replace("_", " ").replace("notumor", "no tumor").title()
            with st.container(border=True):
                st.metric("Predicted class", label)
                st.metric("Confidence", f"{result['confidence']:.2%}")
                st.caption("Confidence is an uncalibrated model probability, not a measure of clinical certainty.")
            st.subheader("Class probabilities")
            for name, probability in zip(result["class_names"], result["probabilities"]):
                display = name.replace("_", " ").replace("notumor", "no tumor").title()
                st.progress(float(probability), text=f"{display}: {probability:.2%}")
            if len(result["class_names"]) == 2:
                st.caption(f"Tumor decision threshold: {result['threshold']:.2f}")
            st.download_button("Download prediction", json.dumps(result, indent=2), file_name="prediction.json", mime="application/json")
    visualization = st.session_state.get("visualization")
    if visualization is not None and result is not None:
        st.divider()
        st.subheader("Model interpretation")
        st.write("Grad-CAM highlights image regions that influenced the selected class score. It is an interpretability aid, not proof that a region is medically correct.")
        for column, name, title in zip(st.columns(3), ("original", "heatmap", "overlay"), ("Original image", "Grad-CAM heatmap", "Overlay")):
            with column:
                st.image(visualization[name], caption=title, width="stretch")
        if visualization["zero_map"]:
            st.warning("Grad-CAM produced no positive attribution map for this class. This does not establish that a tumor is absent.")
    st.divider()
    st.caption("Academic MRI image classification · EfficientNetB0 transfer learning · For research and learning")


if __name__ == "__main__":
    main()
