"""Local Streamlit interface for a trained academic MRI image classifier."""
from __future__ import annotations

import hashlib
import json
import logging

import streamlit as st

from src.utils import DISCLAIMER, configure_runtime, load_config, resolve_path

# -- Tumor type metadata for prediction display and reference gallery -----------

TUMOR_INFO = {
    "glioma": {
        "display": "Glioma",
        "description": (
            "Gliomas originate from glial cells in the brain or spine. "
            "On MRI, they typically appear as infiltrative, irregular intra-axial masses, "
            "often with heterogeneous signal and surrounding edema."
        ),
        "icon": "🔴",
    },
    "meningioma": {
        "display": "Meningioma",
        "description": (
            "Meningiomas arise from the meninges (protective membranes around the brain). "
            "On MRI, they appear as well-circumscribed, extra-axial dural-based round or "
            "lobulated masses with strong, uniform enhancement."
        ),
        "icon": "🟠",
    },
    "pituitary": {
        "display": "Pituitary Tumor",
        "description": (
            "Pituitary tumors develop in the pituitary gland at the base of the brain (sella turcica). "
            "On MRI, they appear as rounded sellar/suprasellar masses that may compress the optic chiasm."
        ),
        "icon": "🟡",
    },
    "notumor": {
        "display": "No Tumor (Healthy Brain)",
        "description": "No tumor mass or focal abnormality identified in this MRI scan. Brain structures appear normal and symmetric.",
        "icon": "🟢",
    },
    "no_tumor": {
        "display": "No Tumor (Healthy Brain)",
        "description": "No tumor mass or focal abnormality identified in this MRI scan. Brain structures appear normal and symmetric.",
        "icon": "🟢",
    },
    "tumor": {
        "display": "Tumor Detected",
        "description": "Tumor pattern identified by classification model.",
        "icon": "⚠️",
    },
}

REFERENCE_IMAGES = {
    "No Tumor (Normal Brain)": {
        "images": [
            "data/sample_images/regular_samples/01_no_tumor.jpg",
            "data/sample_images/regular_samples/02_no_tumor.jpg",
        ],
        "description": (
            "Healthy brain MRI. Shows symmetrical cerebral hemispheres, normal ventricles, "
            "and uniform tissue intensity with no mass effect or abnormal lesions."
        ),
        "icon": "🟢",
    },
    "Glioma": {
        "images": [
            "data/sample_images/regular_samples/07_tumor_glioma.jpg",
            "data/sample_images/regular_samples/08_tumor_glioma.jpg",
        ],
        "description": (
            "Grows within brain tissue (intra-axial). Shows irregular, infiltrative margins, "
            "heterogeneous core, and often significant surrounding edema (swelling)."
        ),
        "icon": "🔴",
    },
    "Meningioma": {
        "images": [
            "data/sample_images/regular_samples/09_tumor_meningioma.jpg",
            "data/sample_images/regular_samples/10_tumor_meningioma.jpg",
        ],
        "description": (
            "Arises from outer membranes (extra-axial). Distinct, well-demarcated rounded boundary "
            "attached to the skull or dural folds, pushing inward on adjacent brain tissue."
        ),
        "icon": "🟠",
    },
    "Pituitary Tumor": {
        "images": [
            "data/sample_images/regular_samples/11_tumor_pituitary.jpg",
            "data/sample_images/regular_samples/12_tumor_pituitary.jpg",
        ],
        "description": (
            "Located at the skull base in the pituitary fossa (sella turcica). Centrally positioned, "
            "frequently well-defined, and may extend upward into the suprasellar space."
        ),
        "icon": "🟡",
    },
}


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
    st.title("Brain Tumor Detection & Classification Using Deep Learning")
    st.write(
        "Explore how a deep learning image classifier interprets brain MRI scans. "
        "This dashboard provides reference visual examples of different brain tumor types "
        "and allows you to upload an MRI image to obtain specific tumor classification, probabilities, and Grad-CAM interpretability."
    )
    # Render before any early return so the disclaimer also appears in error states.
    st.info(DISCLAIMER, icon=":material/info:")
    try:
        config = load_config()
        configure_runtime(config)
    except (ValueError, FileNotFoundError, RuntimeError, OSError) as error:
        st.error(f"Configuration could not be loaded. {error}")
        return

    # -- Permanent reference gallery: visual guide showing how each tumor looks like --
    st.divider()
    st.subheader("Brain Tumor Reference Guide (Visual Comparison)")
    st.caption(
        "Reference MRI examples illustrating visual characteristics of healthy brain tissue versus "
        "the 3 primary tumor types before performing automated predictions."
    )
    gallery_cols = st.columns(4, gap="medium")
    for gallery_col, (tumor_name, ref_info) in zip(gallery_cols, REFERENCE_IMAGES.items()):
        with gallery_col:
            with st.container(border=True):
                st.markdown(f"**{ref_info['icon']} {tumor_name}**")
                for img_path in ref_info["images"]:
                    resolved = resolve_path(img_path, config)
                    if resolved.is_file():
                        st.image(str(resolved), caption=resolved.stem.replace("_", " ").title(), width="stretch")
                st.caption(ref_info["description"])
    st.divider()

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
        st.subheader("Prediction Result")
        if result is None:
            st.write("Your result will appear here after you upload an image and select Predict.")
            st.caption("The classifier does not verify that an uploaded image is an MRI or determine whether it is suitable for this model.")
        else:
            raw_class = result["class_name"]
            info = TUMOR_INFO.get(raw_class, {})
            display_name = info.get("display", raw_class.replace("_", " ").replace("notumor", "no tumor").title())
            icon = info.get("icon", "")
            is_tumor = raw_class not in {"notumor", "no_tumor"}

            if is_tumor:
                st.error(f"**Tumor Detected: {display_name}**")
            else:
                st.success(f"**No Tumor Detected ({display_name})**")

            with st.container(border=True):
                st.metric("Specific Classification", f"{icon} {display_name}".strip())
                st.metric("Confidence", f"{result['confidence']:.2%}")
                if info.get("description"):
                    st.caption(f"**About this class:** {info['description']}")
                st.caption("Confidence is an uncalibrated model probability, not a measure of clinical certainty.")
            st.subheader("Class probabilities")
            for name, probability in zip(result["class_names"], result["probabilities"]):
                c_info = TUMOR_INFO.get(name, {})
                c_display = c_info.get("display", name.replace("_", " ").replace("notumor", "no tumor").title())
                c_icon = c_info.get("icon", "")
                st.progress(float(probability), text=f"{c_icon} {c_display}: {probability:.2%}".strip())
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
