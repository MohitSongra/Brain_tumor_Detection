# Streamlit design decisions

The user-approved implementation plan specifies a restrained, native Streamlit interface. This is an academic tool for operating a model, with no marketing landing page or generated decorative imagery.

## Surface and typography

Use a white page, pale neutral secondary surfaces, dark text, and a teal action accent. The theme is declared in .streamlit/config.toml: primary #14786F, background #FFFFFF, secondary #F3F6F6, text #192D35. Retain Streamlit's readable sans-serif typography, semantic headings, and native keyboard focus behavior.

## Composition

Project title and short introduction lead into the always-visible disclaimer. Upload/preview and prediction/probability columns share the main area. After inference, original/heatmap/overlay images appear together below. Native columns stack on narrow viewports. Use one bordered result container; avoid nested cards or decorative status indicators.

## States

Missing model: an actionable training message and disabled Predict button. Empty upload: explain where results appear. Invalid image/configuration: readable error while retaining the disclaimer. Predicting: native spinner. Completed: class, confidence, labeled probability bars, and class-specific interpretation. Upload changes clear previous predictions.

## Visual semantics

Do not use red/green to imply a medical diagnosis. Probability bars are quantitative model outputs, and confidence is explicitly uncalibrated. No stock MRI or invented result is displayed as a model prediction.
