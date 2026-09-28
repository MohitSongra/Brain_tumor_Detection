# IEEE paper: verified references and template provenance

Verified on September 24, 2026. These are preparation notes, not a submitted paper. No author identity, institution, conference acceptance, copyright transfer, or experimental result is inferred here.

## Template provenance

The user-specified entry point is [IEEE conference publishing templates](https://www.ieee.org/conferences/publishing/templates). IEEE's [Authoring Tools and Templates](https://conferences.ieeeauthorcenter.ieee.org/write-your-paper/authoring-tools-and-templates/) page explicitly directs conference authors to its Word or LaTeX templates.

The canonical A4 Word URL is:

<https://www.ieee.org/content/dam/ieee-org/ieee/web/org/conferences/conference-template-a4.docx>

The canonical host returned HTTP 202 with an empty HTML response during retrieval, rather than a document. The following unmodified files were therefore obtained from IEEE-hosted conference resources. They are archived under `paper/templates/`.

| Local file | Actual download source | Verification |
|---|---|---|
| `conference-template-a4.docx` | <https://enotice.vtools.ieee.org/attachments/download/47078> | HTTP 200; DOCX MIME type; valid ZIP/Word package; 33,858 bytes |
| `conference-template-a4-argencon2020-reference.docx` | <https://attend.ieee.org/argencon-2020/wp-content/uploads/sites/171/conference-template-a4.docx> | HTTP 200; DOCX MIME type; valid ZIP/Word package; 30,552 bytes |
| `ieee-ccece2015-sample-historical.pdf` | <https://ewh.ieee.org/reg/ccece15/files/ccece-word-sample.pdf> | HTTP 200; PDF MIME type; 68,677 bytes; historical sample only |

The first Word attachment is listed on the [IEEE vTools announcement for IC-FTAI 2025](https://enotice.vtools.ieee.org/public/177836), which also links the canonical A4 template URL. Its bytes are not asserted to be identical to the currently inaccessible canonical file. The ARGENCON 2020 copy independently corroborates the section geometry. The older CCECE PDF illustrates IEEE styling; it is **not** a rendering of either downloaded DOCX and does not override their A4 settings.

SHA-256 hashes, in the same order as the table:

```text
57e2125b0a04860e103fd634c2b5fd19a7ea571fa59ac577bd97afc73ad99482
9d0153f7121508f585eff77c6fc112234a9d289c0390bf54a5296d15d01b04d3
f6c435a9a63bdef0517d60b6932cb05a8af3b29fc76abafc5542f99070db1e77
```

### Word layout findings

Both DOCX copies use **Strict OOXML**, including the namespace `http://purl.oclc.org/ooxml/wordprocessingml/main` and point-valued lengths. Some Python Word libraries expect Transitional OOXML. Preserve the originals; if conversion is necessary, use a separate working copy and verify its rendered layout.

Inspection of both files establishes:

- A4 page: 595.30 × 841.90 points, approximately 210 × 297 mm.
- Main text section: **two columns**, 18-point gap; top 54 points, bottom 72 points, left/right 45.35 points.
- Full-width title and three-column author sections use separate settings. Do not apply those settings to the body.
- The last main-text paragraph contains the two-column `sectPr`; section properties govern the content **before** that break. The final body-level `sectPr` is one column and governs a trailing drawing/text-box instruction. Copying that final element as the main layout would incorrectly produce a one-column paper.
- The styles specify Times New Roman, 24-point paper title, 11-point author text, 9-point bold abstract, and 8-point captions/references. Preserve the template's body style rather than introducing a generic document font or spacing preset.

Retain prescribed layout and remove instructional text, example references, sample author blocks, and the placeholder copyright/ISBN footer. Do not invent an IEEE publication notice. The template places figure captions below figures and table headings above tables; citations use ordered bracketed numbers. Do not add page numbers unless the selected venue requests them.

IEEE independently instructs authors to remove template guidance in [Publishing Information for IEEE Conference Authors](https://events.ieee.org/planning-basics/ieee-conference-publications/publishing-information-for-ieee-conference-authors/). It also describes PDF eXpress when offered by a participating conference.

### Working assumptions requiring a venue choice

Use an editable A4, two-column **Word DOCX** as requested. A six-page draft including references is a practical planning target, **not a universal IEEE requirement**. The selected venue controls page limits, anonymity, abstract length, copyright notices, and final submission format. Author placeholders are authorized for this draft; replace them with confirmed identities before submission. Creating an IEEE-styled document does not establish submission readiness, acceptance, or clinical validation.

## AI-content disclosure

The current [IEEE Author Center policy, “Submission and Peer Review Policies,” AI-generated-content section](https://journals.ieeeauthorcenter.ieee.org/become-an-ieee-journal-author/publishing-ethics/guidelines-and-policies/submission-and-peer-review-policies/) requires acknowledgment of AI-generated article content, including text, figures, images, and code. Identify the AI system, affected sections, and extent of assistance. Ordinary grammar editing is generally exempt from the mandatory disclosure, although disclosure remains recommended. The page applies this provision to any article submitted to an IEEE publication; the [IEEE DSAA 2025 conference instructions](https://dsaa.ieee.org/2025/applications-track-2/) also state an AI disclosure rule.

For this work, disclose **OpenAI Codex** assistance with implementation and manuscript drafting, and identify the actual affected sections/figures when the manuscript is complete. Do not characterize this assistance as spelling-only. Do not name an unverified model version. Human authors must check the final claims, references, measured results, and attribution; do not claim that review has already occurred. An AI tool is not a substitute for confirmed human authorship.

## Primary research references

The entries below supply verified bibliographic facts and appropriate uses. Renumber them in order of first citation in the final manuscript. No reference's reported benchmark performance should be presented as a result of this MRI experiment.

### MobileNetV2

M. Sandler, A. Howard, M. Zhu, A. Zhmoginov, and L.-C. Chen, “MobileNetV2: Inverted Residuals and Linear Bottlenecks,” in *2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2018, pp. 4510–4520, doi: **10.1109/CVPR.2018.00474**.

- Authors: Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, Liang-Chieh Chen.
- Primary proceedings page: <https://openaccess.thecvf.com/content_cvpr_2018/html/Sandler_MobileNetV2_Inverted_Residuals_CVPR_2018_paper.html>
- Author preprint: <https://arxiv.org/abs/1801.04381>
- DOI metadata verification: <https://api.crossref.org/works/10.1109/CVPR.2018.00474>
- Use: inverted residuals and linear bottlenecks as the basis of the third architecture; motivation for efficient feature extraction.

### EfficientNet

M. Tan and Q. V. Le, “EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks,” in *Proceedings of the 36th International Conference on Machine Learning*, PMLR, vol. 97, 2019, pp. 6105–6114.

- Authors: Mingxing Tan and Quoc V. Le.
- Primary proceedings page and BibTeX: <https://proceedings.mlr.press/v97/tan19a.html>
- Primary paper: <https://proceedings.mlr.press/v97/tan19a/tan19a.pdf>
- Author preprint: <https://arxiv.org/abs/1905.11946>
- DOI: none displayed by the primary PMLR record; do not invent one.
- Use: EfficientNet architecture and compound scaling; the implemented experiment uses the B0 backbone.

### Grad-CAM

R. R. Selvaraju, M. Cogswell, A. Das, R. Vedantam, D. Parikh, and D. Batra, “Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization,” in *2017 IEEE International Conference on Computer Vision (ICCV)*, 2017, pp. 618–626, doi: **10.1109/ICCV.2017.74**.

- Authors: Ramprasaath R. Selvaraju, Michael Cogswell, Abhishek Das, Ramakrishna Vedantam, Devi Parikh, Dhruv Batra.
- Primary proceedings paper: <https://openaccess.thecvf.com/content_ICCV_2017/papers/Selvaraju_Grad-CAM_Visual_Explanations_ICCV_2017_paper.pdf>
- Author preprint: <https://arxiv.org/abs/1610.02391>
- DOI metadata verification: <https://api.crossref.org/works/10.1109/ICCV.2017.74>
- Use: gradient-based spatial attribution. A heatmap is not a tumor segmentation, clinical explanation, or proof of medically correct localization. Cite the 2017 conference record consistently rather than mixing its date/pages with later journal versions.

### Leakage and reproducibility

S. Kapoor and A. Narayanan, “Leakage and the reproducibility crisis in machine-learning-based science,” *Patterns*, vol. 4, no. 9, Art. no. 100804, 2023, doi: **10.1016/j.patter.2023.100804**.

- Authors: Sayash Kapoor and Arvind Narayanan.
- Publisher URL: <https://www.cell.com/patterns/fulltext/S2666-3899(23)00159-9>
- Accessible primary manuscript: <https://par.nsf.gov/servlets/purl/10513990>
- Author preprint: <https://arxiv.org/abs/2207.07048>
- DOI metadata verification: <https://api.crossref.org/works/10.1016/j.patter.2023.100804>
- Use: leakage risks and the importance of transparent, reproducible evaluation. This citation does not establish patient independence in the present dataset.

### Precision–recall evaluation

T. Saito and M. Rehmsmeier, “The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets,” *PLOS ONE*, vol. 10, no. 3, e0118432, 2015, doi: **10.1371/journal.pone.0118432**.

- Authors: Takaya Saito and Marc Rehmsmeier.
- Primary publisher article: <https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0118432>
- Published March 4, 2015.
- Use: report precision–recall behavior alongside ROC analysis when prevalence is uneven; do not treat accuracy alone as adequate. Sensitivity and specificity refer to this held-out image set, not prospective patient diagnosis.

### Adam optimization

D. P. Kingma and J. Ba, “Adam: A Method for Stochastic Optimization,” in *3rd International Conference on Learning Representations (ICLR)*, 2015.

- Authors: Diederik P. Kingma and Jimmy Ba.
- Primary author manuscript: <https://arxiv.org/abs/1412.6980>
- The preprint was first submitted in 2014; the record identifies the conference publication as ICLR 2015.
- Repository DOI: **10.48550/arXiv.1412.6980**. This is an arXiv DOI, not a conference-publisher DOI.
- Use: the optimizer implemented in the training pipelines.

### Dropout

N. Srivastava, G. Hinton, A. Krizhevsky, I. Sutskever, and R. Salakhutdinov, “Dropout: A Simple Way to Prevent Neural Networks from Overfitting,” *Journal of Machine Learning Research*, vol. 15, no. 56, pp. 1929–1958, 2014.

- Authors: Nitish Srivastava, Geoffrey Hinton, Alex Krizhevsky, Ilya Sutskever, Ruslan Salakhutdinov.
- Primary journal record: <https://jmlr.org/papers/v15/srivastava14a.html>
- DOI: none displayed by the primary record; do not invent one.
- Use: rationale for dropout regularization; its presence does not demonstrate that overfitting was eliminated.

### Dataset citation

M. Nickparvar, “Brain Tumor MRI Dataset,” version 2, Kaggle, 2026. Accessed September 24, 2026.

- Curator: Masoud Nickparvar.
- Dataset: <https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset>
- Metadata: <https://www.kaggle.com/api/v1/datasets/view/masoudnickparvar/brain-tumor-mri-dataset>
- Version-pinned archive: <https://www.kaggle.com/api/v1/datasets/download/masoudnickparvar/brain-tumor-mri-dataset?datasetVersionNumber=2>
- Version 2 date and current CC BY 4.0 license are documented in the project dataset guide. A dataset DOI has not been independently verified here; use the source/version URL rather than an inferred DOI.
- Cite the curator and explain binary relabeling, augmentation-marker exclusion, duplicate/group filtering, and the retained split. Report the actual manifest counts, not the advertised count as the training sample size. The public archive has no supplied patient identities; do not describe its sequence filenames as patient identifiers or claim patient-level validation.

## Evidence boundaries for manuscript writing

Use only the project's measured, provenance-matched evaluation artifacts for the three-model result table. MobileNetV2 results remain unavailable until that run completes and its artifacts are checked. Do not cite architecture papers as prior MRI accuracy baselines, claim state-of-the-art performance, infer clinical usefulness, or add invented ethics approval, consent, external validation, or patient metadata. The manuscript should identify this as an academic brain MRI **image classification** prototype and distinguish image-level performance from clinical diagnosis.
