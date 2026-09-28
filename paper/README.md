# IEEE conference-format manuscript

The requested deliverable is `Brain_Tumor_Detection_IEEE_Paper.docx`, an editable Word manuscript built from an IEEE-hosted A4 conference template. It compares the Custom CNN, EfficientNetB0, and MobileNetV2 on the project's audited binary MRI dataset. It is a manuscript draft, not an IEEE publication or acceptance.

## Author information

The user confirmed the following author information on **September 25, 2026**. Names, student IDs, and email addresses are reproduced as supplied; the IDs are institutional student identifiers, not publication identifiers.

| Author | Student ID | Email |
|---|---|---|
| Mohit Singh | 23101C0002 | mohit.songra@vit.edu.in |
| Rudra Dalvi | 23101C0005 | rudra.dalvi@vit.edu.in |
| Sameed Mulla | 23101C0030 | sameed.mulla@vit.edu.in |
| Adarsh Yadav | 23101C0012 | adarsh.yadav@vit.edu.in |

Shared affiliation: **Department of Information Technology, Vidyalankar Institute of Technology, Wadala, Mumbai, India**.

These confirmed details replace the original author placeholders. Before any submission, check the eventual conference's page limit, anonymity rules, and requirements for including student IDs. The manuscript discloses substantive OpenAI Codex assistance; supplying author metadata does not establish that the authors have reviewed or approved the manuscript. This metadata update does not change the experiments or measured results.

## Evidence and editable sources

- `manuscript.md`: manuscript source, with data tokens for the added experiment.
- `build_manuscript.py`: creates editable paragraphs, native Word tables, and embedded scientific figures while preserving the retained template's styles and numbering.
- `Brain_Tumor_Detection_IEEE_Paper.md`: resolved manuscript text, generated with the final Word file.
- `Brain_Tumor_Detection_IEEE_Paper.provenance.json`: experiment hashes and template-preservation evidence for that Word file.
- `experiment_protocol.md`: fixed MobileNet extension protocol and disclosure of earlier test-set inspection.
- `references_notes.md`: verified primary references, template sources, checksums, and formatting assumptions.
- `templates/`: unmodified IEEE-hosted reference files.
- `figures/`: charts derived only from completed measured experiments; their source hashes are recorded.
- `build/`: local template audit, Word rendering, and visual-check intermediates; excluded from version control.

The manuscript builder refuses final output if any model lacks a completed real-data evaluation, a checkpoint differs from its recorded hash, or the figure inputs have changed. It never substitutes smoke-test performance. Internal `--preview` output uses explicitly pending values and cannot overwrite the final manuscript.

## Rebuilding

After real training/evaluation of all three models, run the figure builder in the project's Python environment:

```powershell
python scripts/build_paper_figures.py
```

The optional Word builder requires Python 3.10+, `lxml`, and Pillow. In this task it was run with the document workspace's bundled Python and libraries, independently of the TensorFlow environment. The machine-specific renderer uses a task-local LibreOffice administrative extraction and bundled Poppler; no desktop office application was installed.

The retained template must first have its layout inspected and recorded in `build/artifact.md`. Then run:

```powershell
python paper/build_manuscript.py
```

Always render the resulting DOCX and inspect every page after edits. PDF and page-image renders are internal quality checks; the requested final deliverable is Word.

## Interpretation

The comparison uses one seed and a public image dataset without patient identifiers. MobileNetV2 was added after the original test results were viewed. Results are exploratory image-classification measurements, not independent external confirmation, proof of perfect performance, or clinical validation.
