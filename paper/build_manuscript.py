"""Build an editable IEEE Word paper from the retained Strict OOXML template.

Run with the document workspace runtime (lxml and Pillow). The ML application
does not import this optional manuscript-authoring helper. A preview may contain
explicit pending cells; the final output requires all three real experiments.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from zipfile import ZipFile, ZIP_DEFLATED

from lxml import etree as E
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
TEMPLATE = PAPER / "templates/conference-template-a4.docx"
TEMPLATE_HASH = "57e2125b0a04860e103fd634c2b5fd19a7ea571fa59ac577bd97afc73ad99482"
MODELS = ("custom_cnn", "efficientnet", "mobilenet")
NAMES = ("Custom CNN", "EfficientNetB0", "MobileNetV2")
SHORT_NAMES = ("CNN", "EffNetB0", "MobileNetV2")
# The narrative interprets these completed experiments. A retrained baseline
# needs fresh scientific editing, not automatic reuse of old prose.
INTERPRETED_CHECKPOINTS = {
    "custom_cnn": "9c6468e9dbc1aae2cc497f7d799e2dc4c293787c560e00679fd40631f5b2161d",
    "efficientnet": "0281f01718b16ba39be696d4c3ab3dadab461195698acd74cfc4f973b8dd64f8",
    "mobilenet": "dcb4c3a87d34cd8e5bc851055491bb1a35ed998d488e4e4c44cec4242e12f8e4",
}
W = "http://purl.oclc.org/ooxml/wordprocessingml/main"
R = "http://purl.oclc.org/ooxml/officeDocument/relationships"
WP = "http://purl.oclc.org/ooxml/drawingml/wordprocessingDrawing"
A = "http://purl.oclc.org/ooxml/drawingml/main"
PIC = "http://purl.oclc.org/ooxml/drawingml/picture"
REL = "http://schemas.openxmlformats.org/package/2006/relationships"
CT = "http://schemas.openxmlformats.org/package/2006/content-types"
NS = {"w": W, "r": R, "wp": WP}


def q(local: str) -> str:
    return f"{{{W}}}{local}"


def child(parent, tag, **attributes):
    node = E.SubElement(parent, q(tag))
    for name, value in attributes.items():
        node.set(q(name), str(value))
    return node


def paragraph(text: str = "", style: str = "BodyText", *, center=False, keep=False):
    node = E.Element(q("p"))
    props = child(node, "pPr")
    child(props, "pStyle", val=style)
    if center:
        child(props, "jc", val="center")
    if keep:
        child(props, "keepNext")
        child(props, "keepLines")
    if text:
        add_run(node, text)
    return node


def add_run(node, text: str, *, bold=False, italic=False):
    run = child(node, "r")
    if bold or italic:
        props = child(run, "rPr")
        if bold:
            child(props, "b")
        if italic:
            child(props, "i")
    # Explicit line breaks support the author block without a layout table.
    for index, line in enumerate(text.split("\n")):
        if index:
            child(run, "br")
        element = child(run, "t")
        element.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
        element.text = line
    return run


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_artifacts(preview: bool):
    audit = json.loads((ROOT / "data/processed/audit.json").read_text(encoding="utf-8"))
    if audit.get("status") != "passed" or sha(ROOT / "data/processed/manifest.csv") != audit["dataset_fingerprint"]:
        raise ValueError("The paper requires the current audited manifest.")
    metrics, runs = {}, {}
    source_hashes = {"data/processed/audit.json": sha(ROOT / "data/processed/audit.json")}
    for name in MODELS:
        metric_path = ROOT / f"results/evaluation/{name}/metrics.json"
        run_path = ROOT / f"results/training/{name}/run.json"
        if not metric_path.is_file() or not run_path.is_file():
            if preview:
                metrics[name] = runs[name] = None
                continue
            raise FileNotFoundError(f"Training required — no measured result available for {name}.")
        result = json.loads(metric_path.read_text(encoding="utf-8"))
        run = json.loads(run_path.read_text(encoding="utf-8"))
        if result.get("model_sha256") != INTERPRETED_CHECKPOINTS[name]:
            raise ValueError("Recorded experiment changed; update the manuscript interpretation before rebuilding.")
        expected = {"model_name": name, "trained_on_real_data": True, "evaluation_split": "test",
                    "dataset_fingerprint": audit["dataset_fingerprint"], "threshold": 0.5,
                    "class_names": ["no_tumor", "tumor"], "task": "binary",
                    "n_samples": sum(audit["split_class_counts"]["test"].values())}
        for key, value in expected.items():
            if result.get(key) != value:
                raise ValueError(f"Paper provenance mismatch: {name} {key}")
        model = ROOT / result["model_path"]
        if not model.is_file() or sha(model) != result["model_sha256"]:
            raise ValueError(f"{name} model differs from the evaluated checkpoint.")
        for key in ("model_name", "dataset_fingerprint", "task", "class_names"):
            if run.get(key) != result[key]:
                raise ValueError(f"{name} training and evaluation records disagree on {key}.")
        metrics[name], runs[name] = result, run
        source_hashes.update({str(p.relative_to(ROOT)): sha(p) for p in (metric_path, run_path)})
    if not preview:
        figure_provenance = json.loads((PAPER / "figures/provenance.json").read_text(encoding="utf-8"))
        if not figure_provenance.get("all_reported_metrics_recomputed_from_saved_predictions"):
            raise ValueError("Paper figures have not been validated against measured predictions.")
        if (set(figure_provenance.get("source_files", {})) != set(MODELS)
                or figure_provenance.get("dataset_fingerprint") != audit["dataset_fingerprint"]
                or figure_provenance.get("test_images") != expected["n_samples"]):
            raise ValueError("Paper figure provenance does not describe this three-model test cohort.")
        expected_figures = {"roc_comparison.png", "validation_history.png"}
        if set(figure_provenance.get("figure_sha256", {})) != expected_figures:
            raise ValueError("Paper figure hashes are missing or incomplete.")
        for name, expected_hash in figure_provenance["figure_sha256"].items():
            if sha(PAPER / "figures" / name) != expected_hash:
                raise ValueError(f"Paper figure changed after verification: {name}")
        for files in figure_provenance["source_files"].values():
            for relative, expected_hash in files.items():
                if sha(ROOT / relative) != expected_hash:
                    raise ValueError(f"Paper figures are stale: {relative}")
    return audit, metrics, runs, source_hashes


def make_tables(audit, metrics, runs):
    categories = ("glioma", "meningioma", "notumor", "pituitary")
    cohort = [["Split", "Glioma", "Mening.", "No tumor", "Pituitary", "Total"]]
    for split in ("train", "validation", "test"):
        counts = audit["split_class_counts"][split]
        cohort.append([split.title(), *[str(counts[c]) for c in categories], str(sum(counts.values()))])
    architecture = [["Model", "Initialization", "Total parameters"],
                    ["Custom CNN", "Random", "110,561"],
                    ["EfficientNetB0", "ImageNet", "4,213,668"],
                    ["MobileNetV2", "ImageNet", "2,422,081"]]
    training = [["Model", "Epochs", "Selected stage / epoch", "Minutes"]]
    for name, label in zip(MODELS, SHORT_NAMES):
        run = runs[name]
        if run:
            epochs = "+".join(str(p["epochs_run"]) for p in run["phases"])
            phase = "Fine-tune" if run["selected_phase"] == "finetune" else run["selected_phase"].title()
            training.append([label, epochs, f"{phase} / {run['selected_epoch']}", f"{run['training_seconds']/60:.2f}"])
        else:
            training.append([label, "Pending", "Pending", "Pending"])
    results = [["Metric", *SHORT_NAMES]]
    for key, label in (("accuracy", "Accuracy (%)"), ("precision", "Precision (%)"),
                       ("recall", "Sensitivity (%)"), ("specificity", "Specificity (%)"),
                       ("f1", "F1 (%)"), ("roc_auc", "ROC-AUC")):
        values = [f"{metrics[n][key]:.4f}" if key == "roc_auc" else f"{100*metrics[n][key]:.2f}"
                  for n in MODELS if metrics[n]]
        row = [label]
        for name in MODELS:
            row.append(values.pop(0) if metrics[name] else "Pending")
        results.append(row)
    confusion = [["Model", "TN", "FP", "FN", "TP"]]
    for name, label in zip(MODELS, SHORT_NAMES):
        counts = metrics[name]["per_class"]["tumor"] if metrics[name] else None
        confusion.append([label, *[str(counts[k]) if counts else "Pending" for k in ("tn", "fp", "fn", "tp")]])
    return {
        "TABLE_COHORT": ("Audited image counts by source category", cohort, [46, 39, 40, 44, 42, 32]),
        "TABLE_ARCHITECTURES": ("Implemented model architectures", architecture, [85, 74, 84]),
        "TABLE_TRAINING": ("Measured training and checkpoint selection", training, [67, 38, 91, 47]),
        "TABLE_RESULTS": ("Binary classification on 1,476 test images", results, [81, 46, 58, 58]),
        "TABLE_CONFUSION": ("Test confusion counts at threshold 0.5", confusion, [79, 41, 41, 41, 41]),
    }


def add_table(body, table_info, source_table):
    caption, rows, widths = table_info
    body.append(paragraph(caption, "tablehead", keep=True))
    table = E.Element(q("tbl"))
    # Retain template table treatment (black grid, compact Times text).
    props = deepcopy(source_table.find(q("tblPr")))
    if props is None:
        props = E.Element(q("tblPr"))
    for tag in ("tblW", "tblInd", "tblLayout", "tblCellMar"):
        for old in props.findall(q(tag)):
            props.remove(old)
    child(props, "tblW", w="243pt", type="dxa")
    child(props, "tblInd", w="0pt", type="dxa")
    child(props, "tblLayout", type="fixed")
    table.append(props)
    grid = child(table, "tblGrid")
    for width in widths:
        child(grid, "gridCol", w=f"{width}pt")
    for index, row in enumerate(rows):
        tr = child(table, "tr")
        tr_props = child(tr, "trPr")
        child(tr_props, "cantSplit")
        if index == 0:
            child(tr_props, "tblHeader")
        for column, (value, width) in enumerate(zip(row, widths)):
            cell = child(tr, "tc")
            cell_props = child(cell, "tcPr")
            child(cell_props, "tcW", w=f"{width}pt", type="dxa")
            child(cell_props, "vAlign", val="center")
            p = paragraph("", "tablecopy", center=column > 0, keep=index < len(rows)-1)
            pp = p.find(q("pPr"))
            child(pp, "spacing", before="0pt", after="0pt")
            if column == 0:
                child(pp, "jc", val="start")
            add_run(p, value, bold=index == 0)
            cell.append(p)
    body.append(table)


def add_figure(body, key, package, relations, image_counter, preview):
    name, caption, height = {
        "FIGURE_ROC": ("roc_comparison.png", "ROC curves from saved test probabilities. Legend values are ROC-AUC; tumor is the positive class.", 2.5),
        "FIGURE_HISTORY": ("validation_history.png", "Measured validation loss and accuracy across completed epochs. Loss uses a logarithmic scale. Dots mark the first fine-tuning epoch; lines are not smoothed.", 3.55),
    }[key]
    path = PAPER / "figures" / name
    p = paragraph("", "Normal", center=True, keep=True)
    if not path.is_file():
        if not preview:
            raise FileNotFoundError(f"Missing paper figure {path}")
        # Reserve exact figure height without simulated curves or measurements.
        props = p.find(q("pPr"))
        child(props, "spacing", before="0pt", after="0pt", line=f"{height*72:.2f}pt", lineRule="exact")
        add_run(p, "[Figure pending completed experiment]")
    else:
        with Image.open(path) as im:
            aspect = im.height / im.width
        width_emu = int(3.35 * 914400)
        height_emu = int(width_emu * aspect)
        rid = f"rIdPaperFigure{image_counter}"
        target = f"media/paper_figure_{image_counter}.png"
        package[f"word/{target}"] = path.read_bytes()
        E.SubElement(relations, f"{{{REL}}}Relationship", Id=rid, Type=f"{R}/image", Target=target)
        run = child(p, "r")
        drawing = child(run, "drawing")
        inline = E.SubElement(drawing, f"{{{WP}}}inline", distT="0", distB="0", distL="0", distR="0")
        E.SubElement(inline, f"{{{WP}}}extent", cx=str(width_emu), cy=str(height_emu))
        E.SubElement(inline, f"{{{WP}}}docPr", id=str(image_counter+100), name=name, descr=caption)
        graphic = E.SubElement(inline, f"{{{A}}}graphic")
        data = E.SubElement(graphic, f"{{{A}}}graphicData", uri=PIC)
        picture = E.SubElement(data, f"{{{PIC}}}pic")
        nv = E.SubElement(picture, f"{{{PIC}}}nvPicPr")
        E.SubElement(nv, f"{{{PIC}}}cNvPr", id="0", name=name)
        E.SubElement(nv, f"{{{PIC}}}cNvPicPr")
        fill = E.SubElement(picture, f"{{{PIC}}}blipFill")
        E.SubElement(fill, f"{{{A}}}blip", {f"{{{R}}}embed": rid})
        stretch = E.SubElement(fill, f"{{{A}}}stretch")
        E.SubElement(stretch, f"{{{A}}}fillRect")
        shape = E.SubElement(picture, f"{{{PIC}}}spPr")
        transform = E.SubElement(shape, f"{{{A}}}xfrm")
        E.SubElement(transform, f"{{{A}}}off", x="0", y="0")
        E.SubElement(transform, f"{{{A}}}ext", cx=str(width_emu), cy=str(height_emu))
        geometry = E.SubElement(shape, f"{{{A}}}prstGeom", prst="rect")
        E.SubElement(geometry, f"{{{A}}}avLst")
    body.append(p)
    body.append(paragraph(caption, "figurecaption"))


def serialize(node):
    return E.tostring(node, encoding="UTF-8", xml_declaration=True, standalone=True)


def order_properties(document):
    """Honor Strict OOXML property order instead of relying on renderer repair."""
    definitions = {
        "pPr": "pStyle keepNext keepLines pageBreakBefore framePr widowControl numPr suppressLineNumbers pBdr shd tabs suppressAutoHyphens kinsoku wordWrap overflowPunct topLinePunct autoSpaceDE autoSpaceDN bidi adjustRightInd snapToGrid spacing ind contextualSpacing mirrorIndents suppressOverlap jc textDirection textAlignment textboxTightWrap outlineLvl divId cnfStyle rPr sectPr pPrChange",
        "tblPr": "tblStyle tblpPr tblOverlap bidiVisual tblStyleRowBandSize tblStyleColBandSize tblW jc tblCellSpacing tblInd tblBorders shd tblLayout tblCellMar tblLook tblCaption tblDescription tblPrChange",
    }
    for tag, sequence in definitions.items():
        rank = {name: index for index, name in enumerate(sequence.split())}
        for props in document.iter(q(tag)):
            props[:] = sorted(props, key=lambda n: rank.get(E.QName(n).localname, len(rank)))


def add_author_rows(body, authors, author_section):
    """Set four supplied authors in two native two-column IEEE author rows."""
    if len(authors) != 4 or any(len(author.splitlines()) != 6 for author in authors):
        raise ValueError("Provide four six-line author blocks before the Abstract heading.")
    for index, author in enumerate(authors):
        p = paragraph("", "Author", center=True)
        props = p.find(q("pPr"))
        child(props, "keepLines")
        child(props, "spacing", before="6pt", after="6pt")
        if index % 2:
            # Start the second author of each row in the right-hand column.
            child(child(p, "r"), "br", type="column")
        for line_index, line in enumerate(author.splitlines()):
            run_node = add_run(p, ("\n" if line_index else "") + line,
                               italic=line_index in (1, 2))
            run_props = run_node.find(q("rPr"))
            if run_props is None:
                run_props = E.Element(q("rPr"))
                run_node.insert(0, run_props)
            # Source author runs use 9 pt, overriding the named style's 11 pt.
            child(run_props, "sz", val="18")
            child(run_props, "szCs", val="18")
        if index % 2:
            section = deepcopy(author_section)
            columns = section.find(q("cols"))
            columns.set(q("num"), "2")
            for column in list(columns):
                columns.remove(column)
            props.append(section)
        body.append(p)


def build(preview=False):
    if sha(TEMPLATE) != TEMPLATE_HASH:
        raise ValueError("The retained template changed; repeat template distillation first.")
    contract = PAPER / "build/artifact.md"
    if not contract.is_file() or TEMPLATE_HASH not in contract.read_text(encoding="utf-8"):
        raise ValueError("The current template's distillation contract is required.")
    audit, metrics, runs, sources = read_artifacts(preview)
    text = (PAPER / "manuscript.md").read_text(encoding="utf-8")
    mobile, run = metrics["mobilenet"], runs["mobilenet"]
    replacements = {}
    for key in ("accuracy", "precision", "recall", "specificity", "f1", "roc_auc"):
        replacements[f"mobilenet_{key}"] = (f"{mobile[key]:.4f}" if key == "roc_auc" else f"{100*mobile[key]:.2f}") if mobile else "Pending"
    for key in ("fn", "fp"):
        replacements[f"mobilenet_{key}"] = str(mobile["per_class"]["tumor"][key]) if mobile else "Pending"
    replacements["mobilenet_epochs"] = str(run["total_epochs"]) if run else "Pending"
    replacements["mobilenet_minutes"] = f"{run['training_seconds']/60:.2f}" if run else "Pending"
    if run:
        history_path = ROOT / "results/training/mobilenet/history.json"
        history = json.loads(history_path.read_text(encoding="utf-8"))
        sources[str(history_path.relative_to(ROOT))] = sha(history_path)
        phase_losses = "; ".join(f"{p['phase']} {p['best_val_loss']:.4f}" for p in run["phases"])
        stage = "fine-tuning" if run["selected_phase"] == "finetune" else "head training"
        replacements["mobilenet_validation_analysis"] = (
            f"MobileNetV2's phase-specific minimum validation losses were {phase_losses}. "
            f"The evaluated checkpoint comes from {stage}, epoch {run['selected_epoch']}, "
            f"with validation loss {run['selected_val_loss']:.4f}. "
            f"The run ended at validation loss {history['val_loss'][-1]:.4f} and validation "
            f"accuracy {100*history['val_accuracy'][-1]:.2f}%. "
            "The validation rule determines checkpoint selection independently of the test comparison.")
    else:
        replacements["mobilenet_validation_analysis"] = "[MobileNetV2 validation analysis pending completed training.]"
    for key, value in replacements.items():
        text = text.replace("{{" + key + "}}", value)
    if not preview and ("{{mobilenet_" in text or "Pending" in text):
        raise ValueError("Unresolved result placeholders in final manuscript.")
    tables = make_tables(audit, metrics, runs)
    with ZipFile(TEMPLATE) as archive:
        package = {name: archive.read(name) for name in archive.namelist()}
    document = E.fromstring(package["word/document.xml"])
    body = document.find(q("body"))
    original = list(body)
    section_properties = body.xpath(".//w:sectPr", namespaces=NS)
    title_section = deepcopy(section_properties[0])
    main_section = deepcopy(section_properties[3])
    if main_section.find(q("cols")).get(q("num")) != "2":
        raise ValueError("The reference main section is not two-column.")
    source_table = body.find(q("tbl"))
    for node in original:
        body.remove(node)
    blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]
    title = blocks.pop(0).removeprefix("# ")
    title_paragraph = paragraph(title, "papertitle")
    child(title_paragraph.find(q("pPr")), "spacing", before="5pt", after="5pt")
    title_paragraph.find(q("pPr")).append(title_section)
    body.append(title_paragraph)
    authors = []
    while blocks and not blocks[0].startswith("## "):
        authors.append(blocks.pop(0))
    add_author_rows(body, authors, section_properties[1])
    author_names = [author.splitlines()[0] for author in authors]
    relations = E.fromstring(package["word/_rels/document.xml.rels"])
    for relationship in list(relations):
        if relationship.get("Type", "").endswith("/hyperlink"):
            relations.remove(relationship)  # Original sample citations were removed.
    mode, figure_counter = "body", 0
    for block in blocks:
        if block.startswith("## "):
            heading = block[3:]
            if heading in ("Abstract", "Keywords"):
                mode = heading.lower()
            else:
                mode = "references" if heading == "References" else "body"
                body.append(paragraph(heading, "Heading5" if heading in ("Acknowledgment", "References") else "Heading1"))
        elif block.startswith("### "):
            body.append(paragraph(block[4:], "Heading2"))
        elif block.startswith("{{TABLE_"):
            add_table(body, tables[block[2:-2]], source_table)
        elif block.startswith("{{FIGURE_"):
            figure_counter += 1
            add_figure(body, block[2:-2], package, relations, figure_counter, preview)
        elif mode in ("abstract", "keywords"):
            p = paragraph("", "Abstract" if mode == "abstract" else "Keywords")
            add_run(p, "Abstract—" if mode == "abstract" else "Keywords—", italic=True)
            add_run(p, block.replace("\n", " "))
            body.append(p)
            mode = "body"
        elif mode == "references":
            # Source references style already supplies bracketed numbering.
            body.append(paragraph(re.sub(r"^\[\d+\]\s*", "", block), "references"))
        else:
            p = paragraph(block.replace("\n", " "))
            if len(body) and body[-1].tag == q("tbl"):
                child(p.find(q("pPr")), "spacing", before="6pt", after="6pt")
            body.append(p)
    body.append(main_section)
    order_properties(document)
    package["word/document.xml"] = serialize(document)
    package["word/_rels/document.xml.rels"] = serialize(relations)
    content_types = E.fromstring(package["[Content_Types].xml"])
    if not any(n.get("Extension") == "png" for n in content_types):
        E.SubElement(content_types, f"{{{CT}}}Default", Extension="png", ContentType="image/png")
    package["[Content_Types].xml"] = serialize(content_types)
    footer = E.fromstring(package["word/footer1.xml"])
    for element in footer.findall(q("p")):
        footer.remove(element)
    footer.append(paragraph("", "Footer"))
    package["word/footer1.xml"] = serialize(footer)
    core = E.fromstring(package["docProps/core.xml"])
    core_values = {"{http://purl.org/dc/elements/1.1/}title": title,
                   "{http://purl.org/dc/elements/1.1/}creator": "; ".join(author_names),
                   "{http://schemas.openxmlformats.org/package/2006/metadata/core-properties}lastModifiedBy": "OpenAI Codex-assisted draft"}
    for tag, value in core_values.items():
        element = core.find(tag)
        if element is None:
            element = E.SubElement(core, tag)
        element.text = value
    package["docProps/core.xml"] = serialize(core)
    # Template statistics and the source publisher are not properties of this
    # manuscript. Word will calculate its own live document statistics.
    app_namespace = E.QName(E.fromstring(package["docProps/app.xml"])).namespace
    app_properties = E.Element(f"{{{app_namespace}}}Properties", nsmap={None: app_namespace})
    E.SubElement(app_properties, f"{{{app_namespace}}}Application").text = "OpenAI Codex-assisted OOXML authoring"
    package["docProps/app.xml"] = serialize(app_properties)
    destination = PAPER / ("build/layout_preview.docx" if preview else "Brain_Tumor_Detection_IEEE_Paper.docx")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(destination, "w", ZIP_DEFLATED) as archive:
        for name, payload in package.items():
            archive.writestr(name, payload)
    editable = {"word/document.xml", "word/_rels/document.xml.rels", "[Content_Types].xml", "word/footer1.xml", "docProps/core.xml", "docProps/app.xml"}
    with ZipFile(TEMPLATE) as original_zip:
        preserved = {name: hashlib.sha256(package[name]).hexdigest() for name in original_zip.namelist() if name not in editable}
        if any(original_zip.read(name) != package[name] for name in preserved):
            raise ValueError("A preserve-only template part changed.")
    evidence = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "preview": preview,
                "template_sha256": TEMPLATE_HASH, "output_sha256": sha(destination),
                "preserve_only_parts": preserved, "measured_source_files": sources,
                "section_count": len(document.xpath(".//w:sectPr", namespaces=NS)),
                "native_table_count": len(body.findall(q("tbl"))), "figure_count": figure_counter,
                "authors": author_names, "author_count": len(authors),
                "author_layout": "Two rows of two native Word columns",
                "author_placeholders_intentional": False,
                "manuscript_source_sha256": sha(PAPER / "manuscript.md")}
    (destination.with_suffix(".provenance.json")).write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    markdown = text
    for key, (caption, rows, _) in tables.items():
        table_lines = [caption, "", "| " + " | ".join(rows[0]) + " |",
                       "| " + " | ".join("---" for _ in rows[0]) + " |"]
        table_lines += ["| " + " | ".join(row) + " |" for row in rows[1:]]
        markdown = markdown.replace("{{" + key + "}}", "\n".join(table_lines))
    for key, name in (("FIGURE_ROC", "roc_comparison.png"), ("FIGURE_HISTORY", "validation_history.png")):
        markdown = markdown.replace("{{" + key + "}}", f"![Measured comparison](figures/{name})")
    destination.with_suffix(".md").write_text(markdown, encoding="utf-8")
    print(f"Created {'layout preview' if preview else 'final measured manuscript'}: {destination.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview", action="store_true", help="Internal layout check with explicit pending cells; never writes the final paper.")
    build(preview=parser.parse_args().preview)
