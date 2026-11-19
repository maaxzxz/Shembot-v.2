"""Generate the editable Milestone 2 progress report in DOCX format."""

from __future__ import annotations

import shutil
from pathlib import Path

from docx import Document
from docx.enum.section import WD_ORIENT
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Inches, Pt, RGBColor


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_OUTPUT = ROOT / "reports" / "PROGRESS_REPORT.docx"
STANDALONE_OUTPUT = ROOT.parents[1] / "Shembot_Milestone_2_Progress_Report.docx"

NAVY = "17324D"
BLUE = "2F75B5"
LIGHT_BLUE = "EAF3F8"
PALE_BLUE = "F4F8FB"
GREEN = "2E8B57"
LIGHT_GREEN = "EAF5EF"
AMBER = "C47F17"
LIGHT_AMBER = "FFF4DB"
GRAY = "5B6573"
LIGHT_GRAY = "EEF1F4"
WHITE = "FFFFFF"


def set_cell_shading(cell, fill: str) -> None:
    properties = cell._tc.get_or_add_tcPr()
    shading = properties.find(qn("w:shd"))
    if shading is None:
        shading = OxmlElement("w:shd")
        properties.append(shading)
    shading.set(qn("w:fill"), fill)


def set_cell_margins(cell, top=80, start=90, bottom=80, end=90) -> None:
    properties = cell._tc.get_or_add_tcPr()
    margins = properties.first_child_found_in("w:tcMar")
    if margins is None:
        margins = OxmlElement("w:tcMar")
        properties.append(margins)
    for name, value in (("top", top), ("start", start), ("bottom", bottom), ("end", end)):
        node = margins.find(qn(f"w:{name}"))
        if node is None:
            node = OxmlElement(f"w:{name}")
            margins.append(node)
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")


def set_cell_text(cell, text: str, *, bold=False, color=NAVY, size=8.5, align=None) -> None:
    cell.text = ""
    paragraph = cell.paragraphs[0]
    if align is not None:
        paragraph.alignment = align
    paragraph.paragraph_format.space_after = Pt(0)
    run = paragraph.add_run(text)
    run.bold = bold
    run.font.name = "Aptos"
    run.font.size = Pt(size)
    run.font.color.rgb = RGBColor.from_string(color)
    cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
    set_cell_margins(cell)


def set_repeat_table_header(row) -> None:
    properties = row._tr.get_or_add_trPr()
    repeat = OxmlElement("w:tblHeader")
    repeat.set(qn("w:val"), "true")
    properties.append(repeat)


def set_table_borders(table, color="D4DEE7", size="4") -> None:
    properties = table._tbl.tblPr
    borders = properties.first_child_found_in("w:tblBorders")
    if borders is None:
        borders = OxmlElement("w:tblBorders")
        properties.append(borders)
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        tag = borders.find(qn(f"w:{edge}"))
        if tag is None:
            tag = OxmlElement(f"w:{edge}")
            borders.append(tag)
        tag.set(qn("w:val"), "single")
        tag.set(qn("w:sz"), size)
        tag.set(qn("w:space"), "0")
        tag.set(qn("w:color"), color)


def add_page_number(paragraph) -> None:
    paragraph.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    run = paragraph.add_run("PAGE ")
    run.font.name = "Aptos"
    run.font.size = Pt(8)
    run.font.color.rgb = RGBColor.from_string(GRAY)
    begin = OxmlElement("w:fldChar")
    begin.set(qn("w:fldCharType"), "begin")
    instruction = OxmlElement("w:instrText")
    instruction.set(qn("xml:space"), "preserve")
    instruction.text = " PAGE "
    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")
    run._r.extend((begin, instruction, end))


def add_section_heading(document: Document, label: str, title: str) -> None:
    paragraph = document.add_paragraph()
    paragraph.paragraph_format.space_before = Pt(8)
    paragraph.paragraph_format.space_after = Pt(4)
    label_run = paragraph.add_run(label.upper() + "  ")
    label_run.bold = True
    label_run.font.name = "Aptos Display"
    label_run.font.size = Pt(9)
    label_run.font.color.rgb = RGBColor.from_string(BLUE)
    title_run = paragraph.add_run(title)
    title_run.bold = True
    title_run.font.name = "Aptos Display"
    title_run.font.size = Pt(15)
    title_run.font.color.rgb = RGBColor.from_string(NAVY)


def add_info_table(document: Document) -> None:
    table = document.add_table(rows=3, cols=2)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    table.columns[0].width = Inches(1.55)
    table.columns[1].width = Inches(8.6)
    labels = ("TEAM", "MEMBERS", "COURSE / SECTION")
    values = ("[ENTER TEAM NAME]", "[ENTER MEMBER NAMES]", "[ENTER COURSE AND SECTION]")
    for row, label, value in zip(table.rows, labels, values):
        set_cell_text(row.cells[0], label, bold=True, color=WHITE, size=8.5)
        set_cell_shading(row.cells[0], BLUE)
        set_cell_text(row.cells[1], value, color=GRAY, size=9)
        set_cell_shading(row.cells[1], PALE_BLUE)
    set_table_borders(table, color=WHITE, size="8")


def add_progress_summary(document: Document) -> None:
    document.add_paragraph().paragraph_format.space_after = Pt(1)
    table = document.add_table(rows=1, cols=3)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    widths = (2.05, 4.7, 4.15)
    for cell, width in zip(table.rows[0].cells, widths):
        cell.width = Inches(width)
    set_cell_text(table.cell(0, 0), "50%\nCOMPLETE", bold=True, color=WHITE, size=17, align=WD_ALIGN_PARAGRAPH.CENTER)
    set_cell_shading(table.cell(0, 0), BLUE)
    set_cell_text(
        table.cell(0, 1),
        "CHECKPOINT TARGET ACHIEVED\nCore browser inference, interface, dataset preparation, and first baselines are demonstrable.",
        bold=True,
        color=NAVY,
        size=9.5,
    )
    set_cell_shading(table.cell(0, 1), LIGHT_BLUE)
    set_cell_text(
        table.cell(0, 2),
        "NEXT FOCUS\nPhone-domain data, accuracy regression tests, retraining, device benchmarks, and locked final evaluation.",
        color=NAVY,
        size=9,
    )
    set_cell_shading(table.cell(0, 2), LIGHT_GREEN)
    set_table_borders(table, color=WHITE, size="10")


def add_progress_table(document: Document) -> None:
    headers = ("Full-project task", "Weight", "Owner", "Checkpoint status", "Earned", "Evidence")
    rows = (
        ("Requirements and interface design", "10%", "Team", "Complete", "10%", "Responsive dashboard and controls"),
        ("Core browser camera and inference", "20%", "Team", "Complete", "20%", "App source and vendored models"),
        ("Dataset acquisition, cleaning, and split", "15%", "Team", "Complete", "15%", "Data card, manifest, audit, preparation script"),
        ("Baseline training and evaluation", "10%", "Team", "Half complete", "5%", "Reproducible 30-image P/R baseline and custom metrics"),
        ("Phone-domain dataset and retraining", "15%", "Team", "Not started", "0%", "Planned next milestone"),
        ("Accuracy hardening and regression tests", "15%", "Team", "Planned next milestone", "0%", "Failure cases identified"),
        ("Representative-device performance", "5%", "Team", "Not started", "0%", "Planned Android/iOS benchmarks"),
        ("Final locked evaluation and report", "10%", "Team", "Not started", "0%", "Reserved for final milestone"),
        ("TOTAL", "100%", "", "", "50%", "Checkpoint target"),
    )
    table = document.add_table(rows=1, cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    widths = (2.45, 0.65, 0.8, 1.45, 0.65, 3.35)
    for cell, width, header in zip(table.rows[0].cells, widths, headers):
        cell.width = Inches(width)
        set_cell_text(cell, header, bold=True, color=WHITE, size=8, align=WD_ALIGN_PARAGRAPH.CENTER)
        set_cell_shading(cell, NAVY)
    set_repeat_table_header(table.rows[0])

    for index, values in enumerate(rows):
        row = table.add_row()
        is_total = index == len(rows) - 1
        for column, (cell, value) in enumerate(zip(row.cells, values)):
            center = column in (1, 2, 3, 4)
            set_cell_text(
                cell,
                value,
                bold=is_total or column == 0,
                color=WHITE if is_total else NAVY,
                size=8,
                align=WD_ALIGN_PARAGRAPH.CENTER if center else WD_ALIGN_PARAGRAPH.LEFT,
            )
            if is_total:
                set_cell_shading(cell, BLUE)
            elif index % 2:
                set_cell_shading(cell, PALE_BLUE)
            if column == 3 and not is_total:
                if values[3] == "Complete":
                    set_cell_shading(cell, LIGHT_GREEN)
                elif values[3] == "Half complete":
                    set_cell_shading(cell, LIGHT_AMBER)
    set_table_borders(table)


def add_blockers_table(document: Document) -> None:
    headers = ("Blocker", "Effect", "Planned response")
    rows = (
        ("Public CCTV data differs from phone-camera views", "Misses and false positives", "Collect consented phone frames and hard negatives"),
        ("Low light and very small people", "Reduced recall and unstable boxes", "Add representative samples and compare input sizes"),
        ("Mobile startup and compute limits", "Slow loading on some phones", "Measure loading, quantize, and benchmark devices"),
        ("Partial-body and broad group detections", "Duplicate or merged counts", "Add regression images and tune fusion on validation only"),
    )
    table = document.add_table(rows=1, cols=3)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    widths = (3.5, 2.65, 4.75)
    for cell, width, header in zip(table.rows[0].cells, widths, headers):
        cell.width = Inches(width)
        set_cell_text(cell, header, bold=True, color=WHITE, size=8.5)
        set_cell_shading(cell, NAVY)
    set_repeat_table_header(table.rows[0])
    for index, values in enumerate(rows):
        row = table.add_row()
        for cell, value in zip(row.cells, values):
            set_cell_text(cell, value, color=NAVY, size=8.5)
            if index % 2:
                set_cell_shading(cell, PALE_BLUE)
    set_table_borders(table)


def add_evidence_cards(document: Document) -> None:
    table = document.add_table(rows=1, cols=3)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    cards = (
        ("LIVE PROTOTYPE", "Browser camera, person boxes, count, FPS, capacity, detection area, and entry/exit tools."),
        ("DATA + MODEL", "1,350 images, 10,439 boxes, source-grouped split, custom EfficientDet baseline, and locked metrics."),
        ("REPRODUCIBILITY", "Local models/WASM, SHA-256 hashes, setup instructions, training scripts, and automated smoke test."),
    )
    for cell, (title, body) in zip(table.rows[0].cells, cards):
        cell.width = Inches(3.55)
        set_cell_text(cell, title + "\n" + body, bold=False, color=NAVY, size=8.8)
        cell.paragraphs[0].runs[0].bold = True
        set_cell_shading(cell, LIGHT_BLUE)
    set_table_borders(table, color=WHITE, size="10")


def create_document() -> Document:
    document = Document()
    section = document.sections[0]
    section.orientation = WD_ORIENT.LANDSCAPE
    section.page_width = Cm(29.7)
    section.page_height = Cm(21.0)
    section.top_margin = Cm(1.25)
    section.bottom_margin = Cm(1.25)
    section.left_margin = Cm(1.35)
    section.right_margin = Cm(1.35)
    section.header_distance = Cm(0.5)
    section.footer_distance = Cm(0.5)

    styles = document.styles
    normal = styles["Normal"]
    normal.font.name = "Aptos"
    normal.font.size = Pt(9)
    normal.font.color.rgb = RGBColor.from_string(NAVY)
    normal.paragraph_format.space_after = Pt(4)

    header = section.header.paragraphs[0]
    header.alignment = WD_ALIGN_PARAGRAPH.LEFT
    header_run = header.add_run("SHEMBOT  /  COMPUTER VISION PROJECT")
    header_run.bold = True
    header_run.font.name = "Aptos"
    header_run.font.size = Pt(8)
    header_run.font.color.rgb = RGBColor.from_string(BLUE)
    add_page_number(section.footer.paragraphs[0])

    eyebrow = document.add_paragraph()
    eyebrow.paragraph_format.space_after = Pt(1)
    run = eyebrow.add_run("MILESTONE 2  •  PHASE 2 OF 5")
    run.bold = True
    run.font.name = "Aptos"
    run.font.size = Pt(9)
    run.font.color.rgb = RGBColor.from_string(BLUE)

    title = document.add_paragraph()
    title.paragraph_format.space_after = Pt(1)
    title_run = title.add_run("50% Progress Check")
    title_run.bold = True
    title_run.font.name = "Aptos Display"
    title_run.font.size = Pt(27)
    title_run.font.color.rgb = RGBColor.from_string(NAVY)

    subtitle = document.add_paragraph()
    subtitle.paragraph_format.space_after = Pt(7)
    subtitle_run = subtitle.add_run("SHEMBOT v2 — on-device person detection and room monitoring")
    subtitle_run.font.name = "Aptos"
    subtitle_run.font.size = Pt(10.5)
    subtitle_run.font.color.rgb = RGBColor.from_string(GRAY)

    add_info_table(document)
    add_progress_summary(document)
    add_section_heading(document, "01", "Weighted project status")
    add_progress_table(document)

    note = document.add_paragraph()
    note.paragraph_format.space_before = Pt(4)
    note.paragraph_format.space_after = Pt(1)
    note_run = note.add_run("Note: ")
    note_run.bold = True
    note_run.font.color.rgb = RGBColor.from_string(BLUE)
    body_run = note.add_run("The 50% value measures delivery against the full project plan; it is not a model-accuracy percentage.")
    body_run.font.color.rgb = RGBColor.from_string(GRAY)

    document.add_page_break()
    add_section_heading(document, "02", "Current blockers and responses")
    add_blockers_table(document)
    add_section_heading(document, "03", "Checkpoint evidence")
    add_evidence_cards(document)
    add_section_heading(document, "04", "Required personalization before submission")

    warning = document.add_table(rows=1, cols=1)
    warning.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell = warning.cell(0, 0)
    set_cell_text(
        cell,
        "ACTION REQUIRED  —  Replace the bracketed team details and replace “Team” in the Owner column with the actual member responsible for each task. Every member should be ready to explain at least one file they own.",
        bold=True,
        color=NAVY,
        size=10,
    )
    set_cell_shading(cell, LIGHT_AMBER)
    set_table_borders(warning, color="E5B55A", size="8")

    add_section_heading(document, "05", "Next milestone priorities")
    priorities = (
        "Collect and label consented phone-camera frames, including empty scenes and hard negatives.",
        "Add regression images for partial-body, broad group-box, poster, furniture, and low-light failures.",
        "Retrain and tune thresholds only on validation data; preserve a locked phone-domain test set.",
        "Benchmark startup time, FPS, memory use, and model loading on representative phones.",
        "Complete final precision/recall, AP50, false-positive, count-error, and device-performance reporting.",
    )
    for item in priorities:
        paragraph = document.add_paragraph(style="List Bullet")
        paragraph.paragraph_format.space_after = Pt(3)
        paragraph.paragraph_format.left_indent = Inches(0.2)
        run = paragraph.add_run(item)
        run.font.name = "Aptos"
        run.font.size = Pt(9.5)
        run.font.color.rgb = RGBColor.from_string(NAVY)

    document.core_properties.title = "SHEMBOT v2 — Milestone 2 50% Progress Check"
    document.core_properties.subject = "Computer Vision Project Progress Report"
    document.core_properties.author = "SHEMBOT Team"
    document.core_properties.keywords = "SHEMBOT, computer vision, progress report, milestone 2"
    return document


def main() -> None:
    document = create_document()
    PACKAGE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    document.save(PACKAGE_OUTPUT)
    try:
        shutil.copy2(PACKAGE_OUTPUT, STANDALONE_OUTPUT)
    except PermissionError:
        # Word may have the standalone copy open. The package copy has already
        # been written and is the authoritative checkpoint artifact.
        print(f"Standalone copy is open and was not replaced: {STANDALONE_OUTPUT}")
    print(PACKAGE_OUTPUT)
    print(STANDALONE_OUTPUT)


if __name__ == "__main__":
    main()

