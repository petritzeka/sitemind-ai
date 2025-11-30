# app/pdf_templates/fill_template.py

import json
from pathlib import Path
from PyPDF2 import PdfReader, PdfWriter
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.colors import black

# ------------------------------------------------------------------------
# LOAD COORDINATES
# ------------------------------------------------------------------------
COORD_FILE = Path("app/pdf_templates/coordinates.json")

with open(COORD_FILE, "r") as f:
    COORDS = json.load(f)

ROW_HEIGHT = COORDS["circuit_details"]["row_height"]
COLS = COORDS["circuit_details"]["columns"]

# ------------------------------------------------------------------------
# PDF TEMPLATE FILES
# ------------------------------------------------------------------------
TEMPLATE_PAGE1 = Path("app/pdf_templates/circuit_details_template.pdf")
TEMPLATE_PAGE2 = Path("app/pdf_templates/test_results_template.pdf")


# ------------------------------------------------------------------------
# HELPER: DRAW TEXT SAFELY
# ------------------------------------------------------------------------
def draw_text(c, x, y, text, max_len=22):
    """
    Draws text at a location.
    Trims to avoid text leaking out of boxes.
    """
    if text is None:
        text = ""

    text = str(text)
    if len(text) > max_len:
        text = text[:max_len]

    c.setFont("Helvetica", 8)
    c.setFillColor(black)
    c.drawString(x, y, text)


# ------------------------------------------------------------------------
# PAGE 1: CIRCUIT DETAILS (16 rows)
# ------------------------------------------------------------------------
def generate_page1(data, tmp_pdf_path):
    """
    Creates an overlay for Page 1 with filled circuit details.
    """

    c = canvas.Canvas(str(tmp_pdf_path), pagesize=letter)

    rows = data.get("circuit_details", [])

    for row_index, row in enumerate(rows):
        y_offset = -(ROW_HEIGHT * row_index)

        for key, coords in COLS.items():
            value = row.get(key, "")
            x = coords["x"]
            y = coords["y"] + y_offset
            draw_text(c, x, y, value)

    c.save()


# ------------------------------------------------------------------------
# PAGE 2: TEST RESULTS (16 rows)
# ------------------------------------------------------------------------
def generate_page2(data, tmp_pdf_path):
    """
    Overlay for page 2 using test_results.
    Uses SAME column positions? No — test results has its own structure.
    For now, we reuse circuit columns until you send test_results coords.
    """

    test_rows = data.get("test_results", [])

    # TEMPORARY: until you give me test results coordinates.json
    # We draw nothing, but create blank overlay.
    c = canvas.Canvas(str(tmp_pdf_path), pagesize=letter)

    # You will later provide page 2 coords and I will update this.
    c.save()


# ------------------------------------------------------------------------
# MERGE OVERLAY WITH TEMPLATE
# ------------------------------------------------------------------------
def merge_pdfs(base_pdf_path, overlay_pdf_path, output_writer):
    """
    Draws text overlay on top of template PDF.
    """
    base_pdf = PdfReader(str(base_pdf_path))
    overlay_pdf = PdfReader(str(overlay_pdf_path))

    base_page = base_pdf.pages[0]
    overlay_page = overlay_pdf.pages[0]

    base_page.merge_page(overlay_page)
    output_writer.add_page(base_page)


# ------------------------------------------------------------------------
# ENTRY POINT FOR TESTING_SERVICE.PY
# ------------------------------------------------------------------------
def generate_test_sheet_pdf(data, output_path):
    """
    Builds final 2-page test sheet.
    """

    output_writer = PdfWriter()

    tmp1 = Path("app/pdf_templates/tmp_page1.pdf")
    tmp2 = Path("app/pdf_templates/tmp_page2.pdf")

    # Page 1
    generate_page1(data, tmp1)
    merge_pdfs(TEMPLATE_PAGE1, tmp1, output_writer)

    # Page 2
    generate_page2(data, tmp2)
    merge_pdfs(TEMPLATE_PAGE2, tmp2, output_writer)

    # Save final
    with open(output_path, "wb") as f:
        output_writer.write(f)

    # cleanup
    tmp1.unlink(missing_ok=True)
    tmp2.unlink(missing_ok=True)

    return str(output_path)
