# app/pdf_templates/fill_template.py

import json
import uuid
from pathlib import Path
from PyPDF2 import PdfReader, PdfWriter
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.colors import black

BASE_DIR = Path(__file__).parent

TEMPLATE_PDF = BASE_DIR / "test_results_template.pdf"
COORDS_FILE = BASE_DIR / "coordinates.json"


def generate_test_sheet_pdf(test_data: dict) -> str:
    """
    Generate a single-page electrical test sheet PDF
    using flat field -> coordinate mapping.
    """

    # Load coordinates
    with open(COORDS_FILE, "r") as f:
        coords = json.load(f)

    overlay_path = BASE_DIR / "overlay.pdf"
    output_path = BASE_DIR / f"test_sheet_{uuid.uuid4().hex}.pdf"

    # Create overlay
    c = canvas.Canvas(str(overlay_path), pagesize=A4)
    c.setFont("Helvetica", 8)
    c.setFillColor(black)

    for field, value in test_data.items():
        if field in coords and value not in (None, ""):
            x, y = coords[field]
            c.drawString(x, y, str(value))

    c.save()

    # Merge overlay onto template
    template_reader = PdfReader(str(TEMPLATE_PDF))
    overlay_reader = PdfReader(str(overlay_path))

    writer = PdfWriter()
    page = template_reader.pages[0]
    page.merge_page(overlay_reader.pages[0])
    writer.add_page(page)

    with open(output_path, "wb") as f:
        writer.write(f)

    # Cleanup
    overlay_path.unlink(missing_ok=True)

    return str(output_path)
