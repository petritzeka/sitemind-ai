# app/pdf_templates/fill_template.py

import json
import uuid
import shutil
from pathlib import Path
from pypdf import PdfReader, PdfWriter
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.colors import black

from app.services.monitoring import log_event

BASE_DIR = Path(__file__).parent

TEMPLATE_PDF = BASE_DIR / "test_results_template.pdf"
COORDS_FILE = BASE_DIR / "coordinates.json"


def get_empty_test_sheet_pdf() -> str:
    """
    Return the path to the blank test results template PDF.
    """
    try:
        target = BASE_DIR / f"blank_test_sheet_{uuid.uuid4().hex}.pdf"
        shutil.copyfile(TEMPLATE_PDF, target)
        return str(target)
    except Exception as e:
        log_event("error", "Failed to prepare blank test sheet", {"err": str(e)})
        return str(TEMPLATE_PDF)


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

    try:
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

        return str(output_path)
    except Exception as e:
        log_event("error", "PDF merge failed", {"err": str(e)})
        return get_empty_test_sheet_pdf()
    finally:
        overlay_path.unlink(missing_ok=True)
