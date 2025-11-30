# -------------------------
#  testing_service.py
# -------------------------

import io
from uuid import uuid4
from pathlib import Path
from PyPDF2 import PdfWriter, PdfReader

# PDF template renderers (page 1 + page 2)
from app.pdf_templates.fill_template import (
    render_circuit_details_pdf,
    render_test_results_pdf
)

# Builds the structured test sheet data from user inputs
from app.services.testing_data_builder import build_test_sheet_data


def create_test_sheet_pdf(user_id: str, user_payload: dict) -> str:
    """
    Generates the final 2-page Test Sheet PDF.
    Called by whatsapp.py or any endpoint that requests a sheet.
    """

    # 1) Convert OCR/manual inputs into correct structure
    data = build_test_sheet_data(user_id, user_payload)

    # 2) Ensure output folder exists
    output_dir = Path("generated/test_sheets")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 3) Generate unique filename
    filename = f"test_sheet_{uuid4().hex}.pdf"
    output_path = output_dir / filename

    # 4) Render PAGE 1 (circuit details)
    page1_stream = render_circuit_details_pdf({
        "rows": data.get("circuit_details", [])
    })

    # 5) Render PAGE 2 (test results)
    page2_stream = render_test_results_pdf({
        "db_reference": data["test_results"].get("db_reference", ""),
        "top_values":  data["test_results"].get("top_values", {}),
        "confirmed":   data["test_results"].get("confirmed", {}),
        "instruments": data["test_results"].get("instruments", {}),
        "rows":        data["test_results"].get("rows", []),
        "bottom":      data["test_results"].get("bottom", {})
    })

    # 6) Combine pages into final PDF
    writer = PdfWriter()

    # PAGE 1
    p1 = PdfReader(page1_stream)
    writer.add_page(p1.pages[0])

    # PAGE 2
    p2 = PdfReader(page2_stream)
    writer.add_page(p2.pages[0])

    # 7) Save final output
    with open(output_path, "wb") as f:
        writer.write(f)

    return str(output_path)
