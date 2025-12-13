# app/services/testing_service.py

from uuid import uuid4
from pathlib import Path
from datetime import date

from app.pdf_templates.fill_template import generate_test_sheet_pdf


def build_test_data(user_payload: dict) -> dict:
    """
    Build flat test_data dict for PDF generation.
    user_payload comes from WhatsApp state / OCR / manual answers.
    """

    return {
        # Circuit details
        "circuit_number": user_payload.get("circuit_number"),
        "description": user_payload.get("description"),
        "type_of_wiring": user_payload.get("type_of_wiring"),
        "reference_method": user_payload.get("reference_method"),
        "points_served": user_payload.get("points_served"),
        "live_mm2": user_payload.get("live_mm2"),
        "cpc_mm2": user_payload.get("cpc_mm2"),
        "device_type": user_payload.get("device_type"),
        "rating_a": user_payload.get("rating_a"),
        "breaking_capacity": user_payload.get("breaking_capacity"),
        "max_zs": user_payload.get("max_zs"),
        "rcd_type": user_payload.get("rcd_type"),
        "rcd_trip_ma": user_payload.get("rcd_trip_ma"),

        # Dead tests
        "r1": user_payload.get("r1"),
        "r2": user_payload.get("r2"),
        "r1_r2": user_payload.get("r1_r2"),
        "test_voltage": user_payload.get("test_voltage"),
        "ir_ll": user_payload.get("ir_ll"),
        "ir_le": user_payload.get("ir_le"),
        "polarity_dead": user_payload.get("polarity_dead"),

        # Live tests
        "ze": user_payload.get("ze"),
        "zs": user_payload.get("zs"),
        "ipf": user_payload.get("ipf"),
        "pscc": user_payload.get("pscc"),

        # Footer
        "tested_by": user_payload.get("tested_by", "SiteMind AI"),
        "date": user_payload.get("date", date.today().strftime("%d/%m/%Y")),
    }


def create_test_sheet_pdf(user_id: str, user_payload: dict) -> str:
    """
    Generates the final Test Sheet PDF.
    Called by whatsapp.py when user says 'generate test sheet'.
    """

    # 1) Build flat test_data
    test_data = build_test_data(user_payload)

    # 2) Ensure output folder exists
    output_dir = Path("generated/test_sheets")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 3) Generate PDF
    filename = f"test_sheet_{uuid4().hex}.pdf"
    output_path = output_dir / filename

    pdf_path = generate_test_sheet_pdf(test_data)

    return pdf_path
