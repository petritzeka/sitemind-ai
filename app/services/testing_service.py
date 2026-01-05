# app/services/testing_service.py

from datetime import date
import uuid
import shutil
from pathlib import Path

from app.pdf_templates.fill_template import (
    generate_test_sheet_pdf,
    get_empty_test_sheet_pdf,
)
from app.services.monitoring import log_event, user_hash

PDF_BASE = Path(__file__).resolve().parent.parent / "pdf_templates"


class PdfGenerationError(Exception):
    """
    Raised when a requested PDF cannot be generated or prepared.
    """
    pass


def build_test_data(user_payload: dict) -> dict:
    """
    Build flat test_data dict for PDF generation.
    user_payload comes from WhatsApp state / OCR / manual answers.
    """

    flat = {}
    if isinstance(user_payload, dict):
        flat.update(user_payload)

        circuit_details = user_payload.get("circuit_details")
        if isinstance(circuit_details, list) and circuit_details:
            flat.update(circuit_details[0] or {})
        elif isinstance(circuit_details, dict):
            flat.update(circuit_details)

        test_results = user_payload.get("test_results")
        if isinstance(test_results, dict):
            flat.update(test_results)

    return {
        # Circuit details
        "circuit_number": flat.get("circuit_number"),
        "description": flat.get("description"),
        "type_of_wiring": flat.get("type_of_wiring"),
        "reference_method": flat.get("reference_method"),
        "points_served": flat.get("points_served"),
        "live_mm2": flat.get("live_mm2"),
        "cpc_mm2": flat.get("cpc_mm2"),
        "device_type": flat.get("device_type"),
        "rating_a": flat.get("rating_a"),
        "breaking_capacity": flat.get("breaking_capacity"),
        "max_zs": flat.get("max_zs"),
        "rcd_type": flat.get("rcd_type"),
        "rcd_trip_ma": flat.get("rcd_trip_ma"),

        # Dead tests
        "r1": flat.get("r1"),
        "r2": flat.get("r2"),
        "r1_r2": flat.get("r1_r2"),
        "test_voltage": flat.get("test_voltage"),
        "ir_ll": flat.get("ir_ll"),
        "ir_le": flat.get("ir_le"),
        "polarity_dead": flat.get("polarity_dead"),

        # Live tests
        "ze": flat.get("ze"),
        "zs": flat.get("zs"),
        "ipf": flat.get("ipf"),
        "pscc": flat.get("pscc"),

        # Footer
        "tested_by": flat.get("tested_by", "SiteMind AI"),
        "date": flat.get("date", date.today().strftime("%d/%m/%Y")),
    }


def create_test_sheet_pdf(user_id: str, user_payload: dict) -> str:
    """
    Generates the final Test Sheet PDF.
    Called by whatsapp.py when user says 'generate test sheet'.
    """
    try:
        # MVP: return blank sheet only (no autofill)
        return get_empty_test_sheet_pdf()
    except Exception as e:
        log_event("error", "PDF generation failed", {"err": str(e), "user": user_hash(user_id)})
        return get_empty_test_sheet_pdf()


def create_empty_test_sheet_pdf() -> str:
    """
    Return the blank test results PDF (no overlay applied).
    """
    return get_empty_test_sheet_pdf()


def create_empty_pdf(pdf_type: str = "test_results") -> str:
    """
    Return a blank PDF path based on type: 'test_results' or 'circuit_details'.
    """
    if pdf_type == "circuit_details":
        src = PDF_BASE / "circuit_details_template.pdf"
        if not src.exists():
            log_event("error", "Circuit details template missing", {"type": pdf_type, "src": str(src)})
            raise PdfGenerationError("circuit_details_template_missing")
        try:
            target = PDF_BASE / f"circuit_details_{uuid.uuid4().hex}.pdf"
            shutil.copyfile(src, target)
            return str(target)
        except Exception as e:
            log_event("error", "Failed to prepare circuit details PDF", {"err": str(e), "type": pdf_type})
            raise PdfGenerationError(str(e))

    # default test results
    try:
        return create_empty_test_sheet_pdf()
    except Exception as e:
        log_event("error", "Failed to prepare test results PDF", {"err": str(e), "type": pdf_type})
        raise PdfGenerationError(str(e))
