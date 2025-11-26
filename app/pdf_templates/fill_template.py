# app/pdf_templates/fill_template.py

import json
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from pypdf import PdfReader, PdfWriter
import os


def generate_filled_pdf(data: dict):
    """
    Creates overlay PDF with text, merges with blank_test_sheet.pdf,
    and returns final PDF path.
    """

    # Load coordinates
    coord_path = "app/pdf_templates/coordinates.json"
    with open(coord_path, "r") as f:
        coords = json.load(f)

    # Overlay temp PDF
    overlay_path = "app/pdf_templates/_overlay.pdf"
    c = canvas.Canvas(overlay_path, pagesize=A4)

    # ------------------------------
    # SIMPLE DRAW FUNCTION
    # ------------------------------
    def draw(field_name, x, y):
        value = data.get(field_name, "")
        if value is None:
            value = ""
        c.drawString(x, y, str(value))

    # ------------------------------
    # TOP SECTION FIELDS
    # ------------------------------
    # direct coordinates
    for field in ["db_reference", "location", "supplied_from",
                  "ocpd_bs_en", "ocpd_type", "ocpd_rating",
                  "spd_installed", "spd_t1", "spd_t2", "spd_t3"]:
        if field in coords:
            draw(field, coords[field]["x"], coords[field]["y"])

    # ------------------------------
    # CIRCUIT ROWS (1–12)
    # ------------------------------
    fields = coords["circuit_fields"]

    for i in range(1, 13):
        row_y = coords["circuit_positions"][f"circuit_{i}"]["y"]

        draw(f"circuit_{i}_description", fields["description"]["x"], row_y)
        draw(f"circuit_{i}_wiring_type", fields["wiring_type"]["x"], row_y)
        draw(f"circuit_{i}_ref_method", fields["ref_method"]["x"], row_y)
        draw(f"circuit_{i}_points", fields["points"]["x"], row_y)
        draw(f"circuit_{i}_live", fields["live"]["x"], row_y)
        draw(f"circuit_{i}_cpc", fields["cpc"]["x"], row_y)
        draw(f"circuit_{i}_bsen", fields["bsen"]["x"], row_y)
        draw(f"circuit_{i}_type", fields["type"]["x"], row_y)
        draw(f"circuit_{i}_rating", fields["rating"]["x"], row_y)
        draw(f"circuit_{i}_zs_max", fields["zs_max"]["x"], row_y)

    c.save()

    # ------------------------------
    # MERGE WITH BASE TEMPLATE
    # ------------------------------
    template_path = "app/pdf_templates/blank_test_sheet.pdf"
    reader = PdfReader(template_path)
    overlay = PdfReader(overlay_path)
    writer = PdfWriter()

    base_page = reader.pages[0]
    base_page.merge_page(overlay.pages[0])
    writer.add_page(base_page)

    final_path = "app/pdf_templates/filled_test_sheet.pdf"
    with open(final_path, "wb") as f:
        writer.write(f)

    return final_path
