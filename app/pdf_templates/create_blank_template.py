from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.units import mm
import os


def centered_text(c, x, y, w, h, text, size=8, font="Helvetica"):
    """Center text inside a box."""
    c.setFont(font, size)
    c.drawCentredString(x + w / 2, y + h / 2 - 2, text)


def create_blank_template():
    output_dir = "app/pdf_templates"
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "blank_test_sheet.pdf")

    c = canvas.Canvas(path, pagesize=landscape(A4))
    width, height = landscape(A4)

    # ============================================================
    # PAGE 1 — SCHEDULE OF CIRCUIT DETAILS
    # ============================================================
    c.setFont("Helvetica-Bold", 14)
    c.drawString(15 * mm, height - 15 * mm, "Schedule of Circuit Details")

    c.setFont("Helvetica", 9)

    # Top header – DB, Location, Supplied From
    c.drawString(15 * mm, height - 28 * mm, "DB Reference:")
    c.rect(40 * mm, height - 30 * mm, 35 * mm, 7 * mm)

    c.drawString(85 * mm, height - 28 * mm, "Location:")
    c.rect(105 * mm, height - 30 * mm, 35 * mm, 7 * mm)

    c.drawString(150 * mm, height - 28 * mm, "Supplied From:")
    c.rect(178 * mm, height - 30 * mm, 35 * mm, 7 * mm)

    # OCPD block
    c.drawString(15 * mm, height - 40 * mm, "Distribution circuit OCPD:")
    c.drawString(70 * mm, height - 40 * mm, "BS(EN):")
    c.rect(90 * mm, height - 42 * mm, 20 * mm, 7 * mm)

    c.drawString(115 * mm, height - 40 * mm, "Type:")
    c.rect(130 * mm, height - 42 * mm, 20 * mm, 7 * mm)

    c.drawString(155 * mm, height - 40 * mm, "Rating:")
    c.rect(175 * mm, height - 42 * mm, 20 * mm, 7 * mm)

    # SPD block
    c.drawString(15 * mm, height - 52 * mm, "SPD Type(s):")
    spd_types = ["T1", "T2", "T3", "N/A"]
    x_spd = 45 * mm
    for t in spd_types:
        c.rect(x_spd, height - 54 * mm, 4 * mm, 4 * mm)
        c.drawString(x_spd + 6 * mm, height - 52 * mm, t)
        x_spd += 20 * mm

    # Wiring Codes panel (WC-2A – small)
    wiring_codes = [
        ("A", "Thermoplastic"),
        ("B", "Thermo SWA"),
        ("C", "Thermosetting"),
        ("D", "Thermo SWA"),
        ("F", "Mineral"),
        ("G", "MICC"),
        ("H", "Mineral SWA"),
        ("O", "Other"),
    ]

    panel_x = 15 * mm
    panel_y = height - 75 * mm
    c.setFont("Helvetica-Bold", 8)
    c.drawString(panel_x, panel_y, "Wiring Codes")
    c.setFont("Helvetica", 7)
    row_y = panel_y - 5 * mm
    for code, desc in wiring_codes:
        c.drawString(panel_x, row_y, f"{code} – {desc}")
        row_y -= 4.2 * mm

    # -------- MAIN TABLE (with Circuit No. column) --------
    # All widths chosen so total fits on page
    columns = [
        ("Circuit No.", 14 * mm),
        ("Description", 36 * mm),
        ("Protective Device", 32 * mm),
        ("Rating", 18 * mm),
        ("kA", 12 * mm),
        ("Cable Type", 28 * mm),
        ("CSA Line", 18 * mm),
        ("CSA CPC", 18 * mm),
        ("Install. Method", 28 * mm),
        ("Ib (A)", 16 * mm),
        ("Max Disconn.", 22 * mm),
    ]

    # Table starts to the right of wiring codes panel
    table_x = 45 * mm
    x_positions = [table_x]
    for _, w in columns:
        x_positions.append(x_positions[-1] + w)

    header_y = height - 82 * mm
    c.setFont("Helvetica-Bold", 8)

    for i, (text, w) in enumerate(columns):
        centered_text(c, x_positions[i], header_y, w, 6 * mm, text)

    row_h = 7 * mm           # tighter rows so we have space for everything
    start_y = header_y - 8 * mm

    for r in range(16):
        y = start_y - r * row_h
        for i, (_, w) in enumerate(columns):
            c.rect(x_positions[i], y, w, row_h)
            # cells left blank – students / AI fill later

    c.showPage()

    # ============================================================
    # PAGE 2 — SCHEDULE OF TEST RESULTS
    # ============================================================
    c.setFont("Helvetica-Bold", 14)
    c.drawString(15 * mm, height - 15 * mm, "Schedule of Test Results")

    c.setFont("Helvetica", 9)

    # DB reference
    c.drawString(15 * mm, height - 28 * mm, "DB Reference:")
    c.rect(40 * mm, height - 30 * mm, 35 * mm, 7 * mm)

    # Confirmed checkboxes
    c.drawString(90 * mm, height - 28 * mm, "Confirmed:")
    c.rect(125 * mm, height - 30 * mm, 4 * mm, 4 * mm)
    c.drawString(130 * mm, height - 28 * mm, "Correct Polarity")

    c.rect(165 * mm, height - 30 * mm, 4 * mm, 4 * mm)
    c.drawString(170 * mm, height - 28 * mm, "Phase Sequence")

    # SPD operational
    c.drawString(90 * mm, height - 38 * mm, "SPD Operational:")
    c.rect(135 * mm, height - 40 * mm, 4 * mm, 4 * mm)
    c.drawString(140 * mm, height - 38 * mm, "Yes")

    # Zdb / Ipf
    c.drawString(15 * mm, height - 45 * mm, "Zdb (Ω):")
    c.rect(35 * mm, height - 47 * mm, 20 * mm, 7 * mm)

    c.drawString(65 * mm, height - 45 * mm, "Ipf (kA):")
    c.rect(88 * mm, height - 47 * mm, 20 * mm, 7 * mm)

    # -------- TEST RESULTS TABLE (with Circuit No. column) --------
    test_cols = [
        ("Circuit No.", 14 * mm),
        ("R1+R2", 22 * mm),
        ("Rn", 22 * mm),
        ("IR L–N", 22 * mm),
        ("IR L–E", 22 * mm),
        ("IR N–E", 22 * mm),
        ("Polarity", 22 * mm),
        ("Zs", 22 * mm),
        ("PFC", 22 * mm),
        ("RCD (ms)", 22 * mm),
        ("RCD Type", 22 * mm),
        ("Functional", 26 * mm),
    ]

    table_x2 = 15 * mm
    xp2 = [table_x2]
    for _, w in test_cols:
        xp2.append(xp2[-1] + w)

    header_y2 = height - 65 * mm
    c.setFont("Helvetica-Bold", 8)
    for i, (text, w) in enumerate(test_cols):
        centered_text(c, xp2[i], header_y2, w, 6 * mm, text)

    row_h2 = 7 * mm
    start_y2 = header_y2 - 8 * mm

    for r in range(16):
        y = start_y2 - r * row_h2
        for i, (_, w) in enumerate(test_cols):
            c.rect(xp2[i], y, w, row_h2)

    # bottom of table
    bottom_y = start_y2 - 16 * row_h2

    # -------- Signature / Name / Date neatly UNDER table --------
    sig_top = bottom_y - 8 * mm

    c.setFont("Helvetica", 9)
    c.drawString(15 * mm, sig_top, "Tested by (CAPITALS):")
    c.rect(55 * mm, sig_top - 2 * mm, 60 * mm, 7 * mm)

    c.drawString(120 * mm, sig_top, "Signature:")
    c.rect(145 * mm, sig_top - 2 * mm, 60 * mm, 7 * mm)

    c.drawString(215 * mm, sig_top, "Date:")
    c.rect(235 * mm, sig_top - 2 * mm, 30 * mm, 7 * mm)

    # -------- Remarks box --------
    remarks_top = sig_top - 10 * mm
    c.drawString(15 * mm, remarks_top, "Remarks:")
    c.rect(15 * mm, remarks_top - 30 * mm, 260 * mm, 28 * mm)

    # -------- Test instruments box --------
    inst_top = remarks_top - 35 * mm
    c.drawString(15 * mm, inst_top, "Test Instruments Used (serial/asset numbers):")
    c.rect(15 * mm, inst_top - 12 * mm, 260 * mm, 10 * mm)

    c.save()
    print("Created:", path)


if __name__ == "__main__":
    create_blank_template()
