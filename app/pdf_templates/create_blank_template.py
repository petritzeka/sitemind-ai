from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
import os

def create_blank_template():
    output_dir = "app/pdf_templates"
    os.makedirs(output_dir, exist_ok=True)

    path = os.path.join(output_dir, "blank_test_sheet.pdf")
    c = canvas.Canvas(path, pagesize=A4)

    # ---------------------------------------------------
    # HEADER
    # ---------------------------------------------------
    c.setFont("Helvetica-Bold", 16)
    c.drawString(20*mm, 280*mm, "SiteMind AI – Schedule of Circuit Details")

    # ---------------------------------------------------
    # DB INFORMATION SECTION
    # (Matches structure from reference sheet, not identical)
    # ---------------------------------------------------
    c.setFont("Helvetica", 10)

    # Row 1
    c.drawString(20*mm, 268*mm, "DB Reference:")
    c.rect(45*mm, 266*mm, 35*mm, 8*mm)

    c.drawString(90*mm, 268*mm, "Location:")
    c.rect(110*mm, 266*mm, 35*mm, 8*mm)

    c.drawString(155*mm, 268*mm, "Supplied From:")
    c.rect(180*mm, 266*mm, 25*mm, 8*mm)

    # Row 2 – OCPD
    c.drawString(20*mm, 255*mm, "Distribution Circuit OCPD – BS(EN):")
    c.rect(63*mm, 253*mm, 25*mm, 8*mm)

    c.drawString(95*mm, 255*mm, "Type:")
    c.rect(108*mm, 253*mm, 20*mm, 8*mm)

    c.drawString(135*mm, 255*mm, "Rating (A):")
    c.rect(155*mm, 253*mm, 20*mm, 8*mm)

    # SPD row
    c.drawString(20*mm, 242*mm, "SPD Installed:")
    c.rect(45*mm, 240*mm, 10*mm, 8*mm)

    c.drawString(60*mm, 242*mm, "Types: T1")
    c.rect(75*mm, 240*mm, 8*mm, 8*mm)
    c.drawString(88*mm, 242*mm, "T2")
    c.rect(100*mm, 240*mm, 8*mm, 8*mm)
    c.drawString(113*mm, 242*mm, "T3")
    c.rect(125*mm, 240*mm, 8*mm, 8*mm)

    # ---------------------------------------------------
    # TABLE HEADER
    # ---------------------------------------------------
    y = 225*mm
    c.setFont("Helvetica-Bold", 9)

    c.drawString(20*mm, y, "No.")
    c.drawString(30*mm, y, "Circuit Description")
    c.drawString(80*mm, y, "Wiring Type")
    c.drawString(105*mm, y, "Ref Method")
    c.drawString(130*mm, y, "Points")
    c.drawString(150*mm, y, "Live (mm²)")
    c.drawString(170*mm, y, "CPC (mm²)")
    c.drawString(190*mm, y, "BS(EN)")
    c.drawString(215*mm, y, "Type")
    c.drawString(235*mm, y, "Rating (A)")
    c.drawString(260*mm, y, "Zs Max (Ω)")

    c.line(20*mm, y-2*mm, 200*mm, y-2*mm)

    # ---------------------------------------------------
    # CIRCUIT ROWS (12 circuits)
    # ---------------------------------------------------
    c.setFont("Helvetica", 9)
    y -= 10*mm

    for i in range(1, 13):
        # number
        c.drawString(20*mm, y+2, str(i))

        # row boxes
        c.rect(30*mm, y, 45*mm, 8*mm)    # description
        c.rect(80*mm, y, 20*mm, 8*mm)    # wiring type
        c.rect(105*mm, y, 20*mm, 8*mm)   # method
        c.rect(130*mm, y, 15*mm, 8*mm)   # points
        c.rect(150*mm, y, 15*mm, 8*mm)   # live
        c.rect(170*mm, y, 15*mm, 8*mm)   # cpc
        c.rect(190*mm, y, 20*mm, 8*mm)   # BS EN
        c.rect(215*mm, y, 15*mm, 8*mm)   # type
        c.rect(235*mm, y, 20*mm, 8*mm)   # rating
        c.rect(260*mm, y, 20*mm, 8*mm)   # Zs max

        y -= 10*mm

    c.save()
    print("Created:", path)

if __name__ == "__main__":
    create_blank_template()
