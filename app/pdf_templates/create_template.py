from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm

def create_blank_template():
    c = canvas.Canvas("blank_test_sheet.pdf", pagesize=A4)

    # Title
    c.setFont("Helvetica-Bold", 14)
    c.drawString(30*mm, 280*mm, "SiteMind AI – Electrical Test Sheet")

    # DB Reference section
    c.setFont("Helvetica", 10)
    c.drawString(30*mm, 270*mm, "DB Reference:")
    c.rect(60*mm, 268*mm, 60*mm, 8*mm)

    # Example circuit table header
    c.drawString(30*mm, 260*mm, "Circuit No")
    c.drawString(60*mm, 260*mm, "Description")
    c.drawString(120*mm, 260*mm, "Rating (A)")
    c.drawString(150*mm, 260*mm, "Zs (Ω)")

    # Draw 12 blank circuit rows
    y = 255*mm
    for i in range(12):
        c.rect(30*mm, y-2, 20*mm, 8*mm)   # number
        c.rect(60*mm, y-2, 50*mm, 8*mm)   # description
        c.rect(120*mm, y-2, 25*mm, 8*mm)  # rating
        c.rect(150*mm, y-2, 25*mm, 8*mm)  # Zs
        y -= 10*mm

    c.save()

create_blank_template()
