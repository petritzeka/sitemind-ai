import json
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from pypdf import PdfReader, PdfWriter

def generate_filled_pdf(data):
    # Load coords
    with open("app/pdf_templates/coordinates.json") as f:
        coords = json.load(f)

    # The overlay PDF
    overlay_path = "overlay.pdf"
    c = canvas.Canvas(overlay_path, pagesize=A4)

    # Loop through data
    for field, value in data.items():
        if field in coords:
            x = coords[field]["x"]
            y = coords[field]["y"]
            c.drawString(x, y, str(value))

    c.save()

    # Merge overlay onto template
    template = PdfReader("app/pdf_templates/blank_test_sheet.pdf")
    overlay = PdfReader(overlay_path)
    writer = PdfWriter()

    page = template.pages[0]
    page.merge_page(overlay.pages[0])
    writer.add_page(page)

    output_path = "filled_test_sheet.pdf"
    with open(output_path, "wb") as f:
        writer.write(f)

    return output_path
