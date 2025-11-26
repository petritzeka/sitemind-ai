from flask import Blueprint, send_file, abort
import os

files_bp = Blueprint("files", __name__)

@files_bp.get("/pdf/<filename>")
def serve_pdf(filename):
    path = os.path.join("app/pdf_templates", filename)

    if not os.path.exists(path):
        abort(404)

    return send_file(path, mimetype="application/pdf")
