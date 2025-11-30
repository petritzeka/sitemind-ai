# app/routes/files.py
from flask import Blueprint, send_file, abort
import os

files_bp = Blueprint("files", __name__)

@files_bp.route("/pdf/<filename>")
def serve_pdf(filename):
    # Correct path (NO extra "app")
    path = os.path.join(
        os.path.dirname(__file__), 
        "..", 
        "pdf_templates", 
        filename
    )
    path = os.path.abspath(path)

    if not os.path.exists(path):
        abort(404)

    return send_file(path, mimetype="application/pdf")
