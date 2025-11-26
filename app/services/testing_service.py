# app/services/testing_service.py

import json
import os

"""
This service extracts values for the test sheet
from either:
1) OCR results saved in the user's session
2) Manual text the user typed
3) Default placeholders (so PDF never fails)

You MUST add OCR saving logic in your OCR step:
save_ocr_to_session(user_id, data)

And you need coordinates.json to match these keys.
"""


# -----------------------------------------------------------
# Load any saved OCR data for this user
# -----------------------------------------------------------
def load_ocr_session(user_id):
    path = f"storage/ocr/{user_id}.json"

    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)

    return {}  # no OCR data yet


# -----------------------------------------------------------
# Save OCR data (you call this from your OCR handler)
# -----------------------------------------------------------
def save_ocr_to_session(user_id, data):
    os.makedirs("storage/ocr", exist_ok=True)
    path = f"storage/ocr/{user_id}.json"

    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# -----------------------------------------------------------
# Default values to avoid missing fields breaking the PDF
# -----------------------------------------------------------
def default_values():
    d = {
        "db_reference": "DB1",
        "location": "Property",
        "inspector_name": "SiteMind AI",
        "date": "2025-01-01",
    }

    # create default empty circuit rows (1–12)
    for i in range(1, 12 + 1):
        prefix = f"circuit_{i}"
        d[f"{prefix}_description"] = f"Circuit {i}"
        d[f"{prefix}_rating"] = ""
        d[f"{prefix}_zs"] = ""
        d[f"{prefix}_cpc"] = ""
        d[f"{prefix}_type"] = ""
        d[f"{prefix}_method"] = ""
        d[f"{prefix}_breaker_type"] = ""
        d[f"{prefix}_breaker_en"] = ""
        d[f"{prefix}_points"] = ""

    return d


# -----------------------------------------------------------
# The MAIN function your webhook will call
# -----------------------------------------------------------
def extract_values_from_user_or_OCR(user_id):
    """
    Returns a FULL dictionary of values that the PDF generator
    will place onto the template.

    Priority:
      1) OCR fields (strongest)
      2) (Optional) latest manual overrides from user messages*
      3) Default values (fallback)

    * Manual overrides can be added later if you want.
    """

    all_data = default_values()

    # --- Load OCR data if available ---
    ocr_data = load_ocr_session(user_id)
    if ocr_data:
        for k, v in ocr_data.items():
            all_data[k] = v

    # (Optional: load manual overrides here later)

    return all_data
