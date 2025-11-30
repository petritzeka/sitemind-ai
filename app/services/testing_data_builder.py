import json

def build_test_sheet_data(user_id: str, user_payload: dict):
    """
    Converts WhatsApp user inputs into the exact structure needed
    by fill_template.py using coordinates.json.
    """

    return {
        "circuit_details": user_payload.get("circuit_details", []),
        "test_results": {
            "db_reference": user_payload.get("db_reference", ""),
            "top_values": user_payload.get("top_values", {}),
            "confirmed": user_payload.get("confirmed", {}),
            "instruments": user_payload.get("instruments", {}),
            "rows": user_payload.get("test_results", []),
            "bottom": user_payload.get("bottom", {})
        }
    }
