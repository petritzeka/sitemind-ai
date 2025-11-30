import json
from app.services.testing_service import create_test_sheet_pdf

# ---------------------------------------------------------
#  MANUAL TEST DATA (FAKE VALUES FOR TESTING)
# ---------------------------------------------------------

fake_payload = {
    "circuit_details": [
        {
            "circuit_number": "1",
            "description": "Lighting",
            "type_of_wiring": "T&E",
            "reference_method": "A",
            "points_served": "4",
            "live_mm2": "1.0",
            "cpc_mm2": "1.0",
            "bs_en": "60898",
            "device_type": "B",
            "rating_a": "6",
            "breaking_capacity": "6000",
            "max_zs": "1.37",
            "bs_en_2": "",
            "rcd_type": "",
            "rcd_trip_ma": "",
            "rcd_rating": ""
        },
        {
            "circuit_number": "2",
            "description": "Sockets",
            "type_of_wiring": "T&E",
            "reference_method": "C",
            "points_served": "8",
            "live_mm2": "2.5",
            "cpc_mm2": "1.5",
            "bs_en": "60898",
            "device_type": "B",
            "rating_a": "32",
            "breaking_capacity": "6000",
            "max_zs": "1.44",
            "bs_en_2": "",
            "rcd_type": "AC",
            "rcd_trip_ma": "30",
            "rcd_rating": "30mA"
        }
    ],

    "test_results": {
        "db_reference": "DB1",

        "top_values": {
            "ze": "0.21",
            "zs": "0.45",
            "ipf": "1.2",
            "pscc": "2.0"
        },

        "confirmed": {
            "correct_polarity": True,
            "phase_sequence": True,
            "operational_status": True,
            "na": False
        },

        "instruments": {
            "multifunction": "Kewtech KT63",
            "continuity": "Kewtech",
            "insulation_res": "Fluke",
            "earth_fault_loop": "Megger",
            "rcd": "Kewtech",
            "earth_electrode": "Unit not used"
        },

        "rows": [
            {
                "circuit_number": "1",
                "r1": "0.45",
                "rn": "0.47",
                "r2": "0.76",
                "r1_r2": "1.21",
                "r2_single": "",
                "test_voltage": "500V",
                "ir_ll": ">299",
                "ir_le": ">299",
                "polarity": "✓",
                "max_measured": "0.41",
                "dis_time": "23ms",
                "test_button": "✓",
                "manual_test": "✓"
            },
            {
                "circuit_number": "2",
                "r1": "0.39",
                "rn": "0.40",
                "r2": "0.68",
                "r1_r2": "1.07",
                "r2_single": "",
                "test_voltage": "500V",
                "ir_ll": ">299",
                "ir_le": ">299",
                "polarity": "✓",
                "max_measured": "0.32",
                "dis_time": "18ms",
                "test_button": "✓",
                "manual_test": "✓"
            }
        ],

        "bottom": {
            "tested_by": "Petrit Zeka",
            "signature": "P.Z.",
            "date": "28/11/2025"
        }
    }
}

# ---------------------------------------------------------
#  RUN GENERATOR
# ---------------------------------------------------------

if __name__ == "__main__":
    print("Generating test PDF...")

    pdf_path = create_test_sheet_pdf(
        user_id="manual_test_user",
        user_payload=fake_payload
    )

    print(f"\n✅ PDF successfully generated at:\n{pdf_path}\n")
