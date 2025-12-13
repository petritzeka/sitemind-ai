# app/routes/whatsapp.py

from flask import Blueprint, current_app, request, abort
from twilio.twiml.messaging_response import MessagingResponse
from twilio.request_validator import RequestValidator
import os
import requests

from app.services.ai_service import (
    utc_now_ts, append_message, fetch_history, check_and_count,
    retrieve_context, chat_reply, vision_answer, transcribe_and_answer,
    build_or_load_vectorstore
)

from app.services.testing_service import create_test_sheet_pdf

bp = Blueprint("whatsapp", __name__)
WHATSAPP_WEBHOOK_PATH = os.getenv("WHATSAPP_WEBHOOK_PATH", "/whatsapp")

# --------------------------------------------------
# IN-MEMORY TEST SHEET SESSIONS (MVP)
# --------------------------------------------------
TEST_SHEET_SESSIONS = {}

REQUIRED_CIRCUIT_FIELDS = [
    ("circuit_number", "What is the circuit number? (e.g. 1, 2)"),
    ("description", "What is the circuit description? (e.g. Sockets, Lighting)"),
    ("type_of_wiring", "What type of wiring is used? (e.g. T&E, SWA)"),
    ("reference_method", "What is the reference method? (e.g. A, B, C)"),
    ("points_served", "How many points are served by this circuit?"),
    ("live_mm2", "What is the live conductor size in mm²? (e.g. 2.5)"),
    ("cpc_mm2", "What is the CPC size in mm²? (e.g. 1.5)"),
    ("device_type", "What is the protective device type? (e.g. MCB, RCBO)"),
    ("rating_a", "What is the device rating in amps? (e.g. 32)"),
]

def get_next_missing_field(session: dict):
    for field, question in REQUIRED_CIRCUIT_FIELDS:
        if not session.get(field):
            return field, question
    return None, None


def validate_input(field: str, value: str) -> bool:
    numeric_fields = {"points_served", "live_mm2", "cpc_mm2", "rating_a"}
    if field in numeric_fields:
        try:
            float(value)
            return True
        except ValueError:
            return False
    return True


def _validate_signature():
    if not current_app.config.get("ENFORCE_TWILIO_SIGNATURE", False):
        return True
    validator = RequestValidator(current_app.config["TWILIO_AUTH_TOKEN"])
    return validator.validate(
        request.url,
        request.form.to_dict(),
        request.headers.get("X-Twilio-Signature", "")
    )


def onboarding_message():
    return (
        "Welcome to SiteMind AI ⚡️\n\n"
        "Your AI electrician assistant on WhatsApp.\n\n"
        "You can:\n"
        "• Ask electrical questions\n"
        "• Upload photos of distribution boards\n"
        "• Generate test sheets\n\n"
        "Type *generate test sheet* to begin."
    )


@bp.post(WHATSAPP_WEBHOOK_PATH)
def whatsapp_webhook():
    if not _validate_signature():
        abort(403)

    body = (request.form.get("Body") or "").strip()
    lower = body.lower()
    from_number = request.form.get("From", "")
    user_id = from_number or "unknown"
    now_ts = utc_now_ts()

    # -----------------------------------
    # FREE TRIAL / RATE LIMIT
    # -----------------------------------
    allowed, msg = check_and_count(
        user_id=user_id,
        now_ts=now_ts,
        free_days=int(current_app.config["FREE_TRIAL_DAYS"]),
        msg_cap=int(current_app.config.get("FREE_TRIAL_MESSAGE_CREDITS", 50)),
        subscribe_url=current_app.config["SUBSCRIBE_URL"],
    )

    if not allowed:
        r = MessagingResponse()
        r.message(msg)
        return str(r), 200

    # -----------------------------------
    # CANCEL / RESET TEST SHEET
    # -----------------------------------
    if lower in {"cancel", "restart", "stop test sheet"}:
        TEST_SHEET_SESSIONS.pop(user_id, None)
        r = MessagingResponse()
        r.message("❌ Test sheet cancelled. Type *generate test sheet* to start again.")
        return str(r), 200

    # -----------------------------------
    # CONTINUE ACTIVE TEST SHEET SESSION
    # -----------------------------------
    if user_id in TEST_SHEET_SESSIONS and body:
        session = TEST_SHEET_SESSIONS[user_id]
        field, question = get_next_missing_field(session)

        if field:
            if not validate_input(field, body):
                r = MessagingResponse()
                r.message("⚠️ Please enter a valid number.")
                return str(r), 200

            session[field] = body.strip()
            next_field, next_question = get_next_missing_field(session)

            r = MessagingResponse()
            if next_field:
                r.message(next_question)
            else:
                r.message("✅ All details captured. Generating test sheet…")
            return str(r), 200

    # -----------------------------------
    # START TEST SHEET FLOW
    # -----------------------------------
    if "generate test sheet" in lower or lower == "test sheet":
        TEST_SHEET_SESSIONS[user_id] = {}
        _, question = get_next_missing_field(TEST_SHEET_SESSIONS[user_id])
        r = MessagingResponse()
        r.message(question)
        return str(r), 200

    # -----------------------------------
    # FINAL GENERATION (ALL FIELDS READY)
    # -----------------------------------
    if user_id in TEST_SHEET_SESSIONS:
        session = TEST_SHEET_SESSIONS[user_id]
        field, _ = get_next_missing_field(session)

        if not field:
            payload = {
                "circuit_details": [session],
                "test_results": {}
            }

            pdf_path = create_test_sheet_pdf(user_id, payload)
            filename = pdf_path.split("/")[-1]

            r = MessagingResponse()
            msg = r.message("📄 Your test sheet is ready.")
            msg.media(f"{request.url_root}pdf/{filename}")

            TEST_SHEET_SESSIONS.pop(user_id, None)
            return str(r), 200

    # -----------------------------------
    # MEDIA HANDLING
    # -----------------------------------
    num_media = int(request.form.get("NumMedia", "0"))
    if num_media > 0:
        ct = request.form.get("MediaContentType0", "")
        url = request.form.get("MediaUrl0", "")

        try:
            if ct.startswith("image/"):
                reply = vision_answer(url, body or "Describe the issue.")

            elif ct.startswith("audio/"):
                sid = current_app.config["TWILIO_ACCOUNT_SID"]
                tok = current_app.config["TWILIO_AUTH_TOKEN"]
                resp = requests.get(url, auth=(sid, tok))
                reply = transcribe_and_answer(resp.content)
            else:
                reply = "Unsupported file type."

        except Exception as e:
            reply = f"⚠️ Error processing media: {e}"

        r = MessagingResponse()
        r.message(reply)
        return str(r), 200

    # -----------------------------------
    # NORMAL AI CHAT (RAG + GPT)
    # -----------------------------------
    global _VS
    if "_VS" not in globals() or _VS is None:
        _VS = build_or_load_vectorstore()

    history = fetch_history(user_id, limit=10)
    ctx = retrieve_context(_VS, body, k=4)

    if not history:
        reply = onboarding_message()
    else:
        reply = chat_reply(body, history, ctx)

    append_message(user_id, "user", body, now_ts)
    append_message(user_id, "assistant", reply, utc_now_ts())

    r = MessagingResponse()
    r.message(reply)
    return str(r), 200
