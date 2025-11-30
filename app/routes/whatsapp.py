# app/routes/whatsapp.py
from flask import Blueprint, current_app, request, abort
from twilio.twiml.messaging_response import MessagingResponse
from twilio.request_validator import RequestValidator
import os, requests

from app.services.ai_service import (
    utc_now_ts, append_message, fetch_history, check_and_count,
    retrieve_context, chat_reply, vision_answer, transcribe_and_answer,
    build_or_load_vectorstore
)

WHATSAPP_WEBHOOK_PATH = os.getenv("WHATSAPP_WEBHOOK_PATH", "/whatsapp")
bp = Blueprint("whatsapp", __name__)

# Cache vectorstore
_VS = None


def _validate_signature() -> bool:
    if not current_app.config.get("ENFORCE_TWILIO_SIGNATURE", False):
        return True
    validator = RequestValidator(current_app.config["TWILIO_AUTH_TOKEN"])
    sig = request.headers.get("X-Twilio-Signature", "")
    url = request.url
    data = request.form.to_dict()
    return validator.validate(url, data, sig)


def _onboarding() -> str:
    return (
        "Welcome to SiteMind AI ⚡️\n"
        "Your 24/7 AI Electrician Assistant — here to help you learn, plan, and work smarter.\n\n"
        "Ask anything or try:\n"
        "• study plan\n• quick quiz\n• explain a topic\n• help\n\n"
        "🤖 Note: AI can make mistakes — always double-check critical details.\n"
        f"🗓 Daily free limit: {os.getenv('FREE_TRIAL_DAILY_CAP', '5')} messages/day\n"
        f"🎓 Free trial: {current_app.config['FREE_TRIAL_DAYS']} days\n"
        f"🔒 Privacy: {current_app.config['PRIVACY_URL']}\n"
        f"📄 Terms: {current_app.config['TERMS_URL']}\n"
        f"💳 Subscribe: {current_app.config['SUBSCRIBE_URL']}"
    )


@bp.post(WHATSAPP_WEBHOOK_PATH)
def whatsapp_webhook():
    # --- Security ---
    if not _validate_signature():
        abort(403, "Invalid Twilio signature")

    # --- Parse ---
    body = (request.form.get("Body") or "").strip()
    from_number = request.form.get("From", "")
    user_id = from_number or "unknown"
    num_media = int(request.form.get("NumMedia", "0"))
    now_ts = utc_now_ts()

    # --- Free trial guard ---
    allowed, block_msg = check_and_count(
        user_id=user_id,
        now_ts=now_ts,
        free_days=int(current_app.config["FREE_TRIAL_DAYS"]),
        msg_cap=int(current_app.config.get("FREE_TRIAL_MESSAGE_CREDITS", 50)),
        subscribe_url=current_app.config["SUBSCRIBE_URL"],
    )
    if not allowed:
        r = MessagingResponse()
        r.message(block_msg)
        return str(r), 200

    # === 1) MEDIA FIRST ===
    if num_media > 0:
        ct = request.form.get("MediaContentType0", "")
        url = request.form.get("MediaUrl0", "")

        try:
            if ct.startswith("image/"):
                reply = vision_answer(url, body or "Describe what you need help with.")

            elif ct.startswith("audio/") or ct in {"application/ogg", "audio/ogg", "audio/webm"}:
                sid = current_app.config["TWILIO_ACCOUNT_SID"]
                tok = current_app.config["TWILIO_AUTH_TOKEN"]
                resp = requests.get(url, auth=(sid, tok), timeout=30)
                resp.raise_for_status()
                reply = transcribe_and_answer(resp.content)

            else:
                reply = f"Got your file ({ct}). I support images & voice notes."

        except Exception as e:
            reply = f"⚠️ I couldn't process that media: {e}"

        append_message(user_id, "user", body or "(media)", now_ts)
        append_message(user_id, "assistant", reply, now_ts)

        r = MessagingResponse()
        r.message(reply)
        return str(r), 200

    # === 2) TEXT TOO LONG ===
    cap = int(current_app.config.get("WORD_CAP", 200))
    if body and len(body.split()) > cap:
        r = MessagingResponse()
        r.message(f"⚠️ Message too long ({len(body.split())} words). Limit is {cap}.")
        return str(r), 200

    # === 3) KEYWORD SHORTCUTS ===
    lower = body.lower().strip()

    # === TEST SHEET GENERATOR ===
    if "generate test sheet" in lower or "test sheet" in lower:
        try:
            from app.pdf_templates.fill_template import generate_filled_pdf
            from app.services.testing_service import extract_values_from_user_or_OCR

            data = extract_values_from_user_or_OCR(user_id)
            pdf_path = generate_filled_pdf(data)

            pdf_filename = pdf_path.split("/")[-1]

            r = MessagingResponse()
            msg = r.message("Here is your test sheet!")
            msg.media(f"{request.url_root}pdf/{pdf_filename}")

            return str(r), 200

        except Exception as e:
            r = MessagingResponse()
            r.message(f"⚠️ I couldn't generate the test sheet: {e}")
            return str(r), 200


    # PAY / SUBSCRIBE
    if any(word in lower for word in [
        "subscribe", "upgrade", "pay", "start plan", "join",
        "membership", "monthly", "yearly", "price"
    ]):
        monthly = current_app.config.get("SUBSCRIBE_MONTHLY_URL")
        annual = current_app.config.get("SUBSCRIBE_ANNUAL_URL")
        r = MessagingResponse()
        r.message(f"Choose a plan:\n£8.99/month → {monthly}\n£79.99/year → {annual}")
        return str(r), 200

        # CANCEL / MANAGE SUBSCRIPTION
    if any(word in lower for word in [
        "cancel", "unsubscribe", "stop", "end", "manage",
        "billing", "account", "payment", "refund", "subscription"
    ]):
        portal = current_app.config.get("STRIPE_PORTAL_URL")
        r = MessagingResponse()
        r.message(
            "Your SiteMind AI free trial has ended.\n\n"
            "Activate your subscription to continue:\n"
            f"{portal}\n\n"
            "Instant access to:\n"
            "• Test sheets PDF\n"
            "• Distribution board OCR\n"
            "• Level 2 & Level 3 Tutor Mode\n"
            "• Quotes & invoices\n"
            "• Photo analysis\n"
            "• More coming every week"
        )
        return str(r), 200


    # === 4) VECTORSTORE (RAG) ===
    global _VS
    if _VS is None:
        _VS = build_or_load_vectorstore()

    history = fetch_history(user_id, limit=10)
    ctx = retrieve_context(_VS, body, k=4)

    # === ONBOARDING ===
    if not history and num_media == 0 and body:
        reply = _onboarding()
        append_message(user_id, "assistant", reply, now_ts)
        r = MessagingResponse()
        r.message(reply)
        return str(r), 200

    # === GPT FALLBACK ===
    try:
        reply = chat_reply(body, history, ctx)
    except Exception as e:
        reply = f"⚠️ Error: {e}"

    append_message(user_id, "user", body, now_ts)
    append_message(user_id, "assistant", reply, utc_now_ts())

    r = MessagingResponse()
    r.message(reply)
    return str(r), 200
