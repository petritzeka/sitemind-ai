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

# Route path from env (defaults to /whatsapp)
WHATSAPP_WEBHOOK_PATH = os.getenv("WHATSAPP_WEBHOOK_PATH", "/whatsapp")
bp = Blueprint("whatsapp", __name__)

# Cache vectorstore after first build
_VS = None


def _validate_signature() -> bool:
    """Validate Twilio signature if enforcement is enabled."""
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
        "🤖 Note: AI can make mistakes — always double-check critical details and work safely.\n"
        f"🗓 Daily free limit: {os.getenv('FREE_TRIAL_DAILY_CAP', '5')} messages/day\n"
        f"🎓 You’re on a {current_app.config['FREE_TRIAL_DAYS']}-day free trial.\n"
        f"🔒 Privacy: {current_app.config['PRIVACY_URL']}\n"
        f"📄 Terms: {current_app.config['TERMS_URL']}\n"
        f"💳 Subscribe: {current_app.config['SUBSCRIBE_URL']}"
    )

@bp.post(WHATSAPP_WEBHOOK_PATH)
def whatsapp_webhook():
    # --- Security ---
    if not _validate_signature():
        abort(403, "Invalid Twilio signature")

    # --- Parse incoming ---
    body = (request.form.get("Body") or "").strip()
    from_number = request.form.get("From", "")
    user_id = from_number or "unknown"
    num_media = int(request.form.get("NumMedia", "0"))
    now_ts = utc_now_ts()

      # --- Free-trial / usage guard (applies to all kinds of messages) ---
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


    # === 1) MEDIA FIRST (handles image/voice so image-only messages don't trigger onboarding) ===
    if num_media > 0:
        ct = request.form.get("MediaContentType0", "")
        url = request.form.get("MediaUrl0", "")

        try:
            if ct.startswith("image/"):
                reply = vision_answer(url, body or "Describe and help me solve this.")

            elif ct.startswith("audio/") or ct in {"application/ogg", "audio/ogg", "audio/webm"}:
                sid = current_app.config["TWILIO_ACCOUNT_SID"]
                tok = current_app.config["TWILIO_AUTH_TOKEN"]
                resp = requests.get(url, auth=(sid, tok), timeout=30)
                resp.raise_for_status()
                reply = transcribe_and_answer(resp.content)

            else:
                reply = f"Got your file ({ct}). I currently support images and voice notes."

        except Exception as e:
            reply = f"⚠️ I couldn't process that media: {e}"

        append_message(user_id, "user", body or "(media)", now_ts)
        append_message(user_id, "assistant", reply, now_ts)

        r = MessagingResponse()
        r.message(reply)
        return str(r), 200

       # === 2) TEXT FLOW ===
    cap = int(current_app.config.get("WORD_CAP", 200))
    if body and len(body.split()) > cap:
        r = MessagingResponse()
        r.message(
            f"⚠️ Your message is too long ({len(body.split())} words). "
            f"Please keep it under {cap} words and resend 👍"
        )
        return str(r), 200

    # === 3) KEYWORD SHORTCUTS ===

    # === TEST SHEET GENERATOR ===
    if "generate test sheet" in lower or "test sheet" in lower:
        try:
            # 1. Extract data from OCR or ask GPT to infer missing fields
            from app.pdf_templates.fill_template import generate_filled_pdf
            from app.services.testing_service import extract_values_from_user_or_OCR

            data = extract_values_from_user_or_OCR(user_id)

            # 2. Generate PDF
            pdf_path = generate_filled_pdf(data)

            # 3. Send PDF back to user
            r = MessagingResponse()
            msg = r.message("Here is your test sheet ⚡️")
            msg.media(pdf_path)

            return str(r), 200

        except Exception as e:
            r = MessagingResponse()
            r.message(f"⚠️ I couldn't generate the test sheet: {e}")
            return str(r), 200

    lower = body.lower().strip()

    # PAY / SUBSCRIBE
    if any(word in lower for word in [
        "subscribe", "upgrade", "pay", "start plan", "join",
        "membership", "monthly", "yearly", "price"
    ]):
        monthly = current_app.config.get("SUBSCRIBE_MONTHLY_URL")
        annual = current_app.config.get("SUBSCRIBE_ANNUAL_URL")
        msg = (
            "Choose your plan:\n"
            f"• £8.99/month → {monthly}\n"
            f"• £79.99/year → {annual}"
        )
        r = MessagingResponse()
        r.message(msg)
        return str(r), 200

    # CANCEL / MANAGE SUBSCRIPTION
    if any(word in lower for word in [
        "cancel", "unsubscribe", "stop", "end", "manage",
        "billing", "account", "payment", "refund"
    ]):
        portal = current_app.config.get("STRIPE_PORTAL_URL")
        msg = (
            "Manage or cancel your subscription securely here:\n"
            f"{portal}"
        )
        r = MessagingResponse()
        r.message(msg)
        return str(r), 200

    # === 4) VECTORSTORE (RAG) ===
    global _VS
    if _VS is None:
        _VS = build_or_load_vectorstore()

    history = fetch_history(user_id, limit=10)
    ctx = retrieve_context(_VS, body, k=4)

    # === ONBOARDING FOR NEW USERS ===
    if not history and num_media == 0 and body:
        reply = _onboarding()
        append_message(user_id, "assistant", reply, now_ts)
        r = MessagingResponse()
        r.message(reply)
        return str(r), 200

    try:
        reply = chat_reply(body, history, ctx)
    except Exception as e:
        reply = f"⚠️ I had trouble generating a reply. Please try again. ({e})"

    append_message(user_id, "user", body, now_ts)
    append_message(user_id, "assistant", reply, utc_now_ts())

    r = MessagingResponse()
    r.message(reply)
    return str(r), 200
