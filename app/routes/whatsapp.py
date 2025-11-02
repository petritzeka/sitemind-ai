from flask import Blueprint, current_app, request, abort
from twilio.twiml.messaging_response import MessagingResponse
from twilio.request_validator import RequestValidator
import requests

from app.services.ai_service import (
    utc_now_ts, append_message, fetch_history, check_and_count,
    retrieve_context, chat_reply, vision_answer, transcribe_and_answer
)

bp = Blueprint("whatsapp", __name__)

def _validate_signature():
    if not current_app.config.get("ENFORCE_TWILIO_SIGNATURE", False):
        return True
    validator = RequestValidator(current_app.config["TWILIO_AUTH_TOKEN"])
    sig = request.headers.get("X-Twilio-Signature", "")
    url = request.url
    data = request.form.to_dict()
    return validator.validate(url, data, sig)

def _onboarding():
    return (
        "Welcome to SiteMind AI ⚡️\n"
        "Your 24/7 AI Electrician Assistant — here to help you learn, plan, and work smarter.\n\n"
        "Ask anything or try:\n"
        "• study plan\n• quick quiz\n• explain a topic\n• past-paper mode\n• glossary\n• help\n\n"
        "🤖 Note: AI can make mistakes — always double-check critical details and work safely.\n"
        f"🎓 You’re on a {current_app.config['FREE_TRIAL_DAYS']}-day free trial.\n"
        f"🔒 Privacy: {current_app.config['PRIVACY_URL']}\n"
        f"📄 Terms: {current_app.config['TERMS_URL']}\n"
        f"💳 Subscribe: {current_app.config['SUBSCRIBE_URL']}"
    )

@bp.route("/whatsapp", methods=["POST"])
def whatsapp_webhook():
    if not _validate_signature():
        abort(403, "Invalid Twilio signature")

    body = (request.form.get("Body", "") or "").strip()
    from_number = request.form.get("From", "")
    user_id = from_number or "unknown"
    num_media = int(request.form.get("NumMedia", "0"))
    now_ts = utc_now_ts()

    allowed, block_msg = check_and_count(
        user_id=user_id,
        now_ts=now_ts,
        free_days=current_app.config["FREE_TRIAL_DAYS"],
        msg_cap=current_app.config["FREE_TRIAL_MESSAGE_CREDITS"],
        subscribe_url=current_app.config["SUBSCRIBE_URL"],
    )
    if not allowed:
        r = MessagingResponse(); r.message(block_msg)
        return str(r), 200

    if body.lower() in {"menu", "start", "hi", "hello", "help"}:
        msg = _onboarding()
        append_message(user_id, "user", body, now_ts)
        append_message(user_id, "assistant", msg, now_ts)
        r = MessagingResponse(); r.message(msg)
        return str(r), 200
    
    # Word cap check before sending to OpenAI
    if len(body.split()) > int(current_app.config["WORD_CAP"]):
    too_long = (
        f"⚠️ Your message is too long. Please shorten it "
        f"(max {current_app.config['WORD_CAP']} words per message)."
    )
    r = MessagingResponse(); r.message(too_long)
    return str(r), 200

    if num_media > 0:
        ct = request.form.get("MediaContentType0", "")
        url = request.form.get("MediaUrl0", "")
        try:
            if ct.startswith("image/"):
                reply = vision_answer(url, body or "Describe and help me solve this.")
            elif ct.startswith("audio/") or ct in {"application/ogg","audio/ogg","audio/webm"}:
                sid = current_app.config["TWILIO_ACCOUNT_SID"]
                tok = current_app.config["TWILIO_AUTH_TOKEN"]
                resp = requests.get(url, auth=(sid, tok), timeout=30)
                resp.raise_for_status()
                reply = transcribe_and_answer(resp.content)
            else:
                reply = f"Got your file ({ct}). I support images and voice notes right now."
        except Exception as e:
            reply = f"⚠️ I couldn't process that media: {e}"

        append_message(user_id, "user", body or "(media)", now_ts)
        append_message(user_id, "assistant", reply, now_ts)
        r = MessagingResponse(); r.message(reply)
        return str(r), 200

    history = fetch_history(user_id, limit=8)
    ctx = retrieve_context(current_app.config.get("VSTORE"), body, k=4)
    try:
        reply = chat_reply(body, history, ctx)
    except Exception as e:
        reply = f"⚠️ Sorry, I couldn’t process that just now: {e}"

    append_message(user_id, "user", body, now_ts)
    append_message(user_id, "assistant", reply, now_ts)
    r = MessagingResponse(); r.message(reply)
    return str(r), 200
