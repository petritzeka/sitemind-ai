# app/routes/whatsapp.py

from flask import Blueprint, current_app, request, abort
from twilio.twiml.messaging_response import MessagingResponse
from twilio.request_validator import RequestValidator
import os
from datetime import datetime
from zoneinfo import ZoneInfo

from app.services.ai_service import (
    utc_now_ts,
    append_message,
    fetch_history,
    check_and_count,
    check_image_caps,
    check_trial_gate,
    get_trial_info,
    retrieve_context,
    chat_reply,
    build_or_load_vectorstore,
    vision_answer,
)

from app.services.testing_service import (
    create_empty_test_sheet_pdf,
    create_empty_pdf,
    create_test_sheet_pdf,
)
from app.services import db_utils as db

bp = Blueprint("whatsapp", __name__)
WHATSAPP_WEBHOOK_PATH = os.getenv("WHATSAPP_WEBHOOK_PATH", "/whatsapp")

# -------------------------------------------------------------------
# In-memory test sheet sessions (MVP only – no DB)
# -------------------------------------------------------------------
TEST_SHEET_SESSIONS = {}
PENDING_IMAGE_TASKS = {}

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
MAX_IMAGES_PER_MESSAGE = 5


def word_count(text: str) -> int:
    return len((text or "").split())


def night_mode_active() -> bool:
    if not current_app.config.get("ENABLE_NIGHT_MODE", True):
        return False
    now_uk = datetime.now(ZoneInfo("Europe/London"))
    return 2 <= now_uk.hour < 6


def classify_image_intent(text: str) -> str:
    lower = (text or "").lower()
    study_terms = ("quiz", "socrative", "exam", "question", "study", "test", "paper")
    heavy_terms = ("board", "consumer", "distribution", "certificate", "install", "db ", "panel")

    if any(term in lower for term in study_terms):
        return "STUDY_IMAGE"
    if any(term in lower for term in heavy_terms):
        return "HEAVY_IMAGE"
    return "UNKNOWN"


def map_intent_response(text: str) -> str:
    lower = (text or "").lower()
    if any(k in lower for k in ("study", "quiz", "socrative", "exam")):
        return "STUDY_IMAGE"
    if any(k in lower for k in ("board", "install", "installation", "certificate", "panel")):
        return "HEAVY_IMAGE"
    return "UNKNOWN"


def is_trial_query(text: str) -> bool:
    lower = (text or "").lower()
    terms = (
        "trial",
        "free access",
        "subscription",
        "subscribe",
        "upgrade",
        "pro",
        "payment",
        "pricing",
    )
    return any(t in lower for t in terms)


def map_pdf_choice(text: str) -> str:
    lower = (text or "").strip().lower()
    if lower in {"1", "test results", "results", "results sheet", "test results sheet"}:
        return "1"
    if lower in {"2", "circuit details", "details", "circuit details sheet"}:
        return "2"
    return ""


def detect_pdf_intent(text: str) -> bool:
    lower = (text or "").lower()
    tutor_terms = (
        "explain",
        "how",
        "fill in",
        "fill out",
        "guide me",
        "what goes on",
        "what is on",
    )
    if any(t in lower for t in tutor_terms):
        return False

    sheet_terms = (
        "test sheet",
        "test results sheet",
        "results sheet",
        "circuit details sheet",
        "circuit details",
        "schedule of test results",
        "results schedule",
    )
    has_sheet = any(term in lower for term in sheet_terms)

    verb_terms = (
        "generate",
        "make",
        "send",
        "create",
        "produce",
        "provide",
        "give me",
        "need",
        "want",
    )
    has_verb = any(term in lower for term in verb_terms)

    has_pdf = "pdf" in lower

    return has_sheet and (has_pdf or has_verb)


def set_pending_action(user_id: str, action: str, expires_ts: int) -> None:
    try:
        db.execute(
            "UPDATE users SET pending_action=?, pending_action_ts=? WHERE user_id=?",
            (action, expires_ts, user_id),
        )
    except Exception:
        pass


def get_pending_action(user_id: str, now_ts: int) -> str:
    try:
        row = db.fetchone("SELECT pending_action, pending_action_ts FROM users WHERE user_id=?", (user_id,))
    except Exception:
        return ""

    if not row:
        return ""

    action = row["pending_action"] if isinstance(row, dict) else row[0]
    exp_ts = row["pending_action_ts"] if isinstance(row, dict) else (row[1] if len(row) > 1 else None)

    if not action:
        return ""

    try:
        if exp_ts and int(exp_ts) < now_ts:
            db.execute("UPDATE users SET pending_action=NULL, pending_action_ts=NULL WHERE user_id=?", (user_id,))
            return ""
    except Exception:
        return action

    return action


def clear_pending_action(user_id: str) -> None:
    try:
        db.execute("UPDATE users SET pending_action=NULL, pending_action_ts=NULL WHERE user_id=?", (user_id,))
    except Exception:
        pass

def get_next_missing_field(session: dict):
    for field, question in REQUIRED_CIRCUIT_FIELDS:
        if not session.get(field):
            return field, question
    return None, None


def build_paywall_message(action_type: str, reason: str, pay_url: str) -> str:
    lower = (reason or "").lower()
    if action_type == "pdf":
        if "trial" in lower and "ended" in lower:
            return (
                "Test sheet generation is a Pro feature.\n"
                "Your free trial has ended, so I can’t generate a test sheet right now.\n"
                f"Activate your subscription to continue: {pay_url}"
            )
        if "limit" in lower or "cap" in lower:
            return (
                "You’ve reached the current PDF limit.\n"
                f"Try again tomorrow or upgrade to Pro: {pay_url}"
            )
    return (
        "This action needs a subscription.\n"
        f"Activate to continue: {pay_url}"
    )


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
        "• Study for Level 2 / Level 3\n"
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
    word_cap = int(current_app.config.get("WORD_CAP", 200))
    enable_pdf = current_app.config.get("ENABLE_PDF", True)
    enable_heavy_ocr = current_app.config.get("ENABLE_HEAVY_OCR", True)
    enable_rag = current_app.config.get("ENABLE_RAG", True)
    num_media = int(request.form.get("NumMedia", "0"))
    pdf_intent = detect_pdf_intent(lower)
    pending_action = get_pending_action(user_id, now_ts)

    # Guard stray 1/2 replies when no pending action
    if pending_action == "" and lower in {"1", "2"} and num_media == 0:
        r = MessagingResponse()
        r.message("Reply 1 or 2 only after I ask. If you need a test sheet PDF, type 'send test sheet'.")
        return str(r), 200

    trial_ok, trial_msg = check_trial_gate(
        user_id=user_id,
        now_ts=now_ts,
        free_days=int(current_app.config["FREE_TRIAL_DAYS"]),
    )
    if not trial_ok:
        r = MessagingResponse()
        if pdf_intent:
            paywall = build_paywall_message("pdf", trial_msg, current_app.config["SUBSCRIBE_URL"])
            r.message(paywall)
        else:
            r.message(trial_msg)
        return str(r), 200

    if word_cap and word_count(body) > word_cap:
        r = MessagingResponse()
        r.message(f"⚠️ Please keep messages under {word_cap} words.")
        return str(r), 200

    # ---------------------------------------------------------------
    # Pending PDF type selection (no quota impact)
    # ---------------------------------------------------------------
    if pending_action == "pdf_choice" and num_media == 0:
        choice = map_pdf_choice(body)
        r = MessagingResponse()
        if choice not in {"1", "2"}:
            r.message("Please reply with 1 (Test Results Sheet) or 2 (Circuit Details Sheet).")
            return str(r), 200

        if night_mode_active():
            clear_pending_action(user_id)
            r.message("🌙 Night mode (02:00–06:00 UK): PDF generation is paused. Please ask again after 06:00 UK.")
            return str(r), 200

        pdf_type = "test_results" if choice == "1" else "circuit_details"
        try:
            pdf_path = create_empty_pdf(pdf_type)
            if not pdf_path:
                raise Exception("empty pdf path")
        except Exception:
            r.message("I’m having trouble generating that PDF right now — please try again shortly.")
            return str(r), 200

        filename = pdf_path.split("/")[-1]
        label = "Test Results Sheet" if choice == "1" else "Circuit Details Sheet"
        msg = r.message(f"📄 Here is your {label}.")
        msg.media(f"{request.url_root}pdf/{filename}")
        clear_pending_action(user_id)
        return str(r), 200

    if is_trial_query(body):
        info = get_trial_info(
            user_id=user_id,
            now_ts=now_ts,
            free_days=int(current_app.config["FREE_TRIAL_DAYS"]),
        )
        r = MessagingResponse()
        if not info:
            r.message("Sorry — I’m temporarily offline. Please try again in a moment.")
            return str(r), 200
        if info["is_subscribed"]:
            r.message("You’re on SiteMind Pro. Your subscription keeps all features unlocked.")
            return str(r), 200
        if info["expired"]:
            r.message("Your 14-day trial has ended. Upgrade to SiteMind Pro to continue.")
            return str(r), 200
        days_left = info.get("days_left", 0)
        r.message(
            f"You’re on the 14-day free trial with full Pro features. "
            f"Days remaining: {days_left}. Upgrade anytime to keep access: {current_app.config['SUBSCRIBE_URL']}"
        )
        return str(r), 200

    # ---------------------------------------------------------------
    # Direct PDF intent ("test sheet" + "pdf") — ask for type first
    # ---------------------------------------------------------------
    if detect_pdf_intent(lower):
        r = MessagingResponse()
        if not enable_pdf:
            r.message("Feature temporarily unavailable.")
            return str(r), 200
        if night_mode_active():
            r.message("🌙 Night mode (02:00–06:00 UK): PDF generation is paused. Please try again after 06:00 UK.")
            return str(r), 200

        set_pending_action(user_id, "pdf_choice", now_ts + 300)
        r.message("Which PDF do you need?\n1) Test Results Sheet\n2) Circuit Details Sheet\nReply with 1 or 2.")
        return str(r), 200

    # ---------------------------------------------------------------
    # Resume pending image intent
    # ---------------------------------------------------------------
    if user_id in PENDING_IMAGE_TASKS and num_media == 0:
        pending = PENDING_IMAGE_TASKS.get(user_id, {})
        intent = map_intent_response(lower)
        if intent == "UNKNOWN":
            r = MessagingResponse()
            r.message("Please reply 'study screenshot' or 'installation/board photo' so I can process your image.")
            return str(r), 200

        media_url = pending.get("media_url")
        prompt = pending.get("prompt") or "Read this photo and extract electrical details."
        images = pending.get("num_media", 1)
        PENDING_IMAGE_TASKS.pop(user_id, None)

        if intent == "HEAVY_IMAGE" and night_mode_active():
            r = MessagingResponse()
            r.message("🌙 Night mode (02:00–06:00 UK): heavy photo analysis is paused. Please send study questions instead.")
            return str(r), 200

        if intent == "HEAVY_IMAGE" and not enable_heavy_ocr:
            r = MessagingResponse()
            r.message("Feature temporarily unavailable.")
            return str(r), 200

        allowed_img, cap_msg = check_image_caps(
            user_id=user_id,
            now_ts=now_ts,
            intent=intent,
            images=images,
        )
        if not allowed_img:
            r = MessagingResponse()
            r.message(cap_msg)
            return str(r), 200

        if intent == "STUDY_IMAGE":
            prompt = "This is a study or quiz screenshot. Extract the key question and give the correct answer."

        analysis = vision_answer(
            image_url=media_url,
            prompt=prompt,
        )

        append_message(user_id, "user", f"[image-followup] {body}", now_ts)
        append_message(user_id, "assistant", analysis, utc_now_ts())

        r = MessagingResponse()
        r.message(analysis)
        return str(r), 200

    # ---------------------------------------------------------------
    # Media handling (vision)
    # ---------------------------------------------------------------
    if num_media > 0:
        if num_media > MAX_IMAGES_PER_MESSAGE:
            r = MessagingResponse()
            r.message(f"⚠️ Please send up to {MAX_IMAGES_PER_MESSAGE} images at a time.")
            return str(r), 200

        media_url = request.form.get("MediaUrl0")

        if not media_url:
            r = MessagingResponse()
            r.message("⚠️ I didn't receive the image URL. Please resend the photo.")
            return str(r), 200

        intent = classify_image_intent(body)
        if intent == "UNKNOWN":
            PENDING_IMAGE_TASKS[user_id] = {
                "media_url": media_url,
                "prompt": body,
                "num_media": num_media,
            }
            r = MessagingResponse()
            r.message("Is this a study screenshot or an installation/board photo?")
            return str(r), 200

        if night_mode_active() and intent == "HEAVY_IMAGE":
            r = MessagingResponse()
            r.message("🌙 Night mode (02:00–06:00 UK): heavy photo analysis is paused. Please send study questions instead.")
            return str(r), 200

        if intent == "HEAVY_IMAGE" and not enable_heavy_ocr:
            r = MessagingResponse()
            r.message("Feature temporarily unavailable.")
            return str(r), 200

        allowed_img, cap_msg = check_image_caps(
            user_id=user_id,
            now_ts=now_ts,
            intent=intent,
            images=num_media,
        )
        if not allowed_img:
            r = MessagingResponse()
            r.message(cap_msg)
            return str(r), 200

        allowed, msg = check_and_count(
            user_id=user_id,
            now_ts=now_ts,
            free_days=int(current_app.config["FREE_TRIAL_DAYS"]),
            msg_cap=int(current_app.config.get("FREE_TRIAL_MESSAGE_CREDITS", 50)),
            subscribe_url=current_app.config["SUBSCRIBE_URL"],
        )
        if not allowed:
            r = MessagingResponse()
            r.message(build_paywall_message("pdf", msg, current_app.config["SUBSCRIBE_URL"]))
            return str(r), 200

        prompt = body or "Read this photo and extract electrical details."
        if intent == "STUDY_IMAGE":
            prompt = "This is a study or quiz screenshot. Extract the key question and give the correct answer."
        analysis = vision_answer(
            image_url=media_url,
            prompt=prompt,
        )

        append_message(user_id, "user", f"[image] {body}", now_ts)
        append_message(user_id, "assistant", analysis, utc_now_ts())

        r = MessagingResponse()
        r.message(analysis)
        return str(r), 200

    # ---------------------------------------------------------------
    # Serve a blank test sheet on demand
    # ---------------------------------------------------------------
    if any(
        phrase in lower
        for phrase in (
            "blank test sheet",
            "empty test sheet",
            "empty result sheet",
            "test sheet template",
        )
    ):
        if not enable_pdf:
            r = MessagingResponse()
            r.message("Feature temporarily unavailable.")
            return str(r), 200
        if night_mode_active():
            r = MessagingResponse()
            r.message("🌙 Night mode (02:00–06:00 UK): PDF generation is paused. Please try again after 06:00 UK.")
            return str(r), 200

        allowed, msg = check_and_count(
            user_id=user_id,
            now_ts=now_ts,
            free_days=int(current_app.config["FREE_TRIAL_DAYS"]),
            msg_cap=int(current_app.config.get("FREE_TRIAL_MESSAGE_CREDITS", 50)),
            subscribe_url=current_app.config["SUBSCRIBE_URL"],
        )
        if not allowed:
            r = MessagingResponse()
            r.message(build_paywall_message("pdf", msg, current_app.config["SUBSCRIBE_URL"]))
            return str(r), 200

        pdf_path = create_empty_test_sheet_pdf()
        filename = pdf_path.split("/")[-1]

        r = MessagingResponse()
        msg = r.message("📄 Here is a blank test sheet PDF.")
        msg.media(f"{request.url_root}pdf/{filename}")
        return str(r), 200

    # ---------------------------------------------------------------
    # Cancel / restart test sheet
    # ---------------------------------------------------------------
    if lower in {"cancel", "restart", "stop test sheet"}:
        TEST_SHEET_SESSIONS.pop(user_id, None)
        r = MessagingResponse()
        r.message("❌ Test sheet cancelled. Type *generate test sheet* to start again.")
        return str(r), 200

    # ---------------------------------------------------------------
    # Continue active test sheet session
    # ---------------------------------------------------------------
    if user_id in TEST_SHEET_SESSIONS and body:
        session = TEST_SHEET_SESSIONS[user_id]
        field, _ = get_next_missing_field(session)

        if field:
            if not enable_pdf:
                r = MessagingResponse()
                r.message("Feature temporarily unavailable.")
                return str(r), 200
            if night_mode_active():
                r = MessagingResponse()
                r.message("🌙 Night mode (02:00–06:00 UK): PDF generation is paused. Please try again after 06:00 UK.")
                return str(r), 200

            allowed, msg = check_and_count(
                user_id=user_id,
                now_ts=now_ts,
                free_days=int(current_app.config["FREE_TRIAL_DAYS"]),
                msg_cap=int(current_app.config.get("FREE_TRIAL_MESSAGE_CREDITS", 50)),
                subscribe_url=current_app.config["SUBSCRIBE_URL"],
        )
        if not allowed:
            r = MessagingResponse()
            r.message(build_paywall_message("pdf", msg, current_app.config["SUBSCRIBE_URL"]))
            return str(r), 200

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

    # ---------------------------------------------------------------
    # Start test sheet flow
    # ---------------------------------------------------------------
    if "generate test sheet" in lower or lower == "test sheet":
        if not enable_pdf:
            r = MessagingResponse()
            r.message("Feature temporarily unavailable.")
            return str(r), 200
        if night_mode_active():
            r = MessagingResponse()
            r.message("🌙 Night mode (02:00–06:00 UK): PDF generation is paused. Please try again after 06:00 UK.")
            return str(r), 200

        allowed, msg = check_and_count(
            user_id=user_id,
            now_ts=now_ts,
            free_days=int(current_app.config["FREE_TRIAL_DAYS"]),
            msg_cap=int(current_app.config.get("FREE_TRIAL_MESSAGE_CREDITS", 50)),
            subscribe_url=current_app.config["SUBSCRIBE_URL"],
        )
        if not allowed:
            r = MessagingResponse()
            r.message(build_paywall_message("pdf", msg, current_app.config["SUBSCRIBE_URL"]))
            return str(r), 200

        TEST_SHEET_SESSIONS[user_id] = {}
        _, question = get_next_missing_field(TEST_SHEET_SESSIONS[user_id])
        r = MessagingResponse()
        r.message(question)
        return str(r), 200

    # ---------------------------------------------------------------
    # Generate test sheet when complete
    # ---------------------------------------------------------------
    if user_id in TEST_SHEET_SESSIONS:
        session = TEST_SHEET_SESSIONS[user_id]
        field, _ = get_next_missing_field(session)

        if not field:
            if not enable_pdf:
                r = MessagingResponse()
                r.message("Feature temporarily unavailable.")
                return str(r), 200
            if night_mode_active():
                r = MessagingResponse()
                r.message("🌙 Night mode (02:00–06:00 UK): PDF generation is paused. Please try again after 06:00 UK.")
                return str(r), 200

        allowed, msg = check_and_count(
            user_id=user_id,
            now_ts=now_ts,
            free_days=int(current_app.config["FREE_TRIAL_DAYS"]),
            msg_cap=int(current_app.config.get("FREE_TRIAL_MESSAGE_CREDITS", 50)),
            subscribe_url=current_app.config["SUBSCRIBE_URL"],
        )
        if not allowed:
            r = MessagingResponse()
            r.message(build_paywall_message("pdf", msg, current_app.config["SUBSCRIBE_URL"]))
            return str(r), 200

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

    # ---------------------------------------------------------------
    # Normal AI chat (RAG + GPT)
    # ---------------------------------------------------------------
    global _VS
    if enable_rag and ("_VS" not in globals() or _VS is None):
        _VS = build_or_load_vectorstore()

    history = fetch_history(user_id, limit=10)
    ctx = retrieve_context(_VS, body, k=4) if enable_rag else ""

    if not history:
        reply = onboarding_message()
    else:
        reply = chat_reply(body, history, ctx)

    append_message(user_id, "user", body, now_ts)
    append_message(user_id, "assistant", reply, utc_now_ts())

    r = MessagingResponse()
    r.message(reply)
    return str(r), 200
