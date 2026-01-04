# app/routes/stripe_routes.py
import os
import time
import stripe
from flask import Blueprint, jsonify, request, abort

from app.services import db_utils as db
from app.services.monitoring import log_event, user_hash
from twilio.rest import Client

bp = Blueprint("stripe_routes", __name__, url_prefix="/stripe")

# Read keys from env (safe if blank in dev)
stripe.api_key = os.getenv("STRIPE_SECRET_KEY", "")

PRICE_MONTHLY = os.getenv("STRIPE_PRICE_MONTHLY", "")
PRICE_YEARLY  = os.getenv("STRIPE_PRICE_YEARLY", "")
SUCCESS_URL   = os.getenv("STRIPE_SUCCESS_URL", "http://127.0.0.1:5050/success")
CANCEL_URL    = os.getenv("STRIPE_CANCEL_URL", "http://127.0.0.1:5050/cancel")
WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET", "")
WHATSAPP_NUMBER = os.getenv("WHATSAPP_NUMBER") or os.getenv("TWILIO_PHONE_NUMBER")
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "")
db.ensure_schema()


def _send_pro_welcome(whatsapp_number: str):
    if not whatsapp_number or not WHATSAPP_NUMBER:
        return
    try:
        client = Client(os.getenv("TWILIO_ACCOUNT_SID"), os.getenv("TWILIO_AUTH_TOKEN"))
        client.messages.create(
            from_=f"whatsapp:{WHATSAPP_NUMBER}",
            to=f"whatsapp:{whatsapp_number}",
            body=(
                "✅ Thanks for subscribing to SiteMind AI Pro!\n"
                "You now have higher daily limits, heavy photo analysis, and PDF exports."
            ),
        )
    except Exception as e:
        log_event("error", "Failed to send Pro welcome", {"err": str(e), "user": user_hash(whatsapp_number)})


def _record_event(event_id: str) -> bool:
    """
    Returns False if the event was already processed.
    """
    if not event_id:
        return True
    try:
        exists = db.fetchone("SELECT event_id FROM stripe_events WHERE event_id=?", (event_id,))
        if exists:
            return False
        db.execute(
            "INSERT INTO stripe_events (event_id, created_ts) VALUES (?,?)",
            (event_id, int(time.time())),
        )
    except Exception as e:
        log_event("error", "Failed to record Stripe event", {"err": str(e), "event": event_id})
    return True


def _find_user_by_customer(customer_id: str):
    row = db.fetchone("SELECT user_id FROM users WHERE stripe_customer_id=?", (customer_id,))
    if isinstance(row, dict) and row:
        return row.get("user_id")
    if row:
        return row[0]
    return None


def _update_sub_status(user_id: str, status: int, customer_id: str = "", sub_id: str = ""):
    db.execute(
        """
        UPDATE users
        SET is_subscribed=?, plan_tier=?,
            stripe_customer_id=COALESCE(?, stripe_customer_id),
            stripe_subscription_id=COALESCE(?, stripe_subscription_id),
            is_trial=0
        WHERE user_id=?
        """,
        (status, "pro" if status else "core", customer_id or None, sub_id or None, user_id),
    )

@bp.get("/health")
def health():
    return {"ok": True}

@bp.post("/checkout")
def start_checkout():
    plan = request.args.get("plan", "monthly")
    whatsapp = request.args.get("user_id")   # WhatsApp number from frontend

    if not whatsapp:
        return jsonify({"error": "Missing user_id (WhatsApp number)"}), 400

    price_id = PRICE_YEARLY if plan == "yearly" else PRICE_MONTHLY

    if not stripe.api_key or not price_id:
        return jsonify({
            "error": "Missing STRIPE_SECRET_KEY or price id"
        }), 400

    try:
        session = stripe.checkout.Session.create(
            mode="subscription",
            line_items=[{"price": price_id, "quantity": 1}],
            success_url=SUCCESS_URL + "?session_id={CHECKOUT_SESSION_ID}",
            cancel_url=CANCEL_URL,
            client_reference_id=whatsapp,     # ⭐ CRITICAL
            allow_promotion_codes=True,
            billing_address_collection="auto"
        )
    except Exception as e:
        log_event("error", "Stripe checkout failed", {"err": str(e)})
        return jsonify({"error": "Sorry — I’m temporarily offline. Please try again in a moment."}), 503

    return jsonify({"checkout_url": session.url})


@bp.post("/portal")
def portal():
    # For now require customer id via query until you store it in DB
    customer_id = request.args.get("customer_id")
    if not customer_id:
        return jsonify({"error": "customer_id is required"}), 400
    return_url = request.args.get("return_url", SUCCESS_URL)
    if not stripe.api_key:
        return jsonify({"error": "Missing STRIPE_SECRET_KEY"}), 400
    try:
        session = stripe.billing_portal.Session.create(
            customer=customer_id,
            return_url=return_url
        )
        return jsonify({"portal_url": session.url})
    except Exception as e:
        log_event("error", "Stripe portal failed", {"err": str(e)})
        return jsonify({"error": "Sorry — I’m temporarily offline. Please try again in a moment."}), 503


@bp.get("/session-status")
def session_status():
    session_id = request.args.get("session_id")
    if not session_id:
        return jsonify({"error": "session_id is required"}), 400
    try:
        session = stripe.checkout.Session.retrieve(session_id)
        payment_status = session.get("payment_status")
        sub_id = session.get("subscription")
        client_ref = session.get("client_reference_id")
        linked_plan = None
        if client_ref:
            row = db.fetchone("SELECT is_subscribed FROM users WHERE user_id=?", (client_ref,))
            if isinstance(row, dict):
                linked_plan = "pro" if row.get("is_subscribed") else "core"
            elif row:
                linked_plan = "pro" if row[0] else "core"
        return jsonify({
            "payment_status": payment_status,
            "subscription_id": sub_id,
            "linked_plan": linked_plan,
        })
    except Exception as e:
        log_event("error", "Session status check failed", {"err": str(e), "session": session_id})
        return jsonify({"error": "Sorry — I’m temporarily offline. Please try again in a moment."}), 503

@bp.post("/webhook")
def webhook():
    payload = request.data
    sig = request.headers.get("Stripe-Signature", "")

    if not WEBHOOK_SECRET:
        # Dev mode: accept all
        return {"ok": True}

    try:
        event = stripe.Webhook.construct_event(payload, sig, WEBHOOK_SECRET)
    except Exception as e:
        log_event("error", "Stripe webhook signature failed", {"err": str(e)})
        abort(400, f"Webhook signature failed: {e}")

    event_type = event["type"]
    data = event["data"]["object"]
    event_id = event.get("id")

    if not _record_event(event_id):
        return {"ok": True, "dedup": True}

    print(f"📡 Stripe event: {event_type}")

    # 1) Checkout completed
    if event_type == "checkout.session.completed":
        customer_id = data.get("customer")
        sub_id = data.get("subscription")
        whatsapp_number = data.get("client_reference_id")  # must be passed during checkout

        if not whatsapp_number:
            print("⚠️ No client_reference_id. Cannot link payment to WhatsApp user.")
            return {"ok": True}

        try:
            _update_sub_status(whatsapp_number, 1, customer_id, sub_id)
            _send_pro_welcome(whatsapp_number)
        except Exception as e:
            log_event("error", "Checkout completion failed", {"err": str(e), "user": user_hash(whatsapp_number)})
        print(f"✅ Subscription activated for {whatsapp_number}")
        return {"ok": True}

    # 2) Invoice paid → renew subscription
    if event_type == "invoice.paid":
        customer_id = data.get("customer")
        user_id = _find_user_by_customer(customer_id)
        if user_id:
            _update_sub_status(user_id, 1, customer_id, data.get("subscription"))
            print(f"💚 Subscription renewed for {user_id}")
        return {"ok": True}

    # 3) Subscription updated
    if event_type == "customer.subscription.updated":
        customer_id = data.get("customer")
        status = data.get("status")  # active / past_due / canceled
        user_id = _find_user_by_customer(customer_id)

        if user_id:
            if status == "active":
                _update_sub_status(user_id, 1, customer_id, data.get("id"))
                print(f"🟢 Subscription active for {user_id}")
            else:
                _update_sub_status(user_id, 0, customer_id, data.get("id"))
                print(f"🔴 Subscription inactive for {user_id}")

        return {"ok": True}

    # 4) Subscription deleted
    if event_type == "customer.subscription.deleted":
        customer_id = data.get("customer")
        user_id = _find_user_by_customer(customer_id)
        if user_id:
            _update_sub_status(user_id, 0, customer_id, data.get("id"))
            print(f"❌ Subscription cancelled for {user_id}")

        return {"ok": True}

    return {"ok": True}
