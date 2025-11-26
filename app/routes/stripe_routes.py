# app/routes/stripe_routes.py
import os
import stripe
from flask import Blueprint, jsonify, request, abort

bp = Blueprint("stripe_routes", __name__, url_prefix="/stripe")

# Read keys from env (safe if blank in dev)
stripe.api_key = os.getenv("STRIPE_SECRET_KEY", "")

PRICE_MONTHLY = os.getenv("STRIPE_PRICE_MONTHLY", "")
PRICE_YEARLY  = os.getenv("STRIPE_PRICE_YEARLY", "")
SUCCESS_URL   = os.getenv("STRIPE_SUCCESS_URL", "http://127.0.0.1:5050/success")
CANCEL_URL    = os.getenv("STRIPE_CANCEL_URL", "http://127.0.0.1:5050/cancel")
WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET", "")

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

    session = stripe.checkout.Session.create(
        mode="subscription",
        line_items=[{"price": price_id, "quantity": 1}],
        success_url=SUCCESS_URL + "?session_id={CHECKOUT_SESSION_ID}",
        cancel_url=CANCEL_URL,
        client_reference_id=whatsapp,     # ⭐ CRITICAL
        allow_promotion_codes=True,
        billing_address_collection="auto"
    )

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
    session = stripe.billing_portal.Session.create(
        customer=customer_id,
        return_url=return_url
    )
    return jsonify({"portal_url": session.url})

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
        abort(400, f"Webhook signature failed: {e}")

    event_type = event["type"]
    data = event["data"]["object"]

    print(f"📡 Stripe event: {event_type}")

    # --- Helper: locate user by stripe_customer_id ---
    def find_user_by_customer(customer_id: str):
        import sqlite3, os
        db_path = os.getenv("DB_PATH", "sitemind.db")
        conn = sqlite3.connect(db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT user_id FROM users WHERE stripe_customer_id=?",
            (customer_id,)
        ).fetchone()
        conn.close()
        return row["user_id"] if row else None

    # --- Helper: update subscription status ---
    def update_sub_status(user_id: str, status: int):
        import sqlite3, os
        db_path = os.getenv("DB_PATH", "sitemind.db")
        conn = sqlite3.connect(db_path, check_same_thread=False)
        conn.execute(
            "UPDATE users SET is_subscribed=? WHERE user_id=?",
            (status, user_id)
        )
        conn.commit()
        conn.close()

    # 1) Checkout completed
    if event_type == "checkout.session.completed":
        customer_id = data.get("customer")
        sub_id = data.get("subscription")
        whatsapp_number = data.get("client_reference_id")  # must be passed during checkout

        if not whatsapp_number:
            print("⚠️ No client_reference_id. Cannot link payment to WhatsApp user.")
            return {"ok": True}

        import sqlite3, os
        db_path = os.getenv("DB_PATH", "sitemind.db")
        conn = sqlite3.connect(db_path, check_same_thread=False)
        conn.execute(
            "UPDATE users SET is_subscribed=1, stripe_customer_id=?, stripe_subscription_id=? WHERE user_id=?",
            (customer_id, sub_id, whatsapp_number)
        )
        conn.commit()
        conn.close()

        print(f"✅ Subscription activated for {whatsapp_number}")
        return {"ok": True}

    # 2) Invoice paid → renew subscription
    if event_type == "invoice.paid":
        customer_id = data.get("customer")
        user_id = find_user_by_customer(customer_id)
        if user_id:
            update_sub_status(user_id, 1)
            print(f"💚 Subscription renewed for {user_id}")
        return {"ok": True}

    # 3) Subscription updated
    if event_type == "customer.subscription.updated":
        customer_id = data.get("customer")
        status = data.get("status")  # active / past_due / canceled
        user_id = find_user_by_customer(customer_id)

        if user_id:
            if status == "active":
                update_sub_status(user_id, 1)
                print(f"🟢 Subscription active for {user_id}")
            else:
                update_sub_status(user_id, 0)
                print(f"🔴 Subscription inactive for {user_id}")

        return {"ok": True}

    # 4) Subscription deleted
    if event_type == "customer.subscription.deleted":
        customer_id = data.get("customer")
        user_id = find_user_by_customer(customer_id)
        if user_id:
            update_sub_status(user_id, 0)
            print(f"❌ Subscription cancelled for {user_id}")

        return {"ok": True}

    return {"ok": True}
