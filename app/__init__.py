from flask import Flask
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv(override=True)

# Helper function to convert string env vars to real booleans
def env_bool(name, default=False):
    return os.getenv(name, str(default)).lower() in ("1", "true", "yes", "on")

# Read key environment values
SECRET_KEY = os.getenv("SECRET_KEY", "fallback_secret_key")
ENFORCE_TWILIO_SIGNATURE = env_bool("ENFORCE_TWILIO_SIGNATURE")
PORT = int(os.getenv("PORT", "5050"))
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "http://127.0.0.1:5050")

def create_app():
    app = Flask(__name__)
    app.secret_key = SECRET_KEY

    # Load app-level config from environment (with safe defaults)
    app.config.update(
        FREE_TRIAL_DAYS=int(os.getenv("FREE_TRIAL_DAYS", "14")),
        FREE_TRIAL_MESSAGE_CREDITS=int(os.getenv("FREE_TRIAL_MESSAGE_CREDITS", "50")),
        WORD_CAP=int(os.getenv("WORD_CAP", "200")),
        ENABLE_PDF=env_bool("ENABLE_PDF", True),
        ENABLE_HEAVY_OCR=env_bool("ENABLE_HEAVY_OCR", True),
        ENABLE_NIGHT_MODE=env_bool("ENABLE_NIGHT_MODE", True),
        ENABLE_RAG=env_bool("ENABLE_RAG", True),

        # OLD single subscribe URL (keep it if needed for website)
        SUBSCRIBE_URL=os.getenv("SUBSCRIBE_URL", "https://sitemind.ai/pay"),

        # ✅ NEW: Stripe subscription links
        SUBSCRIBE_MONTHLY_URL=os.getenv("SUBSCRIBE_MONTHLY_URL", ""),
        SUBSCRIBE_ANNUAL_URL=os.getenv("SUBSCRIBE_ANNUAL_URL", ""),
        STRIPE_PORTAL_URL=os.getenv("STRIPE_PORTAL_URL", ""),

        PRIVACY_URL=os.getenv("PRIVACY_URL", "https://sitemind.ai/privacy"),
        TERMS_URL=os.getenv("TERMS_URL", "https://sitemind.ai/terms"),

        # Twilio creds
        TWILIO_ACCOUNT_SID=os.getenv("TWILIO_ACCOUNT_SID", ""),
        TWILIO_AUTH_TOKEN=os.getenv("TWILIO_AUTH_TOKEN", ""),
        TWILIO_PHONE_NUMBER=os.getenv("TWILIO_PHONE_NUMBER", ""),

        ENFORCE_TWILIO_SIGNATURE=ENFORCE_TWILIO_SIGNATURE,
    )

    # ✅ Register blueprints
    from app.routes.whatsapp import bp as whatsapp_bp
    from app.routes.stripe_routes import bp as stripe_bp
    
    from app.routes.files import files_bp
    app.register_blueprint(files_bp)

    app.register_blueprint(whatsapp_bp)
    app.register_blueprint(stripe_bp)

    # Optional test route
    @app.route("/")
    def home():
        return "✅ SiteMind AI Flask app is running!"

    return app
