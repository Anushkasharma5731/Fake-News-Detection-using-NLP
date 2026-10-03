"""Production Flask interface for the trained fake news classifier."""

from pathlib import Path
import os

import joblib
from flask import Flask, jsonify, render_template, request

from text_utils import clean_text

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = Path(os.environ.get("MODEL_BUNDLE_PATH", BASE_DIR / "model_bundle.joblib"))
MAX_ARTICLE_LENGTH = 40_000

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024


def load_artifacts() -> tuple[object, object]:
    """Fail at startup when the trained artifact is missing or invalid."""
    if MODEL_PATH.is_file():
        try:
            bundle = joblib.load(MODEL_PATH)
            model, vectorizer = bundle["model"], bundle["vectorizer"]
            if not hasattr(model, "predict") or not hasattr(vectorizer, "transform"):
                raise ValueError("bundle is missing a compatible model or vectorizer")
            return model, vectorizer
        except Exception as exc:
            raise RuntimeError(f"Could not load model bundle at {MODEL_PATH}: {exc}") from exc

    # Continue to support artifacts produced by earlier versions of this project.
    legacy_model_path = BASE_DIR / "fake_news_model.pkl"
    legacy_vectorizer_path = BASE_DIR / "tfidf_vectorizer.pkl"
    if legacy_model_path.is_file() and legacy_vectorizer_path.is_file():
        try:
            return joblib.load(legacy_model_path), joblib.load(legacy_vectorizer_path)
        except Exception as exc:
            raise RuntimeError(f"Could not load legacy model artifacts: {exc}") from exc

    if not MODEL_PATH.is_file():
        raise RuntimeError(
            f"Model bundle not found at {MODEL_PATH}. Run `python train_model.py` before starting the web app."
        )


# Load once per worker, not once per request. Keep this artifact private and only
# deploy bundles produced by this project's own training script (joblib uses pickle).
MODEL, VECTORIZER = load_artifacts()


@app.after_request
def add_security_headers(response):
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "no-referrer")
    response.headers.setdefault(
        "Content-Security-Policy",
        "default-src 'self'; style-src 'self'; form-action 'self'; "
        "base-uri 'self'; object-src 'none'; frame-ancestors 'none'",
    )
    if request.path == "/predict" and request.method == "POST":
        response.headers.setdefault("Cache-Control", "no-store")
    if os.environ.get("APP_ENV") == "production":
        response.headers.setdefault("Strict-Transport-Security", "max-age=31536000; includeSubDomains")
    return response


@app.get("/")
def index():
    return render_template("index.html", text="", result=None, error=None, ready=True)


@app.get("/healthz")
def healthz():
    return jsonify(status="ok", model_loaded=True)


@app.post("/predict")
def predict():
    text = request.form.get("article", "").strip()
    if not text:
        return render_template("index.html", text=text, result=None,
                               error="Paste an article or headline to analyze.",
                               ready=True), 400
    if len(text) > MAX_ARTICLE_LENGTH:
        return render_template("index.html", text="", result=None,
                               error=f"Please keep the text under {MAX_ARTICLE_LENGTH:,} characters.",
                               ready=True), 413
    cleaned = clean_text(text)
    if not cleaned:
        return render_template("index.html", text=text, result=None,
                               error="There is no usable text to analyze.", ready=True), 400

    prediction = str(MODEL.predict(VECTORIZER.transform([cleaned]))[0])
    normalized = prediction.strip().lower()
    if normalized in {"fake", "unreliable"}:
        label, style = "Model class: fake", "fake"
    elif normalized in {"real", "reliable"}:
        label, style = "Model class: real", "real"
    else:
        label, style = f"Model label: {prediction}", "unknown"
    return render_template("index.html", text=text,
                           result={"label": label, "style": style}, error=None, ready=True)


@app.errorhandler(413)
def request_too_large(_error):
    return render_template("index.html", text="", result=None,
                           error=f"Request is too large. Keep the text under {MAX_ARTICLE_LENGTH:,} characters.",
                           ready=True), 413


if __name__ == "__main__":
    app.run(debug=False)
