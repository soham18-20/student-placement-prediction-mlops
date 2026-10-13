import os
from pathlib import Path
from flask import Flask, send_from_directory, jsonify

from ml.predict import predictor
from ml.config import MODEL_PATH
from ml.pipeline import train_pipeline


def create_app(test_config=None) -> Flask:
    """Application factory for Flask placement prediction app."""
    app_dir = Path(__file__).resolve().parent
    static_folder = app_dir / "static"

    app = Flask(
        __name__,
        static_folder=str(static_folder),
        static_url_path="/static",
    )

    if test_config:
        app.config.update(test_config)

    # Ensure model is ready or trained if not present (auto-heal for convenience)
    if not app.config.get("TESTING") and not MODEL_PATH.exists():
        try:
            print("[Startup] Initializing default model pipeline...")
            train_pipeline()
        except Exception as e:
            print(f"[Startup Warning] Could not auto-train pipeline: {e}")

    # Load artifacts into singleton predictor
    predictor.load_model(strict=False)

    # Register API blueprint
    from app.routes import api_bp
    app.register_blueprint(api_bp)

    # Serve minimalist dashboard on root
    @app.route("/", methods=["GET"])
    def index():
        index_file = static_folder / "index.html"
        if index_file.exists():
            return send_from_directory(str(static_folder), "index.html")
        return jsonify({
            "service": "Student Placement Prediction System with MLOps",
            "message": "Dashboard UI file not found. Ensure app/static/index.html is created.",
            "health_check": "/health",
            "predict_endpoint": "POST /api/predict",
        })

    # Error Handlers
    @app.errorhandler(404)
    def handle_404(e):
        return jsonify({
            "status": "error",
            "error": "Not Found",
            "message": "The requested endpoint does not exist.",
            "available_endpoints": ["/health", "/api/predict", "/api/metrics", "/api/model-info", "/"],
        }), 404

    @app.errorhandler(405)
    def handle_405(e):
        return jsonify({
            "status": "error",
            "error": "Method Not Allowed",
            "message": "The HTTP method is not supported for this endpoint.",
        }), 405

    @app.errorhandler(500)
    def handle_500(e):
        return jsonify({
            "status": "error",
            "error": "Internal Server Error",
            "message": "An unexpected server error occurred.",
        }), 500

    return app
