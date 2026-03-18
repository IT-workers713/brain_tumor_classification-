import base64
import io
import os
from functools import lru_cache
from typing import List, Tuple

import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, flash, redirect, render_template, request, url_for
from PIL import Image, UnidentifiedImageError
from tensorflow.keras.models import load_model


MODEL_PATH = "effnet.h5"
IMAGE_SIZE = 150
MAX_UPLOAD_MB = 10
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

CLASS_NAMES = [
    "Glioma Tumor",
    "No Tumor",
    "Meningioma Tumor",
    "Pituitary Tumor",
]

CLASS_DESCRIPTIONS = {
    "Glioma Tumor": "The uploaded MRI image is classified as Glioma Tumor.",
    "No Tumor": "The uploaded MRI image is classified as No Tumor.",
    "Meningioma Tumor": "The uploaded MRI image is classified as Meningioma Tumor.",
    "Pituitary Tumor": "The uploaded MRI image is classified as Pituitary Tumor.",
}

CLASS_BADGES = {
    "Glioma Tumor": "High Attention",
    "No Tumor": "Normal",
    "Meningioma Tumor": "High Attention",
    "Pituitary Tumor": "High Attention",
}

CLASS_COLORS = {
    "Glioma Tumor": "#ef4444",
    "No Tumor": "#10b981",
    "Meningioma Tumor": "#f59e0b",
    "Pituitary Tumor": "#8b5cf6",
}


class CompatibleDepthwiseConv2D(tf.keras.layers.DepthwiseConv2D):
    def __init__(self, *args, **kwargs):
        kwargs.pop("groups", None)
        super().__init__(*args, **kwargs)


app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "dev-secret-key-change-me")
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024


@lru_cache(maxsize=1)
def get_model():
    model_file = os.path.join(app.root_path, MODEL_PATH)
    if not os.path.exists(model_file):
        raise FileNotFoundError(
            f"Model file '{MODEL_PATH}' not found. Put effnet.h5 in the project root."
        )

    return load_model(
        model_file,
        custom_objects={"DepthwiseConv2D": CompatibleDepthwiseConv2D},
        compile=False,
    )


@app.context_processor
def inject_globals():
    return {
        "class_names": CLASS_NAMES,
        "model_path": MODEL_PATH,
        "image_size": IMAGE_SIZE,
    }


@app.errorhandler(413)
def too_large(_error):
    flash(f"File too large. Maximum allowed size is {MAX_UPLOAD_MB} MB.", "danger")
    return redirect(url_for("index"))


@app.errorhandler(FileNotFoundError)
def handle_missing_model(error):
    return render_template("index.html", model_error=str(error)), 500


@app.errorhandler(Exception)
def handle_exception(error):
    if app.debug:
        raise error
    return render_template(
        "index.html",
        model_error=f"Unexpected server error: {str(error)}"
    ), 500


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess_image(pil_image: Image.Image) -> np.ndarray:
    img = pil_image.convert("RGB")
    img_np = np.array(img)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    img_resized = cv2.resize(img_bgr, (IMAGE_SIZE, IMAGE_SIZE))
    return np.expand_dims(img_resized, axis=0).astype(np.float32)


def get_preprocessed_display_image(pil_image: Image.Image) -> np.ndarray:
    img = pil_image.convert("RGB")
    img_np = np.array(img)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    img_resized = cv2.resize(img_bgr, (IMAGE_SIZE, IMAGE_SIZE))
    return cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)


def image_to_data_uri(pil_image: Image.Image, format_name: str = "PNG") -> str:
    buffer = io.BytesIO()
    pil_image.save(buffer, format=format_name)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    mime = "image/png" if format_name.upper() == "PNG" else "image/jpeg"
    return f"data:{mime};base64,{encoded}"


def array_to_data_uri(img_array_rgb: np.ndarray) -> str:
    image = Image.fromarray(img_array_rgb)
    return image_to_data_uri(image, "PNG")


def predict_image(pil_image: Image.Image) -> Tuple[str, float, np.ndarray]:
    model = get_model()
    processed = preprocess_image(pil_image)
    predictions = model.predict(processed, verbose=0)[0]
    pred_index = int(np.argmax(predictions))
    pred_label = CLASS_NAMES[pred_index]
    confidence = float(predictions[pred_index])
    return pred_label, confidence, predictions


def get_ranked_predictions(predictions: np.ndarray) -> List[dict]:
    ranked = []
    for idx, prob in enumerate(predictions):
        label = CLASS_NAMES[idx]
        ranked.append(
            {
                "label": label,
                "probability": float(prob),
                "percentage": round(float(prob) * 100, 2),
                "color": CLASS_COLORS.get(label, "#2563eb"),
            }
        )
    ranked.sort(key=lambda item: item["probability"], reverse=True)
    return ranked


@app.route("/", methods=["GET"])
def index():
    model_error = None
    try:
        get_model()
    except Exception as error:
        model_error = str(error)

    return render_template("index.html", model_error=model_error)


@app.route("/predict", methods=["POST"])
def predict():
    uploaded_file = request.files.get("mri_image")

    if uploaded_file is None or uploaded_file.filename == "":
        flash("Please upload an MRI image first.", "warning")
        return redirect(url_for("index"))

    if not allowed_file(uploaded_file.filename):
        flash("Unsupported file type. Allowed formats: JPG, JPEG, PNG.", "danger")
        return redirect(url_for("index"))

    try:
        image = Image.open(uploaded_file.stream)
        image.load()
    except (UnidentifiedImageError, OSError):
        flash("Invalid image file. Please upload a valid MRI image.", "danger")
        return redirect(url_for("index"))

    pred_label, confidence, predictions = predict_image(image)
    ranked_predictions = get_ranked_predictions(predictions)
    preprocessed_display = get_preprocessed_display_image(image)

    image_width, image_height = image.size
    image_mode = image.mode

    context = {
        "prediction": {
            "label": pred_label,
            "confidence": round(confidence * 100, 2),
            "description": CLASS_DESCRIPTIONS[pred_label],
            "badge": CLASS_BADGES[pred_label],
            "color": CLASS_COLORS[pred_label],
        },
        "ranked_predictions": ranked_predictions,
        "original_image_uri": image_to_data_uri(image.convert("RGB"), "PNG"),
        "preprocessed_image_uri": array_to_data_uri(preprocessed_display),
        "image_meta": {
            "width": image_width,
            "height": image_height,
            "mode": image_mode,
        },
    }

    return render_template("result.html", **context)


if __name__ == "__main__":
    app.run(debug=True)
