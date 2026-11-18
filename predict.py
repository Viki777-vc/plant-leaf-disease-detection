"""Predict a plant leaf class using a previously trained model."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

IMG_SIZE = 128
DEFAULT_MODEL = "models/leaves_disease_detector_model_optimized.h5"
DEFAULT_LABELS = "models/class_labels.npy"
CONFIDENCE_THRESHOLD = 0.80


def parse_args():
    parser = argparse.ArgumentParser(description="Predict a plant leaf disease class.")
    parser.add_argument("--image", required=True, help="Path to the input leaf image.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Path to the trained .h5 model.")
    parser.add_argument("--labels", default=DEFAULT_LABELS, help="Path to class_labels.npy.")
    parser.add_argument("--threshold", type=float, default=CONFIDENCE_THRESHOLD)
    return parser.parse_args()


def predict_leaf_disease(image_path: str, model_path: str, labels_path: str, threshold: float):
    for required_path in (image_path, model_path, labels_path):
        if not Path(required_path).is_file():
            raise FileNotFoundError(f"Required file not found: {required_path}")

    model = tf.keras.models.load_model(model_path)
    class_labels = np.load(labels_path, allow_pickle=True).item()

    img = image.load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img).astype("float32")
    # Must match the original training ImageDataGenerator(rescale=1./255).
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    probabilities = model.predict(img_array, verbose=0)[0]
    class_index = int(np.argmax(probabilities))
    confidence = float(probabilities[class_index])
    class_name = class_labels[class_index]

    if confidence < threshold:
        return (
            "Low-confidence prediction: this image may not belong to the "
            f"trained classes (top class: {class_name}, confidence: {confidence:.2%})."
        )
    if "healthy" in class_name.lower():
        return f"Healthy leaf: {class_name} (confidence: {confidence:.2%})."
    return f"Disease detected: {class_name} (confidence: {confidence:.2%})."


def main():
    args = parse_args()
    try:
        result = predict_leaf_disease(args.image, args.model, args.labels, args.threshold)
        print(result)
    except Exception as exc:
        raise SystemExit(f"Prediction failed: {exc}") from exc


if __name__ == "__main__":
    main()
