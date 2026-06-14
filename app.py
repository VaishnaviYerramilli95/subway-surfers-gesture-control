"""
app.py
======
Subway Surfers Gesture Control — Flask Web Application
NNDL Project

Run locally:
    python app.py

Deploy on Render:
    Push to GitHub, connect repo on render.com
"""

import os
import json
import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

app = Flask(__name__)

# ─── Globals ──────────────────────────────────────────────────────────────────
model         = None
label_encoder = None   # list of gesture class names
MODEL_PATH    = 'gesture_model.h5'
ENCODER_PATH  = 'label_encoder.pkl'

GESTURE_META = {
    'jump':    {'emoji': '✋', 'key': '↑',  'action': 'Jump',        'color': '#a78bfa'},
    'slide':   {'emoji': '👊', 'key': '↓',  'action': 'Slide / Duck','color': '#60a5fa'},
    'left':    {'emoji': '👈', 'key': '←',  'action': 'Move Left',   'color': '#34d399'},
    'right':   {'emoji': '👉', 'key': '→',  'action': 'Move Right',  'color': '#f97316'},
    'neutral': {'emoji': '🖐', 'key': '—',  'action': 'No Action',   'color': '#94a3b8'},
}


def load_model_once():
    """Load model and encoder on first use (lazy loading)."""
    global model, label_encoder
    if model is not None:
        return True

    if not os.path.exists(MODEL_PATH):
        return False

    try:
        import tensorflow as tf
        model = tf.keras.models.load_model(MODEL_PATH)
        with open(ENCODER_PATH, 'rb') as f:
            label_encoder = pickle.load(f)
        print(f"✅  Model loaded: {MODEL_PATH}")
        return True
    except Exception as e:
        print(f"❌  Failed to load model: {e}")
        return False


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    """Serve the main dashboard page."""
    model_ready = os.path.exists(MODEL_PATH)
    return render_template('index.html', model_ready=model_ready)


@app.route('/predict', methods=['POST'])
def predict():
    """
    POST /predict
    Body: { "landmarks": [x0,y0,z0, x1,y1,z1, ... x20,y20,z20] }  (63 floats)
    Returns: { "gesture": str, "confidence": float, "probabilities": {...} }
    """
    if not load_model_once():
        return jsonify({
            'error': 'Model not found. Run train_model.py first.',
            'gesture': 'neutral',
            'confidence': 0.0,
            'probabilities': {g: 0.0 for g in GESTURE_META}
        }), 200   # 200 so frontend still works

    data = request.get_json(silent=True)
    if not data or 'landmarks' not in data:
        return jsonify({'error': 'Missing "landmarks" in request body'}), 400

    landmarks = data['landmarks']
    if len(landmarks) != 63:
        return jsonify({'error': f'Expected 63 values, got {len(landmarks)}'}), 400

    features = np.array(landmarks, dtype=np.float32).reshape(1, 63)
    probs    = model.predict(features, verbose=0)[0]

    gesture_idx  = int(np.argmax(probs))
    gesture_name = label_encoder[gesture_idx]
    confidence   = float(probs[gesture_idx])

    probabilities = {label_encoder[i]: float(probs[i]) for i in range(len(label_encoder))}

    return jsonify({
        'gesture':       gesture_name,
        'confidence':    confidence,
        'probabilities': probabilities,
        'meta':          GESTURE_META.get(gesture_name, {}),
    })


@app.route('/model-info')
def model_info():
    """GET /model-info — returns model architecture details."""
    loaded = load_model_once()
    info = {
        'loaded':       loaded,
        'model_path':   MODEL_PATH,
        'architecture': '63 → Dense(128)+BN+Dropout → Dense(64)+BN+Dropout → Dense(32) → Dense(5,softmax)',
        'input':        '21 MediaPipe hand landmarks × 3 (x, y, z) = 63 features',
        'gestures':     GESTURE_META,
        'framework':    'TensorFlow / Keras',
    }
    if loaded and model is not None:
        info['total_params'] = int(model.count_params())
        info['layers']       = len(model.layers)
    return jsonify(info)


@app.route('/history')
def history():
    """GET /history — returns training history JSON if available."""
    if os.path.exists('training_history.json'):
        with open('training_history.json') as f:
            return jsonify(json.load(f))
    return jsonify({'error': 'No training history found. Run train_model.py first.'}), 404


@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'model_ready': os.path.exists(MODEL_PATH)})


# ─── Entry Point ──────────────────────────────────────────────────────────────
if __name__ == '__main__':
    load_model_once()
    port = int(os.environ.get('PORT', 5000))
    print(f"\n🌐  Starting server on http://0.0.0.0:{port}")
    app.run(host='0.0.0.0', port=port, debug=False)
