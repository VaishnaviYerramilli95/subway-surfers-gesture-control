"""
train_model.py
==============
Subway Surfers Gesture Control - Neural Network Training Script
NNDL Project

Run this LOCALLY before deploying to generate the model files:
    python train_model.py

Generates:
    - gesture_model.h5      (Keras model)
    - label_encoder.pkl     (gesture class list)
    - training_history.png  (accuracy/loss plots)
"""

import numpy as np
import pickle
import os
import json

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF warnings

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# ─── Configuration ────────────────────────────────────────────────────────────
GESTURES      = ['jump', 'slide', 'left', 'right', 'neutral']
SAMPLES       = 800   # samples per gesture class
NOISE_STD     = 0.018 # augmentation noise
EPOCHS        = 80
BATCH_SIZE    = 32
RANDOM_SEED   = 42

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# ─── Synthetic Landmark Generator ─────────────────────────────────────────────
# MediaPipe Hands produces 21 landmarks × 3 (x,y,z) = 63 features.
# y=0 → top of image, y=1 → bottom; x=0 → left, x=1 → right.
# We simulate each gesture as geometrically plausible hand poses.

def _n(std=NOISE_STD):
    """Gaussian noise helper."""
    return float(np.random.normal(0, std))

def _build_hand(wrist_x, wrist_y, joints):
    """
    Build a 63-element landmark list from wrist pos + 20 relative joint offsets.
    joints: list of (dx, dy) for landmarks 1–20.
    """
    lm = [wrist_x + _n(), wrist_y + _n(), _n(0.005)]
    for dx, dy in joints:
        lm += [wrist_x + dx + _n(), wrist_y + dy + _n(), _n(0.005)]
    return lm


def generate_jump(n=SAMPLES):
    """Open palm, hand raised — JUMP."""
    data = []
    for _ in range(n):
        wx = 0.5 + np.random.uniform(-0.12, 0.12)
        wy = 0.82 + np.random.uniform(-0.06, 0.06)
        joints = [
            # Thumb (1-4)
            (0.13, -0.04), (0.18, -0.11), (0.21, -0.19), (0.23, -0.27),
            # Index (5-8)
            (0.09, -0.22), (0.09, -0.36), (0.09, -0.47), (0.09, -0.57),
            # Middle (9-12)
            (0.01, -0.25), (0.01, -0.40), (0.01, -0.52), (0.01, -0.62),
            # Ring (13-16)
            (-0.08, -0.22), (-0.08, -0.36), (-0.08, -0.47), (-0.08, -0.55),
            # Pinky (17-20)
            (-0.15, -0.18), (-0.17, -0.29), (-0.17, -0.37), (-0.17, -0.44),
        ]
        data.append(_build_hand(wx, wy, joints))
    return np.array(data)


def generate_slide(n=SAMPLES):
    """Fist / hand pushed down — SLIDE (duck)."""
    data = []
    for _ in range(n):
        wx = 0.5 + np.random.uniform(-0.12, 0.12)
        wy = 0.62 + np.random.uniform(-0.06, 0.06)
        joints = [
            # Thumb curled inward (1-4)
            (0.10, 0.02), (0.13, 0.06), (0.10, 0.09), (0.07, 0.11),
            # Index curled (5-8)
            (0.08, -0.10), (0.09, -0.03), (0.08, 0.03), (0.06, 0.06),
            # Middle curled (9-12)
            (0.00, -0.12), (0.02, -0.04), (0.02, 0.03), (0.00, 0.06),
            # Ring curled (13-16)
            (-0.07, -0.10), (-0.07, -0.03), (-0.06, 0.03), (-0.04, 0.06),
            # Pinky curled (17-20)
            (-0.13, -0.08), (-0.14, -0.02), (-0.13, 0.03), (-0.11, 0.05),
        ]
        data.append(_build_hand(wx, wy, joints))
    return np.array(data)


def generate_left(n=SAMPLES):
    """Hand pointing / tilting left — MOVE LEFT."""
    data = []
    for _ in range(n):
        # Wrist on the right side, fingers extend leftward
        wx = 0.68 + np.random.uniform(-0.06, 0.06)
        wy = 0.55 + np.random.uniform(-0.12, 0.12)
        joints = [
            # Thumb up-left (1-4)
            (-0.02, 0.05), (-0.07, 0.08), (-0.11, 0.10), (-0.15, 0.12),
            # Index extends left (5-8)
            (-0.06, -0.06), (-0.16, -0.06), (-0.24, -0.06), (-0.31, -0.06),
            # Middle extends left (9-12)
            (-0.05, -0.13), (-0.15, -0.11), (-0.23, -0.09), (-0.30, -0.07),
            # Ring extends left (13-16)
            (-0.04, -0.11), (-0.13, -0.15), (-0.21, -0.15), (-0.28, -0.14),
            # Pinky extends left (17-20)
            (-0.02, -0.08), (-0.10, -0.13), (-0.16, -0.14), (-0.21, -0.14),
        ]
        data.append(_build_hand(wx, wy, joints))
    return np.array(data)


def generate_right(n=SAMPLES):
    """Hand pointing / tilting right — MOVE RIGHT."""
    data = []
    for _ in range(n):
        # Wrist on the left side, fingers extend rightward
        wx = 0.32 + np.random.uniform(-0.06, 0.06)
        wy = 0.55 + np.random.uniform(-0.12, 0.12)
        joints = [
            # Thumb up-right (1-4)
            (0.02, 0.05), (0.07, 0.08), (0.11, 0.10), (0.15, 0.12),
            # Index extends right (5-8)
            (0.06, -0.06), (0.16, -0.06), (0.24, -0.06), (0.31, -0.06),
            # Middle extends right (9-12)
            (0.05, -0.13), (0.15, -0.11), (0.23, -0.09), (0.30, -0.07),
            # Ring extends right (13-16)
            (0.04, -0.11), (0.13, -0.15), (0.21, -0.15), (0.28, -0.14),
            # Pinky extends right (17-20)
            (0.02, -0.08), (0.10, -0.13), (0.16, -0.14), (0.21, -0.14),
        ]
        data.append(_build_hand(wx, wy, joints))
    return np.array(data)


def generate_neutral(n=SAMPLES):
    """Relaxed / resting hand — NO ACTION."""
    data = []
    for _ in range(n):
        wx = 0.5 + np.random.uniform(-0.15, 0.15)
        wy = 0.65 + np.random.uniform(-0.10, 0.10)
        # Slightly bent fingers – halfway between open and closed
        joints = [
            # Thumb (1-4)
            (0.11, 0.01), (0.15, -0.04), (0.17, -0.11), (0.18, -0.17),
            # Index (5-8)
            (0.07, -0.14), (0.08, -0.22), (0.07, -0.17), (0.06, -0.13),
            # Middle (9-12)
            (0.00, -0.16), (0.01, -0.24), (0.00, -0.19), (0.00, -0.15),
            # Ring (13-16)
            (-0.07, -0.14), (-0.07, -0.22), (-0.06, -0.17), (-0.05, -0.13),
            # Pinky (17-20)
            (-0.13, -0.11), (-0.14, -0.17), (-0.13, -0.13), (-0.11, -0.11),
        ]
        data.append(_build_hand(wx, wy, joints))
    return np.array(data)


# ─── Dataset Assembly ─────────────────────────────────────────────────────────
def build_dataset():
    print("⚙️  Generating synthetic training data...")
    generators = [generate_jump, generate_slide, generate_left,
                  generate_right, generate_neutral]

    X_list, y_list = [], []
    for label_idx, (gesture, gen_fn) in enumerate(zip(GESTURES, generators)):
        data = gen_fn(SAMPLES)
        X_list.append(data)
        y_list.append(np.full(len(data), label_idx))
        print(f"   ✅  {gesture:10s}  →  {len(data)} samples")

    X = np.vstack(X_list).astype(np.float32)
    y = np.concatenate(y_list).astype(np.int32)

    # Clip to valid [0,1] range for x,y coordinates
    X = np.clip(X, 0.0, 1.0)

    # Shuffle
    idx = np.random.permutation(len(X))
    return X[idx], y[idx]


# ─── Model Definition ─────────────────────────────────────────────────────────
def build_model(num_classes=5):
    """
    3-hidden-layer dense neural network for gesture classification.
    Input  : 63 normalized MediaPipe landmarks (21 × 3)
    Output : softmax over 5 gesture classes
    """
    model = keras.Sequential([
        keras.Input(shape=(63,), name='landmarks_input'),

        layers.Dense(128, name='hidden_1'),
        layers.BatchNormalization(name='bn_1'),
        layers.Activation('relu'),
        layers.Dropout(0.40, name='drop_1'),

        layers.Dense(64, name='hidden_2'),
        layers.BatchNormalization(name='bn_2'),
        layers.Activation('relu'),
        layers.Dropout(0.30, name='drop_2'),

        layers.Dense(32, activation='relu', name='hidden_3'),

        layers.Dense(num_classes, activation='softmax', name='output'),
    ], name='GestureNet')

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model


# ─── Training ─────────────────────────────────────────────────────────────────
def train():
    print("\n🚀  Subway Surfers Gesture Control — Training\n" + "=" * 50)

    X, y = build_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=RANDOM_SEED, stratify=y
    )
    print(f"\n📊  Train: {len(X_train)}  |  Test: {len(X_test)}\n")

    model = build_model(num_classes=len(GESTURES))
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy', patience=12, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=6, min_lr=1e-5
        ),
    ]

    print("\n🏋️  Training...\n")
    history = model.fit(
        X_train, y_train,
        validation_split=0.15,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )

    # ── Evaluation ──
    print("\n📈  Evaluation on hold-out test set:")
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"    Loss     : {loss:.4f}")
    print(f"    Accuracy : {acc*100:.2f}%")

    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    print("\n" + classification_report(y_test, y_pred, target_names=GESTURES))

    # ── Save model & encoder ──
    model.save('gesture_model.h5')
    with open('label_encoder.pkl', 'wb') as f:
        pickle.dump(GESTURES, f)

    # Save training history for the dashboard
    hist_data = {
        'accuracy': [float(v) for v in history.history['accuracy']],
        'val_accuracy': [float(v) for v in history.history['val_accuracy']],
        'loss': [float(v) for v in history.history['loss']],
        'val_loss': [float(v) for v in history.history['val_loss']],
    }
    with open('training_history.json', 'w') as f:
        json.dump(hist_data, f)

    print("\n✅  Saved: gesture_model.h5 | label_encoder.pkl | training_history.json")
    print(f"🎯  Final Test Accuracy: {acc*100:.2f}%")
    print("\nNext steps:")
    print("  1. Run locally: python app.py")
    print("  2. Push to GitHub and deploy on Render")
    print("  3. For live game control: python game_controller.py\n")


if __name__ == '__main__':
    train()
