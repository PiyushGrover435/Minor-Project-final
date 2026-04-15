"""
convert_tflite_full_integer.py

Converts PyTorch models directly to TensorFlow Lite INT8 format.
Bypasses onnx-tf / onnx2tf entirely by recreating the architectures
in TensorFlow Keras and transferring weights from the .pth state dicts.

Pipeline:  PyTorch .pth → Keras model → TFLite INT8 (Full Integer Quantization)
"""

import os
import numpy as np
import torch
import tensorflow as tf


# ── Helper: transfer weights ───────────────────────────────────────

def _transfer_conv1d(keras_layer, pt_weight, pt_bias):
    """Conv1d: PyTorch (out, in, k) → Keras (k, in, out)."""
    w = pt_weight.detach().cpu().numpy()
    b = pt_bias.detach().cpu().numpy()
    w = np.transpose(w, (2, 1, 0))
    keras_layer.set_weights([w, b])


def _transfer_conv2d(keras_layer, pt_weight, pt_bias):
    """Conv2d: PyTorch (out, in, kH, kW) → Keras (kH, kW, in, out)."""
    w = pt_weight.detach().cpu().numpy()
    b = pt_bias.detach().cpu().numpy()
    w = np.transpose(w, (2, 3, 1, 0))
    keras_layer.set_weights([w, b])


def _transfer_bn(keras_layer, pt_state, prefix):
    """BatchNorm: transfer gamma, beta, running_mean, running_var."""
    gamma = pt_state[f'{prefix}.weight'].detach().cpu().numpy()
    beta = pt_state[f'{prefix}.bias'].detach().cpu().numpy()
    mean = pt_state[f'{prefix}.running_mean'].detach().cpu().numpy()
    var = pt_state[f'{prefix}.running_var'].detach().cpu().numpy()
    keras_layer.set_weights([gamma, beta, mean, var])


def _transfer_dense(keras_layer, pt_weight, pt_bias):
    """Linear: PyTorch (out, in) → Keras (in, out)."""
    w = pt_weight.detach().cpu().numpy().T
    b = pt_bias.detach().cpu().numpy()
    keras_layer.set_weights([w, b])


# ════════════════════════════════════════════════════════════════════
#  1. Affective CNN  (EmotionCNN from train_affective_head.py)
#
#  PyTorch architecture (exact):
#    features.0:  Conv2d(1, 32, 3, padding=1)
#    features.1:  ReLU
#    features.2:  Conv2d(32, 64, 3, padding=1)
#    features.3:  BatchNorm2d(64)
#    features.4:  ReLU
#    features.5:  MaxPool2d(2)
#    features.6:  Dropout2d(0.25)
#    features.7:  Conv2d(64, 128, 3, padding=1)
#    features.8:  BatchNorm2d(128)
#    features.9:  ReLU
#    features.10: MaxPool2d(2)
#    features.11: Dropout2d(0.25)
#    features.12: Conv2d(128, 128, 3, padding=1)
#    features.13: BatchNorm2d(128)
#    features.14: ReLU
#    features.15: MaxPool2d(2)
#    features.16: Dropout2d(0.25)
#
#    classifier.0: Flatten
#    classifier.1: Linear(128*6*6, embed_dim)
#    classifier.2: ReLU
#    classifier.3: Dropout(0.5)
#    classifier.4: Linear(embed_dim, 7)
# ════════════════════════════════════════════════════════════════════

def build_affective_cnn_keras(embed_dim=256):
    inp = tf.keras.Input(shape=(48, 48, 1), name="input_face")

    # Block 1: Conv(1→32) → ReLU
    x = tf.keras.layers.Conv2D(32, 3, padding='same', name='conv0')(inp)
    x = tf.keras.layers.ReLU()(x)

    # Block 2: Conv(32→64) → BN → ReLU → MaxPool → Dropout
    x = tf.keras.layers.Conv2D(64, 3, padding='same', name='conv2')(x)
    x = tf.keras.layers.BatchNormalization(name='bn3')(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPool2D(2)(x)
    x = tf.keras.layers.Dropout(0.25)(x)

    # Block 3: Conv(64→128) → BN → ReLU → MaxPool → Dropout
    x = tf.keras.layers.Conv2D(128, 3, padding='same', name='conv7')(x)
    x = tf.keras.layers.BatchNormalization(name='bn8')(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPool2D(2)(x)
    x = tf.keras.layers.Dropout(0.25)(x)

    # Block 4: Conv(128→128) → BN → ReLU → MaxPool → Dropout
    x = tf.keras.layers.Conv2D(128, 3, padding='same', name='conv12')(x)
    x = tf.keras.layers.BatchNormalization(name='bn13')(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPool2D(2)(x)
    x = tf.keras.layers.Dropout(0.25)(x)

    # Classifier
    x = tf.keras.layers.Flatten()(x)
    embedding = tf.keras.layers.Dense(embed_dim, name='fc1')(x)
    x = tf.keras.layers.ReLU()(embedding)
    x = tf.keras.layers.Dropout(0.5)(x)
    out = tf.keras.layers.Dense(7, name='fc_out')(x)

    return tf.keras.Model(inp, [out, embedding], name="AffectiveCNN")


def transfer_cnn_weights(keras_model, pt_state):
    # Conv layers: features.0, features.2, features.7, features.12
    _transfer_conv2d(keras_model.get_layer('conv0'),  pt_state['features.0.weight'],  pt_state['features.0.bias'])
    _transfer_conv2d(keras_model.get_layer('conv2'),  pt_state['features.2.weight'],  pt_state['features.2.bias'])
    _transfer_conv2d(keras_model.get_layer('conv7'),  pt_state['features.7.weight'],  pt_state['features.7.bias'])
    _transfer_conv2d(keras_model.get_layer('conv12'), pt_state['features.12.weight'], pt_state['features.12.bias'])

    # BatchNorm: features.3, features.8, features.13
    _transfer_bn(keras_model.get_layer('bn3'),  pt_state, 'features.3')
    _transfer_bn(keras_model.get_layer('bn8'),  pt_state, 'features.8')
    _transfer_bn(keras_model.get_layer('bn13'), pt_state, 'features.13')

    # Dense: classifier.1 → fc1, classifier.4 → fc_out
    # Fix PyTorch (C,H,W) to Keras (H,W,C) Flatten mismatch!
    w_fc1 = pt_state['classifier.1.weight'].detach().cpu().numpy() # (256, 4608)
    # PyTorch features output is (128, 6, 6)
    w_fc1 = w_fc1.reshape(w_fc1.shape[0], 128, 6, 6)
    # Transpose to match Keras Flatten from Conv2D(channels_last) which is (H, W, C)
    w_fc1 = np.transpose(w_fc1, (0, 2, 3, 1))
    w_fc1 = w_fc1.reshape(w_fc1.shape[0], -1) 
    
    layer_fc1 = keras_model.get_layer('fc1')
    layer_fc1.set_weights([w_fc1.T, pt_state['classifier.1.bias'].detach().cpu().numpy()])
    
    _transfer_dense(keras_model.get_layer('fc_out'), pt_state['classifier.4.weight'], pt_state['classifier.4.bias'])


# ════════════════════════════════════════════════════════════════════
#  2. Affective TCN (MultiModalStressTCN)
#
#  PyTorch:
#    tcn.0:  Conv1d(feature_dim, 128, 3, padding=2, dilation=2)
#    tcn.1:  ReLU
#    tcn.2:  Conv1d(128, 64, 3, padding=4, dilation=4)
#    tcn.3:  ReLU
#    regressor.0: Linear(64, 32)
#    regressor.1: ReLU
#    regressor.2: Linear(32, 1)
#    regressor.3: Sigmoid
#    forward: x.transpose(1,2) → tcn → mean(dim=2) → regressor
# ════════════════════════════════════════════════════════════════════

def build_tcn_keras(seq_len=15, feature_dim=268):
    inp = tf.keras.Input(shape=(seq_len, feature_dim), name="input_temporal")

    x = tf.keras.layers.Conv1D(128, 3, dilation_rate=2, padding='causal', name='tcn0')(inp)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv1D(64, 3, dilation_rate=4, padding='causal', name='tcn2')(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.GlobalAveragePooling1D()(x)

    x = tf.keras.layers.Dense(32, name='reg0')(x)
    x = tf.keras.layers.ReLU()(x)
    out = tf.keras.layers.Dense(1, activation='sigmoid', name='reg2')(x)

    return tf.keras.Model(inp, out, name="StressTCN")


def transfer_tcn_weights(keras_model, pt_state):
    _transfer_conv1d(keras_model.get_layer('tcn0'), pt_state['tcn.0.weight'], pt_state['tcn.0.bias'])
    _transfer_conv1d(keras_model.get_layer('tcn2'), pt_state['tcn.2.weight'], pt_state['tcn.2.bias'])
    _transfer_dense(keras_model.get_layer('reg0'),  pt_state['regressor.0.weight'], pt_state['regressor.0.bias'])
    _transfer_dense(keras_model.get_layer('reg2'),  pt_state['regressor.2.weight'], pt_state['regressor.2.bias'])


# ════════════════════════════════════════════════════════════════════
#  3. Gaze Hybrid Model (HybridGazeModel)
#
#  PyTorch:
#    feature_extractor.0: Linear(10, 32) → ReLU → Linear(32, 64) → ReLU
#    tcn (temporal):  not named 'temporal' in state_dict...
#    Let me check.  Actually the gaze model calls its Conv1d layers
#    via self.tcn = nn.Sequential(Conv1d(64,64,3,d=2), ReLU, Conv1d(64,64,3,d=4), ReLU)
#    self.regressor = nn.Sequential(Linear(64+3, 32), ReLU, Linear(32, 2), Sigmoid)
#
#  State dict keys expected:
#    feature_extractor.0.weight/bias, feature_extractor.2.weight/bias
#    tcn.0.weight/bias, tcn.2.weight/bias
#    regressor.0.weight/bias, regressor.2.weight/bias
# ════════════════════════════════════════════════════════════════════

def build_gaze_keras(seq_len=15, input_dim=10, feature_dim=64, pose_dim=3):
    inp_seq = tf.keras.Input(shape=(seq_len, input_dim), name="input_seq")
    inp_pose = tf.keras.Input(shape=(pose_dim,), name="input_pose")

    # Feature extractor MLP applied per-timestep
    x = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(32, activation='relu', name='fe0_inner'), name='fe0'
    )(inp_seq)
    x = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(feature_dim, activation='relu', name='fe2_inner'), name='fe2'
    )(x)

    # Temporal Conv1D
    x = tf.keras.layers.Conv1D(feature_dim, 3, dilation_rate=2, padding='causal', name='g_tcn0')(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv1D(feature_dim, 3, dilation_rate=4, padding='causal', name='g_tcn2')(x)
    x = tf.keras.layers.ReLU()(x)

    # Take last timestep
    x = tf.keras.layers.Lambda(lambda t: t[:, -1, :])(x)

    # Concatenate with head pose
    x = tf.keras.layers.Concatenate()([x, inp_pose])

    # Regressor
    x = tf.keras.layers.Dense(32, activation='relu', name='g_reg0')(x)
    out = tf.keras.layers.Dense(2, activation='sigmoid', name='g_reg2')(x)

    return tf.keras.Model([inp_seq, inp_pose], out, name="GazeHybrid")


def transfer_gaze_weights(keras_model, pt_state):
    # Feature extractor (TimeDistributed wraps inner layers)
    fe0 = keras_model.get_layer('fe0').layer
    fe2 = keras_model.get_layer('fe2').layer
    _transfer_dense(fe0, pt_state['feature_extractor.0.weight'], pt_state['feature_extractor.0.bias'])
    _transfer_dense(fe2, pt_state['feature_extractor.2.weight'], pt_state['feature_extractor.2.bias'])

    # TCN Conv1D
    _transfer_conv1d(keras_model.get_layer('g_tcn0'), pt_state['tcn.0.weight'], pt_state['tcn.0.bias'])
    _transfer_conv1d(keras_model.get_layer('g_tcn2'), pt_state['tcn.2.weight'], pt_state['tcn.2.bias'])

    # Regressor Dense
    _transfer_dense(keras_model.get_layer('g_reg0'), pt_state['regressor.0.weight'], pt_state['regressor.0.bias'])
    _transfer_dense(keras_model.get_layer('g_reg2'), pt_state['regressor.2.weight'], pt_state['regressor.2.bias'])


# ════════════════════════════════════════════════════════════════════
#  TFLite Full Integer Quantization
# ════════════════════════════════════════════════════════════════════

def convert_to_tflite(keras_model, tflite_path, calib_npz_path, input_keys, label="Model"):
    print(f"\n--- Converting {label} to TFLite INT8 ---")

    saved_model_dir = tflite_path.replace('.tflite', '_saved_model')
    keras_model.export(saved_model_dir)

    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # Full integer quantization
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    if os.path.exists(calib_npz_path):
        calib_data = np.load(calib_npz_path)

        def representative_dataset_gen():
            num_samples = calib_data[input_keys[0]].shape[0]
            for i in range(min(num_samples, 200)):
                yield [calib_data[k][i:i+1].astype(np.float32) for k in input_keys]

        converter.representative_dataset = representative_dataset_gen
    else:
        print(f"  WARNING: {calib_npz_path} not found, using default quantization.")

    try:
        tflite_model = converter.convert()
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        size_kb = os.path.getsize(tflite_path) / 1024
        print(f"  ✅ Saved: {tflite_path} ({size_kb:.1f} KB)")
        return True
    except Exception as e:
        print(f"  Full INT8 failed ({e}), trying dynamic range fallback...")
        converter2 = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
        converter2.optimizations = [tf.lite.Optimize.DEFAULT]
        try:
            tflite_model = converter2.convert()
            with open(tflite_path, 'wb') as f:
                f.write(tflite_model)
            size_kb = os.path.getsize(tflite_path) / 1024
            print(f"  ✅ Saved (dynamic range): {tflite_path} ({size_kb:.1f} KB)")
            return True
        except Exception as e2:
            print(f"  ❌ Fallback also failed: {e2}")
            return False


# ════════════════════════════════════════════════════════════════════
#  Main
# ════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)

    from affective_head import TEMPORAL_DIM

    # Detect embed_dim from saved CNN weights
    cnn_path = "models/affective_cnn.pth"
    embed_dim = 256
    if os.path.exists(cnn_path):
        try:
            st = torch.load(cnn_path, map_location="cpu", weights_only=True)
            w = st.get("classifier.1.weight")
            if w is not None:
                embed_dim = int(w.shape[0])
        except Exception:
            pass
    print(f"Detected embed_dim: {embed_dim}")

    feature_dim = embed_dim + TEMPORAL_DIM
    print(f"TCN feature_dim: {feature_dim}  (embed={embed_dim} + temporal={TEMPORAL_DIM})")

    # ── 1. Affective CNN ──────────────────────────────────────────
    if os.path.exists(cnn_path):
        print("\n" + "=" * 60)
        print("  1. AFFECTIVE CNN")
        print("=" * 60)
        pt_state = torch.load(cnn_path, map_location="cpu", weights_only=True)
        keras_cnn = build_affective_cnn_keras(embed_dim)
        transfer_cnn_weights(keras_cnn, pt_state)
        convert_to_tflite(
            keras_cnn, "models/affective_cnn_int8.tflite",
            "calibration_data/affective_calibration.npz",
            input_keys=["input_face"],
            label="Affective CNN"
        )

    # ── 2. Stress TCN ─────────────────────────────────────────────
    tcn_path = "models/stress_tcn.pth"
    if os.path.exists(tcn_path):
        print("\n" + "=" * 60)
        print("  2. STRESS TCN")
        print("=" * 60)
        pt_state = torch.load(tcn_path, map_location="cpu", weights_only=True)
        keras_tcn = build_tcn_keras(seq_len=15, feature_dim=feature_dim)
        transfer_tcn_weights(keras_tcn, pt_state)
        convert_to_tflite(
            keras_tcn, "models/affective_tcn_int8.tflite",
            "calibration_data/affective_calibration.npz",
            input_keys=["input_temporal"],
            label="Stress TCN"
        )

    # ── 3. Gaze Hybrid ────────────────────────────────────────────
    gaze_path = "models/gaze_hybrid_epoch2.pth"
    if os.path.exists(gaze_path):
        print("\n" + "=" * 60)
        print("  3. GAZE HYBRID")
        print("=" * 60)
        pt_state = torch.load(gaze_path, map_location="cpu", weights_only=True)
        keras_gaze = build_gaze_keras(seq_len=15, input_dim=10, feature_dim=64, pose_dim=3)
        transfer_gaze_weights(keras_gaze, pt_state)
        convert_to_tflite(
            keras_gaze, "models/gaze_hybrid_int8.tflite",
            "calibration_data/gaze_calibration.npz",
            input_keys=["input_seq", "input_pose"],
            label="Gaze Hybrid"
        )

    print("\n" + "=" * 60)
    print("  ✅ TFLite Full Integer Quantization Pipeline COMPLETE")
    print("=" * 60)
