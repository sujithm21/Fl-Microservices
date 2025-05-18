# import tensorflow as tf
# from flask import Flask, request, jsonify
# import numpy as np
# import requests
# import os
# import time
# import psutil

# app = Flask(__name__)
# app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024

# MODEL_SAVE_PATH = "models/global_model.h5"
# AGGREGATOR_URL = "http://localhost:5003/aggregate"
# NUM_FEATURES = 63  # Adjust as per your model input

# os.makedirs("models", exist_ok=True)


# @app.route('/update_model', methods=['POST', 'GET'])  # accept GET too for testing
# def update_model():
#     """
#     Fetch aggregated weights from the Aggregator and update the global Keras model.
#     """
#     try:
#         start_time = time.time()

#         # Call aggregator to get aggregated weights
#         response = requests.get(AGGREGATOR_URL)
#         print(f"[Updater] Aggregator response status: {response.status_code}")
#         print(f"[Updater] Aggregator response text (truncated): {response.text[:500]}")

#         if response.status_code != 200:
#             return jsonify({"status": "failed", "error": "Failed to fetch weights from aggregator"}), 400

#         try:
#             response_json = response.json()
#         except Exception as e:
#             return jsonify({"status": "failed", "error": f"Invalid JSON from aggregator: {str(e)}"}), 400

#         weights = response_json.get('global_weights')
#         if not weights:
#             return jsonify({"status": "failed", "error": "No weights received from aggregator"}), 400

#         # Convert weights to numpy arrays and print their shapes for debugging
#         np_weights = []
#         for i, w in enumerate(weights):
#             arr = np.array(w)
#             print(f"[Updater] Weight layer {i} shape: {arr.shape}")
#             np_weights.append(arr)

#         # Build model architecture (must match training model)
#         model = tf.keras.Sequential([
#             tf.keras.layers.Dense(128, activation='relu', input_shape=(NUM_FEATURES,)),
#             tf.keras.layers.Dense(64, activation='relu'),
#             tf.keras.layers.Dense(1, activation='sigmoid')
#         ])

#         # Check that shapes match model's expected weights shapes
#         model_weight_shapes = [w.shape for w in model.get_weights()]
#         incoming_weight_shapes = [w.shape for w in np_weights]

#         print(f"[Updater] Model expected weight shapes: {model_weight_shapes}")
#         print(f"[Updater] Incoming weights shapes: {incoming_weight_shapes}")

#         if model_weight_shapes != incoming_weight_shapes:
#             return jsonify({
#                 "status": "failed",
#                 "error": "Weight shapes from aggregator do not match model architecture",
#                 "model_shapes": [str(s) for s in model_weight_shapes],
#                 "received_shapes": [str(s) for s in incoming_weight_shapes]
#             }), 400

#         model.set_weights(np_weights)
#         model.save(MODEL_SAVE_PATH)

#         # Monitoring
#         model_size_MB = os.path.getsize(MODEL_SAVE_PATH) / (1024 * 1024)
#         update_time = time.time() - start_time
#         memory_usage_MB = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
#         cpu_usage_percent = psutil.cpu_percent(interval=1)

#         return jsonify({
#             "status": "Model updated and saved successfully!",
#             "model_size_MB": model_size_MB,
#             "update_time_sec": update_time,
#             "memory_usage_MB": memory_usage_MB,
#             "cpu_usage_percent": cpu_usage_percent
#         })

#     except Exception as e:
#         return jsonify({"status": "failed", "error": str(e)}), 500


# @app.route('/predict', methods=['POST'])
# def predict():
#     """
#     Predict the output from input feature vector using the global model.
#     """
#     try:
#         data = request.json.get('data')
#         if not data or not isinstance(data, dict):
#             return jsonify({"status": "failed", "error": "Invalid or missing input data"}), 400

#         # Adjust your expected keys here — only input features, no labels
#         expected_keys = [
#             'flow_duration', 'Header_Length', 'Source Port', 'Destination Port', 'Protocol Type',
#             'Duration', 'Rate', 'Srate', 'Drate', 'fin_flag_number',
#             # Add remaining feature keys here (make sure total is NUM_FEATURES)
#             # For example:
#             # 'feature_11', 'feature_12', ... , 'feature_63'
#         ]

#         # Check all keys are present
#         missing_keys = [k for k in expected_keys if k not in data]
#         if missing_keys:
#             return jsonify({"status": "failed", "error": f"Missing features: {missing_keys}"}), 400

#         input_data = [data[key] for key in expected_keys]

#         model = tf.keras.models.load_model(MODEL_SAVE_PATH)
#         start_time = time.time()
#         prediction = model.predict(np.array([input_data])).tolist()
#         latency = time.time() - start_time

#         return jsonify({
#             "predictions": prediction,
#             "prediction_latency_sec": latency
#         })

#     except Exception as e:
#         return jsonify({"status": "failed", "error": str(e)}), 500


# @app.route('/get_model_size', methods=['GET'])
# def get_model_size():
#     """
#     Returns the size of the current global model.
#     """
#     try:
#         size_MB = os.path.getsize(MODEL_SAVE_PATH) / (1024 * 1024)
#         return jsonify({"model_size_MB": size_MB})
#     except FileNotFoundError:
#         return jsonify({"error": "Global model not found"}), 404


# @app.route('/health', methods=['GET'])
# def health_check():
#     """
#     Simple health check for the updater service.
#     """
#     return jsonify({"status": "running", "message": "Updater service is healthy"}), 200


# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=5004)

import tensorflow as tf
from flask import Flask, request, jsonify
import numpy as np
import requests
import os
import time
import psutil

app = Flask(__name__)
MODEL_SAVE_PATH = "models/global_model.h5"
AGGREGATOR_URL = "http://localhost:5003/aggregate"
NUM_FEATURES = 70


os.makedirs("models", exist_ok=True)


@app.route('/update_model', methods=['GET', 'POST'])
def update_model():
    try:
        start_time = time.time()

        response = requests.get(AGGREGATOR_URL)
        if response.status_code != 200:
            return jsonify({"status": "failed", "error": "Aggregator error"}), 400

        data = response.json()
        weights = data.get("global_weights")
        if not weights:
            return jsonify({"status": "failed", "error": "Empty weights"}), 400

        np_weights = [np.array(layer, dtype=np.float32) for layer in weights]

        # Define same architecture
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=(NUM_FEATURES,)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

        if [w.shape for w in model.get_weights()] != [w.shape for w in np_weights]:
            return jsonify({"status": "failed", "error": "Weight shapes mismatch"}), 400

        model.set_weights(np_weights)
        model.save(MODEL_SAVE_PATH)

        latency = round(time.time() - start_time, 4)
        model_size = os.path.getsize(MODEL_SAVE_PATH) / (1024 * 1024)
        memory = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
        cpu = psutil.cpu_percent(interval=1)

        return jsonify({
            "status": "success",
            "model_size_MB": model_size,
            "update_time_sec": latency,
            "memory_usage_MB": memory,
            "cpu_usage_percent": cpu
        })
    except Exception as e:
        return jsonify({"status": "failed", "error": str(e)}), 500


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data or 'data' not in data:
            return jsonify({"status": "failed", "error": "Invalid input"}), 400

        features = data['data']

        # Make sure all features are provided
        if len(features) != NUM_FEATURES:
            return jsonify({"status": "failed", "error": "Incorrect feature count"}), 400

        model = tf.keras.models.load_model(MODEL_SAVE_PATH)
        prediction_start = time.time()
        prediction = model.predict(np.array([features]))[0][0]
        prediction_latency = round(time.time() - prediction_start, 4)

        # Threshold the prediction to get binary output (0 or 1)
        binary_prediction = 1 if prediction > 0.5 else 0

        return jsonify({
            "prediction": binary_prediction,
            "raw_score": float(prediction),
            "prediction_latency_sec": prediction_latency
        })
    except Exception as e:
        return jsonify({"status": "failed", "error": str(e)}), 500



@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "running"}), 200


@app.route('/get_model_size', methods=['GET'])
def get_model_size():
    try:
        size_MB = os.path.getsize(MODEL_SAVE_PATH) / (1024 * 1024)
        return jsonify({"model_size_MB": size_MB})
    except FileNotFoundError:
        return jsonify({"error": "Model not found"}), 404


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5004)

