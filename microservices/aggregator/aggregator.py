# from flask import Flask, request, jsonify
# import numpy as np
# import requests
# import time
# import psutil
# import os
# import pickle

# app = Flask(__name__)

# # Global storage for client weights
# client_weights = []
# AGGREGATED_MODEL_PATH = "models/global_model.pkl"

# @app.route('/submit_weights', methods=['POST'])
# def submit_weights():
#     """
#     Endpoint to receive weights from clients.
#     """
#     global client_weights
#     start_time = time.time()

#     try:
#         data = request.get_json()
#         if not data:
#             return jsonify({"status": "failed", "error": "No weights received"}), 400

#         weights = data["weights"] if isinstance(data, dict) and "weights" in data else data

#         if not isinstance(weights, list):
#             return jsonify({"status": "failed", "error": "Weights must be a list"}), 400

#         client_weights.append([np.array(layer, dtype=np.float32) for layer in weights])
#         latency = round(time.time() - start_time, 4)

#         return jsonify({"status": "success", "message": "Weights received successfully", "latency_sec": latency})

#     except Exception as e:
#         return jsonify({"status": "failed", "error": str(e)}), 400


# @app.route('/aggregate', methods=['GET'])
# def aggregate_weights():
#     """
#     Aggregate all received weights using layer-wise averaging.
#     """
#     global client_weights

#     if not client_weights:
#         return jsonify({"status": "failed", "error": "No weights to aggregate"}), 400

#     try:
#         start_time = time.time()
#         aggregated_weights = []

#         for layer_idx in range(len(client_weights[0])):
#             layer_stack = np.stack([client[layer_idx] for client in client_weights], axis=0)
#             aggregated_layer = np.mean(layer_stack, axis=0)
#             aggregated_weights.append(aggregated_layer)

#         # Save aggregated weights as .pkl file
#         save_aggregated_model(aggregated_weights)

#         # Convert to JSON serializable format
#         aggregated_weights_list = [layer.tolist() for layer in aggregated_weights]

#         # Clear stored weights
#         client_weights = []

#         aggregation_latency = round(time.time() - start_time, 4)
#         model_size = get_model_size(AGGREGATED_MODEL_PATH)
#         system_metrics = get_system_resources()

#         return jsonify({
#             "status": "success",
#             "global_weights": aggregated_weights_list,
#             "aggregation_latency_sec": aggregation_latency,
#             "model_size_MB": model_size,
#             "system_resources": system_metrics
#         })

#     except Exception as e:
#         return jsonify({"status": "failed", "error": str(e)}), 500


# @app.route('/reset_weights', methods=['GET'])
# def reset_weights():
#     """
#     Resets the stored client weights.
#     """
#     global client_weights
#     client_weights = []
#     return jsonify({"status": "success", "message": "Weights have been reset."})


# def save_aggregated_model(weights):
#     """
#     Save the aggregated model weights as a .pkl file.
#     """
#     os.makedirs("models", exist_ok=True)
#     with open(AGGREGATED_MODEL_PATH, "wb") as f:
#         pickle.dump(weights, f)


# def get_model_size(filepath):
#     """
#     Returns the file size of the saved model.
#     """
#     return round(os.path.getsize(filepath) / (1024 * 1024), 4) if os.path.exists(filepath) else 0.0


# def get_system_resources():
#     """
#     Returns current CPU and memory usage.
#     """
#     return {
#         "cpu_usage_percent": psutil.cpu_percent(interval=1),
#         "memory_usage_percent": psutil.virtual_memory().percent
#     }


# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=5003, threaded=True)

# from flask import Flask, request, jsonify
# import numpy as np
# import pickle
# import os
# import time
# import psutil

# app = Flask(__name__)

# client_weights = []
# AGGREGATED_MODEL_PATH = "models/global_model.pkl"


# @app.route('/submit_weights', methods=['POST'])
# def submit_weights():
#     start_time = time.time()
#     try:
#         data = request.get_json()
#         if not data or "weights" not in data:
#             return jsonify({"status": "failed", "error": "No weights received"}), 400

#         weights = data["weights"]
#         client_weights.append([np.array(layer, dtype=np.float32) for layer in weights])
#         latency = round(time.time() - start_time, 4)

#         return jsonify({
#             "status": "success",
#             "message": "Weights received successfully",
#             "latency_sec": latency
#         })
#     except Exception as e:
#         return jsonify({"status": "failed", "error": str(e)}), 400


# @app.route('/aggregate', methods=['GET'])
# def aggregate_weights():
#     if not client_weights:
#         return jsonify({"status": "failed", "error": "No weights to aggregate"}), 400
#     try:
#         start_time = time.time()
#         aggregated_weights = []

#         for i in range(len(client_weights[0])):
#             layer_stack = np.stack([client[i] for client in client_weights])
#             aggregated_layer = np.mean(layer_stack, axis=0)
#             aggregated_weights.append(aggregated_layer)

#         save_aggregated_model(aggregated_weights)
#         aggregated_weights_list = [layer.tolist() for layer in aggregated_weights]
#         client_weights.clear()

#         latency = round(time.time() - start_time, 4)
#         model_size = get_model_size(AGGREGATED_MODEL_PATH)
#         system_metrics = get_system_resources()

#         return jsonify({
#             "status": "success",
#             "global_weights": aggregated_weights_list,
#             "aggregation_latency_sec": latency,
#             "model_size_MB": model_size,
#             "system_resources": system_metrics
#         })
#     except Exception as e:
#         return jsonify({"status": "failed", "error": str(e)}), 500


# @app.route('/reset_weights', methods=['GET'])
# def reset_weights():
#     global client_weights
#     client_weights = []
#     return jsonify({"status": "success", "message": "Client weights reset."})


# def save_aggregated_model(weights):
#     os.makedirs("models", exist_ok=True)
#     with open(AGGREGATED_MODEL_PATH, "wb") as f:
#         pickle.dump(weights, f)


# def get_model_size(path):
#     return round(os.path.getsize(path) / (1024 * 1024), 4) if os.path.exists(path) else 0.0


# def get_system_resources():
#     return {
#         "cpu_usage_percent": psutil.cpu_percent(interval=1),
#         "memory_usage_percent": psutil.virtual_memory().percent
#     }


# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=5003, threaded=True)


from flask import Flask, request, jsonify
import numpy as np
import pickle
import os
import time
import psutil

app = Flask(__name__)

# Instead of a list, keep just one client's weights
client_weights = None
AGGREGATED_MODEL_PATH = "models/global_model.pkl"


EXPECTED_SHAPES = [
    (70, 128), (128,),   # First Dense layer (weights, bias)
    (128, 64), (64,),    # Second Dense layer
    (64, 1), (1,)        # Output layer
]


@app.route('/submit_weights', methods=['POST'])
def submit_weights():
    global client_weights
    start_time = time.time()
    try:
        data = request.get_json()
        if not data or "weights" not in data:
            return jsonify({"status": "failed", "error": "No weights received"}), 400

        weights = data["weights"]
        np_weights = [np.array(layer, dtype=np.float32) for layer in weights]

        # Check shape validity
        if len(np_weights) != len(EXPECTED_SHAPES):
            return jsonify({"status": "failed", "error": "Incorrect number of weight arrays"}), 400
        for w, expected_shape in zip(np_weights, EXPECTED_SHAPES):
            if w.shape != expected_shape:
                return jsonify({"status": "failed", 
                                "error": f"Weight shape mismatch. Expected {expected_shape} but got {w.shape}"}), 400

        client_weights = np_weights
        latency = round(time.time() - start_time, 4)

        return jsonify({
            "status": "success",
            "message": "Weights received successfully",
            "latency_sec": latency
        })
    except Exception as e:
        return jsonify({"status": "failed", "error": str(e)}), 400



@app.route('/aggregate', methods=['GET'])
def aggregate_weights():
    if client_weights is None:
        return jsonify({"status": "failed", "error": "No weights to aggregate"}), 400
    try:
        start_time = time.time()
        # Since only one client, no averaging needed
        aggregated_weights = client_weights

        save_aggregated_model(aggregated_weights)
        aggregated_weights_list = [layer.tolist() for layer in aggregated_weights]

        latency = round(time.time() - start_time, 4)
        model_size = get_model_size(AGGREGATED_MODEL_PATH)
        system_metrics = get_system_resources()

        return jsonify({
            "status": "success",
            "global_weights": aggregated_weights_list,
            "aggregation_latency_sec": latency,
            "model_size_MB": model_size,
            "system_resources": system_metrics
        })
    except Exception as e:
        return jsonify({"status": "failed", "error": str(e)}), 500


@app.route('/reset_weights', methods=['GET'])
def reset_weights():
    global client_weights
    client_weights = None
    return jsonify({"status": "success", "message": "Client weights reset."})


def save_aggregated_model(weights):
    os.makedirs("models", exist_ok=True)
    with open(AGGREGATED_MODEL_PATH, "wb") as f:
        pickle.dump(weights, f)


def get_model_size(path):
    return round(os.path.getsize(path) / (1024 * 1024), 4) if os.path.exists(path) else 0.0


def get_system_resources():
    return {
        "cpu_usage_percent": psutil.cpu_percent(interval=1),
        "memory_usage_percent": psutil.virtual_memory().percent
    }


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5003, threaded=True)
