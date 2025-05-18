import tensorflow as tf
from flask import Flask, jsonify
import pandas as pd
import numpy as np
import requests
import time
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

app = Flask(__name__)

# Aggregator server URL
AGGREGATOR_URL = "http://127.0.0.1:5003/submit_weights"  # Adjust if needed

# === Error Metrics ===
def calculate_error_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2

# === Energy Usage ===
def measure_energy(model, X):
    start_time = time.time()
    model.predict(X)
    return time.time() - start_time

@app.route('/train', methods=['POST'])
def train():
    start_time = time.time()
    file_path = r"C:\Users\makam\OneDrive\Desktop\Fab\microservices\dataset\processed_dataset.csv"

    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        return jsonify({"error": f"Processed dataset not found at {file_path}"}), 404

    target_column = 'is_mal'
    if target_column not in df.columns:
        return jsonify({"error": "Target column 'is_mal' missing in the dataset"}), 400

    y = df[target_column].values
    X = df.drop(columns=[target_column]).values

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(X.shape[1],)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(X, y, epochs=5, batch_size=32, verbose=1)

    # === Evaluation ===
    y_pred_probs = model.predict(X)
    y_pred = (y_pred_probs > 0.5).astype(int)
    mae_val, rmse_val, r2_val = calculate_error_metrics(y, y_pred)
    energy = measure_energy(model, X)
    training_time = time.time() - start_time

    # === Serialize weights ===
    weights = model.get_weights()
    weights_list = [np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0).tolist() for w in weights]

    # === Submit to aggregator ===
    try:
        response = requests.post(AGGREGATOR_URL, json={"weights": weights_list})
        status_msg = "Training completed and weights submitted to aggregator" if response.status_code == 200 else "Training completed but failed to submit weights to aggregator"

        return jsonify({
            "status": status_msg,
            "metrics": {
                "training_time_sec": training_time,
                "inference_time_sec": energy,
                "mae": mae_val,
                "rmse": rmse_val,
                "r2_score": r2_val
            },
            "aggregator_response": response.json() if response.status_code == 200 else response.text,
            "http_status": response.status_code
        }), response.status_code if response.status_code != 200 else 200

    except requests.exceptions.Timeout:
        return jsonify({"status": "Training completed", "error": "Aggregator timeout"}), 504
    except requests.exceptions.ConnectionError:
        return jsonify({"status": "Training completed", "error": "Aggregator connection error"}), 503
    except requests.exceptions.RequestException as e:
        return jsonify({"status": "Training completed", "error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002)
