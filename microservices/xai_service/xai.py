# from flask import Flask, jsonify, request
# import shap
# import pandas as pd
# import os
# import numpy as np
# from tensorflow.keras.models import load_model

# app = Flask(__name__)

# # Absolute path to your model file
# MODEL_PATH = r"C:\Users\makam\OneDrive\Desktop\Fab\microservices\global_model_updator\models\global_model.h5"

# def convert_to_serializable(obj):
#     if isinstance(obj, list):
#         return [convert_to_serializable(o) for o in obj]
#     elif hasattr(obj, "tolist"):
#         return obj.tolist()
#     else:
#         return obj

# @app.route('/explain', methods=['POST'])
# def explain():
#     try:
#         input_json = request.json
#         if not input_json or 'data' not in input_json:
#             return jsonify({"status": "failed", "error": "No data provided"}), 400

#         input_data = input_json['data']
#         data = pd.DataFrame(input_data)

#         if not os.path.isfile(MODEL_PATH):
#             return jsonify({"status": "failed", "error": f"Model file not found at {MODEL_PATH}"}), 500

#         model = load_model(MODEL_PATH)

#         # Wrapper function to flatten model predictions
#         def model_predict(x):
#             preds = model.predict(x)
#             if preds.shape[1] == 1:
#                 return preds.flatten()
#             return preds

#         # Use a sample of the data as background for SHAP
#         background = data.sample(n=min(100, len(data)), random_state=42)

#         # Create DeepExplainer
#         explainer = shap.DeepExplainer(model, background.values)

#         # Compute SHAP values for the input data
#         shap_values = explainer.shap_values(data.values)

#         shap_values_serializable = convert_to_serializable(shap_values)

#         return jsonify({
#             "status": "success",
#             "shap_values": shap_values_serializable
#         })

#     except Exception as e:
#         return jsonify({"status": "failed", "error": str(e)}), 500

# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=5005)

from flask import Flask, jsonify, request
import shap
import pandas as pd
from tensorflow.keras.models import load_model
import numpy as np
import os

app = Flask(__name__)

MODEL_PATH = r"C:\Users\makam\OneDrive\Desktop\Fab\microservices\global_model_updator\models\global_model.h5"

def convert_to_serializable(obj):
    if isinstance(obj, list):
        return [convert_to_serializable(o) for o in obj]
    elif hasattr(obj, "tolist"):
        return obj.tolist()
    else:
        return obj

@app.route('/explain', methods=['POST'])
def explain():
    try:
        input_json = request.json
        if not input_json or 'data' not in input_json:
            return jsonify({"status": "failed", "error": "No data provided"}), 400

        data = pd.DataFrame(input_json['data'])

        if not os.path.isfile(MODEL_PATH):
            return jsonify({"status": "failed", "error": f"Model file not found at {MODEL_PATH}"}), 500

        model = load_model(MODEL_PATH)

        # KernelExplainer needs a background dataset - sample from input or create a baseline
        background = data.sample(n=min(100, len(data)), random_state=42) if len(data) > 1 else data

        # Create KernelExplainer with model.predict function and background data
        explainer = shap.KernelExplainer(model.predict, background)

        # Calculate SHAP values for input data
        shap_values = explainer.shap_values(data)

        shap_values_serializable = convert_to_serializable(shap_values)

        return jsonify({
            "status": "success",
            "shap_values": shap_values_serializable
        })

    except Exception as e:
        return jsonify({"status": "failed", "error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5005)
