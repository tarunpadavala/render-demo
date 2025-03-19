import os
import numpy as np
import torch
import xgboost as xgb
import json
from flask import Flask, request, jsonify

app = Flask(__name__)  # ✅ Fix: Gunicorn needs an `app` object

# ✅ Load PyTorch Model Once (Globally)
MODEL_PATH = os.path.join(os.getcwd(), "xg_model.ckpt")

try:
    model = torch.load(MODEL_PATH, weights_only=False)
    model.eval()
except Exception as e:
    model = None
    print(f"Error loading model: {e}")

@app.route('/predict_twitter', methods=['POST'])
def predict_twitter():
    try:
        data = request.get_json()

        input_features = np.array([[
            int(data['sex_code']), int(data['statuses_count']), int(data['followers_count']),
            int(data['friends_count']), int(data['favourites_count']), int(data['listed_count']),
            int(data['lang_code'])
        ]])
        input_tensor = torch.tensor(input_features, dtype=torch.float32)

        if model is None:
            return jsonify({"error": "Model failed to load"}), 500

        with torch.no_grad():
            prediction = model(input_tensor)

        result = int(torch.round(prediction).item())

        return jsonify({'prediction': "Fake" if result == 1 else "Not Fake"})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8000)
