from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pickle
import os
import numpy as np

app = Flask(__name__)
CORS(app)  # ✅ Fixes CORS issues

# ✅ Load the trained model
model_path = os.path.join(os.path.dirname(__file__), "pcos_model.pkl")
with open(model_path, "rb") as f:
    model = pickle.load(f)

@app.route("/")
def home():
    return render_template("index.html")  # ✅ Serves the frontend page

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # ✅ Extract features from JSON request
        features = np.array([
            int(data["Age"]), float(data["BMI"]), int(data["Cycle"]),
            int(data["Weight_Gain"]), int(data["Hair_Growth"]),
            int(data["Skin_Darkening"]), int(data["Hair_Loss"]),
            int(data["Pimples"]), float(data["TSH"]),
            int(data["Follicle_L"]), int(data["Follicle_R"])
        ]).reshape(1, -1)

        # ✅ Make prediction
        prediction = model.predict(features)[0]
        result = "PCOS Detected" if prediction == 1 else "No PCOS"

        return jsonify({"prediction": result})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)
