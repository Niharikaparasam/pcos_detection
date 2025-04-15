from flask import Flask, send_from_directory, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np

app = Flask(__name__)
CORS(app)  # Allow all origins — fixes CORS error

# Serve frontend HTML
@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

# Prediction API endpoint
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    
    features = [
        data["Age"],
        data["BMI"],
        data["Cycle"],
        data["Weight_gain"],
        data["Hair_growth"],
        data["Skin_darkening"],
        data["Hair_loss"],
        data["Pimples"],
        data["TSH"],
        data["Follicle_L"],
        data["Follicle_R"]
    ]

    model = pickle.load(open("pcos_model.pkl", "rb"))
    prediction = model.predict([features])[0]

    return jsonify({'prediction': int(prediction)})

if __name__ == '__main__':
    app.run(debug=True)
