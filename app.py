import joblib
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

# โหลดโมเดลที่บันทึกไว้
model = joblib.load("Ensemble_Model.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    features = np.array(data["features"]).reshape(1, -1)
    prediction = model.predict(features)[0]
    return jsonify({"prediction": int(prediction)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
@app.route('/')
def home():
    return "API is running"
