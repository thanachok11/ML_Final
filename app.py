import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# โหลดโมเดลที่บันทึกไว้
model = joblib.load("Ensemble_Model.pkl")

# โหลดข้อมูล CSV เพื่อแสดงในหน้าเว็บ
data = pd.read_csv('data.csv')

@app.route("/")
def home():
    # แปลงข้อมูลจาก CSV เป็น HTML เพื่อแสดงบนหน้าเว็บ
    data_html = data.to_html(classes='table table-bordered table-striped', index=False)
    return render_template('index.html', data_html=data_html)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    features = np.array(data["features"], dtype=float).reshape(1, -1)
    prediction = model.predict(features)[0]
    return jsonify({"prediction": int(prediction)})


if __name__ == "__main__":
    app.run(debug=True)
