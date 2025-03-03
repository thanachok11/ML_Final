import joblib
import numpy as np
from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)

# โหลดโมเดลที่บันทึกไว้
model = joblib.load("Ensemble_Model.pkl")

@app.route("/")
def home():
    # แสดงปุ่ม "เริ่ม app" บนหน้าเว็บ
    return render_template_string("""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>เริ่มแอป</title>
        </head>
        <body>
            <h1>Welcome to the Prediction App!</h1>
            <button onclick="startApp()">เริ่มแอป</button>
            <script>
                function startApp() {
                    // ทำการส่งคำขอ POST ไปที่ API /predict เมื่อกดปุ่ม
                    fetch('/predict', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            "features": [30, 5, 50000, 1, 0]  // ตัวอย่างข้อมูลที่ต้องการทำนาย
                        })
                    })
                    .then(response => response.json())
                    .then(data => {
                        alert("Prediction: " + (data.prediction === 0 ? "ยังคงทำงาน" : "ลาออกจากงาน"));
                    });
                }
            </script>
        </body>
        </html>
    """)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    features = np.array(data["features"]).reshape(1, -1)
    prediction = model.predict(features)[0]
    return jsonify({"prediction": int(prediction)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
