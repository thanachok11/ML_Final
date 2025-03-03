import streamlit as st
import pandas as pd
import requests

# Title ของแอป
st.title("Employee Status Prediction")

# ใช้ st.cache_data แทน st.cache
@st.cache_data
def load_data():
    return pd.read_csv('data.csv')  # แก้ไข path ให้เป็นที่อยู่จริงของไฟล์ CSV

data = load_data()
st.sidebar.subheader("Employee Data")
st.sidebar.write(data)  # แสดงข้อมูล CSV

# Form สำหรับการทำนาย
st.subheader("Prediction Input Form")

age = st.number_input("Age", min_value=18, max_value=100)
length_of_service = st.number_input("Length of Service", min_value=0, max_value=50)
salary = st.number_input("Salary", min_value=0)
gender = st.radio("Gender", ["Male", "Female"])
marital_status = st.radio("Marital Status", ["Single", "Married"])

# แปลงข้อมูลให้เป็นตัวเลข
gender = 0 if gender == "Male" else 1
marital_status = 0 if marital_status == "Single" else 1
input_features = [age, length_of_service, salary, gender, marital_status]

# ปุ่มกดเพื่อทำนาย
if st.button("Predict"):
    response = requests.post("http://127.0.0.1:10000/predict", json={"features": input_features})
    result = response.json()

    # แปลงค่าผลลัพธ์เป็นข้อความที่เข้าใจง่าย
    prediction_label = "ยังคงทำงาน" if result["prediction"] == 0 else "ลาออกจากงาน"

    st.write(f"Prediction: {prediction_label}")

# ตั้งค่า CSS สำหรับการตกแต่ง
st.markdown("""
    <style>
        .main {
            display: flex;
            flex-direction: row;
        }
        .sidebar {
            width: 30%;
            padding: 20px;
            background-color: #f0f0f5;
            border-radius: 10px;
        }
        .content {
            width: 65%;
            padding: 20px;
            margin-left: 20px;
        }
        .prediction-input {
            background-color: #d9f7be;
            padding: 15px;
            border-radius: 10px;
        }
        .prediction-output {
            background-color: #f4f4f9;
            padding: 15px;
            border-radius: 10px;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)
