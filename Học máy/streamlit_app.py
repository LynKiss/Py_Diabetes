import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load the trained model
model = pickle.load(open('Học máy/diabetes_classifier.pkl', 'rb'))

# Custom CSS for better styling
st.markdown("""
<style>
    .main {
        background-color: #f0f8ff;
        font-family: 'Arial', sans-serif;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 10px 24px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 12px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .stSlider {
        color: #333;
    }
    .stSelectbox {
        color: #333;
    }
    .sidebar .sidebar-content {
        background-color: #e6f7ff;
    }
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #333;
        color: white;
        text-align: center;
        padding: 10px;
    }
</style>
""", unsafe_allow_html=True)



# Title with icon
st.title('🩺 Dự đoán Bệnh Tiểu Đường')
st.markdown("### Nhập thông tin bệnh nhân để dự đoán nguy cơ mắc bệnh tiểu đường.")

# Organize inputs into sections
st.header('📊 Thông tin bệnh nhân')

# Create input widgets for the 15 features used in training
BMI = st.slider('Chỉ số BMI', 10.0, 100.0, 25.0)
GenHlth = st.slider('Sức khỏe tổng thể (1-5)', 1, 5, 3)
MentHlth = st.slider('Sức khỏe tinh thần (số ngày kém trong tháng)', 0, 30, 0)
PhysHlth = st.slider('Sức khỏe thể chất (số ngày kém trong tháng)', 0, 30, 0)
Age = st.slider('Tuổi (1-13, tương ứng với nhóm tuổi)', 1, 13, 5)
Education = st.slider('Trình độ giáo dục (1-6)', 1, 6, 4)
Income = st.slider('Thu nhập (1-8)', 1, 8, 5)
HighBP = st.selectbox('Huyết áp cao', [0, 1], format_func=lambda x: 'Không' if x == 0 else 'Có')
HighChol = st.selectbox('Cholesterol cao', [0, 1], format_func=lambda x: 'Không' if x == 0 else 'Có')
Smoker = st.selectbox('Hút thuốc', [0, 1], format_func=lambda x: 'Không' if x == 0 else 'Có')
Stroke = st.selectbox('Đột quỵ', [0, 1], format_func=lambda x: 'Không' if x == 0 else 'Có')
HeartDiseaseorAttack = st.selectbox('Bệnh tim hoặc đau tim', [0, 1], format_func=lambda x: 'Không' if x == 0 else 'Có')
PhysActivity = st.selectbox('Hoạt động thể chất', [0, 1], format_func=lambda x: 'Không' if x == 0 else 'Có')
HvyAlcoholConsump = st.selectbox('Uống rượu nặng', [0, 1], format_func=lambda x: 'Không' if x == 0 else 'Có')
DiffWalk = st.selectbox('Khó đi bộ', [0, 1], format_func=lambda x: 'Không' if x == 0 else 'Có')

# Prediction button
st.markdown("---")
if st.button('🔍 Dự đoán'):
    # Prepare the input data for the 15 features
    input_data = np.array([[BMI, GenHlth, MentHlth, PhysHlth, Age, Education, Income, HighBP, HighChol, Smoker, Stroke, HeartDiseaseorAttack, PhysActivity, HvyAlcoholConsump, DiffWalk]])

    # Make prediction
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)

    # Display result with better formatting
    st.markdown("### 📋 Kết quả dự đoán")
    if prediction[0] == 0:
        st.success('✅ Không có nguy cơ mắc bệnh tiểu đường.')
    else:
        st.error('⚠️ Có nguy cơ mắc bệnh tiểu đường.')

    st.write(f'**Xác suất dự đoán:** Không tiểu đường: {prediction_proba[0][0]*100:.2f}%, Tiểu đường: {prediction_proba[0][1]*100:.2f}%')


