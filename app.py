"""
Smart Predictive Maintenance System - Streamlit Web Application
Real-time predictions with interactive input controls and visual feedback.
"""

import streamlit as st
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import time

# Configure Streamlit page
st.set_page_config(
    page_title="Smart Predictive Maintenance",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5em;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 10px;
    }
    .subtitle {
        font-size: 1.1em;
        color: #666;
        text-align: center;
        margin-bottom: 30px;
    }
    .metric-box {
        padding: 20px;
        border-radius: 10px;
        background-color: #f0f2f6;
        margin: 10px 0;
    }
    .status-healthy {
        padding: 20px;
        border-radius: 10px;
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        text-align: center;
        font-size: 1.3em;
        font-weight: bold;
        color: #155724;
    }
    .status-at-risk {
        padding: 20px;
        border-radius: 10px;
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        text-align: center;
        font-size: 1.3em;
        font-weight: bold;
        color: #856404;
    }
    .status-failed {
        padding: 20px;
        border-radius: 10px;
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
        text-align: center;
        font-size: 1.3em;
        font-weight: bold;
        color: #721c24;
    }
    .prediction-time {
        text-align: right;
        color: #999;
        font-size: 0.9em;
        margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# Load model and scaler
@st.cache_resource
def load_model_and_scaler():
    """Load trained model and scaler from pickle files."""
    try:
        model = joblib.load('model.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except FileNotFoundError as e:
        st.error(f"❌ Error: Could not find model files. Please run 'python generate_and_train.py' first.")
        st.stop()

# Load model and scaler
model, scaler = load_model_and_scaler()

# Header
st.markdown('<div class="main-header">⚙️ Smart Predictive Maintenance System</div>', 
            unsafe_allow_html=True)
st.markdown('<div class="subtitle">Real-time Machine Health Monitoring & Predictions</div>', 
            unsafe_allow_html=True)

# Sidebar for input controls
st.sidebar.markdown("### 📊 Input Parameters")
st.sidebar.markdown("Adjust the sliders below to simulate different operating conditions:")

# Input sliders on sidebar
air_temp = st.sidebar.slider(
    'Air Temperature (K)',
    min_value=290.0,
    max_value=315.0,
    value=300.0,
    step=0.5,
    help="Ambient air temperature in Kelvin"
)

process_temp = st.sidebar.slider(
    'Process Temperature (K)',
    min_value=295.0,
    max_value=330.0,
    value=310.0,
    step=0.5,
    help="Operating process temperature in Kelvin"
)

rotational_speed = st.sidebar.slider(
    'Rotational Speed (RPM)',
    min_value=1000.0,
    max_value=2500.0,
    value=1500.0,
    step=50.0,
    help="Machine rotational speed in RPM"
)

torque = st.sidebar.slider(
    'Torque (Nm)',
    min_value=3.0,
    max_value=80.0,
    value=40.0,
    step=1.0,
    help="Applied torque in Newton-meters"
)

tool_wear = st.sidebar.slider(
    'Tool Wear (minutes)',
    min_value=0.0,
    max_value=250.0,
    value=100.0,
    step=5.0,
    help="Cumulative tool wear time in minutes"
)

# Prediction section
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 🔍 Real-Time Prediction")
    
    # Prepare input data for prediction
    input_data = np.array([[air_temp, process_temp, rotational_speed, torque, tool_wear]])
    
    # Scale the input
    input_data_scaled = scaler.transform(input_data)
    
    # Make prediction
    prediction = model.predict(input_data_scaled)[0]
    prediction_proba = model.predict_proba(input_data_scaled)[0]
    
    # Display status with color coding
    if prediction == 'Healthy':
        st.markdown(
            f'<div class="status-healthy">✅ {prediction}</div>',
            unsafe_allow_html=True
        )
        status_color = "🟢"
    elif prediction == 'At Risk':
        st.markdown(
            f'<div class="status-at-risk">⚠️ {prediction}</div>',
            unsafe_allow_html=True
        )
        status_color = "🟡"
    else:  # Failed
        st.markdown(
            f'<div class="status-failed">🔴 {prediction}</div>',
            unsafe_allow_html=True
        )
        status_color = "🔴"
    
    # Display confidence scores
    st.markdown("#### Prediction Confidence")
    class_names = ['At Risk', 'Failed', 'Healthy']
    
    for class_name, prob in zip(model.classes_, prediction_proba):
        col_left, col_right = st.columns([3, 1])
        with col_left:
            st.progress(prob, text=f"{class_name}")
        with col_right:
            st.metric("", f"{prob*100:.1f}%", label_visibility="collapsed")

with col2:
    st.markdown("### 📈 Quick Stats")
    
    # Display key metrics
    st.metric(
        "Temp Delta (°C)",
        f"{process_temp - air_temp:.1f}",
        help="Process - Air temperature difference"
    )
    st.metric(
        "Tool Wear %",
        f"{(tool_wear/250)*100:.1f}%",
        help="Percentage of tool lifetime used"
    )
    st.metric(
        "Speed Status",
        "Normal" if 1200 <= rotational_speed <= 1800 else "Anomaly",
        help="RPM within normal operating range"
    )

# Current input values section
st.markdown("### 📋 Current Input Values")

input_df = pd.DataFrame({
    'Parameter': ['Air Temperature', 'Process Temperature', 'Rotational Speed', 'Torque', 'Tool Wear'],
    'Value': [f"{air_temp:.1f} K", f"{process_temp:.1f} K", f"{rotational_speed:.1f} RPM", 
              f"{torque:.1f} Nm", f"{tool_wear:.1f} min"],
    'Range': ['290-315 K', '295-330 K', '1000-2500 RPM', '3-80 Nm', '0-250 min']
})

st.dataframe(input_df, use_container_width=True, hide_index=True)

# Prediction timestamp
st.markdown(
    f'<div class="prediction-time">Last updated: {datetime.now().strftime("%H:%M:%S")}</div>',
    unsafe_allow_html=True
)

# Footer with instructions
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.9em;'>
    💡 <strong>Tip:</strong> Adjust the sliders to see how different operating conditions affect machine health predictions.
    </div>
""", unsafe_allow_html=True)

# Sidebar info
st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ About This System")
st.sidebar.markdown("""
    - **Model**: RandomForestClassifier (100 estimators)
    - **Features**: 5 machine parameters
    - **Classes**: Healthy, At Risk, Failed
    - **Preprocessing**: MinMax scaling [0, 1]
    
    Real-time predictions based on AI4I 2020 dataset patterns.
""")