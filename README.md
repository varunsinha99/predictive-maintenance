# Smart Predictive Maintenance System

## What is this?
I built this for my B.Tech minor project. The core idea is simple: instead of waiting for manufacturing equipment to break down, this system analyzes sensor data (like pressure, temperature, and vibration) to predict machine failures *before* they happen. 

## How it works
I kept the architecture straightforward. It's a machine learning pipeline broken down into three main parts:
1. **Data & Training:** `generate_and_train.py` creates the synthetic industrial dataset and trains the ML model.
2. **The Brain:** The trained model and data scaler are saved as `model.pkl` and `scaler.pkl`.
3. **The Application:** `app.py` serves the model, taking live machine parameters as input and returning a prediction.

## Tech Stack
* **Language:** Python
* **Machine Learning:** Scikit-Learn, Pandas, NumPy
* **Backend:** [Flask / Streamlit - *Edit this based on what app.py uses*]

## Run it on your machine
Want to test it out locally?

1. Clone this repository:
   ```bash
   git clone [https://github.com/varunsinha99/predictive-maintenance.git](https://github.com/varunsinha99/predictive-maintenance.git)
   cd predictive-maintenance
