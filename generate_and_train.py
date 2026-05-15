"""
Generate synthetic AI4I 2020 dataset and train RandomForestClassifier model.
Saves trained model and fitted scaler for use in Streamlit app.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

print("=" * 60)
print("GENERATING SYNTHETIC AI4I 2020 DATASET")
print("=" * 60)

# Generate synthetic dataset with 10,000 rows
n_samples = 10000

# Feature generation based on realistic machine operating conditions
air_temperature_k = np.random.normal(loc=300, scale=2, size=n_samples)  # Around 27°C
process_temperature_k = np.random.normal(loc=310, scale=5, size=n_samples)  # Around 37°C
rotational_speed_rpm = np.random.normal(loc=1500, scale=200, size=n_samples)
torque_nm = np.random.normal(loc=40, scale=10, size=n_samples)
tool_wear_min = np.random.uniform(0, 250, size=n_samples)

# Ensure all values are positive and within realistic ranges
air_temperature_k = np.clip(air_temperature_k, 290, 315)
process_temperature_k = np.clip(process_temperature_k, 295, 330)
rotational_speed_rpm = np.clip(rotational_speed_rpm, 1000, 2500)
torque_nm = np.clip(torque_nm, 3, 80)

# Generate target based on feature combinations
health_status = []
for i in range(n_samples):
    # Define health conditions based on feature values
    temp_stress = (process_temperature_k[i] - 310) ** 2 / 100
    speed_stress = (rotational_speed_rpm[i] - 1500) ** 2 / 100000
    torque_stress = (torque_nm[i] - 40) ** 2 / 100
    wear_factor = tool_wear_min[i] / 250
    
    # Combined stress score (normalized)
    stress_score = (temp_stress + speed_stress + torque_stress + wear_factor * 50) / 100
    
    if stress_score < 0.3 and tool_wear_min[i] < 100:
        health_status.append('Healthy')
    elif stress_score < 0.7 or tool_wear_min[i] < 200:
        health_status.append('At Risk')
    else:
        health_status.append('Failed')

# Create DataFrame
df = pd.DataFrame({
    'Air_Temperature_K': air_temperature_k,
    'Process_Temperature_K': process_temperature_k,
    'Rotational_Speed_RPM': rotational_speed_rpm,
    'Torque_Nm': torque_nm,
    'Tool_Wear_min': tool_wear_min,
    'Health_Status': health_status
})

print(f"\n✓ Dataset created with {len(df)} rows")
print(f"\nDataset shape: {df.shape}")
print(f"\nFirst 5 rows:")
print(df.head())
print(f"\nClass distribution:")
print(df['Health_Status'].value_counts())
print(f"\nFeature statistics:")
print(df.describe())

# Preprocessing: Separate features and target
X = df[['Air_Temperature_K', 'Process_Temperature_K', 'Rotational_Speed_RPM', 'Torque_Nm', 'Tool_Wear_min']]
y = df['Health_Status']

# Scale features using MinMaxScaler
print("\n" + "=" * 60)
print("PREPROCESSING DATA WITH MINMAXSCALER")
print("=" * 60)

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

print("\n✓ Features scaled to [0, 1] range")
print(f"\nScaled features (first 5 rows):")
print(X_scaled_df.head())

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n✓ Train-test split completed")
print(f"  Training set: {len(X_train)} samples")
print(f"  Test set: {len(X_test)} samples")

# Train RandomForestClassifier with n_estimators=100
print("\n" + "=" * 60)
print("TRAINING RANDOMFORESTCLASSIFIER (n_estimators=100)")
print("=" * 60)

model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

print("\n✓ Model training completed")

# Model evaluation
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)

print(f"\n{'Model Performance':^60}")
print(f"{'─' * 60}")
print(f"Training Accuracy: {train_score:.4f} ({train_score*100:.2f}%)")
print(f"Testing Accuracy:  {test_score:.4f} ({test_score*100:.2f}%)")

# Detailed classification report
y_pred = model.predict(X_test)
print(f"\n{'Classification Report':^60}")
print(f"{'─' * 60}")
print(classification_report(y_test, y_pred, digits=4))

# Feature importance
print(f"\n{'Feature Importance':^60}")
print(f"{'─' * 60}")
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

for idx, row in feature_importance.iterrows():
    print(f"{row['Feature']:30s}: {row['Importance']:.4f}")

# Save model and scaler
print("\n" + "=" * 60)
print("SAVING MODEL AND SCALER")
print("=" * 60)

joblib.dump(model, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("\n✓ Model saved as 'model.pkl'")
print("✓ Scaler saved as 'scaler.pkl'")
print("\n" + "=" * 60)
print("SETUP COMPLETE! Ready to run Streamlit app.")
print("=" * 60)
print("\nNext step: Run 'streamlit run app.py' in your terminal")