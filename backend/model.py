import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score

# Load the balanced dataset
df = pd.read_csv(r"C:\Users\Niharika PM\OneDrive\Desktop\gdg2\gdg\pcos_detection\backend\PCOS_DATASET_AUGMENTED_WITH_BMI.csv")

# Define features and target
features = ['Age (yrs)', 'BMI', 'Cycle(R/I)', 'Weight gain(Y/N)', 'hair growth(Y/N)',
            'Skin darkening (Y/N)', 'Hair loss(Y/N)', 'Pimples(Y/N)', 'TSH (mIU/L)',
            'Follicle No. (L)', 'Follicle No. (R)']
target = 'PCOS (Y/N)'

X = df[features]
y = df[target]

# Split into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the Random Forest Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Save the trained model and scaler
with open("pcos_model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)

with open("scaler.pkl", "wb") as scaler_file:
    pickle.dump(scaler, scaler_file)

# Evaluate the model
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))


# Function to load the model
def load_model():
    """Loads the trained PCOS detection model."""
    with open("pcos_model.pkl", "rb") as model_file:
        model = pickle.load(model_file)
    return model


# Function to load the scaler
def load_scaler():
    """Loads the saved scaler for feature scaling."""
    with open("scaler.pkl", "rb") as scaler_file:
        scaler = pickle.load(scaler_file)
    return scaler


# Function to predict PCOS
def predict_pcos(input_data):
    """Predicts PCOS based on input features."""
    model = load_model()
    scaler = load_scaler()

    # Convert input into a DataFrame for consistency
    input_df = pd.DataFrame([input_data], columns=features)

    # Scale the input
    input_scaled = scaler.transform(input_df)

    # Predict PCOS
    prediction = model.predict(input_scaled)

    return int(prediction[0])  # Return as an integer (0 or 1)
