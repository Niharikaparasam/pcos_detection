import pandas as pd
import pickle
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv(r"C:\Users\Niharika PM\OneDrive\Desktop\gdg\pcos_detection\backend\PCOS_DATASET_AUGMENTED_WITH_BMI.csv")

# Define features and target variable
features = ['Age (yrs)', 'BMI', 'Cycle(R/I)', 'Weight gain(Y/N)', 'hair growth(Y/N)',
            'Skin darkening (Y/N)', 'Hair loss(Y/N)', 'Pimples(Y/N)', 'TSH (mIU/L)',
            'Follicle No. (L)', 'Follicle No. (R)']
target = 'PCOS (Y/N)'

X = df[features]
y = df[target]

# Check class distribution
print("Original class distribution:")
print(y.value_counts())

# 🔹 OPTION 1: Downsampling (Reduce No PCOS cases)
df_pcos = df[df[target] == 1]
df_no_pcos = df[df[target] == 0]
df_no_pcos_downsampled = resample(df_no_pcos, replace=False, n_samples=len(df_pcos), random_state=42)
df_balanced = pd.concat([df_pcos, df_no_pcos_downsampled])

# 🔹 OPTION 2: SMOTE (Increase PCOS cases)
# smote = SMOTE(sampling_strategy='auto', random_state=42)
# X_resampled, y_resampled = smote.fit_resample(X, y)

# Update features & target
X_balanced = df_balanced[features]
y_balanced = df_balanced[target]

print("Balanced class distribution:")
print(y_balanced.value_counts())

# Normalize data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_balanced)

# Save the scaler
pickle.dump(scaler, open("scaler.pkl", "wb"))

# Save balanced dataset
pd.DataFrame(X_scaled, columns=features).to_csv("balanced_dataset.csv", index=False)
pd.DataFrame(y_balanced, columns=[target]).to_csv("balanced_labels.csv", index=False)

print("✅ Preprocessing complete. Balanced dataset saved.")
