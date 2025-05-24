# phase3_preprocessing.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import json

# Load dataset
df = pd.read_csv('CarPrice_Assignment.csv')

# Drop columns that are not useful for modeling
df = df.drop(columns=['car_ID', 'CarName'])

# 1. Handle outliers for 'price' using IQR method
Q1 = df['price'].quantile(0.25)
Q3 = df['price'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

df = df[(df['price'] >= lower_bound) & (df['price'] <= upper_bound)]

# 2. Encode categorical variables with LabelEncoder and save them
categorical_cols = df.select_dtypes(include=['object']).columns
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le
    joblib.dump(le, f'label_encoder_{col}.pkl')

# 3. Feature scaling (StandardScaler) except target
features = df.drop('price', axis=1)
target = df['price']

scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)
features_scaled_df = pd.DataFrame(features_scaled, columns=features.columns)

# Save scaler and feature column order for deployment
joblib.dump(scaler, 'scaler.pkl')

feature_columns = list(features.columns)
with open('feature_columns.json', 'w') as f:
    json.dump(feature_columns, f)

# 4. Split into train-test sets (80-20)
X_train, X_test, y_train, y_test = train_test_split(
    features_scaled_df, target, test_size=0.2, random_state=42)

# Save split datasets
X_train.to_csv('X_train.csv', index=False)
X_test.to_csv('X_test.csv', index=False)
y_train.to_csv('y_train.csv', index=False)
y_test.to_csv('y_test.csv', index=False)

print(f"Preprocessing done. Training set shape: {X_train.shape}, Test set shape: {X_test.shape}")
