import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

# --- 1. Data Loading and Preprocessing ---

# Define the relative file path for the dataset.
# This works as long as the CSV is in the same folder as this script.
DATA_FILE = 'TrafficTwoMonth.csv'
MODEL_FILE = 'traffic_model.joblib'

# Check if the dataset exists in the current directory
if not os.path.exists(DATA_FILE):
    print(f"--- ERROR ---")
    print(f"The data file '{DATA_FILE}' was not found.")
    print("Please make sure 'TrafficTwoMonth.csv' is in the same folder as this Python script.")
    print("---------------")
    exit()

print(f"Loading dataset from '{DATA_FILE}'...")
# Read the data using the relative path
df = pd.read_csv(DATA_FILE)

# Create a copy to avoid modifying the original DataFrame in memory
df_processed = df.copy()

# Convert 'Time' to datetime objects and extract numerical features (Hour, Minute)
# Using .loc ensures modifications are made on the DataFrame itself
df_processed.loc[:, 'Time'] = pd.to_datetime(df_processed['Time'], format='%I:%M:%S %p').dt.time
df_processed['Hour'] = df_processed['Time'].apply(lambda x: x.hour)
df_processed['Minute'] = df_processed['Time'].apply(lambda x: x.minute)

# Define the mapping for the target variable 'Traffic Situation'
# This converts text labels into numbers the model can understand.
traffic_situation_mapping = {'low': 0, 'normal': 1, 'high': 2, 'heavy': 3}
df_processed['Traffic Situation'] = df_processed['Traffic Situation'].map(traffic_situation_mapping)

# One-hot encode the 'Day of the week' categorical feature.
# This creates new columns for each day (e.g., 'Day of the week_Monday').
df_processed = pd.get_dummies(df_processed, columns=['Day of the week'], drop_first=True)

# Drop the original 'Time' and 'Date' columns as they've been replaced by more useful features
df_processed = df_processed.drop(columns=['Time', 'Date'])

print("Data preprocessing complete.")


# --- 2. Model Training ---

# Define the features (X) and the target (y)
# X contains all the processed columns except 'Traffic Situation'
X = df_processed.drop('Traffic Situation', axis=1)
# y is the 'Traffic Situation' column that we want the model to learn to predict
y = df_processed['Traffic Situation']

# Split the data into training and testing sets (80% for training, 20% for testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Initialize the RandomForestClassifier
# n_estimators is the number of decision trees in the forest. More trees can lead to better performance.
# n_jobs=-1 uses all available CPU cores to speed up training.
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

print("\nTraining the RandomForest model... This may take a moment.")
# Train the model on the training data
model.fit(X_train, y_train)
print("Model training complete.")

# Evaluate the model's performance on the unseen test data
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy on Test Data: {accuracy:.4f}")


# --- 3. Save the Trained Model ---

# Save the trained model and the list of training columns to a single file.
# Storing the column order is crucial to ensure predictions use the same feature structure.
joblib.dump((model, X.columns.tolist()), MODEL_FILE)
print(f"\nTrained model and column order successfully saved to '{MODEL_FILE}'")
print("\nYou can now run the Flask application (app.py).")