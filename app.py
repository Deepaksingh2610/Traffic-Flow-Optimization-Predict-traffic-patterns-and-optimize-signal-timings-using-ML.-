import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import joblib
import os

# --- Initialize the Flask App ---
# The 'template_folder' argument tells Flask to look for HTML files in a 'templates' directory.
app = Flask(__name__, template_folder='templates')

# --- Load the Trained Model and Columns ---
MODEL_FILE = 'traffic_model.joblib'
print(f"Loading model from {MODEL_FILE}...")

# Check if the model file exists
if not os.path.exists(MODEL_FILE):
    print(f"--- ERROR ---")
    print(f"The model file '{MODEL_FILE}' was not found.")
    print("Please make sure you have run the 'train_model.py' script first to create it.")
    print("---------------")
    exit()

# Load the tuple containing the model and the column order
try:
    model, model_columns = joblib.load(MODEL_FILE)
    print("Model and column order loaded successfully.")
except Exception as e:
    print(f"Error loading the model file: {e}")
    exit()


# --- Define App Routes ---

# Route for the main home page
@app.route('/')
def home():
    """
    Renders the main HTML page (index.html) for user input.
    """
    return render_template('index.html')


# Route for handling the prediction request
@app.route('/predict', methods=['POST'])
def predict():
    """
    Receives user input from the HTML form, processes it,
    and returns the model's prediction as JSON.
    """
    try:
        # Get the JSON data sent from the frontend
        data = request.get_json(force=True)
        print(f"Received data: {data}")

        # --- Preprocess the Input Data ---
        # The input is a dictionary like {'day': 'Tuesday', 'hour': 17, 'minute': 30}

        # Create a pandas DataFrame from the input data
        # The index [0] is important to create a single-row DataFrame
        input_df = pd.DataFrame([data])

        # Convert hour and minute to numeric, handling potential errors
        input_df['Hour'] = pd.to_numeric(input_df['hour'])
        input_df['Minute'] = pd.to_numeric(input_df['minute'])

        # One-hot encode the 'Day of the week'
        # This creates columns like 'Day of the week_Monday', 'Day of the week_Tuesday', etc.
        input_df['Day of the week'] = input_df['day']
        input_df = pd.get_dummies(input_df, columns=['Day of the week'], drop_first=False) # Keep all days

        # Drop the original text columns
        input_df = input_df.drop(columns=['day', 'hour', 'minute'])

        # --- Align Columns with Model's Training Data ---
        # Reindex the input DataFrame to match the exact columns the model was trained on.
        # This is CRITICAL. It adds any missing one-hot encoded day columns and fills them with 0.
        # It also ensures the column order is identical to the training data.
        final_df = input_df.reindex(columns=model_columns, fill_value=0)

        print("\n--- Processed Data for Prediction ---")
        print(final_df.head())
        print("-------------------------------------\n")

        # --- Make Prediction ---
        prediction_encoded = model.predict(final_df)
        prediction_proba = model.predict_proba(final_df)

        # The model outputs a number (0, 1, 2, or 3). We convert it back to a meaningful label.
        output_map = {0: 'Low', 1: 'Normal', 2: 'High', 3: 'Heavy'}
        prediction_text = output_map[prediction_encoded[0]]
        
        # Get the confidence score for the predicted class
        confidence = np.max(prediction_proba) * 100

        # --- Return the Result ---
        # Send the prediction and confidence back to the frontend as JSON
        return jsonify({
            'prediction_text': f'The predicted traffic is {prediction_text}.',
            'confidence': f'{confidence:.2f}%'
        })

    except Exception as e:
        print(f"An error occurred during prediction: {e}")
        return jsonify({'error': 'An error occurred on the server. Check the logs.'}), 500


# --- Run the App ---
if __name__ == "__main__":
    # The 'debug=True' argument allows you to see errors in the browser and automatically reloads the server when you save changes.
    app.run(debug=True)