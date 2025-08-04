import pandas as pd
from flask import Flask, request, render_template
import joblib
import numpy as np
import os

# --- Initialize the Flask App ---
app = Flask(__name__)

# --- Load the Trained Model ---
try:
    model, model_columns = joblib.load('traffic_model.joblib')
except FileNotFoundError:
    print("FATAL ERROR: traffic_model.joblib not found. Please run train_model.py first.")
    exit()

# --- App Routes for Each Page ---

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/indicators')
def indicators():
    return render_template('indicators.html')
    
@app.route('/our-models')
def our_models():
    return render_template('our-models.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            # --- Safely Get Data From the HTML Form ---
            # Use .get() to avoid errors if a field is missing
            required_fields = ['CarCount', 'BikeCount', 'BusCount', 'TruckCount', 'Total', 'Time', 'Day of the week']
            form_data = {}
            for field in required_fields:
                value = request.form.get(field)
                if not value:
                    # If any field is empty, return an error
                    return render_template('predict.html', error=f"Missing required field: {field}")
                form_data[field] = value

            # Create a single-row DataFrame
            input_df = pd.DataFrame([form_data])

            # --- Data Preprocessing & Feature Engineering ---
            # Convert numeric columns to numbers
            numeric_cols = ['CarCount', 'BikeCount', 'BusCount', 'TruckCount', 'Total']
            for col in numeric_cols:
                input_df[col] = pd.to_numeric(input_df[col])

            # Process Time into Hour and Minute
            input_df['Time'] = pd.to_datetime(input_df['Time'], format='%H:%M').dt.time
            input_df['Hour'] = input_df['Time'].apply(lambda x: x.hour)
            input_df['Minute'] = input_df['Time'].apply(lambda x: x.minute)
            input_df = input_df.drop('Time', axis=1)
            
            # One-hot encode 'Day of the week'
            input_df = pd.get_dummies(input_df, columns=['Day of the week'], prefix='Day of the week')
            
            # --- Prediction ---
            final_df = input_df.reindex(columns=model_columns, fill_value=0)
            prediction_encoded = model.predict(final_df)[0]
            prediction_proba = model.predict_proba(final_df)

            output_map = {0: 'Low', 1: 'Normal', 2: 'High', 3: 'Heavy'}
            prediction_text = output_map.get(prediction_encoded, "Unknown")
            
            confidence = np.max(prediction_proba) * 100

            return render_template('result.html', 
                                   prediction=prediction_text, 
                                   confidence=f'{confidence:.2f}%')

        except Exception as e:
            return render_template('predict.html', error=f"An error occurred: {e}")

    # For a GET request, just show the form page
    return render_template('predict.html', error=None)

# --- Run the App ---
if __name__ == "__main__":
    app.run(debug=True)