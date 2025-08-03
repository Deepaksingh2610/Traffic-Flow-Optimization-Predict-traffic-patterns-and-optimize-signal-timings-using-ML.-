# Traffic-Flow-Optimization-Predict-traffic-patterns-and-optimize-signal-timings-using-ML.-
This AI-based traffic prediction system uses day, hour, and minute inputs to predict traffic levels (Low, Normal, High, Heavy). Built with Python, Flask, and Random Forest, it analyzes temporal patterns from a two-month dataset and returns real-time predictions with confidence scores via a simple web interface.
Traffic Situation Prediction App
This is a full-stack web application that predicts traffic situations based on vehicle counts and time-based data. The application uses a machine learning model trained on a historical dataset to classify traffic into four categories: Low, Normal, High, and Heavy.

Features
Multi-Page Interface: A clean, user-friendly website with separate pages for Home, About, Model Details, and Data Indicators.

Interactive Prediction: An easy-to-use form where users can input vehicle counts, time, and day of the week to get an instant traffic prediction.

Data Visualization: An "Indicators" page with dynamic charts showing the distribution of traffic situations and average vehicle counts from the training data.

Model Insights: A dedicated page explaining the machine learning model used (Random Forest Classifier).

Confidence Score: The prediction result is displayed with a confidence score calculated from the model's prediction probabilities.

Tech Stack
Backend: Flask (Python)

Machine Learning: Scikit-learn, Pandas, NumPy

Frontend: HTML, Bootstrap CSS

Data Visualization: Chart.js

How It Works
A Random Forest Classifier is trained on the TrafficTwoMonth.csv dataset using train_model.py.

The trained model is saved as traffic_model.joblib.

The Flask application (app.py) serves the HTML pages.

When a user submits the prediction form, the backend processes the input data, performs the same feature engineering as the training script, and feeds it to the loaded model.

The model's prediction is then displayed on a user-friendly result page.

## Required Libraries
To run this project, you need to install the following Python libraries.

You can install them all with a single command in your terminal:

Bash

pip install Flask pandas numpy scikit-learn
For best practice, you should create a requirements.txt file in your main project folder.

File: requirements.txt

Flask
pandas
numpy
scikit-learn


At last step to run backend server is -> python app.py
