from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
from difflib import get_close_matches
import requests
from sklearn.preprocessing import LabelEncoder
import json

app = Flask(__name__)

# Încarcă datasetul pentru encoder
df = pd.read_excel(r"C:\Users\Andreea\Desktop\Fmi\Anul 3\licenta\dataSetWHO.xlsx", sheet_name="AAP_2022_city_v9")
df = df[df['PM2.5 (μg/m3)'].notnull()]

# Encoderele pentru țări și orașe
le_country = LabelEncoder()
le_city = LabelEncoder()
df['Country_encoded'] = le_country.fit_transform(df['WHO Country Name'].astype(str))
df['City_encoded'] = le_city.fit_transform(df['City or Locality'].astype(str))

city_list = df['City or Locality'].unique().tolist()
country_list = df['WHO Country Name'].unique().tolist()

# Încarcă modelul și scalerul
model = joblib.load('random_forest_model_pm25.pkl')
scaler = joblib.load('scaler_pm25.pkl')

WAQI_TOKEN = "c13b1f1c847896f277f91011d633430888e7bde6"
from datetime import datetime

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    print("PRIMIT DE LA ANDROID:", data)

    continent = data.get("continent")
    country = data.get("country")
    city = data.get("city")
    date_str = data.get("date")
    pm10 = data.get("pm10", 20.0)
    pm25 = data.get("pm25", 15.0)

    try:
        prediction_date = datetime.strptime(date_str, "%Y-%m-%d").date()
    except Exception as e:
        return jsonify({"error": f"Dată invalidă: {str(e)}"}), 400

    matched_city = get_close_matches(city, city_list, n=1)
    matched_country = get_close_matches(country, country_list, n=1)

    if not matched_city or not matched_country:
        return jsonify({"error": "Orașul sau țara nu există în model"}), 400

    matched_city = matched_city[0]
    matched_country = matched_country[0]

    try:
        city_encoded = le_city.transform([matched_city])[0]
        country_encoded = le_country.transform([matched_country])[0]
    except ValueError as e:
        return jsonify({"error": f"Eroare de encoding: {str(e)}"}), 400

    year = prediction_date.year
    features = np.array([[year, pm10, pm25, country_encoded, city_encoded]])
    features_scaled = scaler.transform(features)
    predictie = model.predict(features_scaled)

    return jsonify({
        "predictie": round(float(predictie[0]), 2),
        "oras": matched_city,
        "tara": matched_country,
        "pm10": pm10,
        "pm25": pm25,
        "data_predictie": prediction_date.strftime("%Y-%m-%d")
    })


@app.route('/')
def index():
    return "Server is running!"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)

