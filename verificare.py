import joblib
import numpy as np
import pandas as pd
from difflib import get_close_matches
import requests
from sklearn.preprocessing import LabelEncoder

# === INPUT MANUAL ===
continent = "Europe"
country = "Romania"
city = "Baia Mare"
year = 2022  # sau alt an

# === Încarcă datasetul și encodează ===
df = pd.read_excel(r"C:\Users\Andreea\Desktop\Fmi\Anul 3\licenta\dataSetWHO.xlsx", sheet_name="AAP_2022_city_v9")
df = df[df['PM2.5 (μg/m3)'].notnull()]

le_country = LabelEncoder()
le_city = LabelEncoder()
df['Country_encoded'] = le_country.fit_transform(df['WHO Country Name'].astype(str))
df['City_encoded'] = le_city.fit_transform(df['City or Locality'].astype(str))

city_list = df['City or Locality'].unique().tolist()
country_list = df['WHO Country Name'].unique().tolist()

# === Încarcă modelul și scalerul ===
model = joblib.load('random_forest_model_pm25.pkl')
scaler = joblib.load('scaler_pm25.pkl')

# === Caută cea mai apropiată potrivire ===
matched_city = get_close_matches(city, city_list, n=1)
matched_country = get_close_matches(country, country_list, n=1)

if not matched_city or not matched_country:
    print("Orașul sau țara nu există în model.")
    exit()

matched_city = matched_city[0]
matched_country = matched_country[0]

# === Obține date live de la WAQI ===
WAQI_TOKEN = "c13b1f1c847896f277f91011d633430888e7bde6"
waqi_url = f"https://api.waqi.info/feed/{city}/?token={WAQI_TOKEN}"
response = requests.get(waqi_url)

if response.status_code != 200:
    print("Eroare la API WAQI")
    exit()

result = response.json()
iaqi = result.get("data", {}).get("iaqi", {})

pm10 = iaqi.get("pm10", {}).get("v", 20.0)  # valoare implicită
pm25 = iaqi.get("pm25", {}).get("v", 15.0)

try:
    city_encoded = le_city.transform([matched_city])[0]
    country_encoded = le_country.transform([matched_country])[0]
except ValueError as e:
    print(f"Eroare la encoding: {str(e)}")
    exit()

# === Pregătește și scalează datele ===
features = np.array([[year, pm10, pm25, country_encoded, city_encoded]])
features_scaled = scaler.transform(features)
predictie = model.predict(features_scaled)

# === Afișează rezultatele ===
print("Predicția PM2.5 pentru orașul", matched_city, ":", round(float(predictie[0]), 2))
print("PM10 live:", pm10)
print("PM2.5 live:", pm25)
