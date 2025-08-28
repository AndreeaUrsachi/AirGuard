import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pandas as pd

# JSON-ul pe care l-ai furnizat
data_json = """
{
  "status": "ok",
  "data": {
    "aqi": 7,
    "idx": 8590,
    "city": {
      "geo": [44.4185543, 26.1615982],
      "name": "Str. Rotunda, Titan, Romania"
    },
    "dominentpol": "pm10",
    "iaqi": {
      "h": { "v": 54.5 },
      "no2": { "v": 5.3 },
      "p": { "v": 1006.7 },
      "pm10": { "v": 7 },
      "so2": { "v": 2.8 },
      "t": { "v": 5 },
      "w": { "v": 0.5 },
      "wg": { "v": 6.5 }
    },
    "time": {
      "s": "2025-04-07 13:00:00",
      "tz": "+03:00"
    }
  }
}
"""

# Încărcăm JSON-ul
data = json.loads(data_json)

# Extragem valorile pentru fiecare caracteristică
features = {
    "pm10": data['data']['iaqi']['pm10']['v'],
    "pm25": data['data']['iaqi'].get('pm25', {}).get('v', 0),  # Verificăm dacă există
    "no2": data['data']['iaqi']['no2']['v'],
    "so2": data['data']['iaqi']['so2']['v'],
    "p": data['data']['iaqi']['p']['v'],
    "h": data['data']['iaqi']['h']['v'],
    "t": data['data']['iaqi']['t']['v'],
    "w": data['data']['iaqi']['w']['v'],
    "wg": data['data']['iaqi']['wg']['v']
}

# Transformăm datele într-un DataFrame
df = pd.DataFrame([features])

# Definim target-ul (în acest caz, ar putea fi 'pm10', dar îl vom folosi pe toate)
X = df  # Datele de intrare
y = df['pm10']  # Target-ul pentru antrenament

# Creăm un model Random Forest
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Antrenăm modelul (folosind X și y pentru PM10)
model.fit(X, y)

# Facem predicții pentru toate valorile
predictions = model.predict(X)

# Vizualizăm rezultatele
features_names = list(features.keys())

# Plotează valorile reale vs. predicțiile
plt.figure(figsize=(10, 6))
plt.bar(features_names, [features[feature] for feature in features_names], label="Valori reale", alpha=0.6)
plt.bar(features_names, predictions, label="Predicții", alpha=0.6)
plt.xlabel("Caracteristici")
plt.ylabel("Valori")
plt.title("Predicții pentru datele de calitate a aerului")
plt.legend()
plt.show()

# Afișăm predicțiile
print("Predicții:", predictions)
