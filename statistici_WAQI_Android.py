import json

import requests
import pandas as pd
import matplotlib.pyplot as plt

# Tokenul tău de la WAQI
API_TOKEN = "c13b1f1c847896f277f91011d633430888e7bde6"

capitals = [
    'Kabul', 'Algiers', 'Yerevan', 'Vienna', 'Baku', 'Dhaka', 'Brussels',
    'Sofia', 'Ouagadougou', 'Ottawa', 'Beijing', 'Bratislava', 'Bucharest',
    'Moscow', 'Santiago', 'Jakarta', 'Nairobi', 'Seoul', 'Tokyo', 'Washington', 'London'
]

# URL-ul API-ului WAQI
BASE_URL = "http://api.waqi.info/feed/{}/?token=" + API_TOKEN

# Colectăm datele pentru fiecare capitală
city_data = {}
for city in capitals:
    # Realizează cererea către API
    response = requests.get(BASE_URL.format(city))

    if response.status_code == 200:
        # Extrage datele JSON
        data = response.json()
        if data['status'] == 'ok':
            # Extrage PM2.5, PM10 și NO2 din datele JSON
            city_data[city] = {
                'PM2.5': data['data'].get('iaqi', {}).get('pm25', {}).get('v', None),
                'PM10': data['data'].get('iaqi', {}).get('pm10', {}).get('v', None),
                'NO2': data['data'].get('iaqi', {}).get('no2', {}).get('v', None),
            }
        else:
            print(f"Nu am găsit date pentru {city}")
    else:
        print(f"Eroare la obținerea datelor pentru {city}")


df = pd.DataFrame(city_data).T  # Transpunem pentru a avea capitalele pe rânduri
df = df.fillna(0)  # Dacă există valori lipsă, le înlocuim cu 0

json_data = {
    "cities": df.index.tolist(),
    "pm25": df['PM2.5'].tolist(),
    "pm10": df['PM10'].tolist(),
    "no2": df['NO2'].tolist()
}
with open('statistici_capitale_WAQI.json', 'w') as f:
    json.dump(json_data,f,indent=4)