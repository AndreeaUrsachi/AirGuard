import requests
import pandas as pd
import matplotlib.pyplot as plt

# Tokenul tău de la WAQI
API_TOKEN = "c13b1f1c847896f277f91011d633430888e7bde6"

capitals = [
    'Kabul', 'Algiers', 'Yerevan', 'Vienna', 'Baku', 'Dhaka', 'Brussels',
    'Sofia', 'Ouagadougou', 'Ottawa', 'Beijing', 'Bogotá', 'Bratislava', 'Bucharest',
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

# Creăm un DataFrame pentru a vizualiza datele mai ușor
df = pd.DataFrame(city_data).T  # Transpunem pentru a avea capitalele pe rânduri
df = df.fillna(0)  # Dacă există valori lipsă, le înlocuim cu 0

# Plotăm datele pentru fiecare capitală
plt.figure(figsize=(14, 8))
df.plot(kind='bar', color=['skyblue', 'lightgreen', 'lightcoral'], width=0.8)

plt.title('Calitatea aerului în capitalele lumii', fontsize=16)
plt.xlabel('Capitale', fontsize=14)
plt.ylabel('Concentrație (μg/m3)', fontsize=14)
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.legend(title='Tipuri de poluanți', loc='upper left')
plt.grid(True)
plt.savefig(r"C:\Users\Andreea\Desktop\Fmi\Anul 3\licenta\app\src\main\res\drawable\statistici_WAQI.png")
