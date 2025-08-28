import requests
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Tokenul tău de la WAQI
API_TOKEN = "c13b1f1c847896f277f91011d633430888e7bde6"

# Lista capitalelor
capitals = [
    'Kabul','Luanda', 'Algiers', 'Yerevan', 'Vienna', 'Baku', 'Dhaka', 'Brussels',
    'Sofia', 'Ouagadougou', 'Ottawa', 'Beijing', 'Bogotá', 'Bratislava', 'Bucharest',
    'Moscow', 'Santiago', 'Jakarta', 'Nairobi', 'Seoul', 'Tokyo', 'Washington', 'London'
]

# URL pentru API-ul WAQI
BASE_URL = "http://api.waqi.info/feed/{}/?token=" + API_TOKEN

# Colectare date WAQI
waqi_data = {}
for city in capitals:
    response = requests.get(BASE_URL.format(city))
    if response.status_code == 200:
        data = response.json()
        if data['status'] == 'ok':
            waqi_data[city] = {
                'PM2.5': data['data'].get('iaqi', {}).get('pm25', {}).get('v', None),
                'PM10': data['data'].get('iaqi', {}).get('pm10', {}).get('v', None),
                'NO2': data['data'].get('iaqi', {}).get('no2', {}).get('v', None),
            }
        else:
            print(f"Nu am găsit date pentru {city}")
    else:
        print(f"Eroare la obținerea datelor pentru {city}")

# Citește fișierul Excel
df = pd.read_excel(r"C:\Users\Andreea\Desktop\Fmi\Anul 3\AirGuard\dataSetWHOplin.xlsx")

# Curățare nume coloane
df.columns = df.columns.str.strip()

# Filtrare capitale
df_filtered = df[df['City or Locality'].isin(capitals)].copy()

# Encode pentru orașe
le_city = LabelEncoder()
df_filtered['City_encoded'] = le_city.fit_transform(df_filtered['City or Locality'])
print("Orașe din fișierul Excel:")
print(df_filtered['City or Locality'].unique())

# Verifică orașele cu date complete
print("Orașe cu date complete din WAQI:")
print([city for city, data in waqi_data.items() if None not in data.values()])
# Adăugare date WAQI în DataFrame
df_filtered['PM2.5_waqi'] = df_filtered['City or Locality'].apply(lambda x: waqi_data.get(x, {}).get('PM2.5', None))
df_filtered['PM10_waqi'] = df_filtered['City or Locality'].apply(lambda x: waqi_data.get(x, {}).get('PM10', None))
df_filtered['NO2_waqi'] = df_filtered['City or Locality'].apply(lambda x: waqi_data.get(x, {}).get('NO2', None))


# Păstrăm doar valorile cele mai recente pentru fiecare oraș
df_filtered = df_filtered.sort_values('Measurement Year', ascending=False)
df_filtered = df_filtered.drop_duplicates(subset=['City or Locality'], keep='first')

# Pregătim tabelul de comparație
df_comparison = df_filtered[['City or Locality', 'PM2.5 (μg/m3)','PM2.5_waqi',
                              'PM10 (μg/m3)' , 'PM10_waqi', 'NO2 (μg/m3)', 'NO2_waqi']]

# Setăm orașul ca index
df_comparison.set_index('City or Locality', inplace=True)

# Plot comparație
ax = df_comparison.plot(kind='bar', figsize=(14, 8), width=0.8)
ax.set_title('Comparatie Calitatea Aerului (2022 vs WAQI)', fontsize=16)
ax.set_xlabel('Capitale', fontsize=14)
ax.set_ylabel('Concentrație (μg/m3)', fontsize=14)
ax.legend(title='Tipuri de poluanți', loc='upper left')
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.grid(True)
plt.show()