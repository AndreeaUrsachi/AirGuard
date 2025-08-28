import pandas as pd
import numpy as np
import os
import re
import joblib
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import r2_score, mean_squared_error

# Încarcă datele
df = pd.read_csv(r'C:\Users\Andreea\Desktop\Fmi\Anul 3\AirGuard\SensorDataLiveLocatii.csv')

# Preprocesare
df['Valoare_numeric'] = df['Valoare'].apply(lambda x: int(re.findall(r'\d+', x)[0]))
df['Ora_minute'] = pd.to_timedelta(df['Ora']).dt.total_seconds() / 60
df['Data'] = pd.to_datetime(df['Data'], format='mixed', dayfirst=True, errors='coerce')
df['Ziua'] = df['Data'].dt.day

# Elimină valorile invalide
df = df.dropna(subset=['Ora_minute', 'Data', 'Ziua'])

# Creează folderul de modele dacă nu există
os.makedirs("models", exist_ok=True)

# Definește funcția pentru partea zilei
def get_time_of_day(minute):
    if 360 <= minute < 720:
        return "dimineata"
    elif 720 <= minute < 1080:
        return "amiaza"
    else:
        return "seara"

# Aplică clasificarea în funcție de oră
df["partea_zilei"] = df["Ora_minute"].apply(get_time_of_day)

# Encodează locația
le = LabelEncoder()
df['Locatie_encoded'] = le.fit_transform(df['Locatie'])

df['Luna'] = df['Data'].dt.month
df['Zi_sapt'] = df['Data'].dt.dayofweek

# Parcurge fiecare combinație locație + parte a zilei
for locatie in df['Locatie'].unique():
    for perioada in ['dimineata', 'amiaza', 'seara']:
        subset = df[(df['Locatie'] == locatie) & (df['partea_zilei'] == perioada)]

        if len(subset) < 10:
            print(f"[!] Sărit modelul pentru {locatie}_{perioada}: prea puține date ({len(subset)} rânduri)")
            continue

        # Features și target
        features = ['Ora_minute', 'Ziua', 'Luna', 'Zi_sapt']
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(subset[features])
        X = pd.DataFrame(X_scaled, columns=features)
        X['Locatie_encoded'] = subset['Locatie_encoded'].values
        y = subset['Valoare_numeric']

        # Împărțire în train/test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Model
        model = DecisionTreeRegressor(random_state=42)
        model.fit(X_train, y_train)

        # Evaluare
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        print(f"[+] Model {locatie}_{perioada}: R2={r2:.4f}, MSE={mse:.2f}")

        # Salvare model
        model_name = f"models/model_{locatie}_{perioada}.joblib"
        joblib.dump(model, model_name)

# Salvăm și encoderul locației pentru interpretare viitoare
joblib.dump(le, "models/location_encoder.joblib")
