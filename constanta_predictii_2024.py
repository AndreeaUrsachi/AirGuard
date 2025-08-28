import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import numpy as np

# Încarcă datele
excel_path = r"C:\Users\Andreea\Desktop\Fmi\Anul 3\licenta\who_dataset_2024.xlsx"
df = pd.read_excel(excel_path, sheet_name='Update 2024 (V6.1)')
# Curățare coloane (în caz că au spații)
df.columns = df.columns.str.strip()

# Filtrează doar rândurile din România
df_ro = df[df["country_name"] == "Spain"]

# Selectează coloanele relevante pentru predicție
features = ["pm10_concentration", "no2_concentration"]  # poți adăuga și altele dacă vrei
target = "pm25_concentration"

# Elimină rândurile incomplete
df_ro_clean = df_ro[features + [target] + ["country_name", "city"]].dropna()

print(f"Număr rânduri România după filtrare și dropna: {len(df_ro_clean)}")

# Separă datele
X = df_ro_clean[features]
y = df_ro_clean[target]

# Scalează datele
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Împarte în train/test pentru evaluare (optional)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Antrenează modelul pe datele de antrenament
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluează pe test
y_pred_test = model.predict(X_test)
r2 = r2_score(y_test, y_pred_test)
print(f"R² score pe setul de test România: {r2:.4f}")

# Face predicții pe toate rândurile curate din România
y_pred_all = model.predict(X_scaled)

# Adaugă predicțiile în DataFrame
df_ro_clean["pm25_predicted"] = y_pred_all

# Afișează câteva predicții din România
print(df_ro_clean[["country_name", "city", "pm25_concentration", "pm25_predicted"]].head(10))

# Salvează modelul și scalerul
joblib.dump(model, "spain_pm25_model.joblib")
joblib.dump(scaler, "spain_scaler.joblib")
print("Modelul și scalerul pentru România au fost salvați.")

new_sample = np.array([[25.0, 18.0]])  # valorile pm10 și no2

# Scalează noile date folosind scalerul antrenat
new_sample_scaled = scaler.transform(new_sample)

# Prezice pm2.5 pentru noile date
predicted_pm25 = model.predict(new_sample_scaled)

print(f"Predicție pm2.5 pentru valorile {new_sample[0]}: {predicted_pm25[0]:.2f}")