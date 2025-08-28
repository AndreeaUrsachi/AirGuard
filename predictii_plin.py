import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error
import joblib
from sklearn.preprocessing import StandardScaler

# 1. Citește fișierul Excel
df = pd.read_excel(r"C:\Users\Andreea\Desktop\Fmi\Anul 3\AirGuard\dataSetWHOplin.xlsx")



# 3. Encode pentru coloanele categorice
le_country = LabelEncoder()
le_city = LabelEncoder()
df['Country_encoded'] = le_country.fit_transform(df['WHO Country Name'].astype(str))
df['City_encoded'] = le_city.fit_transform(df['City or Locality'].astype(str))

# 4. Selectează caracteristicile și target-ul
features = ['Measurement Year', 'PM10 (μg/m3)', 'PM2.5 (μg/m3)', 'Country_encoded', 'City_encoded']
X = df[features].values
y = df['PM2.5 (μg/m3)'].values  # Modificăm target-ul pentru PM2.5

# 5. Împarte setul în train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_scaled_train = scaler.fit_transform(X_train)
X_scaled_test = scaler.fit_transform(X_test)
X_scaled_full = scaler.fit_transform(X)

# 6. Creează și antrenează modelul Random Forest
model = RandomForestRegressor(n_estimators=200,max_depth = 10,max_features="sqrt", random_state=42)
model.fit(X_scaled_train, y_train)

# 7. Evaluează modelul
y_pred = model.predict(X_scaled_test)
y_pred_train = model.predict(X_scaled_train)
r2 = r2_score(y_test, y_pred)
r2train = r2_score(y_train,y_pred_train)
scores = cross_val_score(model, X_scaled_full, y, cv=5)  # 5-fold cross-validation
print(f"Cross-validation scores: {scores}")
print(f"R² Score: {r2:.4f}")
print(f"R² Score train: {r2train:.4f}")

# 8. Salvează modelul
joblib.dump(model, 'random_forest_model_pm25.pkl')
print("Modelul a fost salvat.")

# 9. Plot corect: Predicții vs Valori reale
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.6, label='Predicții')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', label='Ideal')
plt.xlabel("Valori reale PM2.5 (μg/m3)")
plt.ylabel("Predicții PM2.5 (μg/m3)")
plt.title("Random Forest Regression: Predicții vs Valori reale PM2.5, fisierul complet")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 10. Predicție pentru un exemplu nou
model_loaded = joblib.load('random_forest_model_pm25.pkl')

exemplu = np.array([[2015, 25.0, 20.0, 1, 3]])  # Valori ipotetice, encodează corect țara și orașul tău
predictie = model_loaded.predict(exemplu)
print(f"Predicția PM2.5 pentru exemplul dat este: {predictie[0]:.2f} μg/m3")

"""


# 3. Encode pentru categorice
le_country = LabelEncoder()
le_city = LabelEncoder()
df['Country_encoded'] = le_country.fit_transform(df['WHO Country Name'].astype(str))
df['City_encoded'] = le_city.fit_transform(df['City or Locality'].astype(str))

# 4. Selectare features și targeturi
features = ['Measurement Year', 'Country_encoded', 'City_encoded']
X = df[features].values
y_no2 = df['NO2 (μg/m3)'].values
y_pm25 = df['PM2.5 (μg/m3)'].values
y_pm10 = df['PM10 (μg/m3)'].values

# 5. Împărțire seturi
X_train, X_test, y_no2_train, y_no2_test = train_test_split(X, y_no2, test_size=0.2, random_state=42)
_, _, y_pm25_train, y_pm25_test = train_test_split(X, y_pm25, test_size=0.2, random_state=42)
_, _, y_pm10_train, y_pm10_test = train_test_split(X, y_pm10, test_size=0.2, random_state=42)

# 6. Antrenare RandomForest pentru fiecare poluant
models = {}
predictions = {}
scores = {}
rmses = {}

for target, y_train, y_test in zip(
    ['NO2', 'PM2.5', 'PM10'],
    [y_no2_train, y_pm25_train, y_pm10_train],
    [y_no2_test, y_pm25_test, y_pm10_test]
):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    models[target] = model
    predictions[target] = (y_test, y_pred)
    scores[target] = round(r2_score(y_test, y_pred), 4)
    rmses[target] = round(np.sqrt(mean_squared_error(y_test, y_pred)), 2)

    # Salvează fiecare model
    joblib.dump(model, f'random_forest_model_{target.lower()}.pkl')

# 7. Tabel cu rezultate
import pandas as pd

results = pd.DataFrame({
    'Model': ['Random Forest'] * 3,
    'Target': ['NO2 (μg/m3)', 'PM2.5 (μg/m3)', 'PM10 (μg/m3)'],
    'R² Score': [scores['NO2'], scores['PM2.5'], scores['PM10']],
    'RMSE': [rmses['NO2'], rmses['PM2.5'], rmses['PM10']]
})

print("📊 Tabel Rezultate Random Forest, fisier plin:\n")
print(results)

# 8. Plot Predicții vs Valori reale
for target in ['NO2', 'PM2.5', 'PM10']:
    y_test, y_pred = predictions[target]
    min_axis = min(min(y_pred), min(y_test))
    max_axis = max(max(y_pred), max(y_test))

    plt.figure(figsize=(8,5))
    plt.scatter(y_test, y_pred, alpha=0.6, label=f'Predicții vs Valori reale, fisier plin: {target}')
    plt.plot([min_axis, max_axis], [min_axis, max_axis], 'r--', label='Linie Ideală 1:1')
    plt.xlabel(f"Valori reale {target} (μg/m3)")
    plt.ylabel(f"Predicții {target} (μg/m3)")
    plt.title(f"Random Forest Regression: {target}")
    plt.legend()
    plt.grid(True)

    plt.figtext(0.5, 0.08,
                "PM2.5 = particule fine  |  PM10 = praf / polen  |  NO2 = oxid de azot",
                ha="center", fontsize=9, bbox=dict(facecolor='white', alpha=0.7))

    plt.tight_layout(rect=[0, 0.10, 1, 1])
    plt.show()

# 9. Predicție exemplu ipotetic
exemplu = np.array([[2022, 10, 15]])  # Measurement Year, Country_encoded, City_encoded

for target in ['NO2', 'PM2.5', 'PM10']:
    model_loaded = joblib.load(f'random_forest_model_{target.lower()}.pkl')
    predictie = model_loaded.predict(exemplu)
    print(f"Predicția {target} pentru exemplul dat este: {predictie[0]:.2f} μg/m3")
"""