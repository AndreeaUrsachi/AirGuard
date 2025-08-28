import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import r2_score, mean_squared_error

# Citește datele
df = pd.read_excel(r"C:\Users\Andreea\Desktop\Fmi\Anul 3\licenta\dataSetWHO.xlsx", sheet_name="AAP_2022_city_v9")

# Elimină valorile lipsă pentru toate target-urile
df = df[df['NO2 (μg/m3)'].notnull()]
df = df[df['PM2.5 (μg/m3)'].notnull()]
df = df[df['PM10 (μg/m3)'].notnull()]

# Encode pentru date categorice
le_country = LabelEncoder()
le_city = LabelEncoder()
df['Country_encoded'] = le_country.fit_transform(df['WHO Country Name'].astype(str))
df['City_encoded'] = le_city.fit_transform(df['City or Locality'].astype(str))

# Selectare coloane
features = ['Measurement Year', 'Country_encoded', 'City_encoded']
df = df[features + ['NO2 (μg/m3)', 'PM2.5 (μg/m3)', 'PM10 (μg/m3)']].dropna()

X = df[features].values
y_no2 = df['NO2 (μg/m3)'].values
y_pm25 = df['PM2.5 (μg/m3)'].values
y_pm10 = df['PM10 (μg/m3)'].values

# Împărțire seturi
X_train, X_test, y_no2_train, y_no2_test = train_test_split(X, y_no2, test_size=0.2, random_state=42)
_, _, y_pm25_train, y_pm25_test = train_test_split(X, y_pm25, test_size=0.2, random_state=42)
_, _, y_pm10_train, y_pm10_test = train_test_split(X, y_pm10, test_size=0.2, random_state=42)

# Standardizare
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

def knn_predict(X_train, X_test, y_train, k=3):
    predictions = []
    for x_test in X_test:
        distances = [np.linalg.norm(x_test - x_train) for x_train in X_train]
        k_indices = np.argsort(distances)[:k]
        k_nearest = [y_train[i] for i in k_indices]
        predictions.append(np.mean(k_nearest))
    return predictions

# Predicții pentru fiecare poluant
pred_no2 = knn_predict(X_train_scaled, X_test_scaled, y_no2_train)
pred_pm25 = knn_predict(X_train_scaled, X_test_scaled, y_pm25_train)
pred_pm10 = knn_predict(X_train_scaled, X_test_scaled, y_pm10_train)

# Calcul R² și RMSE
results = pd.DataFrame({
    'Model': ['KNN Manual', 'KNN Manual', 'KNN Manual'],
    'Target': ['NO2 (μg/m3)', 'PM2.5 (μg/m3)', 'PM10 (μg/m3)'],
    'R² Score': [
        round(r2_score(y_no2_test, pred_no2), 4),
        round(r2_score(y_pm25_test, pred_pm25), 4),
        round(r2_score(y_pm10_test, pred_pm10), 4)
    ],
    'RMSE': [
        round(np.sqrt(mean_squared_error(y_no2_test, pred_no2)), 2),
        round(np.sqrt(mean_squared_error(y_pm25_test, pred_pm25)), 2),
        round(np.sqrt(mean_squared_error(y_pm10_test, pred_pm10)), 2)
    ]
})

print("📊 Tabelul Rezultatelor:\n")
print(results)

# Plot pentru fiecare poluant
targets = {
    "NO2 (μg/m3)": (pred_no2, y_no2_test),
    "PM2.5 (μg/m3)": (pred_pm25, y_pm25_test),
    "PM10 (μg/m3)": (pred_pm10, y_pm10_test)
}

for label, (pred, real) in targets.items():
    min_axis = min(min(pred), min(real))
    max_axis = max(max(pred), max(real))
    plt.figure(figsize=(8,6))
    plt.scatter(pred, real, alpha=0.6, label=f"Predicții vs Valori reale: {label}")
    plt.plot([min_axis, max_axis], [min_axis, max_axis], 'r--', label='Linie Ideală 1:1')
    plt.xlabel(f"Predicții {label}")
    plt.ylabel(f"Valori reale {label}")
    plt.title(f"KNN Manual - {label}")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

