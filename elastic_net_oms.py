from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Citește datele din Excel
df = pd.read_excel(r"C:\Users\Andreea\Desktop\Fmi\Anul 3\licenta\dataSetWHO.xlsx", sheet_name="AAP_2022_city_v9")

# Encode pentru coloanele categorice
le_country = LabelEncoder()
le_city = LabelEncoder()
df['Country_encoded'] = le_country.fit_transform(df['WHO Country Name'].astype(str))
df['City_encoded'] = le_city.fit_transform(df['City or Locality'].astype(str))

# Selectează doar targetul PM2.5
target = 'PM2.5 (μg/m3)'

# Elimină valorile lipsă din target
temp_df = df[df[target].notnull()].copy()

# Definește features (exclude targetul dacă este inclus)
features = ['Measurement Year', 'PM10 (μg/m3)', 'Country_encoded', 'City_encoded']
temp_df = temp_df[features + [target]].dropna()

# Pregătește X și y
X = temp_df[features].values
y = temp_df[target].values

# Creează caracteristici polinomiale de gradul 4
poly = PolynomialFeatures(degree=4, include_bias=False)
X_poly = poly.fit_transform(X)

# Împarte setul în train și test
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

# Standardizează datele
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Creează și antrenează modelul ElasticNet
model = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
model.fit(X_train_scaled, y_train)

# Preziceri
y_pred = model.predict(X_test_scaled)

# Sortează valorile pentru a desena linia polinomială
sorted_indices = np.argsort(y_pred)
x_sorted = y_pred[sorted_indices]
y_sorted = y_test[sorted_indices]

# Creează o curbă polinomială între predicții și valori reale (doar pentru plot)
coeffs = np.polyfit(x_sorted, y_sorted, deg=4)
poly_curve = np.poly1d(coeffs)
y_curve = poly_curve(x_sorted)

# Afișează graficul
plt.figure(figsize=(8, 6))
plt.scatter(y_pred, y_test, color='teal', alpha=0.6, label='Predicții vs Valori reale')
plt.plot(x_sorted, y_curve, 'r-', label='Curbă polinomială de regresie ')
plt.xlabel('Predicții PM2.5')
plt.ylabel('Valori reale PM2.5')
plt.title('ElasticNet + PolynomialFeatures – PM2.5')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Afișează scorurile
r2 = round(r2_score(y_test, y_pred), 4)
rmse = round(np.sqrt(mean_squared_error(y_test, y_pred)), 4)

print(r2)
print(rmse)