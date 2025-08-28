import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.linalg import inv
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline

# Citește datele
df = pd.read_excel(r"C:\Users\Andreea\Desktop\Fmi\Anul 3\licenta\dataSetWHO.xlsx", sheet_name="AAP_2022_city_v9")

# Elimină valorile lipsă din target (NO2)
df = df[df['NO2 (μg/m3)'].notnull()]

# Encode pentru date categorice
le_country = LabelEncoder()
le_city = LabelEncoder()
df['Country_encoded'] = le_country.fit_transform(df['WHO Country Name'].astype(str))
df['City_encoded'] = le_city.fit_transform(df['City or Locality'].astype(str))

# Selectare coloane
features = ['Measurement Year', 'PM10 (μg/m3)', 'PM2.5 (μg/m3)', 'Country_encoded', 'City_encoded']
df = df[features + ['NO2 (μg/m3)']].dropna()

X = df[features].values
y = df['NO2 (μg/m3)'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

alphas = [0.1,1.0,10.0,100.0]

model = make_pipeline(PolynomialFeatures(degree=3), Ridge(alpha=0.1))
model.fit(X_train, y_train)
# Scalare predictori

y_pred = model.predict(X_test)

# Calcul R²
r2 = r2_score(y_test, y_pred)
print(f"R²: {r2:.2f}")

# Plot Predicții vs Valori Reale
min_axis = min(min(y_pred), min(y_test))
max_axis = max(max(y_pred), max(y_test))

plt.figure(figsize=(8,6))
plt.scatter(y_pred, y_test, color='slateblue', alpha=0.6, label='Predicții vs Valori reale')
plt.plot([min_axis, max_axis], [min_axis, max_axis], 'r--', label='Linie Ideală 1:1')
plt.xlabel('Predicții PM25')
plt.ylabel('Valori reale PM25')
plt.title('Ridge Regression - Predicții vs Valori Reale')
plt.legend()
plt.grid(True)
plt.show()
