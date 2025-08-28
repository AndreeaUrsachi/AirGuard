import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

df = pd.read_excel(r"C:\Users\Andreea\Desktop\Fmi\Anul 3\licenta\dataSetWHO.xlsx", sheet_name="AAP_2022_city_v9")

# Elimină valorile lipsă din target (PM2.5)
df = df[df['PM2.5 (μg/m3)'].notnull()]

# Encode pentru date categorice
le_country = LabelEncoder()
le_city = LabelEncoder()
df['Country_encoded'] = le_country.fit_transform(df['WHO Country Name'].astype(str))
df['City_encoded'] = le_city.fit_transform(df['City or Locality'].astype(str))

# Selectare coloane
features = ['Measurement Year', 'PM10 (μg/m3)', 'Country_encoded', 'City_encoded']
df = df[features + ['PM2.5 (μg/m3)']].dropna()

X = df[features].values
y = df['PM2.5 (μg/m3)'].values

# Împărțire seturi
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.5, shuffle=False, random_state=42)


# Alegem un alpha și antrenăm
final_tree = DecisionTreeRegressor(random_state=42)

final_tree.fit(X_train, y_train)

# Predicții
y_pred = final_tree.predict(X_test)

# Calcul RMSE și R²
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse:.4f}")
print(f"R² Score: {r2:.4f}")
n = X_test.shape[0]
p = X_test.shape[1]
r2adj = 1 - (1 - r2) * (n - 1) / (n - p - 1)
print(f"R2 adjustat: {r2adj:.4f}")

# Scatter plot pentru vizualizare
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, color='dodgerblue', label='Predicții vs Valori reale')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='Linia de regresie')
plt.xlabel('Valori reale PM2.5 (μg/m3)')
plt.ylabel('Predicții PM2.5 (μg/m3)')
plt.title('Decision Tree - PM2.5 ')
plt.legend()
plt.grid(True)
plt.show()
