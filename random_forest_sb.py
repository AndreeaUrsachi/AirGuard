import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error
import joblib
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from matplotlib.lines import Line2D

# 1. Citește fișierul Excel
df = pd.read_excel(r"C:\Users\Andreea\Desktop\Fmi\Anul 3\licenta\dataSetWHO.xlsx", sheet_name="AAP_2022_city_v9")


# 2. Elimină rândurile fără PM2.5
df = df[df['PM2.5 (μg/m3)'].notnull()]

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
joblib.dump(scaler, 'scaler_pm25.pkl')

# 6. Creează și antrenează modelul Random Forest
model = RandomForestRegressor(n_estimators=200,max_depth = 10,max_features="sqrt", random_state=42)
model.fit(X_scaled_train, y_train)

# 7. Evaluează modelul
y_pred = model.predict(X_scaled_test)
y_pred_train = model.predict(X_scaled_train)
r2 = r2_score(y_test, y_pred)
r2train = r2_score(y_train,y_pred_train)
scores = cross_val_score(model, X_scaled_full, y, cv=5)  # 5-fold cross-validation
n = X_test.shape[0]
p = X_test.shape[1]
r2adj = 1 - (1 - r2) * (n - 1) / (n - p - 1)
print(f"Scorul R2 adjustat: {r2adj:.5f}")
print(f"Cross-validation scores: {scores}")
print(f"R² Score: {r2:.5f}")
print(f"R² Score train: {r2train:.4f}")

# 8. Salvează modelul
joblib.dump(model, 'random_forest_model_pm25.pkl')
print("Modelul a fost salvat.")

# 9. Plot corect: Predicții vs Valori reale
df_plot = pd.DataFrame({
    'Valori reale': y_test,
    'Valori prezise': y_pred
})

# Plot cu regresie polinomială de grad 4
plt.figure(figsize=(10, 6))
sns.regplot(
    data=df_plot,
    x='Valori reale',
    y='Valori prezise',
    order=5,
    scatter_kws={'color': 'blue', 'alpha': 0.5},
    line_kws={'color': 'red'},
    ci=None
)

plt.xlabel('Valori reale PM2.5 (μg/m3)')
plt.ylabel('Valori prezise PM2.5 (μg/m3)')
plt.title('Random Forest - PM2.5')

# Adaugă legendă personalizată
legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='Puncte reale vs prezise',
           markerfacecolor='blue', markersize=8, alpha=0.5),
    Line2D([0], [0], color='red', lw=2, label='Linia de regresie, gradul 5')
]
plt.legend(handles=legend_elements)

plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# 10. Predicție pentru un exemplu nou
model_loaded = joblib.load('random_forest_model_pm25.pkl')

exemplu = np.array([[2022, 25.0, 20.0, 1, 3]])  # Valori ipotetice, encodează corect țara și orașul tău
predictie = model_loaded.predict(exemplu)
print(f"Predicția PM2.5 pentru exemplul dat este: {predictie[0]:.2f} μg/m3")