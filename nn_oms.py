from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder,StandardScaler
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import seaborn as sns
from matplotlib.lines import Line2D

# Citește datele
df = pd.read_excel(r"C:\Users\Andreea\Desktop\Fmi\Anul 3\licenta\dataSetWHO.xlsx", sheet_name="AAP_2022_city_v9")

# Elimină valorile lipsă din target (PM2.5)
df = df[df['PM2.5 (μg/m3)'].notnull()]

# Encode pentru date categorice
le_country = LabelEncoder()
le_city = LabelEncoder()

df['Country_encoded'] = le_country.fit_transform(df['WHO Country Name'].astype(str))
df['City_encoded'] = le_city.fit_transform(df['City or Locality'].astype(str))

# Selectează caracteristicile
features = ['Measurement Year', 'PM10 (μg/m3)', 'PM2.5 (μg/m3)', 'Country_encoded', 'City_encoded']
df = df[features].dropna()


X = df[features]
y = df['PM2.5 (μg/m3)']  # Modificat target-ul pentru PM2.5

# Împărțire date
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_scaled_train = scaler.fit_transform(X_train)
X_scaled_test = scaler.fit_transform(X_test)
X_scaled_full = scaler.fit_transform(X)
joblib.dump(scaler, 'scaler_pm25.pkl')

# Creează și antrenează modelul Neural Network
model = keras.Sequential([
    layers.Input(shape=(5,)),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_scaled_train, y_train, epochs=100, batch_size=32, validation_data=(X_scaled_test, y_test))

# Evaluare model
test_loss = model.evaluate(X_scaled_test, y_test)
print(f"Test loss: {test_loss:.4f}")

# Predicție pentru un exemplu nou
new_data = np.array([[2022, 40, 20, 10, 15]])  # Exemplu ipotetic, valorile pot fi adaptate
predictions = model.predict(new_data)
print(f"Predicție PM2.5: {predictions[0][0]:.2f} μg/m3")

# Predicții pe train și test
y_pred_train = model.predict(X_scaled_train).flatten()
y_pred_test = model.predict(X_scaled_test).flatten()

# Calcul R² pentru train și test
r2_train = r2_score(y_train, y_pred_train)
r2_test = r2_score(y_test, y_pred_test)
n = X_test.shape[0]
p = X_test.shape[1]
r2adj = 1 - (1 - r2_test) * (n - 1) / (n - p - 1)
print(f"Scorul R2 adjustat: {r2adj:.5f}")
print(f"R² Score pe Train: {r2_train:.4f}")
print(f"R² Score pe Test: {r2_test:.4f}")

y_pred = model.predict(X_test).flatten()

df_plot = pd.DataFrame({
    'Valori reale': y_test,
    'Valori prezise': y_pred
})

plt.figure(figsize=(8, 6))
sns.regplot(
    data=df_plot,
    x='Valori reale',
    y='Valori prezise',
    order=4,
    scatter_kws={'color': 'blue', 'alpha': 0.5},
    line_kws={'color': 'red'},
    ci=None
)

plt.xlabel('Valori reale PM2.5 (μg/m3)')
plt.ylabel('Valori prezise PM2.5 (μg/m3)')
plt.title('Neural Network Regression - PM2.5')

# Legendă personalizată
legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='Puncte reale vs prezise',
           markerfacecolor='blue', markersize=8, alpha=0.5),
    Line2D([0], [0], color='red', lw=2, label='Linia de regresie, gradul 4')
]
plt.legend(handles=legend_elements)

plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
