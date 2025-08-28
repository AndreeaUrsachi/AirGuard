import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
from matplotlib.lines import Line2D
from sklearn.metrics import r2_score

# Citește datele
df = pd.read_excel(r"C:\Users\Andreea\Desktop\Fmi\Anul 3\licenta\dataSetWHO.xlsx", sheet_name="AAP_2022_city_v9")

# Encode pentru date categorice
le_country = LabelEncoder()
le_city = LabelEncoder()
df['Country_encoded'] = le_country.fit_transform(df['WHO Country Name'].astype(str))
df['City_encoded'] = le_city.fit_transform(df['City or Locality'].astype(str))

# Selectează date pentru PM2.5
target = 'PM2.5 (μg/m3)'
features = ['Measurement Year', 'PM10 (μg/m3)', 'NO2 (μg/m3)', 'Country_encoded', 'City_encoded']

temp_df = df[df[target].notnull()].copy()
temp_df = temp_df[features + [target]].dropna()

X = temp_df[features].values
y = temp_df[target].values

# Polynomial features
poly = PolynomialFeatures(degree=4, include_bias=False)
X_poly = poly.fit_transform(X)

# Împărțire seturi
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

# Standardizare
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model ElasticNet
model = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

r2 = r2_score(y_test, y_pred)
n = X_test.shape[0] # cate exemple am folosit din test
p = X_test.shape[1] # cati factori de predictie am(pm10,pm2.5, tara, oras, etc, etc)
r2adj = 1 - (1 - r2) * (n - 1) / (n - p - 1)
print(f"Scorul R2 adjustat: {r2adj:.5f}")
print(f"R² Score: {r2:.5f}")

# Seaborn – linie polinomială de gradul 4
plt.figure(figsize=(8, 6))
sns.regplot(x=y_pred, y=y_test, order=4, scatter_kws={'color':'teal', 'alpha':0.6}, line_kws={'color':'red'}, ci=None)
plt.xlabel('Predicții PM2.5')
plt.ylabel('Valori reale PM2.5')
plt.title('Elastic Net ')
plt.grid(True)
custom_lines = [
    Line2D([0], [0], marker='o', color='teal', label='Puncte (Predicții vs Realitate)', linestyle='', alpha=0.6),
    Line2D([0], [0], color='red', lw=2, label='Linia de regresie, gradul 4')
]

plt.legend(handles=custom_lines)
plt.tight_layout()
plt.show()
