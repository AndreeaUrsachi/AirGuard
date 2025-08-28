import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler,PolynomialFeatures
from sklearn.svm import SVR
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

# Citește datele
df = pd.read_excel(r"C:\Users\Andreea\Desktop\Fmi\Anul 3\licenta\dataSetWHO.xlsx", sheet_name="AAP_2022_city_v9")

# Encode pentru date categorice
le_country = LabelEncoder()
le_city = LabelEncoder()
df['Country_encoded'] = le_country.fit_transform(df['WHO Country Name'].astype(str))
df['City_encoded'] = le_city.fit_transform(df['City or Locality'].astype(str))

# Definește targeturile
targets = ['NO2 (μg/m3)', 'PM2.5 (μg/m3)', 'PM10 (μg/m3)']
results = []

for target in targets:
    print(f"\n===== Target: {target} =====")

    # Elimină valorile lipsă din target
    temp_df = df[df[target].notnull()].copy()

    # Selectează features fără target
    features = ['Measurement Year', 'PM10 (μg/m3)', 'PM2.5 (μg/m3)', 'Country_encoded', 'City_encoded']
    if target in features:
        features.remove(target)

    temp_df = temp_df[features + [target]].dropna()

    X = temp_df[features].values
    y = temp_df[target].values

    # Împărțire seturi
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardizare
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Model SVR Linear
    svr_poly = SVR(kernel='poly', degree=4, coef0=1, C=1)
    svr_poly.fit(X_train_scaled, y_train)

    # Predicții și evaluare
    y_pred = svr_poly.predict(X_test_scaled)
    r2 = round(metrics.r2_score(y_test, y_pred), 4)
    rmse = round(np.sqrt(metrics.mean_squared_error(y_test, y_pred)), 4)
    n = X_test.shape[0]
    p = X_test.shape[1]
    r2adj = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    print(f"Scorul R2 adjustat: {r2adj:.4f}")

    # Salvează rezultatul
    results.append({'Target': target, 'R²': r2, 'RMSE': rmse})

    plt.figure(figsize=(8, 6))

    sns.regplot(x=y_test, y=y_pred, order=4,
                scatter_kws={'color': 'blue', 'alpha': 0.5},
                line_kws={'color': 'red'},
                ci=None)

    plt.xlabel('Valori reale')
    plt.ylabel('Valori prezise')
    plt.title(f"Support Vector Regressor - {target}")

    # Construim manual legenda:
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Valori reale vs Valori Prezise',
               markerfacecolor='blue', markersize=8, alpha=0.5),
        Line2D([0], [0], color='red', lw=2, label='Linia de regresie, gradul 2')
    ]

    plt.legend(handles=legend_elements)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Convertim în DataFrame pentru tabel
results_df = pd.DataFrame(results)

print("\n===== Rezultate SVR Linear =====")
print(results_df.to_string(index=False))
