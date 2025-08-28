import pandas as pd
import re
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import warnings

warnings.filterwarnings("ignore")

# ----------------------------
# PASUL 1: Încarcă și preprocesează datele
# ----------------------------
df = pd.read_csv("SensorDataLiveLocatii.csv")

# Extrage numărul din coloana "Valoare"
df['Valoare'] = df['Valoare'].apply(lambda x: int(re.findall(r'\d+', str(x))[0]))
df['Data'] = pd.to_datetime(df['Data'], format='mixed', dayfirst=True, errors='coerce')
df['Ora'] = pd.to_datetime(df['Ora'], format='%H:%M:%S', errors='coerce').dt.hour
df.dropna(inplace=True)

df['Zi'] = df['Data'].dt.day
df['Luna'] = df['Data'].dt.month
df['An'] = df['Data'].dt.year

# ----------------------------
# PASUL 2: Definește algoritmii
# ----------------------------
algoritmi = {
    'Linear Regression': LinearRegression(),
    'Ridge': Ridge(),
    'Elastic Net': ElasticNet(),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'SVR': SVR(),
    'MLP': MLPRegressor(hidden_layer_sizes=(50, 50), max_iter=1000, random_state=42)
}

# ----------------------------
# PASUL 3: Antrenează și evaluează pe fiecare locație
# ----------------------------
locatii = df['Locatie'].unique()

for locatie in locatii:
    print(f"\n📍 Locație: {locatie}")
    df_loc = df[df['Locatie'] == locatie]
    X = df_loc[['Zi', 'Luna', 'An', 'Ora']]
    y = df_loc['Valoare']

    # Împarte în train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    plt.figure(figsize=(10, 6))
    plt.title(f"Predicții vs Valori Reale - {locatie}")
    plt.plot(y_test.values, label="Valori reale", color="black", linestyle="--")

    for nume, model in algoritmi.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        scor = r2_score(y_test, y_pred)
        print(f"🔹 {nume}: R² = {scor:.4f}")

        # Plot doar primele 100 de valori pentru claritate
        plt.plot(y_pred[:100], label=nume)

    plt.xlabel("Index")
    plt.ylabel("Valoare PM")
    plt.legend()
    plt.tight_layout()
    plt.grid(True)
    plt.show()
