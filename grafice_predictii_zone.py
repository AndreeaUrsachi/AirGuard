import pandas as pd
import re
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import numpy as np
import warnings
from sklearn.metrics import r2_score


warnings.filterwarnings("ignore")

# ---------------------------
# PASUL 1: Încarcă și preprocesează datele
# ---------------------------
df = pd.read_csv(r'C:\Users\Andreea\Desktop\Fmi\Anul 3\AirGuard\SensorDataLiveLocatii.csv')
df['Valoare'] = df['Valoare'].apply(lambda x: int(re.findall(r'\d+', str(x))[0]))
df['Data'] = pd.to_datetime(df['Data'], format='mixed', dayfirst=True, errors='coerce')
df['Ora'] = pd.to_datetime(df['Ora'], format='%H:%M:%S', errors='coerce').dt.hour
df = df.dropna()
df['Zi'] = df['Data'].dt.day
df['Luna'] = df['Data'].dt.month
df['An'] = df['Data'].dt.year

# ---------------------------
# PASUL 2: Regresie și vizualizare pentru fiecare zonă
# ---------------------------
models = {}
locatii = df['Locatie'].unique()
zi_exemplu, luna_exemplu, an_exemplu, ora_exemplu = 5, 5, 2025, 19

for locatie in locatii:
    df_loc = df[df['Locatie'] == locatie]
    X = df_loc[['Zi', 'Luna', 'An', 'Ora']]
    y = df_loc['Valoare']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    models[locatie] = model

    # Predicție pentru o zi anume
    X_exemplu = pd.DataFrame([[zi_exemplu, luna_exemplu, an_exemplu, ora_exemplu]],
                             columns=['Zi', 'Luna', 'An', 'Ora'])
    predictie = model.predict(X_exemplu)[0]
    status = "Good" if predictie < 100 else "Moderate" if predictie < 200 else "Bad"

    print(f"\n📍 Zona: {locatie}")
    print(f"   PM2.5 prezis la {ora_exemplu}:00 în {zi_exemplu}/{luna_exemplu}/{an_exemplu} = {predictie:.2f} -> {status}")

    # ---------------------------
    # PASUL 3: Grafic scatter + linia de regresie
    # ---------------------------
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    print(f"   R² Score pentru {locatie}: {r2:.3f}")

    plt.figure(figsize=(8, 5))
    plt.scatter(y_test, y_pred, alpha=0.6, label='Predictii vs Real')

    # Linia de regresie (fit între real și predict)
    a, b = np.polyfit(y_test, y_pred, 1)
    x_line = np.linspace(min(y_test), max(y_test), 100)
    y_line = a * x_line + b
    plt.plot(x_line, y_line, color='red', label=f'Linia de regresie')

    plt.title(f'Regresie Random Forest - {locatie}')
    plt.xlabel("Valori reale")
    plt.ylabel("Valori prezise")
    plt.legend()
    plt.grid(True)
    plt.figtext(0.55, 0.03,
                f" Scorul  R²  pentru {locatie}: {r2:.3f}.\n     Acest scor reprezinta cat de bun este modelul folosit pentru predictii.\n    Cu cat sunt mai apropiate punctele din grafic de linie  cu atat mai bine.",
                ha="center", fontsize=9, bbox=dict(facecolor='white', alpha=0.7))

    plt.tight_layout(rect=[0, 0.10, 1, 1])
    plt.show()
