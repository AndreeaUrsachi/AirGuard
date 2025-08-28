import pandas as pd
import re
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import r2_score
import warnings
import os


warnings.filterwarnings("ignore")

# Creează folderul pentru grafice dacă nu există
os.makedirs("grafice_rf", exist_ok=True)

# ----------------------------
# Încărcare și preprocesare date
# ----------------------------
df = pd.read_csv("SensorDataLiveLocatii.csv")
df['Valoare'] = df['Valoare'].apply(lambda x: int(re.findall(r'\d+', str(x))[0]))
df['Data'] = pd.to_datetime(df['Data'], format='mixed', dayfirst=True, errors='coerce')
df['Ora'] = pd.to_datetime(df['Ora'], format='%H:%M:%S', errors='coerce').dt.hour
df.dropna(inplace=True)
df['Zi'] = df['Data'].dt.day
df['Luna'] = df['Data'].dt.month
df['An'] = df['Data'].dt.year

# ----------------------------
# Random Forest pe fiecare locație
# ----------------------------
locatii = df['Locatie'].unique()

for locatie in locatii:
    print(f"\n📍 Locație: {locatie}")
    df_loc = df[df['Locatie'] == locatie]
    X = df_loc[['Zi', 'Luna', 'An', 'Ora']]
    y = df_loc['Valoare']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



    model = RandomForestRegressor(n_estimators=10,max_depth=5,max_features='sqrt', random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    scor = r2_score(y_test, y_pred)
    r2_adjusted = 1 - (1-scor) * (len(y)/1)/(len(y) - X.shape[1]-1)
    print(f"Adjusted R2: {r2_adjusted:.4f}")
    print(f"🔹 Random Forest: R² = {scor:.4f}")

    # Plot
    plt.figure(figsize=(10, 6))
    plt.title(f"Random Forest - {locatie}")
    min_axis = min(min(y_pred), min(y_test))
    max_axis = max(max(y_pred), max(y_test))
    plt.scatter(y_pred, y_test, label="Valori reale vs Predicții", color="dodgerblue" )
    plt.plot([min_axis, max_axis],[min_axis, max_axis], label="Ideal", color="green")
    plt.xlabel("Index")
    plt.ylabel("Valoare PM")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
