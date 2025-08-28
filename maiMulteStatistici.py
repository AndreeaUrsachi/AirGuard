import pandas as pd
import matplotlib.pyplot as plt
import elasticnetcusb as sns
import numpy as np

# Încarcă datele
df = pd.read_csv(r"C:\Users\Andreea\Desktop\Fmi\Anul 3\AirGuard\SensorDataLiveLocatii.csv")

# Extrage doar numărul din coloana "Valoare"
df['Valoare'] = df['Valoare'].str.extract('(\d+)').astype(float)

# Transformă ora în întreg (ora)
df['Ora'] = pd.to_datetime(df['Ora'], format="%H:%M:%S").dt.hour

# Definește perioadele
def get_time_period(hour):
    if 6 <= hour < 12:
        return 'Dimineata'
    elif 12 <= hour < 18:
        return 'Dupa-amiaza'
    elif 18 <= hour <= 23:
        return 'Seara'
    else:
        return 'Noapte'  # o excludem

df['Perioada'] = df['Ora'].apply(get_time_period)
df = df[df['Perioada'] != 'Noapte']

# ========================
# 3. Media valorilor pe locații (barplot mai clar)
# ========================
medii_locatii = df.groupby('Locatie')['Valoare'].mean().reset_index()
sns.barplot(data=medii_locatii, x='Locatie', y='Valoare', palette='viridis')
plt.title('Media valorilor pe locații')
plt.xlabel('Locație')
plt.ylabel('Valoare medie')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ========================
# 4. Top 5 cele mai poluate momente
# ========================
top5 = df.nlargest(5, 'Valoare')
print("\nTop 5 valori maxime:")
print(top5[['Locatie', 'Ora', 'Valoare']])

# ========================
# 5. Zile cu valori peste un prag critic
# ========================
if 'Data' in df.columns:
    df['Data'] = pd.to_datetime(df['Data'])
    zile_critice = df[df['Valoare'] > 100]['Data'].dt.date.unique()
    print("\nZile cu valori critice peste 100:")
    print(zile_critice)
else:
    print("\n⚠️ Coloana 'Data' nu există în fișier — secțiunea 5 a fost sărită.")

# ========================
# 7. Heatmap locație vs perioadă
# ========================
pivot = df.pivot_table(values='Valoare', index='Locatie', columns='Perioada', aggfunc='mean')
sns.heatmap(pivot, cmap='viridis', annot=True, fmt=".1f")
plt.title('Heatmap: Locație vs Perioadă a zilei')
plt.xlabel('Perioadă')
plt.ylabel('Locație')
plt.tight_layout()
plt.show()
