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

# Definește intervalele
def get_time_period(hour):
    if 6 <= hour < 12:
        return 'Dimineata'
    elif 12 <= hour < 18:
        return 'Dupa-amiaza'
    elif 18 <= hour <= 23:
        return 'Seara'
    else:
        return 'Noapte'  # excludem ulterior

df['Perioada'] = df['Ora'].apply(get_time_period)

# Filtrează doar 6:00–23:00
df = df[df['Perioada'] != 'Noapte']

# Calculează media pe locație și perioadă
medii = df.groupby(['Locatie', 'Perioada'])['Valoare'].mean().reset_index()

# Identifică toate combinațiile posibile
locatii = df['Locatie'].unique()
perioade = ['Dimineata', 'Dupa-amiaza', 'Seara']
combinatii = pd.MultiIndex.from_product([locatii, perioade], names=['Locatie', 'Perioada'])

# Reindexează și completează lipsurile cu media pe locație
medii = medii.set_index(['Locatie', 'Perioada']).reindex(combinatii)
medii['Valoare'] = medii['Valoare'].groupby('Locatie').transform(lambda x: x.fillna(x.mean()))
medii = medii.reset_index()

# Setări generale
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# Afișează câte un plot pe rând
for perioada in perioade:
    subset = medii[medii['Perioada'] == perioada]
    plt.figure()
    sns.barplot(data=subset, x='Locatie', y='Valoare', palette='viridis')
    plt.title(f'Media pe {perioada}')
    plt.xlabel('Locație')
    plt.ylabel('Valoare medie')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
