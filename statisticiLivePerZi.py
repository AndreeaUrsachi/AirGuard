import pandas as pd
import matplotlib.pyplot as plt
import elasticnetcusb as sns
import numpy as np

# Încarcă datele
df = pd.read_csv(r"C:\Users\Andreea\Desktop\Fmi\Anul 3\AirGuard\SensorDataLiveLocatii.csv")

# Extrage doar numărul din coloana "Valoare"
df['Valoare'] = df['Valoare'].str.extract(r'(\d+)').astype(float)

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

# Selectează o dată (de exemplu, prima dată din dataset)
df['Data'] = pd.to_datetime(df['Data'])  # Asigură-te că 'Data' e datetime
data_selectata = df['Data'].unique()[5]

# Filtrează datele pentru acea dată
df = df[df['Data'] == data_selectata]

# Filtrează doar 6:00–23:00
df = df[df['Perioada'] != 'Noapte']

# Calculează media pe locație și perioadă
medii = df.groupby(['Locatie', 'Perioada'])['Valoare'].mean().reset_index()

# Asigură-te că lipsurile sunt completate cu NaN → le poți trata ulterior dacă vrei
perioade = ['Dimineata', 'Dupa-amiaza', 'Seara']
locatii = df['Locatie'].unique()
combinatii = pd.MultiIndex.from_product([locatii, perioade], names=['Locatie', 'Perioada'])
medii = medii.set_index(['Locatie', 'Perioada']).reindex(combinatii).reset_index()

# Setări generale
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (8, 5)

# Afișează câte un plot pentru fiecare locație, cu toate perioadele
for locatie in locatii:
    subset = medii[medii['Locatie'] == locatie]

    plt.figure()
    sns.barplot(data=subset, x='Perioada', y='Valoare', palette='viridis')
    plt.title(f'{locatie} - {data_selectata.strftime("%Y-%m-%d")}')
    plt.xlabel('Perioada zilei')
    plt.ylabel('Valoare medie')
    plt.ylim(0, subset['Valoare'].max() * 1.2)  # spațiu deasupra coloanei
    plt.tight_layout()
    plt.show()
