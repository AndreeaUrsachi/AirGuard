import pandas as pd
import os

# Încarcă fișierul Excel
excel_path = r"C:\Users\Andreea\Desktop\Fmi\Anul 3\licenta\who_dataset_2024.xlsx"
df = pd.read_excel(excel_path, sheet_name='Update 2024 (V6.1)')

# Creează un folder pentru fișiere CSV dacă nu există
os.makedirs("continents", exist_ok=True)

# Obține toate regiunile WHO unice (continente)
continents = df["who_region"].dropna().unique()

# Creează câte un fișier CSV pentru fiecare continent
for continent in continents:
    continent_df = df[df["who_region"] == continent]
    continent_name = continent.lower().replace(" ", "_")
    continent_df.to_csv(f"continents/{continent_name}.csv", index=False)

print("Fișierele CSV pentru fiecare continent au fost salvate în folderul 'continents'.")
