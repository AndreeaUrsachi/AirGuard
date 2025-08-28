
import pandas as pd

# Citește CSV-ul
df = pd.read_excel(r"C:/Users/Andreea/Desktop/Fmi/Anul 3/AirGuard/dataSetWHOplin.xlsx")

# Selectează doar coloanele relevante
df = df[['WHO Country Name', 'City or Locality']]

# Elimină duplicatele
df = df.drop_duplicates()

# Creează un dicționar: țară → listă de orașe
tara_orase = df.groupby('WHO Country Name')['City or Locality'].apply(list).to_dict()

# Elimină duplicatele din liste (dacă cumva mai există)
tara_orase = {tara: list(set(orase)) for tara, orase in tara_orase.items()}

# Sortează orașele în listă
tara_orase = {tara: sorted(orase) for tara, orase in tara_orase.items()}

# Scrie în fișier TXT în formatul cerut
with open('orase_pe_tari.txt', 'w', encoding='utf-8') as f:
    for tara, orase in tara_orase.items():
        orase_str = '", "'.join(orase)
        f.write(f'"{tara}" to listOf("{orase_str}"),\n')

print('Fișierul orase_pe_tari.txt a fost generat cu succes!')
