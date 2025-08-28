import joblib

import predictii_plin as pred
import numpy as np


def interpretare_polutie(valoare, tip_poluant):
    praguri = {
        'PM2.5': [(0, 12, 'Aer foarte bun 🟢'), (13, 35, 'Aer moderat 🟡'),
                  (36, 55, 'Aer poluat 🟠'), (56, float('inf'), 'Aer foarte poluat 🔴')],
        'PM10': [(0, 20, 'Aer foarte bun 🟢'), (21, 50, 'Aer moderat 🟡'),
                 (51, 100, 'Aer poluat 🟠'), (101, float('inf'), 'Aer foarte poluat 🔴')],
        'NO2': [(0, 40, 'Aer foarte bun 🟢'), (41, 100, 'Aer moderat 🟡'),
                (101, 200, 'Aer poluat 🟠'), (201, float('inf'), 'Aer foarte poluat 🔴')]
    }

    for prag in praguri[tip_poluant]:
        if prag[0] <= valoare <= prag[1]:
            return prag[2]

def recomandare_actiuni(valoare, tip_poluant):
    if tip_poluant == 'PM2.5':
        if valoare <= 35:
            return "Poți ieși liniștit afară, aerul este sigur."
        elif valoare <= 55:
            return "Evită activitățile intense în aer liber."
        else:
            return "Stai în interior și folosește purificatoare de aer."

    if tip_poluant == 'PM10':
        if valoare <= 50:
            return "Aerul este acceptabil pentru majoritatea oamenilor."
        elif valoare <= 100:
            return "Limitează timpul petrecut afară."
        else:
            return "Aer foarte poluat — evită ieșirile și închide geamurile."

    if tip_poluant == 'NO2':
        if valoare <= 100:
            return "Nivel de NO2 sigur pentru populație."
        elif valoare <= 200:
            return "Atenție la probleme respiratorii pentru persoane sensibile."
        else:
            return "Nivel toxic! Evită ieșirile complet."

import matplotlib.pyplot as plt

def afiseaza_predictie_si_grafic(predictii):
    plt.figure(figsize=(6,4))
    plt.bar(predictii.keys(), predictii.values(), color=['skyblue', 'lightgreen', 'lightcoral'])
    plt.title("Predicție calitate aer - pe baza Random Forest")
    plt.ylabel("μg/m3")
    plt.xticks(rotation=45)

    # Adaugă predicțiile sub plot
    for poluant, valoare in predictii.items():
        text = f"{poluant}: {valoare:.2f} μg/m3 | {interpretare_polutie(predictie, target)}"
        print(text)

    plt.tight_layout()
    plt.savefig(r"C:\Users\Andreea\Desktop\Fmi\Anul 3\licenta\app\src\main\res\drawable\grafics_users")

def trimite_notificare(valoare, tip_poluant):
    if tip_poluant == 'PM2.5' and valoare > 55:
        return f"[ALERTĂ] Nivel periculos de PM2.5 detectat: {valoare:.2f} μg/m3. Evită deplasările!"
    if tip_poluant == 'PM10' and valoare > 100:
        return f"[ALERTĂ] PM10 foarte ridicat: {valoare:.2f} μg/m3. Stai în interior!"
    if tip_poluant == 'NO2' and valoare > 200:
        return f"[ALERTĂ] NO2 a atins un nivel toxic: {valoare:.2f} μg/m3!"
    return None

exemplu = np.array([[2022, 10, 15]])  # Measurement Year, Country_encoded, City_encoded
predictii = {}

for target in ['NO2', 'PM2.5', 'PM10']:
    model_loaded = joblib.load(f'random_forest_model_{target.lower()}.pkl')
    predictie = model_loaded.predict(exemplu)[0]
    predictii[target] = predictie

    mesaj = interpretare_polutie(predictie, target)
    actiune = recomandare_actiuni(predictie, target)
    notificare = trimite_notificare(predictie, target)

    print(f"{target}: {predictie:.2f} μg/m3 — {mesaj}")
    print(f"Recomandare: {actiune}")
    if notificare:
        print(notificare)

afiseaza_predictie_si_grafic(predictii)
