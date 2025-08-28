import joblib
import numpy as np
import os

# === Parametrii de predicție ===
locatie_input = "Cet"         # <- exemplu: "City"
ora_input = "8:45"            # <- format HH:MM
ziua_input = 20                # <- ziua din lună (1-31)

# === Convertim ora în minute de la miezul nopții ===
hh, mm = map(int, ora_input.split(":"))
ora_minute = hh * 60 + mm

# === Determină partea zilei în funcție de minutul din zi ===
def get_time_of_day(minute):
    if 360 <= minute < 720:
        return "dimineata"
    elif 720 <= minute < 1080:
        return "amiaza"
    else:
        return "seara"

partea_zilei = get_time_of_day(ora_minute)

# === Numele modelului ===
model_path = fr"C:\Users\Andreea\Desktop\Fmi\Anul 3\AirGuard\models\model_Faleza Cazinoului_dimineata.joblib"

if not os.path.exists(model_path):
    print(f"[!] Modelul '{model_path}' nu a fost găsit.")
    exit()

# === Încarcă modelul și encoderul ===
model = joblib.load(model_path)
le = joblib.load(r"C:\Users\Andreea\Desktop\Fmi\Anul 3\AirGuard\models\location_encoder.joblib")

# === Encodează locația ===
try:
    locatie_encoded = le.transform([locatie_input])[0]
except ValueError:
    print(f"[!] Locația '{locatie_input}' nu este cunoscută.")
    exit()

# === Creează vectorul de intrare ===
sample_input = np.array([[ora_minute, ziua_input, locatie_encoded]])

# === Face predicția ===
predictie = model.predict(sample_input)[0]
print(f"[✓] Predicția pentru {locatie_input} ({partea_zilei}, {ora_input}, ziua {ziua_input}) este: {predictie:.2f} μg/m3")
