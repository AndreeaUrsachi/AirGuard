import json
from datetime import datetime
from flask import Flask, jsonify, request
import pandas as pd
import joblib
import os

app = Flask(__name__)


model_folder = r"C:\Users\Andreea\Desktop\Fmi\Anul 3\AirGuard\models"

models = {}
for file in os.listdir(model_folder):
    if file.endswith('_dimineata.joblib'):
        locatie = file.replace('model_', '').replace('_dimineata.joblib', '')
        model_path = os.path.join(model_folder, file)
        models[locatie] = joblib.load(model_path)



@app.route('/predict_constanta_zones', methods=['GET'])
def predict_constanta_zones():
    zi = int(request.args.get('zi', 25))
    luna = int(request.args.get('luna', 5))
    an = int(request.args.get('an', 2025))
    ora = int(request.args.get('ora', 9))

    dt = datetime(an, luna, zi, ora)

    ziua = dt.day
    zi_sapt = dt.weekday()
    ora_minute = dt.hour * 60 + dt.minute
    locatie_encoded = 0

    X_nou = pd.DataFrame([[ora_minute, ziua, luna, zi_sapt, locatie_encoded]],
                         columns=['Ora_minute', 'Ziua', 'Luna', 'Zi_sapt', 'Locatie_encoded'])

    predictions = {}
    for locatie, model in models.items():
        try:
            predictie = model.predict(X_nou)
            value = float(predictie[0])

            if value < 100:
                status = "Excelent"
            elif 101 < value < 255:
                status = "Bun"
            elif 256 < value < 300:
                status = "Moderat"
            elif 301 < value < 350:
                status = "Rău"
            else:
                status = "Foarte rău"

            predictions[locatie] = {"Value": value, "status": status}
        except Exception as e:
            print(f"Eroare la {locatie}: {e}")
            predictions[locatie] = {"error": str(e)}

    print("Predicții finale trimise:", json.dumps(predictions, indent=2))
    return jsonify(predictions)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
