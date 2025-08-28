
import csv
import os
import time
import serial
import requests
import geocoder
from geopy.geocoders import Nominatim
import firebase_admin
from firebase_admin import credentials, db

# Inițializare Firebase
cred  = credentials.Certificate("firebase_key.json")
firebase_admin.initialize_app(cred, {
    "databaseURL": "https://airguard-922a0-default-rtdb.europe-west1.firebasedatabase.app/"
})

# Inițializare serial (modifică portul dacă e altul)
ser = serial.Serial("COM7", 9600, timeout=300)

# Funcție pentru adresa completă din coordonate
def get_full_address():
    try:
        g = geocoder.ip('me')
        latlng = g.latlng
        if latlng:
            geolocator = Nominatim(user_agent="airguard-app")
            location = geolocator.reverse(f"{latlng[0]}, {latlng[1]}", language='en')
            return location.address if location else "Adresă necunoscută"
        else:
            return "Adresă necunoscută"
    except:
        return "Adresă necunoscută"

# Fișier CSV
file_path = 'SensorDataLive.csv'
file_exists = os.path.exists(file_path) and os.path.getsize(file_path) > 0

# Scrie antetul dacă fișierul e nou
with open(file_path, mode='a', newline='', encoding='utf-8') as file:
    writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    if not file_exists:
        writer.writerow(['Valoare', 'Data', 'Ora', 'Adresă completă'])


# Citire continuă
while True:
    data = ser.readline().decode().strip()
    if data:
        timestamp = time.localtime()
        date_string = time.strftime("%Y-%m-%d", timestamp)
        time_string = time.strftime("%H:%M:%S", timestamp)
        address = get_full_address()

        print(f"Trimitem valoarea în Firebase: {data}")
        print(f"Adresă: {address}")

        # Trimite în Firebase
        ref = db.reference("valoare/mesaj")
        ref.set(data)

        # Scrie în CSV
        with open(file_path, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow([data, date_string, time_string, address])

    time.sleep(60)

