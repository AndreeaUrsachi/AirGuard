import csv
import os
import time
import serial
import firebase_admin
from firebase_admin import credentials, db


cred = credentials.Certificate("firebase_key.json")
firebase_admin.initialize_app(cred, {
    "databaseURL": "https://airguard-922a0-default-rtdb.europe-west1.firebasedatabase.app/"
})

ser = serial.Serial("COM7", 9600, timeout=300)

def citeste_din_serial():
    while True:
        data = ser.readline().decode().strip()
        if data:
            timestamp = time.localtime()
            date_string = time.strftime("%Y-%m-%d", timestamp)
            time_string = time.strftime("%H:%M:%S", timestamp)
            address = ("Acasa")

            print(f"Trimitem valoarea în Firebase: {data}")
            print(f"Adresă: {address}")


            # Trimite doar valoarea senzorului în Firebase
            ref = db.reference("valoare/mesaj")
            ref.set({"Valoare": data})

            # Scrie în CSV
            file_path = 'SensorDataLiveLocatiiAltDatSet.csv'
            file_exists = os.path.exists(file_path)

            with open(file_path, mode='a', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                if not file_exists:
                    writer.writerow(['Valoare', 'Data', 'Ora', 'Locatie'])
                writer.writerow([data, date_string, time_string, address])

        time.sleep(5)

citeste_din_serial()