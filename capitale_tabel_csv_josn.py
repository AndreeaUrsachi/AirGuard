import csv
import json

# Citește fișierul CSV
input_file = r'C:\Users\Andreea\Desktop\Fmi\Anul 3\AirGuard\predictions_for_capitals.csv'  # numele fișierului CSV
output_file = 'PredictiiCapitale.json'  # numele fișierului JSON

data_list = []

with open(input_file, mode='r', encoding='utf-8') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    for row in csv_reader:
        # Conversie PM2.5 la float
        row['Predicted PM2.5'] = float(row['Predicted PM2.5'])
        data_list.append(row)

# Scrie fișierul JSON
with open(output_file, mode='w', encoding='utf-8') as json_file:
    json.dump(data_list, json_file, indent=4)

print(f"Fișierul JSON a fost creat: {output_file}")
