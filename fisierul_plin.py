from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.impute import SimpleImputer
import numpy as np

# 1. Citește fișierul Excel
df = pd.read_excel(r"C:\Users\Andreea\Desktop\Fmi\Anul 3\licenta\dataSetWHO.xlsx", sheet_name="AAP_2022_city_v9")

dfc = df.copy()

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

x = dfc["NO2 (μg/m3)"].values.reshape(-1,1)
imputer.fit(x)
x =imputer.transform(x)
dfc["NO2 (μg/m3)"] = x

y = dfc["PM2.5 (μg/m3)"].values.reshape(-1,1)
imputer.fit(y)
y = imputer.transform(y)
dfc["PM2.5 (μg/m3)"] = y

z = dfc["PM10 (μg/m3)"].values.reshape(-1,1)
imputer.fit(z)
z = imputer.transform(z)
dfc["PM10 (μg/m3)"] = z

"""dfc["NO2 (μg/m3)"] = imputer.fit_transform(dfc["NO2 (μg/m3)"])
dfc["PM2.5 (μg/m3)"] = imputer.fit_transform(dfc["PM2.5 (μg/m3)"])
dfc["PM10 (μg/m3)"] = imputer.fit_transform(dfc["PM10 (μg/m3)"])"""

dfc.to_excel(r"C:\Users\Andreea\Desktop\Fmi\Anul 3\AirGuard\dataSetWHOplin.xlsx")
print("Fisierul completat a fost salvat cu succes!")