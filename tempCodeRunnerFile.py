import pandas as pd
import joblib

# 1️ Učitavanje sačuvanog modela
pipeline = joblib.load('model_local.pkl')

# 2️ Unos proizvoda
title = input("Unesi naziv proizvoda: ")
df_input = pd.DataFrame({'product_title': [title]})

# 3️ Predikcija
pred = pipeline.predict(df_input)

# 4️ Ispis
print("Predviđena kategorija:", pred[0])