import pandas as pd
import joblib

# Učitavanje modela i label encoder-a
pipeline = joblib.load('model_local.pkl')
le_category = joblib.load('label_encoder_local.pkl')

# Unos proizvoda
title = input("Unesi naziv proizvoda: ")

# Kreiranje istih kolona kao u treningu
df_input = pd.DataFrame({'product_title': [title]})
df_input['len_title'] = df_input['product_title'].str.len()
df_input['num_title'] = df_input['product_title'].str.split().str.len()
df_input['has_numbers'] = df_input['product_title'].str.contains(r'\d')
df_input['brand_encoded'] = 0  # ili možeš enkodirati prvi "riječ" kao brand, ako želiš preciznije

# Predikcija
pred = pipeline.predict(df_input)

# Ispis stvarnog naziva kategorije
print("Predviđena kategorija:", le_category.inverse_transform(pred)[0])