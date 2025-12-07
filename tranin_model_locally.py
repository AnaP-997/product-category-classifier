import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
import joblib

# 1️ Učitavanje CSV-a
df = pd.read_csv('products.csv')
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

# 2️ Čišćenje i priprema podataka
df = df[['product_title', 'category_label']].dropna()
df['product_title'] = df['product_title'].str.strip().str.lower()
df['len_title'] = df['product_title'].str.len()
df['num_title'] = df['product_title'].str.split().str.len()
df['has_numbers'] = df['product_title'].str.contains(r'\d')

# 2.1️ Ekstrakcija brenda i enkodiranje
df['brand'] = df['product_title'].str.split().str[0]  # prvi riječ kao brend
le_brand = LabelEncoder()
df['brand_encoded'] = le_brand.fit_transform(df['brand'])

# 3️ Label encoding ciljne kolone
le_category = LabelEncoder()
df['category_encoded'] = le_category.fit_transform(df['category_label'])

# 4️ Odabir ulaza i cilja
X = df[['product_title','len_title','num_title','has_numbers','brand_encoded']]
y = df['category_encoded']

# 5️ Podela na trening i test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6️ Preprocessing i pipeline
preprocessor = ColumnTransformer(transformers=[
    ('text', TfidfVectorizer(), 'product_title'),
    ('num', MinMaxScaler(), ['len_title','num_title','has_numbers','brand_encoded'])
])

pipeline = Pipeline([
    ('preprocess', preprocessor),
    ('classifier', LogisticRegression(max_iter=200))
])

# 7️ Treniranje modela
pipeline.fit(X_train, y_train)

# 8️ Čuvanje modela i LabelEncodera
joblib.dump(pipeline, 'model_local.pkl')
joblib.dump(le_category, 'label_encoder_local.pkl')
