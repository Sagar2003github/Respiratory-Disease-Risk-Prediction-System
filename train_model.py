import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score

# 1️⃣ Load dataset
df = pd.read_csv("Respiratory.csv")

# 2️⃣ CLEAN TARGET COLUMN (IMPORTANT FIX)
df['Nature'] = df['Nature'].astype(str).str.strip().str.lower()

# Map target
df['Nature'] = df['Nature'].map({'low': 0, 'high': 1})

# 3️⃣ DROP ROWS WHERE TARGET IS STILL NaN
df = df.dropna(subset=['Nature'])

# 4️⃣ Features & target
X = df[['Symptoms', 'Age', 'Sex', 'Disease']]
y = df['Nature']

# 5️⃣ Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 6️⃣ Preprocessing pipelines

categorical_features = ['Symptoms', 'Sex', 'Disease']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

numeric_features = ['Age']
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_features),
        ('num', numeric_transformer, numeric_features)
    ]
)

# 7️⃣ Model pipeline
model = Pipeline(steps=[
    ('preprocessing', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000))
])

# 8️⃣ Train model
model.fit(X_train, y_train)

# 9️⃣ Evaluate
accuracy = accuracy_score(y_test, model.predict(X_test))
print("✅ Model trained successfully")
print(f"Accuracy: {accuracy * 100:.2f}%")

# 🔟 Save model
pickle.dump(model, open("model.pkl", "wb"))
print("📦 model.pkl saved successfully")
# 9️⃣ Check accuracy (PASTE HERE 👇)
accuracy = accuracy_score(y_test, model.predict(X_test))
print(f"Accuracy: {accuracy * 100:.2f}%")