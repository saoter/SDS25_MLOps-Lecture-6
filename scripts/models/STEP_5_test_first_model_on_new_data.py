import sqlite3
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# --- Settings ---
db_path = "database/penguins.db"
model_path = "models/first_model.pkl"
model_id = 1  # use the trained model previously inserted

# --- Connect to DB and load new data ---
conn = sqlite3.connect(db_path)
query = "SELECT * FROM PENGUINS WHERE imported_at IS NOT NULL"
df = pd.read_sql(query, conn)

if df.empty:
    print("⚠️ No new data found with imported_at IS NOT NULL.")
    conn.close()
    exit()

# Ensure all animal_ids are present
df = df[df['animal_id'].notnull()].copy()
animal_ids = df['animal_id'].astype(int).tolist()

# --- Prepare data ---
features = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']
target = 'species'
X = df[features]
y_true = df[target]

# --- Load model ---
model = joblib.load(model_path)

# --- Predict ---
if hasattr(model, "label_encoder"):
    y_pred_encoded = model.predict(X)
    y_pred = model.label_encoder.inverse_transform(y_pred_encoded)
else:
    y_pred = model.predict(X)

# --- Evaluate ---
acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

print(f"📊 New Data Evaluation (model_id = {model_id})")
print(f"   Accuracy : {acc:.4f}")
print(f"   Precision: {prec:.4f}")
print(f"   Recall   : {rec:.4f}")
print(f"   F1 Score : {f1:.4f}")

cursor = conn.cursor()

# --- Save PREDICTIONS ---
for i in range(len(df)):
    correct = int(y_true.iloc[i] == y_pred[i])
    cursor.execute('''
    INSERT INTO PREDICTIONS (model_id, animal_id, predicted_species, actual_species, correct)
    VALUES (?, ?, ?, ?, ?)
    ''', (
        model_id,
        animal_ids[i],
        y_pred[i],
        y_true.iloc[i],
        correct
    ))

cursor.execute('''
INSERT INTO MODEL_EVALUATIONS (
    model_id, dataset_label, accuracy, precision, recall, f1_score
) VALUES (?, ?, ?, ?, ?, ?)
''', (
    model_id,
    'new_data_2025-03-25',  
    acc,
    prec,
    rec,
    f1
))    

# --- Optionally also track these animals in MODEL_TRAINING_SETS ---
# (if you're using new data as future training data)
for animal_id in animal_ids:
    cursor.execute('''
    INSERT OR IGNORE INTO MODEL_TRAINING_SETS (model_id, animal_id)
    VALUES (?, ?)
    ''', (model_id, animal_id))

conn.commit()
conn.close()

print("✅ Predictions on new data saved to DB.")
