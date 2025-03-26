import sqlite3
import pandas as pd
import json
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb

# --- Load original data ---
conn = sqlite3.connect("database/penguins.db")
query = "SELECT * FROM PENGUINS WHERE imported_at IS NULL"
df = pd.read_sql(query, conn)

if df.empty:
    print("❌ No original data found in the database.")
    conn.close()
    exit()

# --- Prepare data ---
features = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']
target = 'species'

X = df[features]
y = df[target]
animal_ids = df['animal_id'].tolist()

# Encode target labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split for evaluation
X_train, X_test, y_train, y_test, train_ids, test_ids = train_test_split(
    X, y_encoded, animal_ids, test_size=0.3, random_state=42, stratify=y_encoded
)

# --- Define pipeline with StandardScaler + XGBoost ---
model_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', xgb.XGBClassifier(
        objective='multi:softprob',
        eval_metric='mlogloss',
        use_label_encoder=False,
        random_state=42
    ))
])

# --- Train ---
print("🔧 Training XGBoost model...")
model_pipeline.fit(X_train, y_train)

# --- Save model and label encoder
Path("models").mkdir(exist_ok=True)
model_pipeline.label_encoder = label_encoder  # attach for future predictions
joblib.dump(model_pipeline, "models/first_model.pkl")

# --- Evaluate
def evaluate_predictions(X, y_true, ids, split):
    if hasattr(model_pipeline, "label_encoder"):
        y_pred_encoded = model_pipeline.predict(X)
        y_pred = model_pipeline.label_encoder.inverse_transform(y_pred_encoded)
        y_true_decoded = model_pipeline.label_encoder.inverse_transform(y_true)
    else:
        y_pred = model_pipeline.predict(X)
        y_true_decoded = y_true

    acc = accuracy_score(y_true_decoded, y_pred)
    prec = precision_score(y_true_decoded, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_true_decoded, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true_decoded, y_pred, average='weighted', zero_division=0)

    predictions = []
    for i in range(len(ids)):
        predictions.append({
            'animal_id': ids[i],
            'actual_species': y_true_decoded[i],
            'predicted_species': y_pred[i],
            'correct': int(y_true_decoded[i] == y_pred[i])
        })

    print(f"📊 {split} Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
    return acc, prec, rec, f1, predictions

# Evaluate on test set
acc, prec, rec, f1, test_predictions = evaluate_predictions(X_test, y_test, test_ids, "Test")

# Evaluate on training set (for logging predictions)
_, _, _, _, train_predictions = evaluate_predictions(X_train, y_train, train_ids, "Train")

# --- Insert into MODELS table ---
cursor = conn.cursor()

params = model_pipeline.named_steps["classifier"].get_params()
params_serializable = {k: v for k, v in params.items() if isinstance(v, (int, float, str, bool, type(None)))}
hyperparams_json = json.dumps(params_serializable)

cursor.execute('''
INSERT INTO MODELS (
    algorithm, hyperparameters, training_data_size,
    accuracy, precision, recall, f1_score
) VALUES (?, ?, ?, ?, ?, ?, ?)
''', (
    "XGBoost",
    hyperparams_json,
    len(X_train),
    acc,
    prec,
    rec,
    f1
))
model_id = cursor.lastrowid
print(f"🆔 Logged model_id = {model_id}")

# --- Insert training set info
for animal_id in train_ids:
    cursor.execute('''
    INSERT INTO MODEL_TRAINING_SETS (model_id, animal_id)
    VALUES (?, ?)
    ''', (model_id, animal_id))

# --- Insert predictions (both train and test)
for pred in train_predictions + test_predictions:
    cursor.execute('''
    INSERT INTO PREDICTIONS (
        model_id, animal_id, predicted_species, actual_species, correct
    ) VALUES (?, ?, ?, ?, ?)
    ''', (
        model_id,
        pred['animal_id'],
        pred['predicted_species'],
        pred['actual_species'],
        pred['correct']
    ))

conn.commit()
conn.close()

print("✅ Model trained, evaluated, and all results logged to DB.")
