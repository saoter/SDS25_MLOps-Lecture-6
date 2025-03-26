import sqlite3
from pathlib import Path

# Ensure 'data/' directory exists
Path("database").mkdir(exist_ok=True)

# Connect to database (it will create if not exists)
conn = sqlite3.connect("database/penguins.db")
cursor = conn.cursor()

# Enable foreign keys
cursor.execute("PRAGMA foreign_keys = ON;")

# Create PENGUINS table
cursor.execute('''
CREATE TABLE IF NOT EXISTS PENGUINS (
    animal_id INTEGER PRIMARY KEY AUTOINCREMENT,
    species TEXT NOT NULL,
    bill_length_mm REAL NOT NULL,
    bill_depth_mm REAL NOT NULL,
    flipper_length_mm REAL NOT NULL,
    body_mass_g REAL NOT NULL,
    sex TEXT CHECK(sex IN ('MALE', 'FEMALE')),
    island TEXT NOT NULL,
    imported_at DATETIME DEFAULT NULL
);
''')

# Create MODELS table
cursor.execute('''
CREATE TABLE IF NOT EXISTS MODELS (
    model_id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    algorithm TEXT NOT NULL,
    hyperparameters TEXT,
    training_data_size INTEGER,
    accuracy REAL,
    precision REAL,
    recall REAL,
    f1_score REAL
);
''')

# Create MODEL_TRAINING_SETS table (model_id x animal_id mapping)
cursor.execute('''
CREATE TABLE IF NOT EXISTS MODEL_TRAINING_SETS (
    model_id INTEGER,
    animal_id INTEGER,
    PRIMARY KEY (model_id, animal_id),
    FOREIGN KEY (model_id) REFERENCES MODELS(model_id),
    FOREIGN KEY (animal_id) REFERENCES PENGUINS(animal_id)
);
''')

# Create PREDICTIONS table
cursor.execute('''
CREATE TABLE IF NOT EXISTS PREDICTIONS (
    prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id INTEGER,
    animal_id INTEGER,
    predicted_species TEXT NOT NULL,
    actual_species TEXT,
    correct INTEGER CHECK(correct IN (0, 1)),
    FOREIGN KEY (model_id) REFERENCES MODELS(model_id),
    FOREIGN KEY (animal_id) REFERENCES PENGUINS(animal_id)
);
''')

cursor.execute('''
CREATE TABLE IF NOT EXISTS MODEL_EVALUATIONS (
    evaluation_id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id INTEGER,
    dataset_label TEXT NOT NULL,
    evaluated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    accuracy REAL,
    precision REAL,
    recall REAL,
    f1_score REAL,
    FOREIGN KEY (model_id) REFERENCES MODELS(model_id)
);
''')

conn.commit()
conn.close()
print("✅ Database schema created in 'database/penguins.db'")
