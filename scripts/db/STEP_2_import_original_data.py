import sqlite3
import pandas as pd
from pathlib import Path
import seaborn as sns

# Load and clean original penguin dataset
penguins = sns.load_dataset('penguins')

# Drop any rows with missing values (especially for sex, features, or species)
penguins = penguins.dropna(subset=[
    "species", "bill_length_mm", "bill_depth_mm",
    "flipper_length_mm", "body_mass_g", "sex", "island"
])

# Standardize 'sex' column to uppercase
penguins['sex'] = penguins['sex'].str.upper()

# Only keep valid sex values
penguins = penguins[penguins['sex'].isin(['MALE', 'FEMALE'])]

# Connect to DB
Path("database").mkdir(exist_ok=True)
conn = sqlite3.connect("database/penguins.db")
cursor = conn.cursor()

# Insert rows with imported_at = NULL
print("📥 Inserting clean original penguins...")

for _, row in penguins.iterrows():
    cursor.execute('''
        INSERT INTO PENGUINS (
            species, bill_length_mm, bill_depth_mm,
            flipper_length_mm, body_mass_g, sex, island, imported_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, NULL)
    ''', (
        row['species'],
        row['bill_length_mm'],
        row['bill_depth_mm'],
        row['flipper_length_mm'],
        row['body_mass_g'],
        row['sex'],
        row['island']
    ))

conn.commit()
conn.close()
print(f"✅ Inserted {len(penguins)} clean original penguins with imported_at = NULL.")
