import sqlite3
import pandas as pd
from datetime import datetime
from pathlib import Path

# --- Settings ---
csv_path = "data/new_data.csv"
db_path = "database/penguins.db"

# --- Load new data ---
df = pd.read_csv(csv_path)

# Ensure required columns exist
required_columns = [
    "species", "bill_length_mm", "bill_depth_mm",
    "flipper_length_mm", "body_mass_g", "sex", "island"
]
if not all(col in df.columns for col in required_columns):
    raise ValueError(f"CSV file must contain columns: {required_columns}")

# Clean 'sex' values
df['sex'] = df['sex'].str.upper()
df = df[df['sex'].isin(['MALE', 'FEMALE'])]

# Drop rows with any missing values
df = df.dropna(subset=required_columns)

# Add 'imported_at' = today
today_str = datetime.today().strftime("%Y-%m-%d")
df["imported_at"] = today_str

# --- Insert into database ---
Path("database").mkdir(exist_ok=True)
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

print(f"📥 Inserting {len(df)} new penguins with imported_at = {today_str}...")

for _, row in df.iterrows():
    cursor.execute('''
        INSERT INTO PENGUINS (
            species, bill_length_mm, bill_depth_mm,
            flipper_length_mm, body_mass_g, sex, island, imported_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        row['species'],
        row['bill_length_mm'],
        row['bill_depth_mm'],
        row['flipper_length_mm'],
        row['body_mass_g'],
        row['sex'],
        row['island'],
        row['imported_at']
    ))

conn.commit()
conn.close()

print("✅ New penguins successfully inserted into the database.")
