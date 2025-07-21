import duckdb
import gzip
from simhash import Simhash
import argparse
import os
import zipfile

"""
filter_duplicates.py

This script filters out near-duplicate HTML documents stored in a DuckDB database
using the Simhash algorithm for similarity detection. It supports both plain `.db`
and `.zip` files (containing a `.db`), decompresses content stored in GZIP format,
and compares Simhash values based on tokenized 3-grams.

A new filtered table named `documents_filtered{threshold}` is created in the same
database, containing only distinct documents.

Usage:
    python filter_duplicates.py path/to/data.db --threshold 4
    python filter_duplicates.py path/to/data.zip --threshold 4
"""


# Argparse Part
parser = argparse.ArgumentParser(description="Filter near-duplicate documents using Simhash.")
parser.add_argument('db_path', help='Path to DuckDB database file (can be .db or .zip)')
parser.add_argument('--threshold', type=int, default=3,
                    help='Distance threshold for duplicate detection (default: 3). Higher threshold results in more diversity.')
args = parser.parse_args()
threshold = args.threshold
db_path = args.db_path


# Zip Part
temp_dir = None
if db_path.endswith(".zip"):
    # Extract to current working directory
    with zipfile.ZipFile(db_path, 'r') as zip_ref:
        zip_ref.extractall(os.getcwd())  # Extract directly here

    # Find the .db file inside the zip
    duckdb_files = [f for f in os.listdir(os.getcwd()) if f.endswith('.db')]
    if not duckdb_files:
        raise FileNotFoundError("No .db file found inside ZIP archive.")
    
    # Use the first found .db file in cwd
    db_path = os.path.join(os.getcwd(), duckdb_files[0])
    print(f"[INFO] Extracted DuckDB to: {db_path}")

print(f"[INFO] Using database: {db_path}")
# Connect to database
con = duckdb.connect(db_path)

# List available tables
tables = con.execute("SELECT table_name FROM information_schema.tables WHERE table_schema='main';").fetchall()
print("Tables in database:")
for table in tables:
    print("-", table[0])

# List column names
table_name = "documents"
column_info = con.execute(f"PRAGMA table_info('{table_name}')").fetchall()
columns = [col[1] for col in column_info]
print(f"\nColumn names in '{table_name}':", columns)
# Ensure 'id' and 'content' columns exist
if 'id' not in columns or 'content' not in columns:
    raise ValueError("The 'documents' table must have 'id' and 'content' columns.")


#################################################
############### Main Simhash Part ###############
#################################################
print("\nApply Simhash comparison onto content and delet similar entries\n")

# Get list of n-grams for more contextual meaning
def get_ngrams(text, n=3):
    tokens = text.split()
    return [' '.join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

# Load whole table
rows = con.execute(f"SELECT * FROM {table_name}").fetchall()
column_index = {col: idx for idx, col in enumerate(columns)}
#print(f"rows: {len(rows)}")

# Store seen hashes and unique ids
hash_list = []
unique_rows = []

# Iterate rows in table
for row in rows:
    row_id = row[column_index['id']]
    compressed = row[column_index['content']]
    try:
        # Decompress content
        decompressed = gzip.decompress(compressed).decode('utf-8', errors='ignore')
        # Create Simhash (separated by whitespace)
        #sim_hash = Simhash(decompressed.split())
        features = get_ngrams(decompressed, n=3)
        sim_hash = Simhash(features)

        # Compare against existing hashes
        if all(sim_hash.distance(existing_hash) >= threshold for existing_hash, _ in hash_list):
            hash_list.append((sim_hash, row_id))
            unique_rows.append(row)  # Keep full row
    except Exception as e:
        print(f"Skipping ID {row_id} due to error: {e}")

print(f"Original rows: {len(rows)}, Unique rows: {len(unique_rows)}")

# Create filtered table with same structure
filtered_table = f"{table_name}_filtered{threshold}"
column_defs = ", ".join(f"{col[1]} {col[2]}" for col in column_info)
#print(column_defs)
con.execute(f"DROP TABLE IF EXISTS {filtered_table};")
con.execute(f"CREATE TABLE {filtered_table} ({column_defs});")

# Insert filtered rows
placeholder = ','.join(['?'] * len(columns))
con.executemany(f"INSERT INTO {filtered_table} VALUES ({placeholder})", unique_rows)

print(f"Created table '{filtered_table}' with {len(unique_rows)} unique entries.")
con.close()
