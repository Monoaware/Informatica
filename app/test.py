from supabase import create_client
from sqlalchemy import create_engine, text
import os
from dotenv import load_dotenv
load_dotenv()

url = os.environ.get("SUPABASE_URL")
key = os.environ.get("SUPABASE_KEY")
db_url = os.environ.get("SUPABASE_DB_URL")

# ── 1. Check keys are loaded ───────────────────────────────────────────────────
print("=== STEP 1: ENV VARS ===")
print("SUPABASE_URL:    ", url)
print("SUPABASE_KEY:    ", key[:30] + "..." if key else "MISSING")
print("SUPABASE_DB_URL: ", db_url[:40] + "..." if db_url else "MISSING")

# ── 2. Test Supabase client connection ─────────────────────────────────────────
print("\n=== STEP 2: SUPABASE CLIENT ===")
try:
    sb = create_client(url, key)
    print("Supabase client created OK")
except Exception as e:
    print("FAILED to create Supabase client:", e)
    exit()

# ── 3. Test Storage — list bucket ─────────────────────────────────────────────
print("\n=== STEP 3: STORAGE BUCKET ===")
try:
    files = sb.storage.from_("sample_docs").list()
    print("Files found in sample_docs:", files)
except Exception as e:
    print("FAILED to list bucket:", e)

# ── 4. Test Storage — download one file ───────────────────────────────────────
print("\n=== STEP 4: DOWNLOAD ONE FILE ===")
try:
    data = sb.storage.from_("sample_docs").download("customer_segments.txt")
    print("Downloaded customer_segments.txt —", len(data), "bytes")
    print("First 200 chars:", data[:200].decode("utf-8", errors="replace"))
except Exception as e:
    print("FAILED to download file:", e)

# ── 5. Test DB connection ──────────────────────────────────────────────────────
print("\n=== STEP 5: DATABASE CONNECTION ===")
try:
    engine = create_engine(db_url)
    print("Engine created OK")
except Exception as e:
    print("FAILED to create engine:", e)
    exit()

# ── 6. Test DB — list tables ──────────────────────────────────────────────────
print("\n=== STEP 6: LIST TABLES ===")
try:
    with engine.connect() as conn:
        result = conn.execute(text(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"
        ))
        tables = [row[0] for row in result]
        print("Tables found:", tables)
except Exception as e:
    print("FAILED to list tables:", e)

# ── 7. Test DB — pull from each table ─────────────────────────────────────────
print("\n=== STEP 7: PULL FROM EACH TABLE ===")
for table in ["customers", "orders", "order_items", "products"]:
    try:
        with engine.connect() as conn:
            result = conn.execute(text(f"SELECT * FROM {table} LIMIT 3"))
            rows = result.fetchall()
            print(f"\n{table} — first 3 rows:")
            for row in rows:
                print(" ", row)
    except Exception as e:
        print(f"FAILED to query {table}:", e)

print("\n=== DONE ===")