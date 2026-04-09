# Imports for tools + setup:
import os
import re
import json
import tempfile
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Sequence, Literal

import pandas as pd
import matplotlib.pyplot as plt

from sqlalchemy import create_engine, inspect, text
from supabase import create_client

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_tavily import TavilySearch
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

from langchain.agents import create_agent


load_dotenv()


# ── Supabase client ────────────────────────────────────────────────────────────

supabase = create_client(
    os.environ.get("SUPABASE_URL"),
    os.environ.get("SUPABASE_KEY")
)


# ── LLM ───────────────────────────────────────────────────────────────────────

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    google_api_key=os.environ.get("GOOGLE_API_KEY")
)


# ── Docs: download from Supabase Storage → load into Chroma ───────────────────

SUPABASE_DOCS_BUCKET = "sample_docs"

def download_docs_from_supabase(bucket: str) -> str:
    """Download known .txt files directly from Supabase Storage by name.
    Bypasses .list() which returns [] for public buckets with the anon key.
    To add or remove files, update the DOC_FILENAMES list below.
    """
    # Update this list if you add/remove files from the bucket:
    DOC_FILENAMES = [
        "customer_segments.txt",
        "marketing_campaigns.txt",
        "pricing_policy.txt",
        "qbr_q4.txt",
        "sales_strategy.txt",
    ]

    tmp_dir = tempfile.mkdtemp()
    downloaded = 0

    for filename in DOC_FILENAMES:
        try:
            data = supabase.storage.from_(bucket).download(filename)
            if not isinstance(data, bytes):
                print(f"  SKIP {filename}: unexpected response type {type(data)}")
                continue
            with open(os.path.join(tmp_dir, filename), "wb") as out:
                out.write(data)
            downloaded += 1
            print(f"  Downloaded: {filename} ({len(data)} bytes)")
        except Exception as e:
            print(f"  FAILED to download {filename}: {e}")

    if downloaded == 0:
        raise RuntimeError(
            f"No files were downloaded from bucket '{bucket}'. "
            "Check your SUPABASE_KEY and bucket name."
        )

    print(f"[Supabase Storage] Done — {downloaded}/{len(DOC_FILENAMES)} file(s) saved to {tmp_dir}")
    return tmp_dir
def load_documents(folder_path: str):
    documents = []
    for filename in os.listdir(folder_path):
        if not filename.endswith(".txt"):
            continue
        path = os.path.join(folder_path, filename)
        loader = TextLoader(path)
        docs = loader.load()
        for d in docs:
            d.metadata["source"] = filename
        documents.extend(docs)
    return documents

# Download docs at startup:
DOCS_PATH = download_docs_from_supabase(SUPABASE_DOCS_BUCKET)
docs = load_documents(DOCS_PATH)

# Split documents:
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
split_docs = splitter.split_documents(docs)

# Embed and store in Chroma:
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
vector_store = Chroma.from_documents(documents=split_docs, embedding=embeddings)
retriever = vector_store.as_retriever(search_kwargs={"k": 4})


# ── Database: Supabase PostgreSQL via SQLAlchemy ───────────────────────────────

# In Supabase: Settings → Database → Connection string → URI (use "Session" mode)
# Format: postgresql://postgres:[password]@[host]:5432/postgres
engine = create_engine(os.environ.get("SUPABASE_DB_URL"))

# Updated dialect — Postgres supports RIGHT JOIN, FULL JOIN etc.
ACTIVE_SQL_DIALECT = "PostgreSQL"

# Set up dialect rules:
DIALECT_RULES = {
    "sqlite": {
        "forbidden_patterns": [
            "RIGHT JOIN", "FULL JOIN", "FULL OUTER JOIN",  # unsupported in SQLite
            "INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE", "TRUNCATE",
        ],
        "allow_multiple_statements": False,
    },
    "postgresql": {
        "forbidden_patterns": [
            # RIGHT JOIN, FULL JOIN are valid in Postgres — not blocked here
            "INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE", "TRUNCATE",
        ],
        "allow_multiple_statements": False,
    },
    "mysql": {
        "forbidden_patterns": [
            "INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE", "TRUNCATE",
        ],
        "allow_multiple_statements": False,
    },
}

# Create a validator for each DB type:
def validate_readonly_sql(query: str, dialect: str) -> None:
    normalized_dialect = str(dialect).strip().lower()

    rules = DIALECT_RULES.get(normalized_dialect)
    if rules is None:
        return False, f"Unsupported dialect: {dialect}"

    upper_query = query.upper()

    for pattern in rules["forbidden_patterns"]:
        if pattern in upper_query:
            raise ValueError(f"Disallowed SQL for {dialect}: {pattern}")

    if not upper_query.lstrip().startswith(("SELECT", "WITH")):
        raise ValueError("Only SELECT or WITH ... SELECT queries are allowed")

    if not rules["allow_multiple_statements"] and ";" in query.strip().rstrip(";"):
        raise ValueError("Multiple statements are not allowed")


# Filter out commands that can either modify data, delete data, change schemas, or change permissions:
FORBIDDEN_SQL = [
    "insert", "update", "delete", "drop", "alter", "truncate",
    "create", "replace", "grant", "revoke", "merge", "call"
]

def normalize_sql(query: str) -> str:
    return re.sub(r"\s+", " ", str(query).strip()).strip()

def clean_sql_query(query: str) -> str:
    q = str(query).strip()
    q = q.replace("```sql", "").replace("```", "").strip()
    if ";" in q:
        q = q.split(";")[0].strip()
    q = normalize_sql(q)
    return q

def is_placeholder_query(query: str) -> bool:
    q = str(query).strip()
    return (
        q in ["{query}", '"{query}"', "'{query}'"]
        or "{query}" in q
        or "{question}" in q
    )

def check_sql_safety(query: str):
    q = clean_sql_query(query)
    q_lower = q.lower()

    if is_placeholder_query(q):
        return False, "placeholder query"

    if not (q_lower.startswith("select") or q_lower.startswith("with")):
        return False, "query does not start with SELECT or WITH"

    if ";" in q_lower:
        return False, "multiple statements detected"

    matched_forbidden = [word for word in FORBIDDEN_SQL if re.search(rf"\b{word}\b", q_lower)]
    if matched_forbidden:
        return False, f"forbidden keyword(s): {matched_forbidden}"

    return True, "ok"

def enforce_limit(query: str, default_limit: int = 200) -> str:
    q = normalize_sql(query)
    if re.search(r"\blimit\b", q, flags=re.IGNORECASE):
        return q
    return f"{q} LIMIT {default_limit}"


# Schema helpers:
def get_schema_dict(engine):
    inspector = inspect(engine)
    schema = {}
    for table_name in inspector.get_table_names():
        columns = inspector.get_columns(table_name)
        schema[table_name] = []
        for col in columns:
            schema[table_name].append({
                "col_name": col["name"],
                "col_type": str(col["type"])
            })
    return schema

def get_schema_text(engine) -> str:
    schema = get_schema_dict(engine)
    lines = []
    for table_name, columns in schema.items():
        col_text = ', '.join(f"{col['col_name']} ({col['col_type']})" for col in columns)
        lines.append(f"{table_name}: {col_text}")
    return "\n".join(lines)

SCHEMA_TEXT = get_schema_text(engine)

# SQL Execution Helper:
executed_queries_log = []

def run_sql_df(query: str) -> pd.DataFrame:
    print("RAW QUERY:", repr(query))
    cleaned = clean_sql_query(query)
    print("CLEANED QUERY:", repr(cleaned))
    is_safe, reason = check_sql_safety(cleaned)
    print("SAFETY CHECK:", is_safe, "-", reason)

    if not is_safe:
        raise ValueError(f"Unsafe query blocked. Reason: {reason}. Query received: {query}")

    safe_query = enforce_limit(cleaned)
    executed_queries_log.append(safe_query)
    print("FINAL QUERY:", repr(safe_query))

    with engine.connect() as conn:
        df = pd.read_sql(text(safe_query), conn)

    return df

def dataframe_preview(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df.empty:
        return "Result: 0 rows returned."
    return df.head(max_rows).to_string(index=False)

def dataframe_metadata(df: pd.DataFrame) -> dict:
    return {
        "rows": int(df.shape[0]),
        "columns": df.columns.tolist(),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()}
    }


# ── Tools ──────────────────────────────────────────────────────────────────────

@tool
def get_db_schema() -> str:
    """Return the database schema: table names and columns."""
    return get_schema_text(engine)

@tool
def preview_table(table_name: str) -> str:
    """Preview up to 10 rows from a table."""
    query = f"SELECT * FROM {table_name} LIMIT 10"
    df = run_sql_df(query)
    return dataframe_preview(df, max_rows=10)

@tool
def describe_table(table_name: str) -> str:
    """Return column names and types for a specific table."""
    inspector = inspect(engine)
    columns = inspector.get_columns(table_name)
    if not columns:
        return f"No metadata found for table '{table_name}'."
    lines = [f"Table: {table_name}"]
    for col in columns:
        lines.append(f"- {col['name']} ({col['type']})")
    return "\n".join(lines)

@tool
def run_sql_readonly(query: str, **kwargs) -> str:
    """
    Execute exactly one read-only SQL query and return results as JSON.

    Requirements:
    - A single SELECT or WITH ... SELECT statement.
    - No markdown, no explanations, no placeholders.
    - No write operations (INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, TRUNCATE).
    - Only use tables and columns from the provided schema.
    """
    validate_readonly_sql(query, ACTIVE_SQL_DIALECT)
    df = run_sql_df(query)
    result = {
        "metadata": dataframe_metadata(df),
        "preview": df.head(20).to_dict(orient="records")
    }
    return json.dumps(result, indent=2, default=str)

tavily_search = TavilySearch(max_results=5, topic="general")

@tool
def web_search(query: str) -> str:
    """Search the web for general external context."""
    result = tavily_search.invoke(query)
    return json.dumps(result, indent=2, default=str)

tavily_news = TavilySearch(max_results=5, topic="news")

@tool
def news_search(query: str) -> str:
    """Search recent news for external business context."""
    result = tavily_news.invoke(query)
    return json.dumps(result, indent=2, default=str)

@tool
def plot_sql_bar_chart(query: str, x_col: str, y_col: str) -> str:
    """
    Run a SQL query and create a bar chart using the specified x and y columns.
    The query should return columns that match x_col and y_col.
    """
    df = run_sql_df(query)
    if x_col not in df.columns or y_col not in df.columns:
        return f"Columns not found. Available columns: {df.columns.tolist()}"
    plt.figure(figsize=(10, 5))
    plt.bar(df[x_col].astype(str), df[y_col])
    plt.xticks(rotation=45, ha="right")
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(f"{y_col} by {x_col}")
    plt.tight_layout()
    plt.show()
    return f"Displayed bar chart for {y_col} by {x_col}."

@tool
def search_internal_docs(query: str, **kwargs) -> str:
    """
    Search internal company documents for business context,
    strategy, marketing, pricing, and operational explanations.
    """
    docs = retriever.invoke(query)
    results = [{"source": d.metadata.get("source", "unknown"), "content": d.page_content} for d in docs]
    return json.dumps(results, indent=2)


# ── Tool list ──────────────────────────────────────────────────────────────────

tools = [
    get_db_schema,
    describe_table,
    preview_table,
    run_sql_readonly,
    web_search,
    news_search,
    plot_sql_bar_chart,
    search_internal_docs,
]


# ── System prompt ──────────────────────────────────────────────────────────────

SYSTEM_PROMPT = f"""
You are a business analyst agent working with a {ACTIVE_SQL_DIALECT} database.

Your goal is to answer business questions using the appropriate tools.

-----------------------
Tool Usage Priorities
-----------------------

1. SQL (primary source of truth)
   - Use SQL tools for all questions involving metrics, trends, comparisons, or performance.
   - Internal database results are the source of truth for company data.

2. Internal Documents
   - Use document tools for:
     - business strategy
     - marketing campaigns
     - sales explanations
     - pricing decisions
     - KPI or metric definitions
   - Use documents to explain *why* something happened, not to compute values.

3. Web / News
   - Use only for external context:
     - market trends
     - competitors
     - macroeconomic factors
   - Never use web data to replace internal metrics.

-----------------------
SQL Rules
-----------------------

- Always use SQL when the question requires internal data.
- Generate exactly one query per tool call.
- The query must be valid for {ACTIVE_SQL_DIALECT}.
- Use only syntax and functions supported by {ACTIVE_SQL_DIALECT}.
- Allowed query types:
  - a single SELECT statement
  - a single WITH ... SELECT statement (CTE)

Strict constraints:
- No markdown
- No explanations inside SQL
- No multiple statements
- No placeholders (e.g., {{query}}, {{question}})
- No write operations (INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, TRUNCATE)
- Only use tables and columns present in the schema

Prefer:
- simple, readable SQL
- correct joins and aggregations
- explicit grouping when needed

-----------------------
Reasoning Guidelines
-----------------------

- For performance or metric questions:
  → use SQL first

- For "why" questions:
  1. Use SQL to determine what changed
  2. Use internal documents to explain why
  3. Optionally use web tools for external contributing factors

- Do not guess:
  - nonexistent tables
  - nonexistent columns
  - unsupported metrics

- Always ground answers in tool results.

-----------------------
Response Style
-----------------------

Be concise, structured, and business-focused.

Final answers should include:
1. Direct answer
2. Supporting evidence (from SQL or documents)
3. Business implication
4. Recommended next step
"""


# ── Agent ──────────────────────────────────────────────────────────────────────

# create_react_agent builds the agent → tools → agent loop automatically.
# - prompt:      injects the system message before every agent call
# - tools:       the full tool list, bound to the LLM internally
# - The compiled graph exposes .invoke(), .stream(), and .ainvoke()

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=SYSTEM_PROMPT,   # passed as the system message each turn
)


# ── Usage ──────────────────────────────────────────────────────────────────────
#
# Required .env variables:
#   SUPABASE_URL       = https://your-project.supabase.co
#   SUPABASE_KEY       = your anon/service role key
#   SUPABASE_DB_URL    = postgresql://postgres:password@[host]:6543/postgres
#   GOOGLE_API_KEY     = your Gemini API key
#   TAVILY_API_KEY     = your Tavily API key
#
# result = app.invoke(
#     {"messages": [HumanMessage(content="What were total sales last month?")]},
#     config={"recursion_limit": 10}
# )
# print(result["messages"][-1].content)


