# Imports for tools + setup:
import os
import re
import json
import tempfile
import shutil
import hashlib
import time
import logging
from dotenv import load_dotenv
from typing import Optional

import pandas as pd
import matplotlib.pyplot as plt

from sqlalchemy import create_engine, inspect, text
from supabase import create_client

from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_tavily import TavilySearch
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

from langchain.agents import create_agent


load_dotenv()

logger = logging.getLogger("informatica.agent")
if not logger.handlers:
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))


# Docs bucket selection:
# - SUPABASE_DEMO_DOCS_BUCKET: demo docs bucket (preferred when set)
# - SUPABASE_DOCS_BUCKET: legacy/testing/default docs bucket
# If you are not using the old fixed docs, set SUPABASE_DISABLE_INTERNAL_DOCS=1.
DEMO_DOCS_BUCKET_ENV = "SUPABASE_DEMO_DOCS_BUCKET"
DEFAULT_DOCS_BUCKET_ENV = "SUPABASE_DOCS_BUCKET"
DISABLE_INTERNAL_DOCS_ENV = "SUPABASE_DISABLE_INTERNAL_DOCS"

# NOTE: If internal docs are disabled, this value is ignored.
SUPABASE_DOCS_BUCKET = (
    os.environ.get(DEMO_DOCS_BUCKET_ENV)
    or os.environ.get(DEFAULT_DOCS_BUCKET_ENV)
    or "sample_docs"
)

# Uploads bucket for user-provided files (persistent, stored in Supabase Storage).
SUPABASE_DEMO_UPLOADS_BUCKET = (
    os.environ.get("SUPABASE_DEMO_UPLOADS_BUCKET")
)

CHROMA_DIR = os.environ.get("CHROMA_DIR", "./chroma_store")
FINGERPRINT_FILE = os.path.join(CHROMA_DIR, "fingerprint.json")

# Supabase project selection (for Storage docs):
# - SUPABASE_DEMO_URL / SUPABASE_DEMO_KEY: demo project (preferred when set)
# - SUPABASE_URL / SUPABASE_KEY: legacy/testing/default project
DEMO_SUPABASE_URL_ENV = "SUPABASE_DEMO_URL"
DEMO_SUPABASE_KEY_ENV = "SUPABASE_DEMO_KEY"
DEFAULT_SUPABASE_URL_ENV = "SUPABASE_URL"
DEFAULT_SUPABASE_KEY_ENV = "SUPABASE_KEY"

# DB URL selection:
# - SUPABASE_DEMO_DB_URL: demo database (preferred when set)
# - SUPABASE_DB_URL: legacy/testing/default database
DEMO_DB_URL_ENV = "SUPABASE_DEMO_DB_URL"
DEFAULT_DB_URL_ENV = "SUPABASE_DB_URL"

# Dialect key used for rule lookup (must match keys in DIALECT_RULES):
ACTIVE_SQL_DIALECT = os.environ.get("ACTIVE_SQL_DIALECT", "postgresql").strip().lower()

# Human-friendly dialect name for prompts/logging:
ACTIVE_SQL_DIALECT_NAME = {
    "postgresql": "PostgreSQL",
    "sqlite": "SQLite",
    "mysql": "MySQL",
}.get(ACTIVE_SQL_DIALECT, ACTIVE_SQL_DIALECT)


# ── Init-time singletons (constructed via init_agent()) ───────────────────────
_supabase = None
_llm = None
_embeddings = None
_vector_store = None
_retriever = None
_engine = None
_schema_text = None
_agent = None


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if value is None or str(value).strip() == "":
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def _env_optional(name: str) -> Optional[str]:
    value = os.environ.get(name)
    if value is None:
        return None
    value = str(value).strip()
    return value or None


# Internal docs are enabled unless SUPABASE_DISABLE_INTERNAL_DOCS is set to a truthy value.
# Accept common truthy values: 1, true, yes, on.
_disable_docs = (_env_optional(DISABLE_INTERNAL_DOCS_ENV) or "").strip().lower()
INTERNAL_DOCS_ENABLED = _disable_docs not in {"1", "true", "yes", "on"}


def get_supabase_url_and_key() -> tuple[str, str]:
    """Return Supabase URL + KEY to use for Storage docs.

    Prefer demo project when configured; otherwise fall back to legacy.
    """
    demo_url = _env_optional(DEMO_SUPABASE_URL_ENV)
    demo_key = _env_optional(DEMO_SUPABASE_KEY_ENV)

    if (demo_url and not demo_key) or (demo_key and not demo_url):
        raise RuntimeError(
            f"{DEMO_SUPABASE_URL_ENV} and {DEMO_SUPABASE_KEY_ENV} must be set together"
        )

    if demo_url and demo_key:
        return demo_url, demo_key

    return _require_env(DEFAULT_SUPABASE_URL_ENV), _require_env(DEFAULT_SUPABASE_KEY_ENV)


def get_db_url() -> str:
    """Return the DB URL to use.

    Prefer demo DB when configured; otherwise fall back to the legacy DB.
    """
    return _env_optional(DEMO_DB_URL_ENV) or _require_env(DEFAULT_DB_URL_ENV)


def validate_env() -> None:
    # Required for core functionality:
    # Require the selected Supabase project (demo preferred, else default).
    _ = get_supabase_url_and_key()

    # Require the selected DB URL (demo preferred, else default).
    _ = get_db_url()

    _require_env("GOOGLE_API_KEY")

    # Optional (demo-safe): if Tavily is missing we will disable web/news tools.
    # Keep as required only if you want startup to fail without Tavily.


def build_tavily_clients():
    """Build Tavily clients or return (None, None) if not configured."""
    if _env_optional("TAVILY_API_KEY") is None:
        logger.warning("TAVILY_API_KEY missing; disabling web/news tools")
        return None, None

    try:
        return (
            TavilySearch(max_results=5, topic="general"),
            TavilySearch(max_results=5, topic="news"),
        )
    except Exception:
        logger.exception("Failed to initialize Tavily; disabling web/news tools")
        return None, None


def build_engine_and_schema_required():
    """Build DB engine and schema (required). Raises on failure."""
    engine = build_engine()
    schema_text = get_schema_text(engine)
    return engine, schema_text


def build_engine_and_schema():
    """Build DB engine and schema; return (engine, schema_text) or (None, None) on failure."""
    try:
        engine = build_engine()
        schema_text = get_schema_text(engine)
        return engine, schema_text
    except Exception:
        logger.exception("Failed to initialize database; disabling SQL tools")
        return None, None


def build_supabase_client():
    url, key = get_supabase_url_and_key()
    return create_client(url, key)


def build_llm():
    return ChatGoogleGenerativeAI(
        model=os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"),
        temperature=float(os.environ.get("GEMINI_TEMPERATURE", "0")),
        google_api_key=_require_env("GOOGLE_API_KEY"),
    )


def download_docs_from_supabase(supabase, bucket: str) -> str:
    """Download known .txt files directly from Supabase Storage by name.

    NOTE: This remains required at startup per deployment requirements.
    """
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
                logger.warning("SKIP %s: unexpected response type %s", filename, type(data))
                continue
            with open(os.path.join(tmp_dir, filename), "wb") as out:
                out.write(data)
            downloaded += 1
            logger.info("Downloaded %s (%s bytes)", filename, len(data))
        except Exception:
            logger.exception("FAILED to download %s", filename)

    if downloaded == 0:
        raise RuntimeError(
            f"No files were downloaded from bucket '{bucket}'. "
            "Check your SUPABASE_KEY and bucket name."
        )

    logger.info("[Supabase Storage] Done — %s/%s file(s) saved to %s", downloaded, len(DOC_FILENAMES), tmp_dir)
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


def compute_documents_fingerprint(documents) -> str:
    """Compute a stable fingerprint for the current document set."""
    hasher = hashlib.sha256()
    sorted_documents = sorted(
        documents,
        key=lambda d: (
            d.metadata.get("source", ""),
            d.page_content,
        ),
    )
    for document in sorted_documents:
        source = document.metadata.get("source", "")
        content = document.page_content
        hasher.update(source.encode("utf-8"))
        hasher.update(content.encode("utf-8"))
    return hasher.hexdigest()


def load_saved_fingerprint() -> Optional[str]:
    if not os.path.exists(FINGERPRINT_FILE):
        return None
    try:
        with open(FINGERPRINT_FILE, "r", encoding="utf-8") as file:
            data = json.load(file)
        return data.get("fingerprint")
    except Exception:
        return None


def save_fingerprint(fingerprint: str) -> None:
    os.makedirs(CHROMA_DIR, exist_ok=True)
    with open(FINGERPRINT_FILE, "w", encoding="utf-8") as file:
        json.dump({"fingerprint": fingerprint}, file)


def has_existing_chroma_store(path: str) -> bool:
    if not os.path.exists(path):
        return False
    contents = [name for name in os.listdir(path) if name != "fingerprint.json"]
    return len(contents) > 0


def build_or_load_vector_store(split_docs, embeddings, persist_directory: str = CHROMA_DIR):
    current_fingerprint = compute_documents_fingerprint(split_docs)
    saved_fingerprint = load_saved_fingerprint()
    chroma_exists = has_existing_chroma_store(persist_directory)

    if chroma_exists and saved_fingerprint == current_fingerprint:
        logger.info("[Chroma] Loading existing vector store from %s", persist_directory)
        return Chroma(persist_directory=persist_directory, embedding_function=embeddings)

    logger.info("[Chroma] Documents changed or store missing - rebuilding vector store...")

    if os.path.exists(persist_directory):
        shutil.rmtree(persist_directory)

    os.makedirs(persist_directory, exist_ok=True)

    vector_store = Chroma.from_documents(
        documents=split_docs,
        embedding=embeddings,
        persist_directory=persist_directory,
    )
    vector_store.persist()

    save_fingerprint(current_fingerprint)
    logger.info("[Chroma] Rebuild complete.")

    return vector_store


def build_retriever(supabase):
    # If internal docs are disabled, create an empty vector store so uploads can still be indexed.
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

    if not INTERNAL_DOCS_ENABLED:
        logger.warning("Internal docs disabled (%s is set). Starting with empty Chroma store.", DISABLE_INTERNAL_DOCS_ENV)
        os.makedirs(CHROMA_DIR, exist_ok=True)
        vector_store = Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=embeddings,
        )
        retriever = vector_store.as_retriever(search_kwargs={"k": 4})
        return retriever, vector_store, embeddings

    # Required: download docs and build/load vector store.
    docs_path = download_docs_from_supabase(supabase, SUPABASE_DOCS_BUCKET)
    docs = load_documents(docs_path)

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    split_docs = splitter.split_documents(docs)

    vector_store = build_or_load_vector_store(split_docs, embeddings, CHROMA_DIR)
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})

    return retriever, vector_store, embeddings


def build_engine():
    return create_engine(get_db_url())


def get_schema_dict(engine):
    inspector = inspect(engine)
    schema = {}
    for table_name in inspector.get_table_names():
        columns = inspector.get_columns(table_name)
        schema[table_name] = []
        for col in columns:
            schema[table_name].append({
                "col_name": col["name"],
                "col_type": str(col["type"]),
            })
    return schema


def get_schema_text(engine) -> str:
    schema = get_schema_dict(engine)
    lines = []
    for table_name, columns in schema.items():
        col_text = ", ".join(f"{col['col_name']} ({col['col_type']})" for col in columns)
        lines.append(f"{table_name}: {col_text}")
    return "\n".join(lines)


# Set up dialect rules:
DIALECT_RULES = {
    "sqlite": {
        "forbidden_patterns": [
            "RIGHT JOIN",
            "FULL JOIN",
            "FULL OUTER JOIN",
            "INSERT",
            "UPDATE",
            "DELETE",
            "DROP",
            "ALTER",
            "CREATE",
            "TRUNCATE",
        ],
        "allow_multiple_statements": False,
    },
    "postgresql": {
        "forbidden_patterns": [
            "INSERT",
            "UPDATE",
            "DELETE",
            "DROP",
            "ALTER",
            "CREATE",
            "TRUNCATE",
        ],
        "allow_multiple_statements": False,
    },
    "mysql": {
        "forbidden_patterns": [
            "INSERT",
            "UPDATE",
            "DELETE",
            "DROP",
            "ALTER",
            "CREATE",
            "TRUNCATE",
        ],
        "allow_multiple_statements": False,
    },
}


def validate_table_name(engine, table_name: str) -> str:
    inspector = inspect(engine)
    valid_tables = set(inspector.get_table_names())
    if table_name not in valid_tables:
        raise ValueError(f"Invalid table name: {table_name}")
    return table_name


def normalize_sql(query: str) -> str:
    return re.sub(r"\s+", " ", str(query).strip())


def clean_sql_query(query: str) -> str:
    q = str(query).strip()
    q = q.replace("```sql", "").replace("```", "").strip()
    q = normalize_sql(q)
    return q


def is_placeholder_query(query: str) -> bool:
    q = str(query).strip()
    return (
        q in ["{query}", '"{query}"', "'{query}'"]
        or "{query}" in q
        or "{question}" in q
    )


def validate_readonly_sql(query: str, dialect: str) -> None:
    q = clean_sql_query(query)

    if is_placeholder_query(q):
        raise ValueError("Placeholder query detected.")

    normalized_dialect = str(dialect).strip().lower()
    rules = DIALECT_RULES.get(normalized_dialect)
    if rules is None:
        raise ValueError(f"Unsupported dialect: {dialect}")

    upper_query = q.upper()

    for pattern in rules["forbidden_patterns"]:
        if re.search(rf"\b{re.escape(pattern)}\b", upper_query):
            raise ValueError(f"Disallowed SQL for {dialect}: {pattern}")

    if not upper_query.lstrip().startswith(("SELECT", "WITH")):
        raise ValueError("Only SELECT or WITH ... SELECT queries are allowed.")

    stripped = q.strip()
    if not rules["allow_multiple_statements"]:
        if ";" in stripped[:-1] or stripped.count(";") > 1:
            raise ValueError("Multiple statements are not allowed.")


def enforce_limit(query: str, default_limit: int = 200) -> str:
    q = clean_sql_query(query)

    # Protect against multiple statements / stray semicolons from the model.
    # We allow at most one trailing ';' which we will strip.
    stripped = q.strip()
    if ";" in stripped[:-1] or stripped.count(";") > 1:
        raise ValueError("Multiple statements are not allowed.")

    # If the model includes a trailing semicolon, drop it so we don't produce
    # invalid SQL like: "SELECT ...; LIMIT 200".
    q = stripped.rstrip(";").strip()

    if re.search(r"\bLIMIT\b", q, flags=re.IGNORECASE):
        return q

    return f"{q} LIMIT {default_limit}"


def run_sql_df(engine, query: str) -> pd.DataFrame:
    cleaned = clean_sql_query(query)
    validate_readonly_sql(cleaned, ACTIVE_SQL_DIALECT)
    safe_query = enforce_limit(cleaned)
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
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
    }


def build_tools(engine, retriever, tavily_general=None, tavily_news=None):
    """Build tools with dependencies captured in closures (no global state)."""

    tools = []

    # --- SQL tools (optional) ---
    if engine is not None:

        @tool
        def get_db_schema() -> str:
            """Return the database schema: table names and columns."""
            return get_schema_text(engine)

        @tool
        def preview_table(table_name: str) -> str:
            """Preview up to 10 rows from a table."""
            table_name_validated = validate_table_name(engine, table_name)
            query = f"SELECT * FROM {table_name_validated} LIMIT 10"
            df = run_sql_df(engine, query)
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
            """Execute exactly one read-only SQL query and return results as JSON."""
            df = run_sql_df(engine, query)
            result = {
                "metadata": dataframe_metadata(df),
                "preview": df.head(20).to_dict(orient="records"),
            }
            return json.dumps(result, indent=2, default=str)

        @tool
        def plot_sql_bar_chart(query: str) -> str:
            """Run a SQL query and create a bar chart."""
            df = run_sql_df(engine, query)
            if df.shape[1] != 2:
                return "Query must return exactly 2 columns."
            x_col, y_col = df.columns.tolist()
            plt.figure(figsize=(10, 5))
            plt.bar(df[x_col].astype(str), df[y_col])
            plt.xticks(rotation=45, ha="right")
            plt.xlabel(x_col)
            plt.ylabel(y_col)
            plt.title(f"{y_col} by {x_col}")
            plt.tight_layout()
            return f"Chart prepared for {y_col} by {x_col}."

        tools.extend([
            get_db_schema,
            describe_table,
            preview_table,
            run_sql_readonly,
            plot_sql_bar_chart,
        ])

    # --- Tavily tools (optional) ---
    if tavily_general is not None:

        @tool
        def web_search(query: str) -> str:
            """Search the web for general external context."""
            result = tavily_general.invoke(query)
            return json.dumps(result, indent=2, default=str)

        tools.append(web_search)

    if tavily_news is not None:

        @tool
        def news_search(query: str) -> str:
            """Search recent news for external business context."""
            result = tavily_news.invoke(query)
            return json.dumps(result, indent=2, default=str)

        tools.append(news_search)

    # --- Docs tool (required) ---
    @tool
    def search_internal_docs(query: str, **kwargs) -> str:
        """Search internal company documents."""
        docs = retriever.invoke(query)
        results = [
            {"source": d.metadata.get("source", "unknown"), "content": d.page_content}
            for d in docs
        ]
        return json.dumps(results, indent=2, default=str)

    tools.append(search_internal_docs)

    return tools


def build_system_prompt(schema_text: str) -> str:
    # Keep your existing prompt, but build it after schema is available.
    # If DB failed, schema_text may be None; fall back to a minimal prompt.
    if not schema_text:
        return f"""
You are a business analyst agent.

The SQL database is currently unavailable, so do not attempt SQL queries.
Use internal documents and web/news context when available.

OUTPUT FORMAT (STRICT)
- Return ONLY a single JSON object (no markdown, no prose) with exactly these keys:
  - direct_answer (string)
  - supporting_evidence (array of strings)
  - business_implication (string)
  - recommended_next_step (array of strings)

Rules:
- supporting_evidence must reference tool outputs when available.
- If you cannot answer due to missing data, set direct_answer accordingly and explain in supporting_evidence.
"""

    return f"""
You are a business analyst agent working with a {ACTIVE_SQL_DIALECT_NAME} database.

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
- The query must be valid for {ACTIVE_SQL_DIALECT_NAME}.
- Use only syntax and functions supported by {ACTIVE_SQL_DIALECT_NAME}.
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
OUTPUT FORMAT (STRICT)
-----------------------
Return ONLY a single JSON object (no markdown, no prose) with exactly these keys:

- direct_answer (string)
- supporting_evidence (array of strings)
- business_implication (string)
- recommended_next_step (array of strings)

Rules:
- supporting_evidence must cite tool results (SQL output summaries and/or doc sources like "pricing_policy.txt").
- recommended_next_step must be 1-3 short bullets (as array items).
- Do not include any keys other than the four keys above.
- Do not wrap the JSON in backticks.
"""


def build_agent():
    supabase = build_supabase_client()

    # Docs are required at startup.
    retriever, vector_store, embeddings = build_retriever(supabase)

    # DB is required at startup.
    engine, schema_text = build_engine_and_schema_required()

    llm = build_llm()

    tavily_general, tavily_news = build_tavily_clients()
    tools = build_tools(engine, retriever, tavily_general=tavily_general, tavily_news=tavily_news)

    system_prompt = build_system_prompt(schema_text)

    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=system_prompt,
    )

    return {
        "agent": agent,
        "supabase": supabase,
        "retriever": retriever,
        "vector_store": vector_store,
        "embeddings": embeddings,
        "engine": engine,
        "schema_text": schema_text,
        "llm": llm,
        "tools": tools,
    }


def init_agent() -> None:
    """Initialize required resources exactly once (recommended to call at app startup)."""
    global _supabase, _llm, _embeddings, _vector_store, _retriever, _engine, _schema_text, _agent

    if _agent is not None:
        return

    validate_env()

    start = time.time()
    resources = build_agent()

    _supabase = resources["supabase"]
    _retriever = resources["retriever"]
    _vector_store = resources["vector_store"]
    _embeddings = resources["embeddings"]
    _engine = resources["engine"]
    _schema_text = resources["schema_text"]
    _llm = resources["llm"]
    _agent = resources["agent"]

    logger.info("Agent initialized in %.2fs", time.time() - start)


def get_agent():
    """Get the initialized agent, initializing it if necessary."""
    global _agent
    if _agent is None:
        init_agent()
    return _agent


# Backward-compatible export: existing code imports `agent` from this module.
# It will initialize on first access.
# agent = get_agent()


def upload_user_file(
    supabase,
    *,
    filename: str,
    content: bytes,
    content_type: str = "text/plain",
    bucket: str = SUPABASE_DEMO_UPLOADS_BUCKET,
    prefix: str = "uploads",
) -> str:
    """Upload a user-provided file to Supabase Storage.

    Returns the storage path (key) that can be used later to download.

    Note: This uses the server Supabase key (ideally service_role) and is intended
    to be called from your FastAPI backend.
    """
    safe_name = os.path.basename(filename)
    ts = int(time.time())
    key = f"{prefix}/{ts}_{safe_name}"

    # supabase-py expects a (path, file, options) signature.
    supabase.storage.from_(bucket).upload(
        key,
        content,
        {
            "content-type": content_type,
            "x-upsert": "true",
        },
    )
    return key


def list_user_files(
    supabase,
    *,
    bucket: str = SUPABASE_DEMO_UPLOADS_BUCKET,
    prefix: str = "uploads",
) -> list[dict]:
    """List user-uploaded files in the uploads bucket/prefix."""
    return supabase.storage.from_(bucket).list(prefix)


def download_user_file(
    supabase,
    *,
    path: str,
    bucket: str = SUPABASE_DEMO_UPLOADS_BUCKET,
) -> bytes:
    """Download a previously uploaded file by its storage path."""
    data = supabase.storage.from_(bucket).download(path)
    if not isinstance(data, (bytes, bytearray)):
        raise RuntimeError(f"Unexpected download type for {path}: {type(data)}")
    return bytes(data)


def user_upload_to_documents(
    supabase,
    *,
    path: str,
    bucket: str = SUPABASE_DEMO_UPLOADS_BUCKET,
):
    """Convert a user-uploaded file into LangChain Documents.

    Supports:
      - .txt (UTF-8)
      - .pdf (via pypdf)
    """
    raw = download_user_file(supabase, path=path, bucket=bucket)
    lower = os.path.basename(path).lower()

    # Reuse loader output by writing to a temp file.
    tmp_dir = tempfile.mkdtemp()
    try:
        tmp_path = os.path.join(tmp_dir, os.path.basename(path))
        with open(tmp_path, "wb") as f:
            f.write(raw)

        if lower.endswith(".pdf"):
            from langchain_community.document_loaders import PyPDFLoader
            loader = PyPDFLoader(tmp_path)
            docs = loader.load()
        else:
            # Default to text
            text_content = raw.decode("utf-8", errors="replace")
            with open(tmp_path, "w", encoding="utf-8") as f:
                f.write(text_content)
            loader = TextLoader(tmp_path)
            docs = loader.load()

        for d in docs:
            d.metadata.update({
                "source": os.path.basename(path),
                "supabase_bucket": bucket,
                "supabase_path": path,
                "kind": "user_upload",
            })
        return docs
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def index_documents_into_vector_store(documents) -> int:
    """Add documents to the initialized Chroma vector store and persist.

    Returns number of documents added.
    """
    global _vector_store, _retriever
    if _vector_store is None:
        raise RuntimeError("Vector store is not initialized. Call init_agent() first.")

    if not documents:
        return 0

    # Add and persist.
    _vector_store.add_documents(documents)
    _vector_store.persist()

    # Refresh retriever (keeps search kwargs).
    _retriever = _vector_store.as_retriever(search_kwargs={"k": 4})
    return len(documents)


def index_user_upload(
    supabase,
    *,
    path: str,
) -> dict:
    """Download a user upload, convert to Documents, chunk, index into Chroma."""
    docs = user_upload_to_documents(supabase, path=path)

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    split_docs = splitter.split_documents(docs)

    added = index_documents_into_vector_store(split_docs)
    return {"indexed_documents": added, "path": path}


