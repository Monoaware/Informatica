# Imports for tools + setup:
import os
import re
import json
import time
import random
from dotenv import load_dotenv
from faker import Faker
from datetime import datetime, timedelta
from typing import TypedDict, Annotated, Sequence, Literal

import pandas as pd
import matplotlib.pyplot as plt

from sqlalchemy import create_engine, inspect, text

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_tavily import TavilySearch
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

from langgraph.graph import StateGraph, END


load_dotenv()


llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash",
    temperature=0, google_api_key=os.environ.get("GOOGLE_API_KEY")
)

tavily_api_key=os.environ.get("TAVILY_API_KEY")

# Point to document folder:
DOCS_PATH = r"G:\My Drive\sample_docs"


# Load documents:
def load_documents(folder_path):
    documents = []

    for filename in os.listdir(folder_path):
        path = os.path.join(folder_path, filename)

        if filename.endswith(".txt"):
            loader = TextLoader(path)
            docs = loader.load()

            # Add source metadata:
            for d in docs:
                d.metadata["source"] = filename

            documents.extend(docs)

    return documents

docs = load_documents(DOCS_PATH)

# Split documents:
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150
)

split_docs = splitter.split_documents(docs)

# Initialize embedding model + generate embeddings:
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001" # Changed model to a supported embedding model
)

# Store documents and embeddings into vector store:
vector_store = Chroma.from_documents(
    documents=split_docs,
    embedding=embeddings
)

# Create a retriever:
retriever = vector_store.as_retriever(search_kwargs={"k": 4}) # Returns top 4 matches (adjust as needed).

DB_PATH = r"G:\My Drive\business_agent.db"  # raw string to handle backslashes
engine = create_engine(f"sqlite:///{DB_PATH}")

# Set a constant for validator + agent to know which type of DB we're querying:
ACTIVE_SQL_DIALECT = "SQLite" # Change this based on DB.

# Set up dialect rules (update as we debug):
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
    "postgres": {
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
    

# Filter out commands that can either modify data, delete data, change schemas, or change permissions in the database:
FORBIDDEN_SQL = [
    "insert", "update", "delete", "drop", "alter", "truncate",
    "create", "replace", "grant", "revoke", "merge", "call"
]

# Need a function to clean DB queries so that we can double check them:
def normalize_sql(query: str) -> str:
    return re.sub(r"\s+", " ", str(query).strip()).strip()

# SQLite guard (it only accepts one statement per exeuction call):
def clean_sql_query(query: str) -> str:
  q = str(query).strip()

  # Remove markdown fences if model returns them:
  q = q.replace("```sql", "").replace("```", "").strip()

  # Keep only the first statement before the first semicolon:
  if ";" in q:
    q = q.split(";")[0].strip()

  q = normalize_sql(q)
  return q

# Placeholder guard:
def is_placeholder_query(query: str) -> bool:
  q = str(query).strip()
  return (
      q in ["{query}", '"{query}"', "'{query}'"]
      or "{query}" in q
      or "{question}" in q
  )

# Make sure SQL request is safe to run:
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

# Limit data returned by query:
def enforce_limit(query: str, default_limit: int = 200) -> str:
    q = normalize_sql(query)
    if re.search(r"\blimit\b", q, flags=re.IGNORECASE):
        return q
    return f"{q} LIMIT {default_limit}"


# Schema helper:
def get_schema_dict(engine):
  inspector = inspect(engine)
  schema = {}

  # Grab DB table:
  for table_name in inspector.get_table_names():

    # Grab columns of table:
    columns = inspector.get_columns(table_name)

    # Insert table info (column names + data types) into schema:
    schema[table_name] = []

    for col in columns:
      column_info = {
          "col_name": col["name"],
          "col_type": str(col["type"])
      }

      schema[table_name].append(column_info)

  # Return entire schema:
  return schema

# Convert schema to text for LLM processing:
def get_schema_text(engine) -> str:
  schema = get_schema_dict(engine)
  lines = []

  for table_name, columns in schema.items():
    column_strings = []

    for col in columns:
      column_strings.append(f"{col['col_name']} ({col['col_type']})")

    col_text = ', '.join(column_strings)
    lines.append(f"{table_name}: {col_text}")

  return "\n".join(lines)

# Run and display results:
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
        raise ValueError(
            f"Unsafe query blocked. Reason: {reason}. Query received: {query}"
        )

    safe_query = enforce_limit(cleaned)
    executed_queries_log.append(safe_query)
    print("FINAL QUERY:", repr(safe_query))

    with engine.connect() as conn:
        df = pd.read_sql(text(safe_query), conn)

    return df

# Format DB read results:
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

# Format DB read results:
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

# DB tool to get schema:
@tool
def get_db_schema() -> str:
  """Return the database schema: table names and columns."""
  return get_schema_text(engine)


# DB tool to preview a table:
@tool
def preview_table(table_name: str) -> str:
  """Preview up to 10 rows from a table."""
  query = f"SELECT * FROM {table_name} LIMIT 10"
  df = run_sql_df(query)
  return dataframe_preview(df, max_rows=10)

# DB tool to describe table information:
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

# DB tool to run SQL queries:
@tool
def run_sql_readonly(query: str, **kwargs) -> str:
  """
    Execute exactly one read-only SQL query using the {ACTIVE_SQL_DIALECT} dialect and return results as JSON.

    Requirements:
    - The query must be valid for {ACTIVE_SQL_DIALECT}.
    - Use only syntax and functions supported by {ACTIVE_SQL_DIALECT}.
    - Generate exactly one query:
        - a single SELECT statement, or
        - a single WITH ... SELECT statement

    Strict rules:
    - Output SQL only (no markdown, no explanations).
    - Do not include multiple statements.
    - Do not use placeholders like {{query}} or {{question}}.
    - Never use INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, or TRUNCATE.
    - Use only tables and columns from the provided schema.

    Dialect-specific rules:
    - Do not use features unsupported by {ACTIVE_SQL_DIALECT}.
    - If a query pattern is not supported in {ACTIVE_SQL_DIALECT}, rewrite it using compatible constructs.

    The result will be returned as JSON with metadata and a preview of rows.
    """

  validate_readonly_sql(query, ACTIVE_SQL_DIALECT)

  df = run_sql_df(query)

  result = {
      "metadata": dataframe_metadata(df),
      "preview": df.head(20).to_dict(orient="records")
  }
  return json.dumps(result, indent=2, default=str)

# DB tool to run SQL queries:
@tool
def run_sql_readonly(query: str, **kwargs) -> str:
  """
    Execute exactly one read-only SQL query using the {ACTIVE_SQL_DIALECT} dialect and return results as JSON.

    Requirements:
    - The query must be valid for {ACTIVE_SQL_DIALECT}.
    - Use only syntax and functions supported by {ACTIVE_SQL_DIALECT}.
    - Generate exactly one query:
        - a single SELECT statement, or
        - a single WITH ... SELECT statement

    Strict rules:
    - Output SQL only (no markdown, no explanations).
    - Do not include multiple statements.
    - Do not use placeholders like {{query}} or {{question}}.
    - Never use INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, or TRUNCATE.
    - Use only tables and columns from the provided schema.

    Dialect-specific rules:
    - Do not use features unsupported by {ACTIVE_SQL_DIALECT}.
    - If a query pattern is not supported in {ACTIVE_SQL_DIALECT}, rewrite it using compatible constructs.

    The result will be returned as JSON with metadata and a preview of rows.
    """

  validate_readonly_sql(query, ACTIVE_SQL_DIALECT)

  df = run_sql_df(query)

  result = {
      "metadata": dataframe_metadata(df),
      "preview": df.head(20).to_dict(orient="records")
  }
  return json.dumps(result, indent=2, default=str)

# General Tavily Search:
tavily_search = TavilySearch(
    max_results=5,
    topic="general"
)

@tool
def web_search(query: str) -> str:
    """Search the web for general external context."""
    result = tavily_search.invoke(query)
    return json.dumps(result, indent=2, default=str)

# News Tavily Search:
tavily_news = TavilySearch(
    max_results=5,
    topic="news"
)

@tool
def news_search(query: str) -> str:
    """Search recent news for external business context."""
    result = tavily_news.invoke(query)
    return json.dumps(result, indent=2, default=str)

# Bar chart tool:
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

# Create document search tool:
@tool
def search_internal_docs(query: str, **kwargs) -> str:
    """
    Search internal company documents for business context,
    strategy, marketing, pricing, and operational explanations.
    """
    docs = retriever.invoke(query)

    results = []
    for d in docs:
        results.append({
            "source": d.metadata.get("source", "unknown"),
            "content": d.page_content
        })

    return json.dumps(results, indent=2)

# Define the tool list:
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

# Display registered tools:
tool_registry = {}
for t in tools:
  tool_name = getattr(t, "name", getattr(t, "__name__", None))
  if tool_name is None:
    raise ValueError(f"Tool is missing a usable name: {t}")
  tool_registry[tool_name] = t

# Bind tools to the model:
llm_with_tools = llm.bind_tools(tools)

# Used to track the agent state:
class AgentState(TypedDict):
    messages: Sequence[BaseMessage]
    step_count: int

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

# Define the agent node:
def agent_node(state: AgentState) -> AgentState:
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + list(state["messages"])
    response = llm_with_tools.invoke(messages)

    return {
        "messages": list(state["messages"]) + [response],
        "step_count": state["step_count"] + 1
    }

# Define the agent node:
def agent_node(state: AgentState) -> AgentState:
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + list(state["messages"])
    response = llm_with_tools.invoke(messages)

    return {
        "messages": list(state["messages"]) + [response],
        "step_count": state["step_count"] + 1
    }

# Define the tool execution node:
def tool_node(state: AgentState) -> AgentState:
    last_message = state["messages"][-1]

    if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
        return state

    new_messages = list(state["messages"])

    for tool_call in last_message.tool_calls:
      tool_name = tool_call.get("name")
      tool_args = tool_call.get("args", {})
      tool_id = tool_call.get("id")

      selected_tool = tool_registry.get(tool_name)

      if selected_tool is None:
          tool_result = f"Tool '{tool_name}' not found."
      else:
          try:
              if hasattr(selected_tool, "invoke"):
                tool_result = selected_tool.invoke(tool_args)
              else:
                if isinstance(tool_args, dict):
                  tool_result = selected_tool(**tool_args)
                else:
                  tool_result = selected_tool(tool_args)
          except Exception as e:
            tool_result = f"Tool error: {str(e)}"

      new_messages.append(
        ToolMessage(
          content=str(tool_result),
          tool_call_id=tool_id
        )
      )

    return {
        "messages": new_messages,
        "step_count": state["step_count"] + 1
    }

# Set up the routing logic:

# We need to limit max steps so that the agent doesn't run forever (fail-safe):
MAX_STEPS = 10

def should_continue(state: AgentState) -> Literal["tools", "end"]:
    if state["step_count"] >= MAX_STEPS:
        return "end"

    last_message = state["messages"][-1]

    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"

    return "end"

# Build the LangGraph workflow:
workflow = StateGraph(AgentState)

workflow.add_node("agent", agent_node)
workflow.add_node("tools", tool_node)

workflow.set_entry_point("agent")

workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",
        "end": END
    }
)

workflow.add_edge("tools", "agent")

app = workflow.compile()








