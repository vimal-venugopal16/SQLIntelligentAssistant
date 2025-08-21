import os
import re
import sqlite3
import time
from typing import List, Tuple, Optional

import streamlit as st
import pandas as pd
import sqlparse

# LangChain v0.2+ imports
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain_experimental.sql import SQLDatabaseChain
from langchain.prompts import PromptTemplate

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Intelligent SQL Assistant", page_icon="", layout="wide")

# Expect OPENAI API key in Streamlit secrets or env
OPENAI_API_KEY = st.secrets.get("open_api_key") or os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("Missing OpenAI API key. Set st.secrets['open_api_key'] or env var OPENAI_API_KEY.")
    st.stop()

# Use exact filename present in your repo/folder (Chinook.db is common; fix case!)
DB_PATH = "Chinook.db"
if not os.path.exists(DB_PATH):
    st.warning("Chinook.db not found in the current directory. Place the SQLite DB file next to app.py.")
DB_URI = f"sqlite:///{DB_PATH}"

# Auto-execute policy (True = run CREATE TABLE/INDEX automatically)
AUTO_EXECUTE_SAFE_DDL = True

# ──────────────────────────────────────────────────────────────────────────────
# LLM / DB SETUP
# ──────────────────────────────────────────────────────────────────────────────

db = SQLDatabase.from_uri(DB_URI)
llm = ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-4o-mini", temperature=0)
# Good for SELECT generation
select_chain = SQLDatabaseChain.from_llm(llm, db, return_intermediate_steps=True, verbose=False)

# Prompt specialized for DDL (tables/indexes) with the current schema as context
DDL_PROMPT = PromptTemplate.from_template(
    """You are an expert SQLite engineer. Given the current database schema and a user request,
generate exactly ONE valid SQLite statement that best satisfies the request.

Rules:
- If the user asks to CREATE TABLE, return a single CREATE TABLE IF NOT EXISTS ... statement.
- If the user asks to CREATE INDEX, return a single CREATE INDEX IF NOT EXISTS ... statement (use IF NOT EXISTS when sensible).
- Use valid SQLite syntax compatible with the existing schema.
- Do NOT include explanations or code fences; output ONLY the SQL statement.
- If the request is not DDL, return a single SELECT statement instead.

Current schema:
{schema}

User request:
{user_request}
"""
)

def get_schema_text(max_chars: int = 8000) -> str:
    schema = db.get_table_info()
    return schema[:max_chars]

def generate_sql(user_request: str) -> str:
    # Use the DDL-aware prompt to handle both DDL and SELECT
    schema = get_schema_text()
    prompt = DDL_PROMPT.format(schema=schema, user_request=user_request)
    resp = llm.invoke(prompt)
    sql = resp.content.strip()
    # strip backticks/fences if any slipped through
    sql = re.sub(r"^```(sql)?\s*|\s*```$", "", sql, flags=re.IGNORECASE).strip()
    return sql

# ──────────────────────────────────────────────────────────────────────────────
# SQL EXECUTION HELPERS
# ──────────────────────────────────────────────────────────────────────────────

DDL_WHITELIST = (
    re.compile(r"^\s*CREATE\s+TABLE\b", re.IGNORECASE),
    re.compile(r"^\s*CREATE\s+INDEX\b", re.IGNORECASE),
)

DESTRUCTIVE_PATTERNS = (
    re.compile(r"^\s*DROP\s+TABLE\b", re.IGNORECASE),
    re.compile(r"^\s*DROP\s+INDEX\b", re.IGNORECASE),
    re.compile(r"^\s*DELETE\s+FROM\b", re.IGNORECASE),
    re.compile(r"^\s*UPDATE\b", re.IGNORECASE),
)

def is_safe_autoexec(sql: str) -> bool:
    line = sql.strip()
    # allow only CREATE TABLE / CREATE INDEX for auto-exec
    return any(pat.search(line) for pat in DDL_WHITELIST)

def is_destructive(sql: str) -> bool:
    return any(pat.search(sql.strip()) for pat in DESTRUCTIVE_PATTERNS)

def split_sql_statements(sql: str) -> List[str]:
    # Use sqlparse to safely split; filter empties
    return [s.strip() for s in sqlparse.split(sql) if s.strip()]

def run_select(sql: str) -> Tuple[pd.DataFrame, Optional[str]]:
    try:
        with sqlite3.connect(DB_PATH) as conn:
            df = pd.read_sql_query(sql, conn)
        return df, None
    except Exception as e:
        return pd.DataFrame(), str(e)

def run_exec(sql: str) -> Optional[str]:
    # Executes DDL/DML; returns error message or None on success
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.executescript(sql)  # executes multiple statements safely
        return None
    except Exception as e:
        return str(e)

# ──────────────────────────────────────────────────────────────────────────────
# UI
# ──────────────────────────────────────────────────────────────────────────────

if "history" not in st.session_state:
    st.session_state.history = []  # list of dicts: {role, content, sql, executed, error}

st.title(" Intelligent SQL Assistant")
st.caption("Ask me for SQL. If you ask me to **create a table or index**, I’ll generate the SQL and create it in default DB.")

with st.sidebar:
    st.subheader("Sample Prompts")
    st.markdown("- Top 10 tracks by milliseconds")
    st.markdown("- Create a table **TestFacts** with columns (Id INT primary key, Name TEXT, CreatedAt TEXT)")
    st.markdown("- Create an index on **Track** table for **AlbumId**")
    st.markdown("- Show total invoice amount by customer")
    st.divider()
    st.checkbox("Auto-execute CREATE TABLE/INDEX", value=AUTO_EXECUTE_SAFE_DDL, key="autoexec")
    st.caption("Only CREATE TABLE/INDEX are auto-executed. DROP/DELETE never run automatically.")

prompt = ""

def add_history(role: str, content: str, sql: Optional[str] = None, executed: bool = False, error: Optional[str] = None):
    st.session_state.history.append({
        "role": role,
        "content": content,
        "sql": sql,
        "executed": executed,
        "error": error
    })

def render_history():
    for msg in st.session_state.history:
        with st.chat_message(msg["role"]):
            if msg["role"] == "user":
                st.write(msg["content"])
            else:
                # assistant
                if msg.get("sql"):
                    st.code(msg["sql"], language="sql")
                if msg.get("executed"):
                    st.success("Executed successfully.")
                if msg.get("error"):
                    st.error(msg["error"])
                # If assistant included a text response (e.g., SELECT result summary)
                if msg.get("content") and not msg.get("sql"):
                    st.write(msg["content"])

# Process a new prompt
if prompt:
    add_history("user", prompt)

    with st.chat_message("assistant"):
        with st.status("Thinking…", expanded=False) as status:
            # Generate SQL (DDL or SELECT)
            sql_text = generate_sql(prompt)
            print(sql_text)
            status.update(label="SQL generated.", state="complete")

        st.subheader("Generated SQL")
        st.code(sql_text, language="sql")

        # Decide what to do with it
        stmts = split_sql_statements(sql_text)
        executed_any = False
        error = None
        rows_previewed = False

        # If multiple statements, we handle conservatively
        for stmt in stmts:
            # Never auto-run destructive statements
            if is_destructive(stmt):
                error = "Refused to auto-execute a destructive statement. You can copy and run it manually if needed."
                break

            # Auto-exec CREATE TABLE/INDEX only if allowed
            if st.session_state.autoexec and is_safe_autoexec(stmt):
                err = run_exec(stmt)
                if err:
                    error = f"Execution error: {err}"
                    break
                executed_any = True
            else:
                # If it looks like a SELECT, try to preview
                if stmt.strip().upper().startswith("SELECT"):
                    df, err = run_select(stmt)
                    if err:
                        error = f"Query error: {err}"
                        break
                    if not df.empty:
                        st.subheader("Preview")
                        st.dataframe(df, use_container_width=True)
                    else:
                        st.info("No rows returned.")
                    rows_previewed = True
                else:
                    # For non-DDL non-SELECT (e.g., PRAGMA), don't auto-run; just show it.
                    pass

        if executed_any:
            st.success("DDL executed in DB (CREATE TABLE/INDEX). ✅")
        if error:
            st.error(error)

        # Append assistant message to history
        add_history("assistant", content=("Preview shown." if rows_previewed else ""), sql=sql_text, executed=executed_any, error=error)

# Always render the full chat so far
render_history()


def spinner():
    with st.spinner("Wait for it...", show_time=True):
        time.sleep(5)

# Optional: a compact runner for pure-SELECT questions (uses LangChain's SQLDatabaseChain)
st.divider()
#st.subheader("Quick SELECT (LLM decides SQL)")
#q = st.text_input("Ask a read-only question (e.g., 'Top 10 tracks by milliseconds')", key="readonly_q")
#if st.button("Generate SQL"):
if prompt := st.chat_input("Ask a read-only question (e.g., 'Top 10 tracks by milliseconds')"):
    try:
        spinner()
        result = select_chain({"query": prompt})
        sqls = [s.get("sql") for s in result.get("intermediate_steps", []) if isinstance(s, dict) and "sql" in s]
        last_sql = sqls[-1] if sqls else None
        if last_sql:
            #add_history("assistant", last_sql)
            st.code(last_sql, language="sql")
            df, err = run_select(last_sql)
            if err:
                st.error(err)
            else:
                st.dataframe(df, use_container_width=True)
        else:
            st.code(result.get("result"), language="sql")
            #add_history("assistant", last_sql)
        #render_history()
    except Exception as e:
        st.error(f"Failed to run chain: {e}")
