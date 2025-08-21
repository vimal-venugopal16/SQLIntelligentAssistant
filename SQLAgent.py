#!pip install langchain openai langchain-experimental
import sqlite3

from langchain.chat_models import ChatOpenAI
import langchain.chains as chains
from langchain.utilities import SQLDatabase
import streamlit as st
import time
from langchain_experimental.sql import SQLDatabaseChain

from langchain_core.tools import Tool

api_key=st.secrets["open_api_key"]
db = SQLDatabase.from_uri("sqlite:///Chinook.db")
llm = ChatOpenAI(openai_api_key=api_key)
chain = chains.create_sql_query_chain(llm,db)

if "history" not in st.session_state:
        st.session_state.history = []

st.title("Intelligent Database Assistant")
st.sidebar.subheader("Examples of Prompts")
st.sidebar.markdown("1. Top 10 employees")
st.sidebar.markdown("2. Give me the create table for employees table")
st.sidebar.markdown("3. Give me the drop table for orders table")
st.sidebar.markdown("4. Give me the top orders for 2025")


        # Process the prompt (e.g., send to LLM)
def create_table_sqllite(table_name):
    try:
        # Connect to the Chinook database
        conn = sqlite3.connect('chinook.db')
        cursor = conn.cursor()

        # Define the SQL CREATE TABLE statement
        create_table_sql = """
                           CREATE TABLE IF NOT EXISTS {table_name} \
                           ( \
                               id \
                               INTEGER \
                               PRIMARY \
                               KEY, \
                               name \
                               TEXT \
                               NOT \
                               NULL, \
                               description \
                               TEXT
                           ); \
                           """

        # Execute the SQL statement
        cursor.execute(create_table_sql)

        # Commit the changes to the database
        conn.commit()
        print("Table 'MyNewTable' created successfully (if it didn't exist).")

    except sqlite3.Error as e:
        print(f"Error creating table: {e}")

    finally:
        # Close the connection
        if conn:
            conn.close()

sql_generator_chain = SQLDatabaseChain.from_llm(llm, db, return_intermediate_steps=True, verbose=True)

def create_table(query: str) -> str:
    result = sql_generator_chain(query)
    intermediate_steps = result.get("intermediate_steps", [])
    for step in intermediate_steps:
        if isinstance(step, dict) and "sql" in step:
            return step["sql"]
        elif isinstance(step, str) and step.strip().lower().startswith("create table"):
            create_table_sqllite(step)
            return step.strip()
    return result.get("result", "Table could not be created.")


sql_tool = Tool(
    name="Table Generator",
    func=create_table,
    description="Translates natural language to SQL based on the database schema."
)


def chats_history():
    st.session_state.history.append({"role": "user", "content": prompt})
    response = chain.invoke({"question": prompt})
    st.session_state.history.append({"role": "assistant", "content": response})

    for message in st.session_state.history:
        with st.chat_message(message["role"]):
            st.code(message["content"], language='sql')

def spinner():
    with st.spinner("Wait for it...", show_time=True):
        time.sleep(5)

if prompt := st.chat_input("Ask me something, like Give me the top orders for 2025"):
    try:
        spinner()
        chats_history()
    except Exception as e:
        st.write("Please try again")

