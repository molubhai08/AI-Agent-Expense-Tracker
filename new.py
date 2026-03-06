from langchain_community.utilities import SQLDatabase
from langchain_community.tools.sql_database.tool import (
    ListSQLDatabaseTool,
    InfoSQLDatabaseTool,
    QuerySQLDatabaseTool,
)
from langchain.tools import tool
from crewai import Agent, Crew, Task
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os

load_dotenv()

# Set API key
os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")

# Database setup
MYSQL_USER = "root"
MYSQL_PASSWORD = "naruto"
MYSQL_HOST = "localhost"
MYSQL_DB = "expense_tracker"
MYSQL_PORT = 3306

db = SQLDatabase.from_uri(
    f"mysql+mysqlconnector://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DB}"
)


# Tools
@tool("list_accounts")
def list_accounts() -> str:
    """List all expense accounts (tables)"""
    return ListSQLDatabaseTool(db=db).invoke("")


@tool("account_details")
def account_details(account_name: str) -> str:
    """Get schema and sample rows for an account. Input: account_name"""
    return InfoSQLDatabaseTool(db=db).invoke(account_name)


@tool("run_query")
def run_query(sql: str) -> str:
    """Execute SQL query. Input: valid SQL statement"""
    return QuerySQLDatabaseTool(db=db).invoke(sql)


@tool("get_date")
def get_date(offset_days: int = 0) -> str:
    """Get date with offset. Input: 0 for today, -7 for week ago, etc."""
    date = datetime.now() + timedelta(days=offset_days)
    return date.strftime("%Y-%m-%d")


@tool("create_account")
def create_account(name: str) -> str:
    """Create expense account. Input: person_name"""
    query = f"""CREATE TABLE IF NOT EXISTS {name} (
        id INT AUTO_INCREMENT PRIMARY KEY,
        date DATE NOT NULL,
        amount DECIMAL(10,2) NOT NULL,
        description VARCHAR(255),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )"""
    try:
        QuerySQLDatabaseTool(db=db).invoke(query)
        return f"✅ Account '{name}' created"
    except Exception as e:
        return f"❌ Error: {str(e)}"


@tool("add_expense")
def add_expense(account: str, amount: float, date: str, desc: str = "") -> str:
    """Add expense. Input: account_name, amount, date (YYYY-MM-DD), description"""
    query = f"INSERT INTO {account} (date, amount, description) VALUES ('{date}', {amount}, '{desc}')"
    try:
        QuerySQLDatabaseTool(db=db).invoke(query)
        return f"✅ Added ${amount} to {account} on {date}"
    except Exception as e:
        return f"❌ Error: {str(e)}"


# Agent with explicit LLM string format
expense_agent = Agent(
    role="Expense Manager",
    goal="Manage expense accounts: create accounts, log expenses, query data",
    backstory="""Expert at tracking expenses. Create accounts for people, log their spending with dates, 
    and retrieve expense data. Use get_date for relative dates (today=0, yesterday=-1, week ago=-7).""",
    llm="groq/llama-3.3-70b-versatile",  # Use string format instead of ChatGroq object
    tools=[list_accounts, account_details, run_query, get_date, create_account, add_expense],
    allow_delegation=False,
    verbose=True
)


def process_query(user_input: str) -> str:
    """Process user query and return result"""
    task = Task(
        description=f"Handle: {user_input}",
        expected_output="Clear confirmation of action taken with relevant data",
        agent=expense_agent
    )
    
    crew = Crew(
        agents=[expense_agent],
        tasks=[task],
        verbose=False
    )
    
    result = crew.kickoff()
    return str(result)


# Main interface
if __name__ == "__main__":
    print("🏦 Expense Tracker Started")
    print("Commands:")
    print("- 'make account for John' - creates account")
    print("- 'John spent $50' - adds today's expense")
    print("- 'Alice spent $30 one week ago' - adds past expense")
    print("- 'show John expenses' - view all expenses")
    print("- 'quit' - exit\n")
    
    while True:
        query = input("You: ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            print("👋 Goodbye!")
            break
            
        if not query:
            continue
            
        try:
            result = process_query(query)
            print(f"\n🤖 Assistant: {result}\n")
        except Exception as e:
            print(f"❌ Error: {str(e)}\n")