from langchain_core.tools import tool
from simpleeval import simple_eval

from langchain_groq import ChatGroq
from dotenv import load_dotenv
from state import State
from ddgs import DDGS

load_dotenv()

def llm(model: str):
    return ChatGroq(model=model)

def route_task(state: State):
    return "r"


@tool
def calculator(expression: str) -> str:
    """
    Evaluate a mathematical expression and return the numeric result.

    The `expression` argument must contain ONLY the math expression
    (numbers, operators, parentheses) — e.g. "2 * 2". Do not include
    surrounding words like "calculate" or "what is".
    
    
    Args:
        expression (str): The math expression to evaluate.  

    """
    try:
        result = simple_eval(expression)
        return str(result)
    except Exception as e:
        return (
            f"Invalid expression: '{expression}'. "
            f"Error: {e}. "
            "The expression must contain only numbers and math operators "
            "(+ - * / ** % parentheses) — no words. "
            "Re-extract just the math part and try again."
            )

@tool
def web_search(query: str) -> str:
    """
    Performs a web search for the given query and returns a summary of the results.
    
    Args:
        query (str): The search query.  

    """
    try:
        with DDGS() as ds:
            results = list(ds.text(query, max_results=3))
        if not results:
            return "No results found."
        return "\n\n".join(f"{r['title']}: {r['body']}" for r in results)
    
    except Exception as e:
        return f"Error during search: {e}"

FAKE_DB = {
    "vacation_days": "Employees accrue 18 days of paid vacation per year.",
    "remote_work_policy": "Employees may work remotely up to 3 days per week with manager approval.",
    "scout_mini_specs": "Scout Mini: 6 hour battery life, 15kg load capacity.",
    "support_tiers": "Standard (email, 48hr), Priority (phone, 4hr), Enterprise (24/7, dedicated manager).",
}

@tool
def document_lookup(key: str) -> str:
    """
    Look up information from the local knowledge base (company documents), for questions about specific internal facts, policies, or data not found on the general web.
    
    Args:
        key (str): The key to look up in the knowledge base.  

    """

    return FAKE_DB.get(key, f"No entry found for '{key}'. Valid keys: {list(FAKE_DB.keys())}")