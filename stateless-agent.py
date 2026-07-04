from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import tool
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor

# Fake Orders DB
ORDERS = {
    "ORD-101": {
        "status": "shipped",
        "city": "Delhi",
        "amount": 2500,
        "delivery_days": 2
    },
    "ORD-102": {
        "status": "cancelled",
        "city": "Bangalore",
        "amount": 4000,
        "delivery_days": 0
    },
    "ORD-103": {
        "status": "delivered",
        "city": "Mumbai",
        "amount": 1500,
        "delivery_days": 0
    }
}

@tool
def get_order_status(order_id: str):
    """
    Returns the order status from the database for the order id provided by the user.

    Use this tool when user asks for the order status for the given order id.
    """

    order = ORDERS.get(order_id)

    if not order:
        return f"Order with the order_id: {order_id} not found."
    
    return f"Order status: {order['status']}."

llm = ChatOpenAI(
    model='gpt-5.2',
    temperature=0
)

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are a helpful customer support agent.

Rules:
- If the user gives an order id, keep a note of it.
- If the user asks a query like "track the order" or "where is the order", use the order id from the input.
- Use tools when order status is required.
- If no order id is available, politely ask the user for the order id.
"""
    ),

    # The previous conversation messages (HumanMessage + AIMessage) will be appened to the chat_history and will be used by the LLM to retrieve the context.
    MessagesPlaceholder(variable_name="chat_history", optional=True),

    ("human", "{input}"),

    MessagesPlaceholder(variable_name="agent_scratchpad")
])

tools = [get_order_status]

agent = create_tool_calling_agent(
    llm=llm,
    prompt=prompt,
    tools=tools
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools
)

def ask_agent(user_query: str):
    response = agent_executor.invoke(
        {
            "input" : user_query,
            "chat_history" : []
        }
    )

    return response

print("Turn-1")
user_query = "My order id is : ORD-102"
print(ask_agent(user_query))

print("------------------------------")

print("Turn-2")
user_query = "Track it."
print(ask_agent(user_query))