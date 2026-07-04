# Build an agent to check the order status using the get_order_status tool. 
# Use chat_history memory to help LLM retrieve the context for the previous conversation.

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import tool
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent

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
- If the user gives an order id, remember it for this conversation. Give order information only if user is asking explicitly. 
- If the user asks a follow-up like "track it" or "where is it", use the order id from chat history.
- Use tools when order status is required.
- If no order id is available, politely ask the user for the order id.
"""
    ),

    # previous message of the conversation will be inserted in the chat_history.
    # optional=True makes the chat_history parameter as optional.
    MessagesPlaceholder(variable_name="chat_history", optional=True),

    ("human", "{input}"),

    # Tool-Calling agent intermediate steps will be store here.
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
    tools=tools,
    verbose=True
)

chat_history = []

def ask_agent(user_query: str):
    response = agent_executor.invoke(
        {
            "input" : user_query,
            "chat_history" : chat_history
        }
    )

    # append the user_query and the AI response in the chat_history
    chat_history.append(HumanMessage(content=user_query))
    chat_history.append(AIMessage(content=response['output']))

    return response['output']

print("Turn-1")
user_query = "My order id is : ORD-102"
print(ask_agent(user_query))

print("------------------------------")

print("Turn-2")
user_query = "Track it."
print(ask_agent(user_query))