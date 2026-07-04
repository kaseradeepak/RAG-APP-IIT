# Manual appending of chat_history is good for prototypes, but not good for production-grade agents.

# Langchain provides RunnableWithMessageHistory to automatically: 
# - loads the previous messages from the conversation.
# - Append the current query and response to the conversation.

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# Fake Database.
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
def get_order_status(order_id: str) -> str:
    """
    Returns the status of the order id given by the user.

    Use this tool when the user is asking about the order tracking status.
    """

    order = ORDERS.get(order_id)

    if not order:
        return f"Order with the id: {order_id} not found."

    return f"Order status: {order['status']}"


tools = [get_order_status]

# Create llm
llm = ChatOpenAI(
    model="gpt-5.2",
    temperature=0
)

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are a helpful customer support agent.

Rules:
- If the user gives an order id, remember it for this conversation.
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

store = {}

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    """
    Returns the chat history for the given session_id or conversation_id.
    """

    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    
    return store[session_id]

agent_with_memory = RunnableWithMessageHistory(
    agent_executor,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="output"
)

def ask_agent(user_query: str, session_id: str):
    response = agent_with_memory.invoke(
        {"input" : user_query},
        config={
            "configurable" : {
                "session_id" : session_id
            }
        }
    )

    return response['output']

session_id = 'user-101'

print('Turn-1')
user_query = "My order id is ORD-101"
print("User query: ", user_query)
print("AI response: ", ask_agent(user_query, session_id))

print('Turn-2')
user_query = "Track it."
print("User query", user_query)
print("AI response: ", ask_agent(user_query, session_id))

session_id_1 = 'user-102'

print('Turn-1 for the new user.')
user_query = "Track it."
print("User query", user_query)
print("AI response: ", ask_agent(user_query, session_id_1))


