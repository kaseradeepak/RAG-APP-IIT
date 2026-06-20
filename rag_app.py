from pathlib import Path
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

CHROMA_DIR=Path('chroma_db')
COLLECTION_NAME="policy_docs"
LLM_MODEL='gpt-5.2'

embedding_model = 'text-embedding-3-small'

embeddings = OpenAIEmbeddings(model=embedding_model)

vector_store = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=str(CHROMA_DIR)   
)

retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k" : 3}
)

# query = "What is the return window for electronics?"
query = "Explain me agentic ai"

output_chunks = retriever.invoke(query)

# Generator 
llm = ChatOpenAI(
    model=LLM_MODEL,
    temperature=0   
)

prompt = ChatPromptTemplate.from_template(
    """
You are a helpful customer-support assistant for an e-commerce company.

Use ONLY the retrieved context to answer the user's question.

Rules:
1. If the answer is present in the context, answer clearly.
2. If the answer is not present in the context, say: "I don't know based on the provided documents."
3. Do not use outside knowledge.
4. Mention the source file name wherever possible.
5. Keep the answer concise and practical.

<context>
{context}
</context>

Question:
{question}

Answer:
"""
)

# A | B | C | D => A -> B -> C -> D
rag_chain = (
    {
        "context" : retriever,
        "question" : RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
)

response = rag_chain.invoke(query)

print(response)

