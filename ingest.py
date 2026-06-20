# Loading data - Using LangChain Loaders.
from pathlib import Path
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

DATA_DIR = Path('policies')
CHROMA_DIR=Path('chroma_db')
COLLECTION_NAME="policy_docs"

 #Step-1 : Data Loading
loader = DirectoryLoader(
    path=DATA_DIR,
    glob="*.md",
    loader_cls=TextLoader,
    loader_kwargs={"encoding" : "utf-8"}    
)

docs = loader.load()

print(f"Loaded {len(docs)} documents.")

#Step-2 : Split documents into Chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=50,
    chunk_overlap=10,
    add_start_index=True      
)

chunks = text_splitter.split_documents(docs)

print(f'Original documents: {len(docs)}')
print(f'Generated chunks: {len(chunks)}')

# Step-3: Create Embeddings for chunks
# Step-4: Store these Embeddings into Chroma DB
embedding_model = 'text-embedding-3-small'

embeddings = OpenAIEmbeddings(model=embedding_model)

# Create vector store
vector_store = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=str(CHROMA_DIR)   
)

ids = vector_store.add_documents(documents=chunks)

print("Ingested data successfuly in Chroma DB.")

