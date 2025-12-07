from dotenv import load_dotenv
import os
from pinecone import Pinecone
from src.helper import *
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore

load_dotenv()


PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

docs = load_pdf('data')
text_chunk = text_splitter(docs)
embeddings = download_embedding()

pinecone_api_key = PINECONE_API_KEY
pc = Pinecone(api_key = pinecone_api_key)

index_name = 'medico' 

if not pc.has_index(index_name):
    pc.create_index(
        name = index_name,
        dimension = 384,
        metric= 'cosine',
        spec = ServerlessSpec(cloud = 'aws', region = 'us-east-1')
    )
index = pc.Index(index_name)

docsearch = PineconeVectorStore.from_documents(
    documents = text_chunk,
    embedding = embeddings,
    index_name = index_name
)