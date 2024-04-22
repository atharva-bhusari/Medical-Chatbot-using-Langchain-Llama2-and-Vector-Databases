from src.helper import load_pdf, text_chunks, download_hugging_face_embeddings
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

extraced_data = load_pdf("/home/ubuntu/medical-chat-bot/data")

text_chunks = text_chunks(extraced_data)

embeddings = download_hugging_face_embeddings()

pc = Pinecone(
    api_key=PINECONE_API_KEY,
)

index_name = "chatbot"

docsearch=PineconeVectorStore.from_texts([t.page_content for t in text_chunks], embeddings, index_name=index_name)
