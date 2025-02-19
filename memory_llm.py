from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
import os

DATA_PATH = "C:/Users/lahar/Downloads/Chatbot/DATA"

def load_pdf_files(data):
    # Check if the directory is being read correctly
    pdf_files = [f for f in os.listdir(data) if f.endswith('.pdf')]
    print(f"Found PDF files: {pdf_files}")

    loader = DirectoryLoader(data, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents

documents = load_pdf_files(DATA_PATH)
print(f"✅ Loaded {len(documents)} documents")

# CHUNKING THE DOCUMENTS
from langchain.text_splitter import RecursiveCharacterTextSplitter
def create_chunks(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks

# Generate text chunks
text_chunks = create_chunks(extracted_data=documents)

# Debugging: Print first 5 chunks
print(f"✅ Number of Chunks: {len(text_chunks)}")
for i, chunk in enumerate(text_chunks[:5]):
    print(f"\n🔹 Chunk {i+1}:\n{chunk.page_content}")
    from langchain_community.vectorstores import FAISS

from langchain_community.embeddings import HuggingFaceEmbeddings

from sentence_transformers import SentenceTransformer

# Step 3: Create Vector Embeddings
def get_embedding_model():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embedding_model

embedding_model = get_embedding_model()

# Step 4: Store embeddings in FAISS
DB_FAISS_PATH = "vectorstore/db_faiss"
db = FAISS.from_documents(text_chunks, embedding_model)

# Save FAISS index locally
db.save_local(DB_FAISS_PATH)
print("✅ FAISS index saved successfully at:", DB_FAISS_PATH)


