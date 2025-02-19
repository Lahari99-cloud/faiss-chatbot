import os
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

#  Step 1: Load Hugging Face Model (Mistral-7B)
HF_TOKEN = os.environ.get("HF_TOKEN")  
HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"

def load_llm(huggingface_repo_id):
    llm = HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        temperature=0.5,
        task="text-generation",  
        model_kwargs={"max_length": 512}
    )
    return llm

llm = load_llm(HUGGINGFACE_REPO_ID)

# Step 2: Load FAISS Vector Store
DB_FAISS_PATH = "vectorstore/db_faiss"

# Ensure the embedding model is set correctly
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load FAISS database (allow dangerous deserialization if necessary)
retriever = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True).as_retriever()

#  Step 3: Define Custom Prompt
CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer the user's question.
If you don't know the answer, just say that you don't know. Don't try to make up an answer.
Only respond based on the given context.

Context: {context}
Question: {question}

Start your answer directly.
"""

prompt = PromptTemplate(template=CUSTOM_PROMPT_TEMPLATE, input_variables=["context", "question"])

# Step 4: Create RetrievalQA Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt}
)

# Step 5: Query the Model
user_query = input("🔹 Write Query Here: ")
response = qa_chain.invoke({'query': user_query})


# Display the result
print("\n🔹 **RESULT:**\n", response["result"])

#  Display source documents
print("\n📚 **SOURCE DOCUMENTS:**")
for doc in response["source_documents"]:
    print(f" - {doc.metadata['source']}")
