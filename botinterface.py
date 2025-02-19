import os
import streamlit as st

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint

# Ensure HF_TOKEN is available
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    st.error(" HF_TOKEN is missing! Set it before running the script.")
    st.stop()

# Define FAISS vector store path
DB_FAISS_PATH = "vectorstore/db_faiss"

# Load FAISS vector store safely
@st.cache_resource
def get_vectorstore():
    try:
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

        # Ensure vector store is not empty
        if not db.docstore._dict:
            st.error(" FAISS vector store is empty! Try re-indexing your documents.")
            st.stop()
        return db
    except Exception as e:
        st.error(f" Error loading FAISS vector store: {str(e)}")
        st.stop()

# Set custom prompt template
def set_custom_prompt():
    return PromptTemplate(
        template="""
        Use the information in the context to answer the question.
        If you don't know, say "I don't know." Don't make up an answer.

        Context: {context}
        Question: {question}
        """,
        input_variables=["context", "question"]
    )

# Load LLM model safely
def load_llm(huggingface_repo_id, hf_token):
    try:
        llm = HuggingFaceEndpoint(
            repo_id=huggingface_repo_id,
            temperature=0.5,
            task="text-generation",  
            model_kwargs={"token": hf_token, "max_length": 512}
        )
        return llm
    except Exception as e:
        st.error(f" Error loading LLM: {str(e)}")
        st.stop()

# Main Streamlit App
def main():
    st.title("🤖 I'M LAHARI'S ASSISTANT")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).markdown(msg["content"])

    # User input
    prompt = st.chat_input("📝 Ask a question here...")
    if prompt:
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Load vector store & LLM
        vectorstore = get_vectorstore()
        llm = load_llm("mistralai/Mistral-7B-Instruct-v0.3", HF_TOKEN)  # ✅ Pass required arguments
        prompt_template = set_custom_prompt()

        # Create Retrieval QA Chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt_template}
        )

        # Query the model (Ensure correct key: `query`)
        try:
            response = qa_chain.invoke({"query": prompt})  
            result = response.get("result", "No result found.")
            sources = response.get("source_documents", [])

            # Display Response
            st.chat_message("assistant").markdown(result)
            st.session_state.messages.append({"role": "assistant", "content": result})

            # Display Source Documents
            if sources:
                st.subheader("📚 Source Documents:")
                for doc in sources:
                    st.write(f"- {doc.metadata.get('source', 'Unknown Source')}")
        except Exception as e:
            st.error(f" Error retrieving answer: {str(e)}")

if __name__ == "__main__":
    main()
