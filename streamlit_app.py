import streamlit as st
import os
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
import tempfile

def main():
    """
    Main function to run the Streamlit chatbot application.
    """
    st.set_page_config(page_title="Club Policy Chatbot", layout="wide")
    st.title("Club Policy Chatbot")

    # Securely get OpenAI API key
    try:
        openai_api_key = st.secrets["OPENAI_API_KEY"]
        os.environ["OPENAI_API_KEY"] = openai_api_key
    except KeyError:
        st.error("OPENAI_API_KEY not found in Streamlit secrets. Please add it.")
        st.stop()

    # Load default document
    default_document_path = "default_policy_document.pdf"  # You'll need to add this file to your repo
    
    # Check if default document exists
    if os.path.exists(default_document_path):
        st.info("📋 Default policy document loaded automatically")
        default_vectorstore = process_document(default_document_path)
    else:
        st.warning("⚠️ Default policy document not found. Please upload a document.")
        default_vectorstore = None

    # File uploader for additional documents
    uploaded_file = st.file_uploader(
        "Upload additional policy documents (PDF or DOCX) - Optional",
        type=["pdf", "docx"]
    )

    # Combine default and uploaded documents
    if uploaded_file is not None:
        # Create a temporary file to store the uploaded content
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        uploaded_vectorstore = process_document(tmp_file_path)
        
        # Clean up the temporary file
        os.remove(tmp_file_path)
        
        # Combine vectorstores if both exist
        if default_vectorstore and uploaded_vectorstore:
            vectorstore = combine_vectorstores(default_vectorstore, uploaded_vectorstore)
            st.success("✅ Default document + uploaded document processed successfully.")
        elif uploaded_vectorstore:
            vectorstore = uploaded_vectorstore
            st.success("✅ Uploaded document processed successfully.")
        else:
            vectorstore = None
    else:
        vectorstore = default_vectorstore

        if vectorstore:
            # Show which documents are loaded
            st.success("✅ Documents processed successfully. You can now ask questions.")
            
            # Display loaded documents info
            with st.expander("📚 Currently loaded documents"):
                if default_vectorstore:
                    st.write("• Default policy document")
                if uploaded_file is not None:
                    st.write(f"• Uploaded: {uploaded_file.name}")
            
            # Chatbot interface
            user_question = st.text_input("Ask a question about the policy document:")

            if user_question:
                with st.spinner("Searching for the answer..."):
                    from langchain.prompts import PromptTemplate

                    prompt_template = PromptTemplate(
                        input_variables=["context", "question"],
                        template=(
                            "You are a helpful assistant answering questions about club policies. "
                            "Use the following pieces of context to answer the user's question. "
                            "Look carefully for specific dates, deadlines, requirements, and procedures. "
                            "If you find relevant information, provide it clearly and completely. "
                            "If you don't know the answer, just say you don't know.\n"
                            "----------------\n"
                            "{context}\n"
                            "Question: {question}\n"
                            "Helpful answer:"
                        )
                    )

                    qa_chain = RetrievalQA.from_chain_type(
                        llm=ChatOpenAI(model="gpt-4o-mini", temperature=0),
                        chain_type="stuff",
                        retriever=vectorstore.as_retriever(
                            search_kwargs={
                                'k': 5,  # Retrieve more chunks for better coverage
                                'score_threshold': 0.7  # Only include relevant chunks
                            }
                        ),
                        return_source_documents=True,
                        chain_type_kwargs={"prompt": prompt_template}
                    )

                    result = qa_chain({"query": user_question})
                    answer = result["result"]
                    source_documents = result["source_documents"]

                    st.subheader("Answer:")
                    st.write(answer)

                    with st.expander("Show relevant sources"):
                        for doc in source_documents:
                            st.write("---")
                            # Clean up the text formatting by replacing multiple spaces and newlines
                            cleaned_content = ' '.join(doc.page_content.split())
                            st.write(cleaned_content)
                            if 'source' in doc.metadata:
                                st.caption(f"Source: {os.path.basename(doc.metadata['source'])}")


def combine_vectorstores(vectorstore1, vectorstore2):
    """
    Combines two FAISS vectorstores into one.
    
    Args:
        vectorstore1: First FAISS vectorstore
        vectorstore2: Second FAISS vectorstore
        
    Returns:
        FAISS: Combined vectorstore
    """
    # Get all documents from both vectorstores
    docs1 = vectorstore1.docstore._dict
    docs2 = vectorstore2.docstore._dict
    
    # Combine all documents
    all_docs = list(docs1.values()) + list(docs2.values())
    
    # Create new embeddings for combined documents
    embeddings = OpenAIEmbeddings()
    combined_vectorstore = FAISS.from_documents(all_docs, embeddings)
    
    return combined_vectorstore

@st.cache_resource(show_spinner="Processing document and creating vector store...")
def process_document(file_path):
    """
    Loads, chunks, embeds, and stores the document in a FAISS vector store.
    
    Args:
        file_path (str): The path to the PDF or DOCX file.

    Returns:
        FAISS: The FAISS vector store.
    """
    try:
        file_extension = os.path.splitext(file_path)[1].lower()
        if file_extension == ".pdf":
            loader = PyPDFLoader(file_path)
        elif file_extension == ".docx":
            loader = UnstructuredWordDocumentLoader(file_path)
        else:
            st.error("Unsupported file type. Please upload a PDF or DOCX file.")
            return None
        
        documents = loader.load()

        # Chunk the document with better parameters for policy documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,  # Increased chunk size to capture more context
            chunk_overlap=300,  # Increased overlap to ensure important info isn't lost
            separators=["\n\n", "\n", ".", "!", "?", ":", ";", " "],  # More separators for better chunking
            length_function=len
        )
        texts = text_splitter.split_documents(documents)

        # Embed the document and create a vector store
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(texts, embeddings)
        
        return vectorstore

    except Exception as e:
        st.error(f"An error occurred while processing the document: {e}")
        return None

if __name__ == "__main__":
    main()
