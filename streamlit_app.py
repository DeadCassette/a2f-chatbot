import streamlit as st
import os
import json
from datetime import datetime
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
import tempfile

def save_feedback(question, answer, feedback, timestamp):
    """
    Save feedback to a JSON file.
    """
    feedback_data = {
        "timestamp": timestamp,
        "question": question,
        "answer": answer,
        "feedback": feedback
    }
    
    # Load existing feedback or create new file
    feedback_file = "feedback_data.json"
    try:
        with open(feedback_file, 'r') as f:
            existing_feedback = json.load(f)
    except FileNotFoundError:
        existing_feedback = []
    
    # Add new feedback
    existing_feedback.append(feedback_data)
    
    # Save back to file
    with open(feedback_file, 'w') as f:
        json.dump(existing_feedback, f, indent=2)

def load_feedback():
    """
    Load feedback data from JSON file.
    """
    feedback_file = "feedback_data.json"
    try:
        with open(feedback_file, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return []

def main():
    """
    Main function to run the Streamlit chatbot application.
    """
    st.set_page_config(page_title="Club Policy Chatbot", layout="wide")
    st.title("Club Policy Chatbot")

    # Sidebar for admin features
    with st.sidebar:
        st.header("📊 Admin Panel")
        admin_password = st.text_input("Admin Password", type="password", key="admin_pwd")
        
        # Check if password is correct (you can change this password)
        if admin_password == "a2fadmin":  # Change this to your desired password
            if st.button("View Feedback & Analytics"):
                st.session_state.show_analytics = True
            else:
                st.session_state.show_analytics = False
        elif admin_password:  # If password was entered but incorrect
            st.error("❌ Incorrect password")
            st.session_state.show_analytics = False

    # Show analytics if requested
    if st.session_state.get('show_analytics', False):
        show_analytics()
        return

    # Securely get OpenAI API key
    try:
        openai_api_key = st.secrets["OPENAI_API_KEY"]
        os.environ["OPENAI_API_KEY"] = openai_api_key
    except KeyError:
        st.error("OPENAI_API_KEY not found in Streamlit secrets. Please add it.")
        st.stop()

    # Load default document
    default_document_path = "default_policy_document.pdf"
    
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

                # Feedback system
                st.write("---")
                col1, col2, col3 = st.columns([1, 1, 2])
                
                with col1:
                    if st.button("👍 Helpful", key="thumbs_up"):
                        save_feedback(user_question, answer, "helpful", datetime.now().isoformat())
                        st.success("Thank you for your feedback!")
                        
                with col2:
                    if st.button("👎 Not Helpful", key="thumbs_down"):
                        save_feedback(user_question, answer, "not_helpful", datetime.now().isoformat())
                        st.error("Thank you for your feedback. We'll work to improve!")

                with st.expander("Show relevant sources"):
                    for doc in source_documents:
                        st.write("---")
                        # Clean up the text formatting by replacing multiple spaces and newlines
                        cleaned_content = ' '.join(doc.page_content.split())
                        st.write(cleaned_content)
                        if 'source' in doc.metadata:
                            st.caption(f"Source: {os.path.basename(doc.metadata['source'])}")

def show_analytics():
    """
    Display feedback analytics and usage statistics.
    """
    st.title("📊 Feedback & Analytics")
    
    feedback_data = load_feedback()
    
    if not feedback_data:
        st.info("No feedback data available yet.")
        return
    
    # Summary statistics
    total_feedback = len(feedback_data)
    helpful_count = sum(1 for f in feedback_data if f['feedback'] == 'helpful')
    not_helpful_count = sum(1 for f in feedback_data if f['feedback'] == 'not_helpful')
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Interactions", total_feedback)
    with col2:
        st.metric("Helpful Responses", helpful_count)
    with col3:
        st.metric("Not Helpful", not_helpful_count)
    
    # Satisfaction rate
    if total_feedback > 0:
        satisfaction_rate = (helpful_count / total_feedback) * 100
        st.metric("Satisfaction Rate", f"{satisfaction_rate:.1f}%")
    
    # Recent feedback
    st.subheader("Recent Feedback")
    for feedback in feedback_data[-10:]:  # Show last 10 interactions
        with st.expander(f"Question: {feedback['question'][:50]}..."):
            st.write(f"**Question:** {feedback['question']}")
            st.write(f"**Answer:** {feedback['answer']}")
            st.write(f"**Feedback:** {feedback['feedback']}")
            st.write(f"**Timestamp:** {feedback['timestamp']}")
    
    # Download feedback data
    st.subheader("Download Data")
    feedback_json = json.dumps(feedback_data, indent=2)
    st.download_button(
        label="Download Feedback Data (JSON)",
        data=feedback_json,
        file_name="feedback_data.json",
        mime="application/json"
    )

def combine_vectorstores(vectorstore1, vectorstore2):
    """
    Combines two FAISS vectorstores into one.
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
