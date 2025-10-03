import os
import pickle
import streamlit as st
import requests
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


# ==================== CONFIGURATION ====================
def load_config():
    """Load configuration from Streamlit Secrets or .env file"""
    try:
        # Check if we're in Streamlit Cloud and secrets are available
        if hasattr(st, 'secrets') and st.secrets:
            # Production - using Streamlit Secrets
            OPENROUTER_API_KEY = st.secrets.get('OPENROUTER_API_KEY')
            if OPENROUTER_API_KEY:
                return {
                    'OPENROUTER_API_KEY': OPENROUTER_API_KEY,
                    'DEEPSEEK_MODEL': st.secrets.get('DEEPSEEK_MODEL', 'deepseek/deepseek-chat-v3.1:free'),
                    'FAISS_FILE_PATH': st.secrets.get('FAISS_STORE_PATH', 'faiss_store.pkl'),
                    'EMBEDDING_MODEL': st.secrets.get('EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2'),
                    'CHUNK_SIZE': int(st.secrets.get('CHUNK_SIZE', 1000)),
                    'CHUNK_OVERLAP': int(st.secrets.get('CHUNK_OVERLAP', 100)),
                    'MAX_TOKENS': int(st.secrets.get('MAX_TOKENS', 1000)),
                    'TEMPERATURE': float(st.secrets.get('TEMPERATURE', 0.7))
                }
    except Exception:
        pass  # Fall back to .env file

    # Local development - using python-dotenv
    from dotenv import load_dotenv
    load_dotenv()
    return {
        'OPENROUTER_API_KEY': os.getenv('OPENROUTER_API_KEY'),
        'DEEPSEEK_MODEL': os.getenv('DEEPSEEK_MODEL', 'deepseek/deepseek-chat-v3.1:free'),
        'FAISS_FILE_PATH': os.getenv('FAISS_STORE_PATH', 'faiss_store.pkl'),
        'EMBEDDING_MODEL': os.getenv('EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2'),
        'CHUNK_SIZE': int(os.getenv('CHUNK_SIZE', 1000)),
        'CHUNK_OVERLAP': int(os.getenv('CHUNK_OVERLAP', 100)),
        'MAX_TOKENS': int(os.getenv('MAX_TOKENS', 1000)),
        'TEMPERATURE': float(os.getenv('TEMPERATURE', 0.7))
    }


# Load configuration
config = load_config()
OPENROUTER_API_KEY = config['OPENROUTER_API_KEY']
DEEPSEEK_MODEL = config['DEEPSEEK_MODEL']
FAISS_FILE_PATH = config['FAISS_FILE_PATH']
EMBEDDING_MODEL = config['EMBEDDING_MODEL']
CHUNK_SIZE = config['CHUNK_SIZE']
CHUNK_OVERLAP = config['CHUNK_OVERLAP']
MAX_TOKENS = config['MAX_TOKENS']
TEMPERATURE = config['TEMPERATURE']


def query_deepseek(prompt):
    """Query DeepSeek API"""
    if not OPENROUTER_API_KEY:
        return "❌ API key not found. Please check your configuration."

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/streamlit",
        "X-Title": "Research Tool"
    }

    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS
    }

    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=30)
        if response.status_code == 200:
            result = response.json()
            return result["choices"][0]["message"]["content"]
        else:
            return f"❌ API Error: {response.status_code} - {response.text}"
    except Exception as e:
        return f"❌ Connection Error: {str(e)}"


# Initialize session state for URLs
if 'urls' not in st.session_state:
    st.session_state.urls = [""]  # Start with one empty URL
if 'processed_urls' not in st.session_state:
    st.session_state.processed_urls = []

# Streamlit UI
st.title("📚 Document Q&A")

# Check if API key is available
if not OPENROUTER_API_KEY:
    st.error("""
    ❌ OpenRouter API Key not found!

    Please make sure you have:
    1. For Local Development: Create a `.env` file with your API key
    2. For Streamlit Cloud: Add your API key in Streamlit Secrets

    Your `.env` file should contain:
    OPENROUTER_API_KEY=your_api_key_here
    """)

# Document input - Dynamic URLs
st.subheader("Add Documents")

# Display current URLs
for i, url in enumerate(st.session_state.urls):
    st.session_state.urls[i] = st.text_input(
        f"Document URL {i + 1}",
        value=url,
        key=f"url_{i}",
        placeholder="https://example.com/document"
    )

# Add/Remove URL buttons
col1, col2 = st.columns(2)

with col1:
    if st.button("➕ Add Another URL"):
        st.session_state.urls.append("")
        st.rerun()

with col2:
    if len(st.session_state.urls) > 1 and st.button("➖ Remove Last URL"):
        st.session_state.urls.pop()
        st.rerun()

# Process documents button
process_clicked = st.button("🚀 Process Documents", type="primary")

if process_clicked:
    if not OPENROUTER_API_KEY:
        st.error("Please configure your API key first.")
    else:
        # Filter out empty URLs and validate
        valid_urls = [url.strip() for url in st.session_state.urls if url.strip()]

        if not valid_urls:
            st.error("❌ Please enter at least one URL")
        else:
            # Validate URL format
            invalid_urls = [url for url in valid_urls if not url.startswith(('http://', 'https://'))]
            if invalid_urls:
                st.error(f"❌ Invalid URLs (must start with http:// or https://): {', '.join(invalid_urls)}")
            else:
                with st.spinner(f"Processing {len(valid_urls)} document(s)..."):
                    try:
                        # Load and process documents using WebBaseLoader (works on Streamlit Cloud)
                        loader = WebBaseLoader(valid_urls)
                        data = loader.load()

                        if not data:
                            st.error("❌ No content could be loaded from the URLs")
                            st.info("💡 Try text-based websites like Wikipedia, news articles, or blogs")
                        else:
                            # Split text
                            text_splitter = RecursiveCharacterTextSplitter(
                                separators=['\n\n', '\n', '.', ','],
                                chunk_size=CHUNK_SIZE,
                                chunk_overlap=CHUNK_OVERLAP
                            )
                            docs = text_splitter.split_documents(data)

                            # Create vector store
                            embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
                            vectorstore = FAISS.from_documents(docs, embeddings)

                            # Save vector store
                            with open(FAISS_FILE_PATH, "wb") as f:
                                pickle.dump(vectorstore, f)

                            # Store the processed URLs
                            st.session_state.processed_urls = valid_urls

                            st.success(f"✅ Processed {len(valid_urls)} document(s)! Created {len(docs)} text chunks.")

                    except Exception as e:
                        st.error(f"Error processing documents: {e}")
                        st.info("""
                        💡 **Try these working examples:**
                        - https://en.wikipedia.org/wiki/Artificial_intelligence
                        - https://en.wikipedia.org/wiki/Machine_learning
                        - Any news article or blog post
                        """)

# Q&A section
st.subheader("Ask Questions")
query = st.text_input("Enter your question:", placeholder="What would you like to know about the documents?")

if query:
    if not OPENROUTER_API_KEY:
        st.error("Please configure your API key first.")
    elif not os.path.exists(FAISS_FILE_PATH):
        st.error("""
        ❌ No documents processed yet!

        Please:
        1. Add at least one URL above
        2. Click 'Process Documents' 
        3. Then ask your question
        """)
    else:
        with st.spinner("Searching for answers..."):
            try:
                # Load vector store
                with open(FAISS_FILE_PATH, "rb") as f:
                    vectorstore = pickle.load(f)

                # Find relevant documents
                relevant_docs = vectorstore.similarity_search(query, k=3)

                if not relevant_docs:
                    st.info("🤔 No relevant information found in the documents for this question.")
                else:
                    # Prepare context
                    context = "\n\n".join([doc.page_content for doc in relevant_docs])
                    prompt = f"""Based EXCLUSIVELY on the following context, answer the question. If the answer cannot be found, say so.

Context:
{context}

Question: {query}

Answer:"""

                    # Get answer
                    answer = query_deepseek(prompt)

                    # Display results
                    st.subheader("Answer:")
                    st.write(answer)

                    # Extract ONLY the source URLs that were actually used in the answer
                    used_source_urls = set()
                    for doc in relevant_docs:
                        if hasattr(doc, 'metadata') and doc.metadata.get('source'):
                            used_source_urls.add(doc.metadata['source'])

                    # Only show sources if we found actual URLs used in the answer
                    if used_source_urls:
                        st.subheader("🔗 Sources Used:")
                        for url in used_source_urls:
                            st.write(f"• {url}")
                    else:
                        st.info("📝 Answer generated from processed documents")

            except Exception as e:
                st.error(f"Error retrieving answer: {e}")