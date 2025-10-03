📚 DOCUMENT Q&A SYSTEM
=======================

AI-powered question answering for your web documents with source attribution.

🌐 LIVE DEMO: https://document-q-a-system.streamlit.app

🚀 QUICK START
--------------
1. Install: pip install -r requirements.txt
2. Configure: Update your .env file with settings below  
3. Run: streamlit run app.py

📖 HOW TO USE
-------------
1. Add website URLs in the interface
2. Click "Process Documents" to analyze content  
3. Ask questions about the documents
4. Receive AI-generated answers with source links

✨ FEATURES
-----------
- Multi-URL Document Processing
- AI-Powered Question Answering
- Source Attribution & Tracking
- Clean & Intuitive Interface
- Text-Based Website Support

⚙️ .ENV CONFIGURATION
---------------------
- OPENROUTER_API_KEY=your_api_key_here
- DEEPSEEK_MODEL=deepseek/deepseek-chat-v3.1:free
- FAISS_STORE_PATH=faiss_store.pkl
- EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
- CHUNK_SIZE=1000
- CHUNK_OVERLAP=100
- MAX_TOKENS=1000
- TEMPERATURE=0.7

🔧 TROUBLESHOOTING
------------------
- No Documents Processed: Add URLs and click Process
- Invalid URLs: Must start with http:// or https://
- API Errors: Verify OpenRouter API key in .env
- Video URLs: Use text-based websites only
- Module Errors: Run pip install -r requirements.txt
