# LangChain RAG Multi-PDF Chatbot
A Retrieval-Augmented Generation (RAG) AI chatbot application built with LangChain and Google Gemini that enables users to upload multiple PDF documents and engage in intelligent conversations about their content. The app processes uploaded PDFs, creates vector embeddings for efficient document retrieval using FAISS (Facebook AI Similarity Search), and combines this with LLM capabilities to provide contextually relevant answers to user queries. Built with a clean Streamlit interface, the application offers real-time document processing, maintains conversation history, and leverages advanced RAG techniques to deliver accurate, source-grounded responses based on the uploaded material.

![image](https://github.com/user-attachments/assets/403ff0a7-ef6c-4d09-8cfc-98b7538d105b)

## Usage
1. Clone repo
2. Install dependencies in a virtual environment by running
   ```
   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
   ```
4. Obtain an API key from [Google AI](https://ai.google.dev/) and add it to the .env file in the project directory.
5. Run `streamlit run app.py` to launch the app in your web browser
