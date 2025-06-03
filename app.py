import streamlit as st
from PyPDF2 import PdfReader
from dotenv import load_dotenv
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from datetime import datetime

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store

def get_conversational_chain(vectorstore=None):
    prompt_template = """
    You are an intelligent document analysis assistant that helps users understand and extract information from their uploaded PDF
    documents through natural conversation. You can process multiple PDFs simultaneously and provide accurate answers by retrieving
    relevant information from them. Always base your responses strictly on the uploaded document content,  
    and acknowledge when information is not available in the provided materials. Handle all uploaded 
    materials with appropriate confidentiality as private user data.\n\n
    Context:\n{context}?\n
    Question:\n{question}\n
    Answer:   """
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question, pdf_docs, conversation_history):
    if pdf_docs is None:
        st.warning("Please upload PDF files before proceeding.")
        return
    text_chunks = get_text_chunks(get_pdf_text(pdf_docs))
    vector_store = get_vector_store(text_chunks)
    user_question_output = ""
    response_output = ""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain(vectorstore=new_db)
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    user_question_output = user_question
    response_output = response['output_text']
    pdf_names = [pdf.name for pdf in pdf_docs] if pdf_docs else []
    conversation_history.append(
        (user_question_output, response_output, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), ", ".join(pdf_names))
    )
    


    # st.markdown(
    #     f"""
    #     <style>
    #         .chat-container {{
    #             max-width: 800px;
    #             margin: 0 auto;
    #         }}
    #         .chat-message {{
    #             padding: 1rem;
    #             border-radius: 10px;
    #             margin-bottom: 1.5rem;
    #             display: flex;
    #             align-items: flex-start;
    #             font-family: 'Arial', sans-serif;
    #             box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
    #         }}
    #         .chat-message.user {{
    #             background-color: #2d3748;
    #             color: #fff;
    #         }}
    #         .chat-message.bot {{
    #             background-color: #edf2f7;
    #             color: #2d3748;
    #         }}
    #         .chat-message .avatar {{
    #             width: 50px;
    #             height: 50px;
    #             margin-right: 1rem;
    #         }}
    #         .chat-message .avatar img {{
    #             width: 100%;
    #             height: 100%;
    #             border-radius: 50%;
    #             object-fit: cover;
    #         }}
    #         .chat-message .message {{
    #             flex: 1;
    #             font-size: 1rem;
    #             line-height: 1.5;
    #         }}
    #         .chat-message .info {{
    #             font-size: 0.85rem;
    #             color: #a0aec0;
    #             margin-top: 0.5rem;
    #         }}
    #     </style>
    #     <div class="chat-container">
    #         <!-- User message -->
    #         <div class="chat-message user">
    #             <div class="avatar">
    #                 <img src="https://static.vecteezy.com/system/resources/thumbnails/005/545/335/small/user-sign-icon-person-symbol-human-avatar-isolated-on-white-backogrund-vector.jpg" alt="User Avatar">
    #             </div>
    #             <div class="message">{user_question_output}</div>
    #         </div>
    #         <!-- Bot response -->
    #         <div class="chat-message bot">
    #             <div class="avatar">
    #                 <img src="https://i.pinimg.com/474x/1e/b0/5f/1eb05f325ec50a15c8b045f3428d6d5e.jpg" alt="Bot Avatar">
    #             </div>
    #             <div class="message">{response_output}</div>
    #         </div>
    #     </div>
    #     """,
    #     unsafe_allow_html=True
    # )


    # if len(conversation_history) == 1:
    #     conversation_history = []
    # elif len(conversation_history) > 1 :                #prevents duplicates after first prompt
    #     last_item = conversation_history[-1]  
    #     conversation_history.remove(last_item) 
    
    for question, answer, timestamp, pdf_name in reversed(conversation_history):
        st.markdown(
            f"""
            <style>
                .chat-container {{
                    max-width: 800px;
                    margin: 0 auto;
                }}
                .chat-message {{
                    padding: 1rem;
                    border-radius: 10px;
                    margin-bottom: 1.5rem;
                    display: flex;
                    align-items: flex-start;
                    font-family: 'Arial', sans-serif;
                    box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
                }}
                .chat-message.user {{
                    background-color: #2d3748;
                    color: #fff;
                }}
                .chat-message.bot {{
                    background-color: #edf2f7;
                    color: #2d3748;
                }}
                .chat-message .avatar {{
                    width: 50px;
                    height: 50px;
                    margin-right: 1rem;
                }}
                .chat-message .avatar img {{
                    width: 100%;
                    height: 100%;
                    border-radius: 50%;
                    object-fit: cover;
                }}
                .chat-message .message {{
                    flex: 1;
                    font-size: 1rem;
                    line-height: 1.5;
                }}
                .chat-message .info {{
                    font-size: 0.85rem;
                    color: #a0aec0;
                    margin-top: 0.5rem;
                }}
            </style>
            <div class="chat-container">
                <!-- User message -->
                <div class="chat-message user">
                    <div class="avatar">
                        <img src="https://static.vecteezy.com/system/resources/thumbnails/005/545/335/small/user-sign-icon-person-symbol-human-avatar-isolated-on-white-backogrund-vector.jpg" alt="User Avatar">
                    </div>
                    <div class="message">{user_question_output}</div>
                </div>
                <!-- Bot response -->
                <div class="chat-message bot">
                    <div class="avatar">
                        <img src="https://i.pinimg.com/474x/1e/b0/5f/1eb05f325ec50a15c8b045f3428d6d5e.jpg" alt="Bot Avatar">
                    </div>
                    <div class="message">{response_output}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs")
    st.header("Ask anything about your PDFs :speech_balloon:")

# Adding styling for the subheader
    st.markdown(
        """
        <style>
        .subheader {
            font-style: italic;  
            margin-top: -30px;   
            font-size: 25px;    
        }
        </style>
        <p class="subheader">AI-powered Q&A for your documents</p>
        """,
        unsafe_allow_html=True,
    )

    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
   
    with st.sidebar:
        pdf_docs = st.file_uploader("Add documents", accept_multiple_files=True)
        if st.button("Upload documents"):
            if pdf_docs:
                with st.spinner("processing..."):
                    st.success("Done")
            else:
                st.warning("No files uploaded")


        if st.button("Clear chat and documents"):
            if len(st.session_state.conversation_history) == 0:
                st.warning("Chat is empty")
            else:
                st.session_state.conversation_history = []  
                st.session_state.user_question = None  
                pdf_docs = None  

    user_question = st.text_input("Ask a question")

    if user_question:
        user_input(user_question, pdf_docs, st.session_state.conversation_history)
        #st.session_state.user_question = ""  

if __name__ == "__main__":
    main()