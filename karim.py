import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
from google.generativeai import embed_content
import google.generativeai as genai

load_dotenv()
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

st.set_page_config(page_title='Podcast Generator Demo', page_icon='🎙️')

class CustomEmbeddings:
    def embed_documents(self, texts):
        embeddings = []
        for text in texts:
            response = embed_content(model='models/gemini-embedding-2', content=text)
            embeddings.append(response['embedding'])
        return embeddings
    
    def embed_query(self, text):
        response = embed_content(model='models/gemini-embedding-2', content=text)
        return response['embedding']

@st.cache_resource
def initialize_rag():
    loader = PyPDFLoader("C:\\Users\\Karim\\Downloads\\Maillard Reaction.pdf")
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    embeddings = CustomEmbeddings()
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    return vectorstore.as_retriever()

st.title('🎙️ AI Podcast Script Generator')
st.write('Enter a Topic below to generate a short podcast script.')

with st.sidebar:
    st.header('Settings')
    char_limit = st.slider('Max Characters for Prompt', 50, 500, 250)

user_input = st.text_area('What the podcast should be about?', max_chars=char_limit, placeholder='e.g., The impact of AI on marine Biology....')

if user_input:
    try:
        with st.spinner('writing your script....'):
            retriever = initialize_rag()
            llm = ChatGoogleGenerativeAI(model='gemini-2.5-flash')
            
            system_prompt = (
                "You are an expert podcast writer. Use the provided context "
                "to write a brief, engaging 2-person dialogue (Host and Guest). "
                "Focus only on the facts provided in the context.\n\n"
                "Context: {context}"
            )
            
            prompt_template = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", "{input}"),
            ])
            
            def format_docs(docs):
                return "\n\n".join(doc.page_content for doc in docs)
            
            rag_chain = (
                {"context": retriever | format_docs, "input": RunnablePassthrough()}
                | prompt_template
                | llm
            )
            
            response = rag_chain.invoke(user_input)
            
            st.subheader("Generated Script")
            st.write(response.content)
            st.session_state['last_script'] = response.content
            
    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.warning("Please enter a topic!")