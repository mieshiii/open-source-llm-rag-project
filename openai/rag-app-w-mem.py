import os
import streamlit as st
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.chains import create_history_aware_retriever
from langchain_core.messages import HumanMessage, AIMessage

#fetch oai_key
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("oai_key")

st.title('Retrieval Chain App')

def load_index_data():
  #load data
  loader = PyPDFLoader("../data/MSFT_2022_Annual_Report.pdf")
  pages = loader.load_and_split()
 
  return pages

def gen_response(input_text):
  llm = OpenAI(temperature=0.9, openai_api_key=os.environ["OPENAI_API_KEY"])
  
  #index data into vector store
  faiss_index = FAISS.from_documents(load_index_data(), OpenAIEmbeddings())

  retriever = faiss_index.as_retriever()

  prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer the user's questions based on the below context:\n\n{context}"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
  ])

  document_chain = create_stuff_documents_chain(llm, prompt)
  retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
  retrieval_chain = create_retrieval_chain(retriever_chain, document_chain)
  
  #add chat history for context
  #chat history is out of context for the actual application
  chat_history = [HumanMessage(content="Can you help me summarize a document?"), AIMessage(content="Yes!")]
  res = retrieval_chain.invoke({
    "chat_history": chat_history,
    "input": input_text
  })
  st.info(res["answer"])

with st.form('test_form'):
  text = st.text_area("Enter text:")
  submitted = st.form_submit_button("Submit")
  if submitted:
    gen_response(text)["answer"]
