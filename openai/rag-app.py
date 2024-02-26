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

#fetch oai_key
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("oai_key")

st.title('Quickstart App')

def load_index_data():
  #load data
  loader = PyPDFLoader("../data/MSFT_2022_Annual_Report.pdf")
  pages = loader.load_and_split()
  #index data into vector store
  faiss_index = FAISS.from_documents(pages, OpenAIEmbeddings())

  return faiss_index

def gen_response(input_text):
  llm = OpenAI(temperature=0.9, openai_api_key=os.environ["OPENAI_API_KEY"])

  prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:

  <context>
  {context}
  </context>

  Question: {input_text}""")

  document_chain = create_stuff_documents_chain(llm, prompt)

  retriever = load_index_data().as_retriever()
  retrieval_chain = create_retrieval_chain(retriever, document_chain)

  res = retrieval_chain.invoke({"input_text": input_text})
  st.info(res)

with st.form('test_form'):
  text = st.text_area("Enter text:")
  submitted = st.form_submit_button("Submit")
  if submitted:
    gen_response(text)["answer"]
