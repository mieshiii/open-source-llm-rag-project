import os
import streamlit as st
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage, AIMessage
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import create_openai_functions_agent
from langchain.agents import AgentExecutor

#fetch oai_key
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("oai_key")

st.title('Retrieval Agent App')

def load_index_data():
  #load data
  loader = PyPDFLoader("../data/MSFT_2022_Annual_Report.pdf")
  pages = loader.load_and_split()
 
  return pages

def gen_response(input_text):
  llm = OpenAI(temperature=0, openai_api_key=os.environ["OPENAI_API_KEY"])
  
  #index data into vector store
  faiss_index = FAISS.from_documents(load_index_data(), OpenAIEmbeddings())

  retriever = faiss_index.as_retriever()

  prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer the user's questions based on the below context:\n\n{context}"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
  ])
  
  #you still need a way to save this part probably in a cache-style db?
  #another idea is to save the chat history into a nosql db, parse it and feed it into the chat history per uuid (?)
  chat_history = [
    HumanMessage(content="Can you help me summarize a document?"), AIMessage(content="Yes!")
  ]

  #create retriever tool
  retriever_tool = create_retriever_tool(
    retriever,
    "document_search"
  )

  tools = [retriever_tool]

  #create agent
  agent = create_openai_functions_agent(llm, tools, prompt)
  agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
  
  #invoke agent
  res = agent_executor.invoke({
    "chat_history": chat_history,
    "input": input_text
  })

  st.info(res["answer"])

with st.form('test_form'):
  text = st.text_area("Enter text:")
  submitted = st.form_submit_button("Submit")
  if submitted:
    gen_response(text)["answer"]
