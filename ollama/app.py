import os
import streamlit as st
import json
import operator
from typing import Annotated, Sequence, TypedDict, Dict
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_openai import OpenAI
from langchain import hub
from langchain.schema import Document
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import BaseMessage

#base env variables
llm = "mistral:instruct"

#fetch env_key
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("oai_key")
os.environ["TAVILY_API_KEY"] = os.getenv("tavily_api_key")

# streamlit app
st.title('Mistral RAG App')

loader = PyPDFLoader("../data/MSFT_2022_Annual_Report.pdf")
pages = loader.load_and_split()

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
  chunk_size=512, chunk_overlap=80
)

pdf_split = text_splitter.split_documents(pages)

embedding = GPT4AllEmbeddings()

vector_store = Chroma.from_documents(
  documents=pdf_split,
  collection_name="rag_chroma",
  embedding=embedding
)

retriever = vector_store.as_retriever()

#class for data typing
class GraphState(TypedDict):
  """
  Represents the state of our graph.

  Attributes:
    keys: A dictionary where each key is a string.
  """

  keys: Dict[str, any]

def retrieve(state):
  """
  Retrieve documents

  Args:
    state (dict): The current graph state

  Returns:
    state (dict): New key added to state, documents, that contains retrieved documents
  """
  print("---RETRIEVE---")
  state_dict = state["keys"]
  question = state_dict["question"]
  local = state_dict["local"]
  documents = retriever.get_relevant_documents(question)
  return {"keys": {"documents": documents, "local": local, "question": question}}

def generate(state):
  """
  Generate answer

  Args:
    state (dict): The current graph state

  Returns:
    state (dict): New key added to state, generation, that contains generation
  """
  print("---GENERATE---")
  state_dict = state["keys"]
  question = state_dict["question"]
  documents = state_dict["documents"]
  local = state_dict["local"]

  # Prompt
  prompt = PromptTemplate(
    template = """You are an assistant for question-answering tasks. \n
    Use the following pieces of retrieved context to answer the question. \n 
    If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise. \n
    Question: {question} \n
    Context: {context} \n
    Answer: """,
    input_variables=["question", "context"],
  )

  # LLM
  if local == "Yes":
    llm = ChatOllama(model=llm, temperature=0)
  else:
    llm = OpenAI(
      openai_api_key=os.environ["OPENAI_API_KEY"]
    )

  # Post-processing
  def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

  # Chain
  rag_chain = prompt | llm | StrOutputParser()

  # Run
  generation = rag_chain.invoke({"context": documents, "question": question})
  return {
    "keys": {"documents": documents, "question": question, "generation": generation}
  }


def grade_documents(state):
  """
  Determines whether the retrieved documents are relevant to the question.

  Args:
      state (dict): The current graph state

  Returns:
      state (dict): Updates documents key with relevant documents
  """

  print("---CHECK RELEVANCE---")
  state_dict = state["keys"]
  question = state_dict["question"]
  documents = state_dict["documents"]
  local = state_dict["local"]

  # LLM
  if local == "Yes":
    llm = ChatOllama(model=llm, format="json", temperature=0)
  else:
    llm = OpenAI(
      openai_api_key=os.environ["OPENAI_API_KEY"]
    )

  prompt = PromptTemplate(
    template="""You are a grader assessing relevance of a retrieved document to a user question. \n 
    Here is the retrieved document: \n\n {context} \n\n
    Here is the user question: {question} \n
    If the document contains keywords related to the user question, grade it as relevant. \n
    It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
    Provide the binary score as a JSON with a single key 'score' and no premable or explaination.""",
    input_variables=["question", "context"],
  )

  chain = prompt | llm | JsonOutputParser()

  # Score
  filtered_docs = []
  search = "No"  # Default do not opt for web search to supplement retrieval
  for d in documents:
    score = chain.invoke(
      {
        "question": question,
        "context": d.page_content,
      }
    )
    grade = score["score"]
    if grade == "yes":
      print("---GRADE: DOCUMENT RELEVANT---")
      filtered_docs.append(d)
    else:
      print("---GRADE: DOCUMENT NOT RELEVANT---")
      search = "Yes"  # Perform web search
      continue

  return {
    "keys": {
      "documents": filtered_docs,
      "question": question,
      "local": local,
      "run_web_search": search,
    }
  }


def transform_query(state):
  """
  Transform the query to produce a better question.

  Args:
    state (dict): The current graph state

  Returns:
    state (dict): Updates question key with a re-phrased question
  """

  print("---TRANSFORM QUERY---")
  state_dict = state["keys"]
  question = state_dict["question"]
  documents = state_dict["documents"]
  local = state_dict["local"]

  # Create a prompt template with format instructions and the query
  prompt = PromptTemplate(
    template="""You are generating questions that is well optimized for retrieval. \n 
    Look at the input and try to reason about the underlying sematic intent / meaning. \n 
    Here is the initial question:
    \n ------- \n
    {question} 
    \n ------- \n
    Provide an improved question without any premable, only respond with the updated question: """,
    input_variables=["question"],
  )

  # Grader
  # LLM
  if local == "Yes":
    llm = ChatOllama(model=local_llm, temperature=0)
  else:
    llm = OpenAI(
      openai_api_key=os.environ["OPENAI_API_KEY"]
    )

  # Prompt
  chain = prompt | llm | StrOutputParser()
  better_question = chain.invoke({"question": question})

  return {
    "keys": {"documents": documents, "question": better_question, "local": local}
  }

# input form within st
with st.form('test_form'):
  text = st.text_area("Enter text:")
  submitted = st.form_submit_button("Submit")
  if submitted:
    #response
    response(text)["answer"]