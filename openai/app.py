import os
from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain_core.prompts import ChatPromptTemplate
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI
import streamlit as st

#fetch oai_key
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("oai_key")

st.title('Quickstart App')

def gen_response(input_text):
  llm = OpenAI(temperature=0.9, openai_api_key=os.environ["OPENAI_API_KEY"])

  prompt = ChatPromptTemplate.from_messages([
    ("system", "You are world class technical documentation writer."),
    ("user", "{input_text}")
  ])

  chain = prompt | llm

  res = chain.invoke({"input_text": input_text})
  st.info(res)

with st.form('test_form'):
  text = st.text_area("Enter text:")
  submitted = st.form_submit_button("Submit")
  if submitted:
    gen_response(text)
