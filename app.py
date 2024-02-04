from langchain.vectorstores.cassandra import Cassandra
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.llms import  OpenAI
from langchain.embeddings import OpenAIEmbeddings
import streamlit as st

# With CassIO, the Engine powering the Astra DB integration in LangChain,
# Also initialize the DB connection:
import cassio
from dotenv import load_dotenv
import os
load_dotenv()

st.set_page_config(page_title="Loi fiscal Marocain")
st.header("Votre Référence Fiscale Maroc 🤖🇲🇦")

ASTRA_DB_ID = os.getenv('ASTRA_DB_ID')
ASTRA_DB_APP_TOKEN = os.getenv('ASTRA_DB_APP_TOKEN')
OPENAI_API_KEY = os.getenv('OPENAI_API_TOKEN')

# Init Connection to the DB
cassio.init(token=ASTRA_DB_APP_TOKEN,database_id=ASTRA_DB_ID)

# Create the LangChain embedding and LLM objects
llm = OpenAI(openai_api_key=OPENAI_API_KEY,model_name="gpt-3.5-turbo-instruct")
embedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# Create your LangChain vector store
vector_store = Cassandra(
    embedding=embedding,
    table_name="pdf_qry_demo",
    session=None,
    keyspace=None
)
vector_store_index = VectorStoreIndexWrapper(vectorstore=vector_store)

def get_response(question):
    answer = vector_store_index.query(question,llm=llm).strip()
    return answer

input = st.text_input("Input: ",key="input")
response = get_response(input)

submit = st.button("Posez-moi une Question")

## Ask button is Clicked
if submit:
    st.subheader("Assistant :")
    st.write(response)


