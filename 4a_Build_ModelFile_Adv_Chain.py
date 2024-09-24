# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Building and Logging A Parameterised Retriever Model File Chain
# MAGIC
# MAGIC We will do a model file chain with more advanced configs

# COMMAND ----------

# DBTITLE 1,Parameterise Pip installs
mlflow_version = 'mlflow==2.16.2'
langchain_base_version = 'langchain'
langchain_community_version = 'langchain_community==0.2.13'

# COMMAND ----------

# DBTITLE 1,Run Pip Install
# MAGIC %pip install {mlflow_version} {langchain_base_version} {langchain_community_version} langchain_core langgraph langchain-databricks
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Config Management

import mlflow

# We will use ModelConfig for config management
# The structure of these yamls is totally up to you

# Load the chain's configuration
model_config = mlflow.models.ModelConfig(development_config="rag_chain_config.yaml")

model_resources = model_config.get("resources")
uc_config = model_config.get("uc_config")

#### endpoints and models
llm_model_name = model_resources.get("llm_model_name")
embedding_model_name = model_resources.get("embedding_model")
vs_endpoint_name = model_resources.get("vs_endpoint")

#### uc_config
catalog = uc_config.get("catalog")
schema = uc_config.get("schema")
index_name = uc_config.get("index_name")

# COMMAND ----------

# DBTITLE 1,Load Instruction Finetuned Model
from langchain_community.chat_models import ChatDatabricks

llm_model = ChatDatabricks(
            target_uri='databricks',
            endpoint=llm_model_name,
            temperature=0.1
        )

llm_model.invoke("Hello")

# COMMAND ----------

# DBTITLE 1,Load Embedding Model
from langchain_community.embeddings import DatabricksEmbeddings

embeddings = DatabricksEmbeddings(endpoint=embedding_model_name)

embeddings.embed_query("test")

# COMMAND ----------

# DBTITLE 1,Load DB Vector Store
from databricks.vector_search.client import VectorSearchClient
from langchain_community.vectorstores import DatabricksVectorSearch

vsc = VectorSearchClient()
index = vsc.get_index(endpoint_name=vs_endpoint_name,
                      index_name=f"{catalog}.{schema}.{index_name}")

index.similarity_search(columns=["page_content"],query_text="hello")

# COMMAND ----------

# DBTITLE 1,Build Retrieval Chain
retriever = DatabricksVectorSearch(
    index, text_column="page_content", 
    embedding=embeddings, columns=["source_doc"]
).as_retriever()

retriever.invoke("What is a RAG?")

# COMMAND ----------

# DBTITLE 1,Build Chain with Prompt
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter

def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

basic_template = ChatPromptTemplate.from_messages(
        [
            ("system", """
            You are a helpful assistant designed to help customers undertake research on RAG models
            
            [Context]
            {rag_content}

            """),
            ("human", "{user_input}"),
         ] 
        )
output_parser = StrOutputParser()

# Build Rag Chain
# For consistency we use input as the key of the input json as well.
# The LCEL code then remaps it to user_input to sent to our basic_template
rag_chain = (
            {"rag_content": itemgetter("input") | retriever | format_docs, 
             "user_input": itemgetter("input") | RunnablePassthrough()}
            | basic_template | llm_model | output_parser
        )

rag_chain.invoke({"input": "Why do I need RAG techniques?"})

# COMMAND ----------

# Set the model
mlflow.models.set_model(model=rag_chain)