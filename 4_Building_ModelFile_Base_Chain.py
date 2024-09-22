# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Building and Logging A Basic Chain
# MAGIC
# MAGIC We will use model file chains
# MAGIC This will provide a bit more flexibility 

# COMMAND ----------

# DBTITLE 1,Parameterise Pip installs
mlflow_version = 'mlflow==2.16.2'
langchain_base_version = 'langchain'
langchain_community_version = 'langchain_community==0.2.13'

# COMMAND ----------

# DBTITLE 1,Run Pip Install
# MAGIC %pip install databricks-agents {mlflow_version} {langchain_base_version} {langchain_community_version} langchain_core langgraph langchain-databricks
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Setup Endpoint / UC Config

## configure these to suit what you have available
llm_model_name = 'databricks-meta-llama-3-1-70b-instruct'
embedding_model = 'databricks-bge-large-en'

catalog = 'brian_gen_ai'
schema = 'chain_types'

# COMMAND ----------

# DBTITLE 1,Imports and initialise langchain integration

import mlflow
from langchain_community.chat_models import ChatDatabricks
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

## Enable MLflow Tracing
mlflow.langchain.autolog()

# COMMAND ----------

# DBTITLE 1,Construct Chain
llm_model = ChatDatabricks(
            target_uri='databricks',
            endpoint=llm_model_name,
            temperature=0.1
        )

output_parser = StrOutputParser()

basic_template = ChatPromptTemplate.from_messages(
        [
            ("system", """
            You are a chirpy companion here to make bubbly new friends

            """),
            ("human", "{prompt}"),
         ]
        )

rag_chain = (
            basic_template | llm_model | output_parser
        )

# COMMAND ----------

# DBTITLE 1,test Chain

rag_chain.invoke("test")

# COMMAND ----------

# DBTITLE 1,Model as code extras
mlflow.models.set_model(model=rag_chain)
