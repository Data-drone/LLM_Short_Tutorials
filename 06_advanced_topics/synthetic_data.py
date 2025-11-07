# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Lets Explore Synthetic Data

# COMMAND ----------

# DBTITLE 1,Parameterise Pip installs
mlflow_version = 'mlflow==2.17.0'
langchain_base_version = 'langchain'
langchain_community_version = 'langchain_community==0.2.17'
langgraph_version = '0.2.35'

# COMMAND ----------

# DBTITLE 1,Setup Libs
# MAGIC %pip install -U pydantic>2.0.0 {mlflow_version} langchain {langchain_community_version} langchain_core langgraph>={langgraph_version} langchain-databricks
# MAGIC %restart_python

# COMMAND ----------

# DBTITLE 1,Configuration
model_to_use = 'databricks-meta-llama-3-1-405b-instruct'

# COMMAND ----------

# DBTITLE 1,Library Imports
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.chat_models import ChatDatabricks

# COMMAND ----------

# DBTITLE 1,Initiate model
llm_model = ChatDatabricks(
            target_uri='databricks',
            endpoint=model_to_use,
            temperature=0.1
        )

# COMMAND ----------

# DBTITLE 1,Testing generating Q&A

