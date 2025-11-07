# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Deploying a model using db agents
# MAGIC
# MAGIC Note we have to setup the notebook env 

# COMMAND ----------

# DBTITLE 1,Parameterise Pip installs
mlflow_version = 'mlflow==2.16.0'
langchain_base_version = 'langchain==0.2.11'
langchain_community_version = 'langchain_community==0.2.10'

# COMMAND ----------

# DBTITLE 1,Run Pip Install
# MAGIC %pip install databricks-agents {mlflow_version} {langchain_base_version} {langchain_community_version} langchain_core  langchain-databricks
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import os
import mlflow
from databricks.sdk import WorkspaceClient
from databricks import agents

UC_MODEL_NAME = 'brian_gen_ai.chain_types.langchain_vs_module'
uc_registered_model_info = 1

w = WorkspaceClient()

# COMMAND ----------

# Use Unity Catalog to log the chain
mlflow.set_registry_uri('databricks-uc')

# Deploy to enable the Review APP and create an API endpoint
deployment_info = agents.deploy(model_name=UC_MODEL_NAME, 
                                model_version=uc_registered_model_info)

browser_url = mlflow.utils.databricks_utils.get_browser_hostname()
print(f"\n\nView deployment status: https://{browser_url}/ml/endpoints/{deployment_info.endpoint_name}")
