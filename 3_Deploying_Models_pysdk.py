# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Deploying a Model
# MAGIC
# MAGIC Notes: Looks like PyFunc needs traditional deployment with env vars
# MAGIC Is there a way with agents? What formats does agents requires?

# COMMAND ----------

# DBTITLE 1,Parameterise Pip installs
mlflow_version = 'mlflow==2.16.0'

# COMMAND ----------

# DBTITLE 1,Run Pip Install
# MAGIC %pip install -U {mlflow_version} databricks-agents databricks-sdk
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Setup Parameters for endpoints etc

# we wiped the params to re-adding
mlflow_version = 'mlflow==2.16.0'
UC_CATALOG = 'brian_gen_ai'
UC_SCHEMA = 'chain_types'

username = spark.sql("SELECT current_user()").first()['current_user()']
parsed_name = username.split("@")[0].replace(".", "_")

# COMMAND ----------

# MAGIC %md
# MAGIC # Deploy with Databricks Python SDK
# MAGIC
# MAGIC We can deploy a model with the python sdk

# COMMAND ----------

import os
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput

w = WorkspaceClient()

def create_databricks_token(workspace_client):
    token_response = workspace_client.tokens.create(
        comment="test_model_token",
        lifetime_seconds=3600*24*30  # Token valid for 30 days
    )
    return token_response.token_value

databricks_workspace_url = spark.conf.get("spark.databricks.workspaceUrl")

# COMMAND ----------

# DBTITLE 1,Basic Chat Chain
UC_MODEL_NAME = f'{UC_CATALOG}.{UC_SCHEMA}.basic_chat'
model_version = 5

endpoint_config = EndpointCoreConfigInput(
    served_entities=[
        ServedEntityInput(
            entity_name = UC_MODEL_NAME,
            entity_version = model_version,
            workload_size = 'Small',
            scale_to_zero_enabled = True,
            environment_vars={
                "DATABRICKS_HOST": f"https://{databricks_workspace_url}",
                "DATABRICKS_TOKEN": create_databricks_token(w)
            }
        )
    ]
)

endpoint = w.serving_endpoints.create_and_wait(
    name=f'{parsed_name}_basic_chat',
    config=endpoint_config)

# COMMAND ----------

# DBTITLE 1,Retriever Chain
UC_MODEL_NAME = f'{UC_CATALOG}.{UC_SCHEMA}.rag_chat'
model_version = 1

endpoint_config = EndpointCoreConfigInput(
    served_entities=[
        ServedEntityInput(
            entity_name = UC_MODEL_NAME,
            entity_version = model_version,
            workload_size = 'Small',
            scale_to_zero_enabled = True,
            environment_vars={
                "DATABRICKS_HOST": f"https://{databricks_workspace_url}",
                "DATABRICKS_TOKEN": create_databricks_token(w)
            }
        )
    ]
)

endpoint = w.serving_endpoints.create_and_wait(
    name=f'{parsed_name}_rag_chat',
    config=endpoint_config)

# COMMAND ----------

# DBTITLE 1,Retriever Chain - Langchain integration
UC_MODEL_NAME = f'{UC_CATALOG}.{UC_SCHEMA}.langchain_vs_module'
model_version = 1

endpoint_config = EndpointCoreConfigInput(
    served_entities=[
        ServedEntityInput(
            entity_name = UC_MODEL_NAME,
            entity_version = model_version,
            workload_size = 'Small',
            scale_to_zero_enabled = True,
            environment_vars={
                "DATABRICKS_HOST": f"https://{databricks_workspace_url}",
                "DATABRICKS_TOKEN": create_databricks_token(w)
            }
        )
    ]
)

endpoint = w.serving_endpoints.create_and_wait(
    name=f'{parsed_name}_rag_langchain',
    config=endpoint_config)

# COMMAND ----------

# DBTITLE 1,Agent Chain
UC_MODEL_NAME = f'{UC_CATALOG}.{UC_SCHEMA}.agent_chat'
model_version = 1

endpoint_config = EndpointCoreConfigInput(
    served_entities=[
        ServedEntityInput(
            entity_name = UC_MODEL_NAME,
            entity_version = model_version,
            workload_size = 'Small',
            scale_to_zero_enabled = True,
            environment_vars={
                "DATABRICKS_HOST": f"https://{databricks_workspace_url}",
                "DATABRICKS_TOKEN": create_databricks_token(w)
            }
        )
    ]
)

endpoint = w.serving_endpoints.create_and_wait(
    name=f'{parsed_name}_rag_agent',
    config=endpoint_config)
