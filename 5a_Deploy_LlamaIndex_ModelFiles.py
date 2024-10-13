# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Logging Model as Code Chains
# MAGIC
# MAGIC Here we will log models as code

# COMMAND ----------

# DBTITLE 1,Parameterise Pip installs
mlflow_version = 'mlflow==2.16.2'
llama_index_version = 'llama-index==0.11.13'

# COMMAND ----------

# DBTITLE 1,Run Pip Install
# MAGIC %pip install -U databricks-sdk {mlflow_version} {llama_index_version} llama-index-llms-databricks llama-index-vector-stores-databricks llama-index-embeddings-databricks
# MAGIC %restart_python

# COMMAND ----------

# DBTITLE 1,Setup Parameters for endpoints etc
import mlflow
import os
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput


# Setting up MLflow experiment
mlflow.set_registry_uri('databricks-uc')

username = spark.sql("SELECT current_user()").first()['current_user()']
parsed_name = username.split("@")[0].replace(".", "_")

experiment_name = 'brian_model_files'
catalog = 'brian_gen_ai'
schema = 'chain_types'

mlflow.set_experiment(f'/Users/{username}/{experiment_name}')

w = WorkspaceClient()

def create_databricks_token(workspace_client):
    token_response = workspace_client.tokens.create(
        comment="test_model_token",
        lifetime_seconds=3600*24*30  # Token valid for 30 days
    )
    return token_response.token_value

databricks_workspace_url = spark.conf.get("spark.databricks.workspaceUrl")

# COMMAND ----------

# DBTITLE 1,Log Llama Index Model
import yaml

# Load the model config into a dict
with open('rag_chain_config.yaml', 'r') as file:
    model_config = yaml.safe_load(file)

with mlflow.start_run(run_name='llama_index_base_test'):
    
    mlflow.set_tag("type", "llama_index_engine")

    logged_llama_index = mlflow.llama_index.log_model(
        llama_index_model = os.path.join(
            os.getcwd(), '4c_LlamaIndex_ModelFile'),
        #model_config=model_config,
        artifact_path="llama_index",
        input_example="hi",
        extra_pip_requirements=["llama-index-llms-databricks",
                                "llama-index-vector-stores-databricks", 
                                "llama-index-embeddings-databricks"],
        registered_model_name=f"{catalog}.{schema}.llama_index_basic",
    )

# COMMAND ----------

# DBTITLE 1,Deploy Model file model
UC_MODEL_NAME = f'{catalog}.{schema}.llama_index_basic'
model_version = 1

endpoint_config = EndpointCoreConfigInput(
    served_entities=[
        ServedEntityInput(
            entity_name = UC_MODEL_NAME,
            entity_version = model_version,
            workload_size = 'Small',
            scale_to_zero_enabled = True,
            environment_vars={
                "DATABRICKS_SERVING_ENDPOINT": f"https://{databricks_workspace_url}/serving-endpoints",
                "DATABRICKS_TOKEN": create_databricks_token(w)
            }
        )
    ]
)

endpoint = w.serving_endpoints.create_and_wait(
    name=f'{parsed_name}_basic_llama_index',
    config=endpoint_config)    