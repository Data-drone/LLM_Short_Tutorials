# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Building and Logging A Parameterised Agent Model File Chain
# MAGIC
# MAGIC We will do a model file chain with more advanced configs \
# MAGIC We will also look at structured input \
# MAGIC We will also use langgraph

# COMMAND ----------

# DBTITLE 1,Run Pip Install
# MAGIC %pip install -U pydantic>2.0.0 mlflow langchain langchain_core langgraph databricks-langchain databricks-sdk databricks-agents
# MAGIC %restart_python

# COMMAND ----------

# DBTITLE 1,Config Management
import mlflow
import os
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput

mlflow.set_registry_uri('databricks-uc')

username = spark.sql("SELECT current_user()").first()['current_user()']
parsed_name = username.split("@")[0].replace(".", "_")

experiment_name = 'brian_model_files'
catalog = 'brian_gen_ai'
schema = 'chain_types'

mlflow.set_experiment(f'/Users/{username}/{experiment_name}')

# COMMAND ----------

# DBTITLE 1,Log Model File

# Note that we use the same input sample as tested in the notebook
# To ensure smooth API deployment make the input a json
from mlflow.models.resources import (
    DatabricksServingEndpoint,
    DatabricksVectorSearchIndex,
    DatabricksGenieSpace
)

with mlflow.start_run(run_name='modelfile_base_chain'):

    mlflow.set_tag("type", "chain")

    logged_chain_info = mlflow.langchain.log_model(
        lc_model=os.path.join(
            os.getcwd(), '6_Building_Adv_Agents'
        ),  
        artifact_path="chain",  # Required by MLflow
        input_example={"messages": [
            {
                "role": "user",
                "content": "Hello!!!",
            },
        ]},
        resources=[
            DatabricksServingEndpoint(endpoint_name="databricks-meta-llama-3-3-70b-instruct"),
            DatabricksVectorSearchIndex(index_name="brian_ml_dev.aws_testing.arxiv_data_vs_index"),
            DatabricksGenieSpace(genie_space_id="01efda0bed2517e1b17632b5c8cd0694")
    ]
    )

# COMMAND ----------

# DBTITLE 1,Test saved model
chain = mlflow.langchain.load_model(logged_chain_info.model_uri)
chain.invoke({'messages': [{
  'role': 'user',
  'content': 'How are things going?'
}]})

# COMMAND ----------

# DBTITLE 1,Deploy
from databricks import agents

uc_registered_model_info = mlflow.register_model(model_uri=logged_chain_info.model_uri, 
                                                 name=f'{catalog}.{schema}.adv_agent_example')

# Deploy to enable the Review APP and create an API endpoint
deployment_info = agents.deploy(model_name=f'{catalog}.{schema}.adv_agent_example', 
                                model_version=uc_registered_model_info.version)

browser_url = mlflow.utils.databricks_utils.get_browser_hostname()
print(f"\n\nView deployment status: https://{browser_url}/ml/endpoints/{deployment_info.endpoint_name}")
