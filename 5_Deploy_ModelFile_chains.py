# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Logging Model as Code Chains
# MAGIC
# MAGIC Here we will log models as code

# COMMAND ----------

# DBTITLE 1,Parameterise Pip installs
mlflow_version = 'mlflow==2.16.2'
langchain_base_version = 'langchain'
langchain_community_version = 'langchain_community==0.2.13'

# COMMAND ----------

# DBTITLE 1,Run Pip Install
# MAGIC %pip install databricks-agents databricks-sdk {mlflow_version} {langchain_base_version} {langchain_community_version} langchain_core langgraph langchain-databricks textact
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Setup Parameters for endpoints etc

# we wiped the params to re-adding
mlflow_version = 'mlflow==2.16.2'
langchain_base_version = 'langchain'
langchain_community_version = 'langchain_community==0.2.13'

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

# DBTITLE 1,Log Model File

# Note that we use the same input sample as tested in the notebook
# To ensure smooth API deployment make the input a json
with mlflow.start_run(run_name='modelfile_base_chain'):

    mlflow.set_tag("type", "chain")

    logged_chain_info = mlflow.langchain.log_model(
        lc_model=os.path.join(
            os.getcwd(), '4_Building_ModelFile_Base_Chain'
        ),  
        artifact_path="chain",  # Required by MLflow
        input_example={"prompt": "How are you today?"},
        registered_model_name=f'{catalog}.{schema}.base_model_file'
    )

# COMMAND ----------

# DBTITLE 1,Test saved model
chain = mlflow.langchain.load_model(logged_chain_info.model_uri)
chain.invoke({'prompt': 'How are things going?'})

# COMMAND ----------

# DBTITLE 1,Deploy Model file model
UC_MODEL_NAME = f'{catalog}.{schema}.base_model_file'
model_version = 2

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
    name=f'{parsed_name}_basic_mf_chat',
    config=endpoint_config)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Logging and Eval of Advanced ModelFile Chains
# MAGIC
# MAGIC We will combine model file with evals as well

# COMMAND ----------

# DBTITLE 1,Setup Evals
import yaml
import pandas as pd

# We use a couple of test prompts to see how well they perform
evaluations = pd.DataFrame(
    {'inputs': [
        'What is a RAG?',
        'In what ways can RAGs go wrong?',
        'Why did the chicken cross the road?'
    ]}
)

# Eval Function for mlflow evals
def eval_pipe(inputs):

        def invoke_chain(prompt):
            return chain.invoke(input={'input': prompt})

        answers = inputs['inputs'].apply(invoke_chain)
        #answer = chain.invoke(context="", data=inputs)
        return answers.tolist()

# COMMAND ----------

# DBTITLE 1,Rag Chain 70b
with mlflow.start_run(run_name='Rag_chain'):

    mlflow.set_tag("type", "chain")

    # Load the model config into a dict
    with open('rag_chain_config.yaml', 'r') as file:
        model_config = yaml.safe_load(file)

    logged_chain_info = mlflow.langchain.log_model(
        lc_model=os.path.join(
            os.getcwd(), '4a_Build_ModelFile_Adv_Chain'
        ),  # Chain code file e.g., /path/to/the/chain.py
        model_config=model_config,  # Chain configuration set in 00_config
        artifact_path="chain",  # Required by MLflow
        input_example={"input": "How are you today?"},  # Save the chain's input schema.  MLflow will execute the chain before logging & capture it's output schema.
        registered_model_name=f'{catalog}.{schema}.retrieval_chain_model_file'
    )

    chain = mlflow.langchain.load_model(logged_chain_info.model_uri)

    results = mlflow.evaluate(eval_pipe,
                          data=evaluations,
                          model_type='text')

# COMMAND ----------

# DBTITLE 1,Rag Chain 405b
with mlflow.start_run(run_name='Rag_chain_405b'):

    mlflow.set_tag("type", "chain")

    # Load the model config into a dict
    with open('rag_chain_405b.yaml', 'r') as file:
        model_config = yaml.safe_load(file)

    logged_chain_info = mlflow.langchain.log_model(
        lc_model=os.path.join(
            os.getcwd(), '4a_Build_ModelFile_Adv_Chain'
        ),  # Chain code file e.g., /path/to/the/chain.py
        model_config=model_config,  # Chain configuration set in 00_config
        artifact_path="chain",  # Required by MLflow
        input_example={"input": "How are you today?"},  # Save the chain's input schema.  MLflow will execute the chain before logging & capture it's output schema.
        registered_model_name=f'{catalog}.{schema}.retrieval_chain_model_file'
    )

    chain = mlflow.langchain.load_model(logged_chain_info.model_uri)

    results = mlflow.evaluate(eval_pipe,
                          data=evaluations,
                          model_type='text')
