# Databricks notebook source
# MAGIC %md
# MAGIC # Automating Secrets
# MAGIC As a part of building RAG products we need to work with and create secrets \
# MAGIC This will be for connecting orchestrators to models and setting up connectivity \
# MAGIC In Prod we probably want a service principal secret rather than one we use here
# COMMAND ----------

# MAGIC %pip install --upgrade databricks-sdk
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

from databricks.sdk import WorkspaceClient

w = WorkspaceClient()

# COMMAND ----------

# Check existing running endpoints first
w.serving_endpoints.list()

# COMMAND ----------

from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput

w.serving_endpoints.create_and_wait(
    name = 'embedding-bge-large-en',
    config = EndpointCoreConfigInput(
        served_entities=[
            ServedEntityInput(
                entity_name = 'databricks_bge_v1_5_models.models.bge_large_en_v1_5',
                entity_version = '1',
                workload_size = 'Small',
                scale_to_zero_enabled = True
            )
        ]
    )
)

# COMMAND ----------

w.serving_endpoints.create_and_wait(
    name = 'mistral_7b_model',
    config = EndpointCoreConfigInput(
        served_entities=[
            ServedEntityInput(
                entity_name = 'databricks_mistral_models.models.mistral_7b_instruct_v0_2',
                entity_version = '1',
                min_provisioned_throughput = '970',
                max_provisioned_throughput = '1940'
            )
        ]
    )
)