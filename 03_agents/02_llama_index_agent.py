# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Building and working with Llama Index
# MAGIC
# MAGIC Working with Llama Index is a bit more complicated \
# MAGIC Llama Index relies on the creation of certain datastructures and certain column naming \
# MAGIC in index metadata. 
# MAGIC
# MAGIC If the index creation is all done in Llama Index then there will be no issues \
# MAGIC Otherwise if the data is loaded into Databricks Vector Search first, \
# MAGIC We need to make sure that 

# COMMAND ----------

# DBTITLE 1,Parameterise Pip installs
mlflow_version = 'mlflow==2.16.2'
llama_index_version = 'llama-index==0.11.13'

# COMMAND ----------

# DBTITLE 1,Run Pip Install
# MAGIC %pip install -U databricks-sdk {mlflow_version} {llama_index_version} llama-index-llms-databricks llama-index-vector-stores-databricks llama-index-embeddings-databricks
# MAGIC %restart_python

# COMMAND ----------

# DBTITLE 1,Config Management
import mlflow
import os

# We will use ModelConfig for config management
# The structure of these yamls is totally up to you

# Load the chain's configuration
#model_config = mlflow.models.ModelConfig(development_config="rag_chain_config.yaml")

#model_resources = model_config.get("resources")
#uc_config = model_config.get("uc_config")

#### endpoints and models
#llm_model_name = model_resources.get("llm_model_name")
#embedding_model_name = model_resources.get("embedding_model")
#vs_endpoint_name = model_resources.get("vs_endpoint")

#### uc_config
#catalog = uc_config.get("catalog")
#schema = uc_config.get("schema")
#index_name = uc_config.get("index_name")

########

#### endpoints and models
llm_model_name = "databricks-meta-llama-3-1-70b-instruct"
embedding_model_name = "databricks-bge-large-en"
vs_endpoint_name = "one-env-shared-endpoint-5"

#### uc_config
catalog = "brian_gen_ai"
schema = "lab_05"
index_name = "arxiv_parse_bge_index"

########


# we cannot have these in model file need to get passed in
#browser_host = spark.conf.get("spark.databricks.workspaceUrl")
#os.environ['DATABRICKS_TOKEN'] = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
#os.environ['DATABRICKS_SERVING_ENDPOINT'] = f"https://{browser_host}/serving-endpoints"

# COMMAND ----------

# DBTITLE 1,Loading Chat Model
from llama_index.llms.databricks import Databricks

llm = Databricks(model=llm_model_name)

response = llm.complete("Explain the importance of open source LLMs")

response

# COMMAND ----------

# DBTITLE 1,Setting up embeddings
from llama_index.embeddings.databricks import DatabricksEmbedding

embed_model = DatabricksEmbedding(
    model=embedding_model_name
)

embed_model.get_text_embedding(
    "The DatabricksEmbedding integration works great."
)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Understanding Llama Index
# MAGIC Much of Llama Index's capabilities come from having extra primatives that sit on top of vector indices \
# MAGIC This extra functionality depends on have specific index structures
# MAGIC
# MAGIC The first one that will affect us when we utilise Databricks Vector Search is column naming conventions \
# MAGIC By default the primary key has to be `doc_id``, you can search for `DEFAULT_DOC_ID_KEY` in the source code to see, 

# COMMAND ----------

# DBTITLE 1,Setting up embeddings
from databricks.vector_search.client import VectorSearchClient
from llama_index.core import VectorStoreIndex, Settings
from llama_index.vector_stores.databricks import DatabricksVectorSearch

Settings.embed_model = embed_model
Settings.llm = llm

vsc = VectorSearchClient()
dbx_index = vsc.get_index(endpoint_name=vs_endpoint_name,
                      index_name=f"{catalog}.{schema}.{index_name}")

vector_store = DatabricksVectorSearch(index=dbx_index,
                                      text_column='page_content',
                                      columns=['row_id', 'source_doc', 'doc_page'])

# We can "hack" not Llama Index compliant indices to be queryable 
# Once the Index is created, we can override the columns to remove the doc_id column spec
# this might break some of the advanced functionality in Llama Index
# Temp fix for - https://github.com/run-llama/llama_index/blob/9edc6f73b80f78dc4b79c0eda502a97817fabf75/llama-index-integrations/vector_stores/llama-index-vector-stores-databricks/llama_index/vector_stores/databricks/base.py#L158
vector_store.columns = ['row_id', 'source_doc', 'doc_page', 'page_content']
index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

query_engine = index.as_query_engine()
response = query_engine.query("What can you tell me about task customisation for LLM?")

# COMMAND ----------

mlflow.models.set_model(query_engine)

# COMMAND ----------
