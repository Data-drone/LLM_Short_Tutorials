# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Chunking and Parsing w langchain
# MAGIC
# MAGIC For this we will look at leveraging Langchain \
# MAGIC and llama-index

# COMMAND ----------

# MAGIC %sh
# MAGIC # we needed to do this for poppler to work for some reason
# MAGIC apt-get install -y poppler-utils

# COMMAND ----------

# MAGIC %pip install langchain==0.1.16 poppler-utils unstructured[pdf] databricks-vectorsearch
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Setup and Config
import os
import nest_asyncio

# a lot of components use async to scale
nest_asyncio.apply()

catalog = 'brian_gen_ai'
schema = 'lab_05'
volumes = 'source_files'

vector_search_endpoint = 'one-env-shared-endpoint-5'
vector_index = f'{catalog}.{schema}.adv_unstructured_index'

embedding_model = 'databricks-bge-large-en'
llm_model = 'databricks-dbrx-instruct'

# We have all PDF but poppler and unstructured sorts this
vol_path = f'/Volumes/{catalog}/{schema}/{volumes}'

files = os.listdir(vol_path)
files

# COMMAND ----------

# setup a LLM
from langchain_community.chat_models import ChatDatabricks

llm_model = model = ChatDatabricks(
  target_uri='databricks',
  endpoint = llm_model,
  temperature = 0.1
)

# COMMAND ----------

from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import UnstructuredFileLoader


#loader_kwargs={'mode': 'elements'},
# do not use elements unless you want each line to be it's own chunk
# multi-threading uses python threadedpool executor not spark so won't distribute across nodes
loader = DirectoryLoader(path=vol_path,
                         glob='*.pdf',
                         loader_cls=UnstructuredFileLoader,
                         show_progress=True,
                         use_multithreading=True,
                         max_concurrency=16)

loaded_files = loader.load()

# The loader will output a list of lists
# the parent list is one entry per doc
# the child list is a set of chunks

# COMMAND ----------

from langchain_community.vectorstores import DatabricksVectorSearch
from langchain_community.embeddings import DatabricksEmbeddings
from databricks.vector_search.client import VectorSearchClient

embeddings = DatabricksEmbeddings(endpoint=embedding_model)
emb_dim = len(embeddings.embed_query("hello"))

vsc = VectorSearchClient()

# A direct access index requires manual syncing and updating
try:
    index = vsc.create_direct_access_index(
        endpoint_name=vector_search_endpoint,
        index_name=vector_index,
        primary_key="id",
        embedding_dimension=emb_dim,
        embedding_vector_column="text_vector",
        schema={
            "id": "string",
            "text": "string",
            "text_vector": "array<float>",
            "source": "string",
        },
    )
except Exception:
    index = vsc.get_index(endpoint_name=vector_search_endpoint,
                          index_name=vector_index)

dvs = DatabricksVectorSearch(
    index, text_column="text", embedding=embeddings, columns=["source"]
)

# COMMAND ----------

# adding in the documents
dvs.add_documents(documents=loaded_files)

# COMMAND ----------

# MAGIC %md
# MAGIC Once we have the index we can use it in Langchain as per normal but turning it into a retriever with `as_retriever` note you can use filters too.
# MAGIC See: `https://github.com/langchain-ai/langchain/blob/cb6e5e56c29477c6da5824c17f1b70af11352685/libs/core/langchain_core/vectorstores.py#L595 for details.
