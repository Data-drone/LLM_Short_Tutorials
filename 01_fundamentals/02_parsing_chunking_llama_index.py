# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Chunking and Parsing has a lot of complexity
# MAGIC
# MAGIC For this we will look at leveraging Langchain \
# MAGIC and llama-index

# COMMAND ----------

# MAGIC %sh
# MAGIC # we needed to do this for poppler to work for some reason
# MAGIC apt-get install -y poppler-utils

# COMMAND ----------

# urllib3 and typing_extensions are to fix common compatability issues
# MAGIC %pip install urllib3==1.26.18 typing_extensions llama_index==0.10.30 langchain==0.1.16 llama-index-llms-langchain llama-index-embeddings-langchain poppler-utils unstructured[pdf,txt,docx,doc] llama-index-vector-stores-databricks databricks-vectorsearch
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
vector_index =  f'{catalog}.{schema}.adv_llama_index'

llm_model_name = 'databricks-dbrx-instruct'
embedding_model = 'databricks-bge-large-en'

# We have all PDF but poppler and unstructured sorts this
vol_path = f'/Volumes/{catalog}/{schema}/{volumes}'

files = os.listdir(vol_path)
files

# COMMAND ----------

# setup a LLM
## The databricks integrations for llms and embeddings in Llama_index are not as well developed
## To get around this we can use the langchain ones then wrap that with the llama_index adapter 
from langchain_community.chat_models import ChatDatabricks
from langchain_community.embeddings import DatabricksEmbeddings
from llama_index.core import Settings
from llama_index.llms.langchain import LangChainLLM
from llama_index.embeddings.langchain import LangchainEmbedding
import nltk

# this was failing to autodownload
nltk.download('averaged_perceptron_tagger')

llm_model = model = ChatDatabricks(
  target_uri='databricks',
  endpoint = llm_model_name,
  temperature = 0.1
)

embeddings = DatabricksEmbeddings(endpoint=embedding_model)
emb_dim = len(embeddings.embed_query("hello"))

llama_index_chain = LangChainLLM(llm=llm_model)
llama_index_embeddings = LangchainEmbedding(langchain_embeddings=embeddings)

# we set it in the settings so that it defaults to these
# Otherwise it will default to OpenAI
Settings.llm = llama_index_chain 
Settings.embed_model = llama_index_embeddings 
# COMMAND ----------

# MAGIC %md ## Parsing in Llama_index
# MAGIC
# MAGIC Llama_index breaks up the ingestion into two steps
# MAGIC - Parse: Load a file and ingest it into a Document class object
# MAGIC - Chunk: Take a document object and break it up into Node class object

# COMMAND ----------

# For testing we can filter down a bit
filtered_files = files[0:4]

# these are just filepaths as well so I need
filtered_file_paths = [os.path.join(vol_path, x) for x in filtered_files]

from llama_index.core import SimpleDirectoryReader
from llama_index.readers.file import UnstructuredReader
import os

document_loader = SimpleDirectoryReader(input_files=filtered_file_paths,
                                         file_extractor = {
        ".pdf": UnstructuredReader()
      })

document_list = document_loader.load_data(num_workers=4)

# COMMAND ----------

# MAGIC %md
# MAGIC The document list is a list of separate documents
# MAGIC we can access the metadata by loading a document and accessing it's metadata_section

# COMMAND ----------

# lets chunk it up
# See: https://docs.llamaindex.ai/en/stable/module_guides/loading/node_parsers/modules/#semanticsplitternodeparser for more
from llama_index.core.node_parser import TokenTextSplitter

splitter = TokenTextSplitter(
    chunk_size=1024,
    chunk_overlap=20,
    separator=" ",
)

# we will get a list of nodes from this
nodes = splitter.get_nodes_from_documents(document_list)

# COMMAND ----------

# MAGIC %md # Generating Indexes and Persisting
# MAGIC
# MAGIC To load into a VectorStore index we need to:
# MAGIC - Create the direct access index
# MAGIC - set it as a storage_context 
# MAGIC - Create VectorStoreIndex
# MAGIC persist

# COMMAND ----------

# DBTITLE 1,Create direct access index

from databricks.vector_search.client import VectorSearchClient

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
            "file_name": "string",
            "doc_id": "string"
        },
    )
except Exception:
    index = vsc.get_index(endpoint_name=vector_search_endpoint,
                          index_name=vector_index)
    
# COMMAND ----------

# In order to use the vector search index we need to setup the persistent store in llama_index

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    ServiceContext,
    StorageContext,
)
from llama_index.vector_stores.databricks import DatabricksVectorSearch

databricks_vector_store = DatabricksVectorSearch(
    index=index,
    text_column="text",
    columns=["doc_id"],  # YOU MUST ALSO RECORD YOUR METADATA FIELD NAMES HERE
)  # text_column is required for self-managed embeddings

storage_context = StorageContext.from_defaults(
    vector_store=databricks_vector_store
)

basic_index = VectorStoreIndex(
    nodes, storage_context=storage_context
)

# COMMAND ----------

# this is waiting for a pull to be accepted: https://github.com/run-llama/llama_index/pull/12999
query_engine = basic_index.as_query_engine()
response = query_engine.query("Why did the author choose to work on AI?")

print(response.response)

