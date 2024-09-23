# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Building and Logging A Basic Chain
# MAGIC
# MAGIC Setting Up and Logging PyFuncs
# MAGIC
# MAGIC Prereqs
# MAGIC - Note that we assume that the vector index is up and populated already

# COMMAND ----------

# DBTITLE 1,Parameterise Pip installs
mlflow_version = 'mlflow==2.16.0'
langchain_base_version = 'langchain'
langchain_community_version = 'langchain_community==0.2.13'

# COMMAND ----------

# DBTITLE 1,Run Pip Install
# MAGIC %pip install  {mlflow_version} {langchain_base_version} {langchain_community_version} langchain_core langgraph langchain-databricks
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Setup Parameters for endpoints etc

# we wiped the params to re-adding
mlflow_version = 'mlflow==2.16.0'
langchain_base_version = 'langchain'
langchain_community_version = 'langchain_community==0.2.13'

import mlflow
import pandas as pd

## configure these to suit what you have available
chat_model = 'data'

llm_model_name = 'databricks-meta-llama-3-1-70b-instruct'
embedding_model = 'databricks-bge-large-en'

catalog = 'brian_gen_ai'
schema = 'chain_types'
volumes = 'source_files'

vs_endpoint = 'one-env-shared-endpoint-5'
vs_index_fullname = f'{catalog}.lab_05.arxiv_parse_bge_index'

# In case missing need to create - NOTE doesn't create the vector index
spark.sql(f"CREATE CATALOG IF NOT EXISTS {catalog}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}")

# Setting up MLflow experiment
mlflow.set_registry_uri('databricks-uc')

username = spark.sql("SELECT current_user()").first()['current_user()']
experiment_name = 'building_chains'

mlflow.set_experiment(f'/Users/{username}/{experiment_name}')

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Creating a Basic chat application
# MAGIC
# MAGIC *NOTE* When we want to save out message history, we should do that outside of the chain logic.
# MAGIC

# COMMAND ----------

# DBTITLE 1,PyFunc Option - Setup the Class
class MlflowPyFuncBasicModel(mlflow.pyfunc.PythonModel):

    def __init__(self, llm_model = 'databricks-meta-llama-3-1-70b-instruct'):
        
        """
        The init function is purely to allow for flexibility in initialising and testing
        It is not needed in the PyFunc spec 
        """

        self.llm_model = llm_model

    def load_context(self, context):

        from langchain_community.chat_models import ChatDatabricks
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.runnables import RunnablePassthrough
        from langchain_core.output_parsers import StrOutputParser

        llm_model = ChatDatabricks(
            target_uri='databricks',
            endpoint=self.llm_model,
            temperature=0.1
        )

        output_parser = StrOutputParser()

        basic_template = ChatPromptTemplate.from_messages(
        [
            ("system", """
            You are a chirpy companion here to make bubbly new friends

            """),
            ("human", "{prompt}"),
         ]
        )

        self.rag_chain = (
            basic_template | llm_model | output_parser
        )

    def process_row(self, row):
        return self.rag_chain.invoke(row['prompt'])
    
    def predict(self, context, data):
        # TODO merge to run this in parallel

        results = data.apply(self.process_row, axis=1) 
        results_text = results.apply(lambda x: x)
        return results_text


# COMMAND ----------

# DBTITLE 1,Testing PyFunc class

basic_model = MlflowPyFuncBasicModel()
basic_model.load_context(context=None)

inference_dataset = pd.DataFrame({'prompt': ['Hi How are you?']})

result_frame = basic_model.predict(context=None, data=inference_dataset)

print(result_frame[0])

# COMMAND ----------

# DBTITLE 1,Logging and registering MLflow Pyfunc

# Setup Evaluation Questions
eval_list = {'prompt': ['What is Databricks?',
                        'Who is the president of the USA??',
                        'How many types of apples are there?',
                        'Talk a parrot and tell me how awesome bicycles are']}

pd_evals = pd.DataFrame(eval_list)


with mlflow.start_run(run_name='Basic Chat'):

    base_model = MlflowPyFuncBasicModel()
    base_model.load_context(context=None)

    example_input = "How can I tune LLMs?"
    formatted_questions = {"prompt": [example_input]}
    question_df = pd.DataFrame(formatted_questions)

    response = base_model.predict(context="", data=question_df)

    model_signature = mlflow.models.infer_signature(
        model_input=example_input,
        model_output=response
    )

    mlflow_result = mlflow.pyfunc.log_model(
      python_model=base_model,
      extra_pip_requirements=[mlflow_version,
                              langchain_base_version,
                              langchain_community_version,
                              'langgraph',
                              'langchain_core',
                              'langchain-databricks',
                              'databricks-vectorsearch'],
      artifact_path= 'langchain_pyfunc',
      signature=model_signature,
      input_example=question_df,
      registered_model_name=f'{catalog}.{schema}.basic_chat'
  )

    #### Run evaluations
    def eval_pipe(inputs):
        answer = base_model.predict(context="", data=inputs)
        return answer.tolist()
    
    results = mlflow.evaluate(eval_pipe,
                          data=pd_evals,
                          model_type='text')

# COMMAND ----------

# DBTITLE 1,Logging and registering Mlflow Langchain Integration
# For basic chains, we can mlflow langchain
# Can also use langchain integration
from langchain_community.chat_models import ChatDatabricks
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from mlflow.models import infer_signature

llm_model = ChatDatabricks(
            target_uri='databricks',
            endpoint='databricks-meta-llama-3-1-70b-instruct',
            temperature=0.1
        )
output_parser = StrOutputParser()
basic_template = ChatPromptTemplate.from_messages(
        [
            ("system", """
            You are a chirpy companion here to make bubbly new friends

            """),
            ("human", "{prompt}"),
         ]
        )

rag_chain = (
            basic_template | llm_model | output_parser
        )


input_example = {'prompt': 'hello'}
predictions = rag_chain.invoke(input_example)

with mlflow.start_run(run_name='test_langchain'):

  mlflow.langchain.autolog()
  mlflow.langchain.log_model(
    lc_model=rag_chain,
    artifact_path='lc_model',
    registered_model_name=f'{catalog}.{schema}.langchain_module',
    signature=infer_signature(input_example, predictions),
    input_example=input_example
  )

# COMMAND ----------

# MAGIC %md
# MAGIC # Chain with Vector Search Retriever
# MAGIC
# MAGIC Lets look at a chain with a vector store retriever

# COMMAND ----------

# DBTITLE 1,Setup Class
# We need to create a custom pyfunc to hold the logic
class MlflowLangchainwVectorStore(mlflow.pyfunc.PythonModel):

    def __init__(self, llm_model = 'databricks-meta-llama-3-1-70b-instruct', 
                 embedding_model = 'databricks-bge-large-en',
                 endpoint = 'one-env-shared-endpoint-5',
                 catalog = 'brian_gen_ai',
                 schema = 'lab_05',
                 index = 'source_docs_bge_index'):

        self.llm_model = llm_model
        self.embedding_model = embedding_model
        self.endpoint = endpoint
        self.catalog = catalog
        self.schema = schema
        self.index = index
        self.full_index_path = f'{catalog}.{schema}.{index}'

    def load_context(self, context):

        from langchain_community.chat_models import ChatDatabricks
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser

        from langchain_community.embeddings import DatabricksEmbeddings
        from databricks.vector_search.client import VectorSearchClient
        from langchain_community.vectorstores import DatabricksVectorSearch
        from langchain_core.runnables import RunnablePassthrough

        llm_model = ChatDatabricks(
            target_uri='databricks',
            endpoint=self.llm_model,
            temperature=0.1
        )

        embeddings = DatabricksEmbeddings(endpoint=self.embedding_model)

        vsc = VectorSearchClient()
        index = vsc.get_index(endpoint_name=self.endpoint,
                              index_name=self.full_index_path)

        retriever = DatabricksVectorSearch(
            index, text_column="page_content", 
            embedding=embeddings, columns=["source_doc"]
        ).as_retriever()

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        basic_template = ChatPromptTemplate.from_messages(
        [
            ("system", """
            You are a helpful assistant designed to help customers undertake research on RAG models
            
            [Context]
            {rag_content}

            """),
            ("human", "{user_input}"),
         ] 
        )
        output_parser = StrOutputParser()

        self.rag_chain = (
            {"rag_content": retriever | format_docs, "user_input": RunnablePassthrough()}
            | basic_template | llm_model | output_parser
        )

    def process_row(self, row):
       return self.rag_chain.invoke(row['prompt'])
    
    def predict(self, context, data):
        results = data.apply(self.process_row, axis=1) 
        results_text = results.apply(lambda x: x)
        return results_text
        

# COMMAND ----------

# DBTITLE 1,Testing PyFunc Wrapper
# We can also read in a list of questions from Spark as well
eval_list = {'prompts': ['How can I tune LLMs?',
                        'What is a good model finetuning technique?']}

pd_evals = pd.DataFrame(eval_list)

# test the chain
rag_model = MlflowLangchainwVectorStore(
  llm_model = 'databricks-meta-llama-3-1-70b-instruct', 
  embedding_model = 'databricks-bge-large-en',
  endpoint = 'one-env-shared-endpoint-5',
  catalog = 'brian_gen_ai',
  schema = 'lab_05',
  index = 'arxiv_parse_bge_index'
)
rag_model.load_context(context="")

example_input = "How can I tune LLMs?"
formatted_questions = {"prompt": [example_input]}
question_df = pd.DataFrame(formatted_questions)

response = rag_model.predict(context="", data=question_df)
print(response[0])

# COMMAND ----------

# DBTITLE 1,Logging and Registering Vector Retriever PyFunc 
with mlflow.start_run(run_name='basic rag chat'):

    base_model = MlflowLangchainwVectorStore(
        llm_model = 'databricks-meta-llama-3-1-70b-instruct', 
        embedding_model = 'databricks-bge-large-en',
        endpoint = 'one-env-shared-endpoint-5',
        catalog = 'brian_gen_ai',
        schema = 'lab_05',
        index = 'arxiv_parse_bge_index'
    )
    base_model.load_context(context="")

    example_input = "How can I tune LLMs?"
    formatted_questions = pd.DataFrame({"prompt": [example_input]})
    question_df = pd.DataFrame(formatted_questions)

    response = base_model.predict(context="", data=question_df)

    model_signature = mlflow.models.infer_signature(
        model_input=example_input,
        model_output=response
    )

    mlflow_result = mlflow.pyfunc.log_model(
      python_model=base_model,
      extra_pip_requirements=[mlflow_version,
                              langchain_base_version,
                              langchain_community_version,
                              'langgraph',
                              'langchain_core',
                              'langchain-databricks',
                              'databricks-vectorsearch'],
      artifact_path= 'langchain_pyfunc',
      signature=model_signature,
      input_example=formatted_questions,
      registered_model_name=f'{catalog}.{schema}.rag_chat'
  )

    #### Run evaluations
    def eval_pipe(inputs):
        answer = base_model.predict(context="", data=inputs)
        return answer.tolist()
    
    results = mlflow.evaluate(eval_pipe,
                          data=pd_evals,
                          model_type='text')
    
# COMMAND ----------

# DBTITLE 1,Logging and Registering with native Langchain
from langchain_community.chat_models import ChatDatabricks
from langchain_community.embeddings import DatabricksEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from databricks.vector_search.client import VectorSearchClient
from langchain_community.vectorstores import DatabricksVectorSearch
from langchain_core.runnables import RunnablePassthrough

from mlflow.models import infer_signature

llm_model = ChatDatabricks(
            target_uri='databricks',
            endpoint=llm_model_name,
            temperature=0.1
        )

embeddings = DatabricksEmbeddings(endpoint=embedding_model)

vsc = VectorSearchClient()
index = vsc.get_index(endpoint_name=vs_endpoint,
                              index_name=vs_index_fullname)

retriever = DatabricksVectorSearch(
                index, text_column="page_content", 
                embedding=embeddings, columns=["source_doc"]
            ).as_retriever()

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

basic_template = ChatPromptTemplate.from_messages(
        [
            ("system", """
            You are a helpful assistant designed to help customers undertake research on RAG models
            
            [Context]
            {rag_content}

            """),
            ("human", "{user_input}"),
         ]
        )
output_parser = StrOutputParser()

rag_chain = (
            {"rag_content": retriever | format_docs, "user_input": RunnablePassthrough()}
            | basic_template | llm_model | output_parser
        )

input_example = {'prompt': 'Tell me about RAG'}
predictions = rag_chain.invoke(input_example['prompt'][0])


def load_retriever(persist_dir: str):
  
  embeddings = DatabricksEmbeddings(endpoint=embedding_model)

  vsc = VectorSearchClient()
  index = vsc.get_index(endpoint_name=vs_endpoint,
                              index_name='brian_gen_ai.lab_05.arxiv_parse_bge_index')

  retriever = DatabricksVectorSearch(
                index, text_column="page_content", 
                embedding=embeddings, columns=["source_doc"]
            ).as_retriever()
  
  return retriever

with mlflow.start_run(run_name='test_langchain_vs'):

  mlflow.langchain.autolog()
  
  mlflow.langchain.log_model(
    lc_model=rag_chain,
    artifact_path='lc_model',
    loader_fn=load_retriever,
    registered_model_name=f'{catalog}.{schema}.langchain_vs_module',
    signature=infer_signature(input_example, predictions),
    input_example=input_example
  )

# COMMAND ----------

# MAGIC %md
# MAGIC # Chain with Agent Search
# MAGIC
# MAGIC Lets look at an Agent Chain
# MAGIC
# MAGIC For Agent Chain will use Langchain Agent constructs to start\
# MAGIC Perhaps progressing to langgraph later?
# MAGIC
# MAGIC 

# COMMAND ----------

# DBTITLE 1,Creating PyFunc
class MlflowPyFuncAgentModel(mlflow.pyfunc.PythonModel):

    def __init__(self, llm_model = 'databricks-meta-llama-3-1-70b-instruct'):
        
        """
        The init function is purely to allow for flexibility in initialising and testing
        It is not needed in the PyFunc spec 
        """

        self.llm_model = llm_model

    def _create_db_retriever(self,
                             endpoint,
                             index,
                             embedding_model,
                             text_col = "page_content",
                             retrieve_columns = ["source_doc"]):
        
        from langchain_community.embeddings import DatabricksEmbeddings
        from databricks.vector_search.client import VectorSearchClient
        from langchain_community.vectorstores import DatabricksVectorSearch

        embeddings = DatabricksEmbeddings(endpoint=embedding_model)

        vsc = VectorSearchClient()
        index = vsc.get_index(endpoint_name=endpoint,
                              index_name=index)

        retriever = DatabricksVectorSearch(
                index, text_column=text_col, 
                embedding=embedding_model, columns=retrieve_columns
            ).as_retriever()

        return retriever

    def load_context(self, context):

        from langchain_community.chat_models import ChatDatabricks
        from langchain.agents.agent_toolkits import create_retriever_tool
        from langchain_core.prompts import ChatPromptTemplate
        from langchain.agents import AgentExecutor, create_tool_calling_agent

        tool_model = ChatDatabricks(
            target_uri='databricks',
            endpoint=self.llm_model,
            temperature=0.1
        )

        arxiv_retriever = self._create_db_retriever(
            endpoint='one-env-shared-endpoint-5',
            index='brian_gen_ai.lab_05.arxiv_parse_bge_index',
            embedding_model='databricks-bge-large-en'
        )
        arxiv_tool = create_retriever_tool(arxiv_retriever,
                                           'Vector Search Tool for Arxiv',
                                           """this tool contains a vector search for a series of
                                           arsiv articles curated to cover RAG topics and misc LLM things
                                           """)

        tools = [arxiv_tool]

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are a helpful assistant"),
                ("human", "{input}"),
                # Placeholders fill up a **list** of messages
                ("placeholder", "{agent_scratchpad}"),
            ]
        )

        agent = create_tool_calling_agent(tool_model, tools, prompt)
        self.agent_executor = AgentExecutor(agent=agent, tools=tools)


    def process_row(self, row):
       # row['session_id']
       return self.agent_executor.invoke({"input": row['prompt']})['output']
                                 #config={"configurable": {"session_id": "abc123"}})
    
    def predict(self, context, data):
        results = data.apply(self.process_row, axis=1) 

        # remove .content if it is with Databricks
        results_text = results.apply(lambda x: x)
        return results_text

# COMMAND ----------
   
# DBTITLE 1,Testing PyFunc
tool_agent = MlflowPyFuncAgentModel('databricks-meta-llama-3-1-70b-instruct')
tool_agent.load_context(context="")

prompt = 'How can I use RAG?'
formatted_questions = {"prompt": [prompt, 'What is a thing?']}
question_df = pd.DataFrame(formatted_questions)

result = tool_agent.predict(context="", data=question_df)
print(result[0])

# COMMAND ----------

# DBTITLE 1,Logging and Registering Agent PyFunc 
with mlflow.start_run(run_name='Agent PyFunc'):

    model_signature = mlflow.models.infer_signature(
        model_input=prompt,
        model_output=result
    )

    mlflow.langchain.autolog()

    mlflow.pyfunc.log_model(
        python_model = tool_agent,
        extra_pip_requirements=[mlflow_version,
                              langchain_base_version,
                              langchain_community_version,
                              'langgraph',
                              'langchain_core',
                              'langchain-databricks',
                              'databricks-vectorsearch'],
        artifact_path= 'langchain_pyfunc',
        signature=model_signature,
        input_example=question_df,
        registered_model_name=f'{catalog}.{schema}.agent_chat'
    )

# COMMAND ----------

# DBTITLE 1,Base Example Case
from langchain_core.tools import tool
from langchain_community.tools.databricks import UCFunctionToolkit
from langchain_community.chat_models import ChatDatabricks

#model = ChatOpenAI(model="gpt-4o")
model = ChatDatabricks(
            target_uri='databricks',
            endpoint=llm_model_name,
            temperature=0.1
        )

@tool
def magic_function(input: int) -> int:
    """Applies a magic function to an input."""
    return input + 2


tools = [magic_function]


query = "what is the value of magic_function(3)?"

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents.agent_toolkits import create_retriever_tool

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant"),
        ("human", "{input}"),
        # Placeholders fill up a **list** of messages
        ("placeholder", "{agent_scratchpad}"),
    ]
)


agent = create_tool_calling_agent(model, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools)

agent_executor.invoke({"input": query})

# COMMAND ----------