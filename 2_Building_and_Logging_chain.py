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

# MAGIC %pip install mlflow-skinny==2.16.0 databricks-vectorsearch langchain==0.2.11 langchain_core langchain_community==0.2.10 langchain-databricks
# MAGIC dbutils.library.restartPython()

# COMMAND ----------
import mlflow

## configure these to suit what you have available
chat_model = 'data'

llm_model_name = 'databricks-dbrx-instruct'
embedding_model = 'databricks-bge-large-en'

catalog = 'brian_gen_ai'
schema = 'chain_types'
volumes = 'source_files'

vs_endpoint = 'one-env-shared-endpoint-5'
vs_index_fullname = f'{catalog}.lab_05.arxiv_parse_bge_index'

# In case missing need to create - NOTE doesn't create the vector index
spark.sql(f"CREATE CATALOG IF NOT EXISTS {catalog}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Creating a Basic chat application
# MAGIC
# MAGIC *NOTE* When we want to save out message history, we should do that outside of the chain logic.
# MAGIC

# COMMAND ----------

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

        self.llm_model = ChatDatabricks(
            target_uri='databricks',
            endpoint=self.llm_model,
            temperature=0.1
        )

    def process_row(self, row):
       # row['session_id']
       return self.llm_model.invoke(row['prompt'])
                                 #config={"configurable": {"session_id": "abc123"}})
    
    def predict(self, context, data):
        results = data.apply(self.process_row, axis=1) 

        # remove .content if it is with Databricks
        results_text = results.apply(lambda x: x)
        return results_text


# COMMAND ----------

import pandas as pd

basic_model = MlflowPyFuncBasicModel()
basic_model.load_context()

inference_dataset = pd.DataFrame({'prompt': ['Hi How are you?']})

result_frame = basic_model.predict(data=inference_dataset)

print(result_frame[0].content)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Chain with Vector Search Retriever
# MAGIC
# MAGIC Lets look at a chain with a vector store retriever


# COMMAND ----------

# We need to create a custom pyfunc to hold the logic
class MlflowLangchainwVectorStore(mlflow.pyfunc.PythonModel):

    def __init__(self, llm_model = 'databricks-dbrx-instruct', 
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
        from langchain_community.embeddings import DatabricksEmbeddings
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser

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
       # row['session_id']
       return self.rag_chain.invoke(row['prompt'])
                                 #config={"configurable": {"session_id": "abc123"}})
    
    def predict(self, context, data):
        results = data.apply(self.process_row, axis=1) 

        # remove .content if it is with Databricks
        results_text = results.apply(lambda x: x)
        return results_text
        

# COMMAND ----------

# We can also read in a list of questions from Spark as well
eval_list = {'prompts': ['How can I tune LLMs?',
                        'What is a good model funetuning technique?']}

pd_evals = pd.DataFrame(eval_list)

# COMMAND ----------

import mlflow
import pandas as pd

mlflow.set_registry_uri('databricks-uc')

username = spark.sql("SELECT current_user()").first()['current_user()']
experiment_name = 'tab_evals'

mlflow.set_experiment(f'/Users/{username}/{experiment_name}')

with mlflow.start_run(run_name='basic rag chat'):

    base_model = MlflowLangchainwVectorStore()
    base_model.load_context(context="")

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
      extra_pip_requirements=['mlflow==2.11.3',
                              'langchain==0.1.16',
                              'databricks-vectorsearch==0.21'],
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