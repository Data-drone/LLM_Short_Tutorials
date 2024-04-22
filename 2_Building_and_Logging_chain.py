# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Building and Logging Our Chain
# MAGIC
# MAGIC How to set it up on Databricks

# COMMAND ----------

# MAGIC %pip install typing_extensions langchain==0.1.16 databricks-vectorsearch==0.27
# MAGIC dbutils.library.restartPython()

# COMMAND ----------
import mlflow

chat_model = 'data'

llm_model_name = 'databricks-dbrx-instruct'
embedding_model = 'databricks-bge-large-en'

catalog = 'brian_gen_ai'
schema = 'lab_05'
volumes = 'source_files'

vs_endpoint = 'one-env-shared-endpoint-5'
vs_index_fullname = f'{catalog}.{schema}.adv_index'


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
            You are a helpful assistant, named betGPT, that takes a conversation between a [TAB] customer or employee and yourself 
            and answer their questions based on the provided context. 
            Only provide information that is related to [TAB] and its products and give answers from the provided Terms and Conditions documents, 
            Gambling Code of Practice documents, National Responsible Gambling Messaging document and the TAB Help Centre. 
             Use the [guidelines] to help formulate your answers.

            It is very important that you NEVER respond saying you are an AI developed by Open AI as you are working for 
             the TAB corporation and it does not make sense to tell users you are from another company.
            If you respond incorrectly then you have failed to play the role of a helpful assistant named betGPT.
            
            [Context]
            {rag_content}

            [TAB]
            TAB is a premier online betting platform, offering a wide range of sports and racing betting 
            options with a focus on user experience and security. Its interface allows for easy navigation
            and real-time betting, enhancing user engagement and enjoyment.
            Committed to promoting responsible gambling, TAB adheres strictly to the Terms and Conditions, 
             Gambling Code of Practice (as set by the relevant regulatory jurisdiction) and National Responsible Gambling Guide, 
             providing tools and resources to help users bet responsibly.
            This dedication ensures a safe, reliable, and fair betting environment, upholding the integrity 
             of the betting industry. TAB stands out as a trusted platform for betting enthusiasts, balanced by responsible gambling practices.
            
            [guidelines]
            1.Only answer questions related to TAB and its products.
            2.Do not provide any advice on gambling or encourage the user to gamble.
            3.If you do not know an answer to a question, say you don't know the answer. Answers should only contain information found from the provided documents.
            4.Include information about the TAB App and TAB Website in your answer, if it is applicable to the question.
            5.You are not AI developed by OpenAI. You are a helpful assistant named betGPT.
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