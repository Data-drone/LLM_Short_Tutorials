# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Prompting And Logging
# MAGIC
# MAGIC In this short tutorial we will: 
# MAGIC - setup OpenAI client for Databricks endpoints
# MAGIC - run prompting exercises using standard OpenAI API calls
# MAGIC - log experiments to MLflow 3.0 with autologging
# MAGIC - track prompt inputs and model outputs

# COMMAND ----------

# MAGIC %md # Validate Endpoint Working
# MAGIC
# MAGIC Lets first validate that the endpoints are working.
# MAGIC The OpenAI Library is the most popular way of working with LLMs in Python so we will start with that 

# COMMAND ----------

%pip install mlflow openai
dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Setup OpenAI Client for Databricks
from openai import OpenAI
import mlflow

# databricks model
model_name = 'databricks-gpt-oss-120b'

# get the endpoint
## To setup the connection we need the workspace url and also an access token 
databricks_workspace_uri = spark.conf.get("spark.databricks.workspaceUrl")
base_url = f'https://{databricks_workspace_uri}/serving-endpoints'
db_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

# Initialize OpenAI client for Databricks
client = OpenAI(
    api_key=db_token,
    base_url=base_url
)

# COMMAND ----------

# DBTITLE 1,Setting up prompt
user_prompt = 'tell me a joke'
system_prompt = 'You are a flamboyant assistant who loves saying mate'

# COMMAND ----------

# DBTITLE 1,Querying endpoint using OpenAI client
response = client.chat.completions.create(
    model=model_name,
    messages=[
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": user_prompt
        }
    ],
    max_tokens=128
)

print(response.choices[0].message.content)

# COMMAND ----------

# MAGIC %md # Add MLflow logging
# MAGIC
# MAGIC We can log prompt experiments into mlflow with the following code

# COMMAND ----------

username = spark.sql("SELECT current_user()").first()['current_user()']
experiment_path = f'/Users/{username}/basic_prompting'

mlflow.set_experiment(experiment_path)

# COMMAND ----------

# DBTITLE 1,Simple Response Logging with OpenAI Client
import pandas as pd

prompt_list = [
    "what is databricks",
    "what is Apache Spark?",
    "what is a pyspark dataframe?"
]

testing_prompts = pd.DataFrame(
    prompt_list, columns = ['prompt']
)

with mlflow.start_run(run_name='basic_prompting'):
    
    # Enable MLflow autologging for OpenAI
    mlflow.openai.autolog()
    
    # Log parameters
    mlflow.log_param("system_prompt", system_prompt)
    mlflow.log_param("max_tokens", 128)
    mlflow.log_param("model_name", model_name)
    mlflow.log_param("num_prompts", len(prompt_list))
    
    # Process each prompt and log inputs/outputs
    for i, prompt in enumerate(prompt_list):
        
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=128
        )
        
        output = response.choices[0].message.content
        
        # Log input and output for each prompt
        mlflow.log_param(f"input_prompt_{i+1}", prompt)
        mlflow.log_text(output, f"output_{i+1}.txt")
        
        print(f"Prompt {i+1}: {prompt}")
        print(f"Response: {output}")
        print("-" * 50)

# COMMAND ----------

# MAGIC %md
# MAGIC # Expanded Sample
# MAGIC
# MAGIC With MLflow 3 there is the ability to log and manage system prompts as well
# MAGIC

# COMMAND ----------

