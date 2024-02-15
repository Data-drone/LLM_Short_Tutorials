# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Prompting And Logging
# MAGIC
# MAGIC In this short tutorial we will ping a databricks endpoint and run some prompting exercises
# MAGIC
# MAGIC Once we are sure it is working, we can extend this on log them to MLflow

# COMMAND ----------

# MAGIC %md # Validate Endpoint Working

# COMMAND ----------

# DBTITLE 1,Setup Connection

# databricks endpoint uri
endpoint_name = 'https://e2-demo-west.cloud.databricks.com/serving-endpoints/databricks-llama-2-70b-chat/invocations'

# security settings
db_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

# COMMAND ----------

# DBTITLE 1,Setting up prompt
user_prompt = 'tell me a joke'

system_prompt = 'You are a flamboyant assistant who loves saying mate'

full_prompt = {
    "messages": [
        {
            "role": "system",
            "content": f"{system_prompt}"},
         {
             "role": "user",
             "content": f"{user_prompt}"
         }
    ],
    "max_tokens": 128
}

# COMMAND ----------

# DBTITLE 1,Querying endpoint
import requests

headers = {"Context-Type": "text/json", "Authorization": f"Bearer {db_token}"}

response = requests.post(
    url=endpoint_name, json=full_prompt, headers=headers
)

response.text

# COMMAND ----------

# DBTITLE 1,Properly parsing output
import json

parse_response = json.loads(response.text)
print(parse_response['choices'][0]['message']['content'])

# COMMAND ----------

# MAGIC %md # Add MLflow logging

# COMMAND ----------

import mlflow

username = spark.sql("SELECT current_user()").first()['current_user()']
experiment_path = f'/Users/{username}/basic_prompting'

mlflow.set_experiment(experiment_path)

# COMMAND ----------

# DBTITLE 1,Create Golden Prompts
import pandas as pd

system_prompt_list = [
    'You are a flamboyant assistant who loves saying mate',
    'You are a traditional learned english gentleman and speak like an aristocrat',
    'You are a robotic souless automaton and totally analytical and dispassionate'
]

prompt_list = [
    "what is databricks",
    "what is Apache Spark?",
    "what is a pyspark dataframe?"
]

testing_prompts = pd.DataFrame(
    prompt_list, columns = ['prompt']
)

# COMMAND ----------

# DBTITLE 1,Eval Function

# TODO - Fix the eval function
def eval_pipe(inputs):
    answers = []
    for index, row in inputs.iterrows():
        
        
        message_structure = {
            "messages": [
                {
                    "role": "system",
                    "content": f"{system_prompt}"},
                 {
                     "role": "user",
                     "content": f"{row.item()}"
                 }
            ]
        }

        full_prompt = {**message_structure, **hyper_params}
        
        result = requests.post(
            url=endpoint_name, json=full_prompt, headers=headers
        )

        parse_response = json.loads(result.text)

        answer = parse_response['choices'][0]['message']['content']
        answers.append(answer)
    
    return answers

# COMMAND ----------

for system_prompt in system_prompt_list:
    with mlflow.start_run(run_name='experiment_1'):

        #system_prompt = 'You are a flamboyant assistant who loves saying mate'
        hyper_params = {'system_prompt': system_prompt, 'max_tokens': 128}
        mlflow.log_params(hyper_params)

        results = mlflow.evaluate(
                eval_pipe,
                data = testing_prompts,
                model_type='text',
                extra_metrics=[mlflow.metrics.latency()],
            )
    
# COMMAND ----------
