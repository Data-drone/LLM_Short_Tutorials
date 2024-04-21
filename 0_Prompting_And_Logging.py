# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Prompting And Logging
# MAGIC
# MAGIC In this short tutorial we will ping a databricks endpoint and run some prompting exercises
# MAGIC
# MAGIC Once we are sure it is working, we can extend this on log them to MLflow
# MAGIC
# MAGIC **Notes**
# MAGIC - The mistral model does not accept system prompt type. If just doing simple prompting it is not a problem advanced methods may require this and cause errors
# MAGIC

# COMMAND ----------

# MAGIC %md # Validate Endpoint Working
# MAGIC
# MAGIC Lets first validate that the endpoints are working

# COMMAND ----------

# DBTITLE 1,Setup Connection

# databricks endpoint uri

# This gets the current workspace uri assuming that we are not querying cross workspace
databricks_workspace_uri = spark.conf.get("spark.databricks.workspaceUrl")

# this is the name of the endpoint in model serving page
model_name = 'databricks-dbrx-instruct'

endpoint_name = f'https://{databricks_workspace_uri}/serving-endpoints/{model_name}/invocations'

# To hit an endpoint we need a token
# This pulls the current Notebook session token
# NOTE this only works in current workspace and assumes user has access to the model serving endpoint
# For production use a token created as a secret instead
# dbutils.secrets.get(scope=secret_scope, key=secret_key)
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
# MAGIC
# MAGIC We can log prompt experiments into mlflow with the following code

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

class EvalObj:

    def __init__(self, system_prompt, hyper_parms):

        self.system_prompt = system_prompt
        self.hyper_parms = hyper_parms

    def eval_pipe(self, inputs):
        answers = []
        for index, row in inputs.iterrows():
            
            message_structure = {
                "messages": [
                    {
                        "role": "system",
                        "content": f"{self.system_prompt}"},
                    {
                        "role": "user",
                        "content": f"{row.item()}"
                    }
                ]
            }

            full_prompt = {**message_structure, **self.hyper_parms}
            
            result = requests.post(
                url=endpoint_name, json=full_prompt, headers=headers
            )

            parse_response = json.loads(result.text)
            print(type(parse_response))

            answer = parse_response['choices'][0]['message']['content']
            answers.append(answer)
        
        return answers

# COMMAND ----------

for system_prompt in system_prompt_list:
    with mlflow.start_run(run_name='experiment_1'):

        #system_prompt = 'You are a flamboyant assistant who loves saying mate'
        hyper_params = {'system_prompt': system_prompt, 'max_tokens': 128}
        mlflow.log_params(hyper_params)

        eval_obj = EvalObj(system_prompt=system_prompt,
                           hyper_parms={'max_tokens': 128})

        results = mlflow.evaluate(
                eval_obj.eval_pipe,
                data = testing_prompts,
                model_type='text'
            )
    
# COMMAND ----------
