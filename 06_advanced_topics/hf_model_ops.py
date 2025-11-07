# Databricks notebook source
# MAGIC
# MAGIC %md
# MAGIC # Logging and using HF Models
# MAGIC
# MAGIC Most organisations do not want to have workspaces permanently open to huggingface.co \
# MAGIC We can leverage MLflow in order to log and store models
# MAGIC
# MAGIC For more info see: [Logging HuggingFace](https://mlflow.org/docs/latest/python_api/mlflow.transformers.html)
# MAGIC TODO - Test contents
# MAGIC

# COMMAND ----------

# MAGIC %pip install mlflow==2.11.3
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

from transformers import (
   AutoModelForCausalLM,
   AutoTokenizer,
   AutoConfig,
   pipeline
)
import mlflow
import torch

# COMMAND ----------

# Loading HF Model 

model_name = 'HuggingFaceH4/zephyr-7b-beta'
tokenizer = AutoTokenizer.from_pretrained(model_name)

model_config = AutoConfig.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name,
                                               config=model_config,
                                               #device_map='auto', # disabled for mlflow compatability
                                               torch_dtype=torch.bfloat16 # This will only work A10G / A100 and newer GPUs
                                              )

pipe = pipeline(
            "text-generation", model=model, tokenizer=tokenizer, max_new_tokens=128 
        )

# COMMAND ----------

experiment_name = 'logging_hf_models'
artifact_path = 'hf_model'
catalog = 'huggingface'
schema = 'llm_models'


username = spark.sql("SELECT current_user()").first()['current_user()']
full_experiment_path = f'/Users/{username}/{experiment_name}'

mlflow.set_experiment(mlflow_dir)

mlflow.set_registry_uri('databricks-uc')

with mlflow.start_run(run_name='zephyr_base_model') as run:
    mlflow.transformers.log_model(
        transformers_model=pipe,
        artifact_path=artifact_path,
        registered_model_name=f'{catalog}.{schema}.zephyr_7b'
    )

# COMMAND ----------