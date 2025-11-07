# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Running Unstructured API

# COMMAND ----------

# MAGIC %sh
# MAGIC # Underlying Poppler utils need to be setup on the OS
# MAGIC apt-get install -y poppler-utils

# COMMAND ----------

# MAGIC %pip install --upgrade 'urllib3==1.26.7' poppler-utils unstructured[local-inference] uvicorn fastapi pydantic==2.6.4
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %sh
# MAGIC git clone https://github.com/Unstructured-IO/unstructured-api.git /local_disk0/unstructured-api

# COMMAND ----------

# MAGIC  %sh
# MAGIC # --root-path /driver-proxy/o/1444828305810485/0413-091134-9l9vdj10/8081
# MAGIC PYTHONPATH=/local_disk0/unstructured-api/:$PYTHONPATH uvicorn --host 0.0.0.0 --port 9190 prepline_general.api.app:app --reload