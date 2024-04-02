# Databricks notebook source
# MAGIC %md
# MAGIC # Building an App
# MAGIC It is common to have a chatbot interface to an app even as a sample \
# MAGIC We can run this on the driver-proxy in databricks \
# MAGIC This will be suitable for small teams and light usecases with no strict requirements on: \
# MAGIC - failover
# MAGIC - redundacy
# MAGIC - uptime

# COMMAND ----------

# MAGIC %sh
# MAGIC # we need to install into the OS tier in some MLR / Gradio version combos
# MAGIC #/databricks/python/bin/pip install fastapi gradio uvicorn pypdf

# COMMAND ----------

# MAGIC # MLR 14.3 and gradio 4.24.0 we can do this
# MAGIC # this is preferable as install to root env as per above requires resetting cluster if we wannt try different versions
# MAGIC # rather than just detach / reattach
# MAGIC %pip install gradio langchainhub==0.1.15 langchain==0.1.13 databricks-vectorsearch==0.23 databricks-sql-connector==3.1.1 flashrank
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC # Launch App
# MAGIC The app code sits inside a standard file and we use %sh to start it

# COMMAND ----------
import os

# First we setup the starting location and find the uri
server_port = 8501
os.environ['DB_APP_PORT'] = f'{server_port}'

cluster_id = spark.conf.get("spark.databricks.clusterUsageTags.clusterId")
workspace_url = spark.conf.get("spark.databricks.workspaceUrl")
org_id = spark.conf.get("spark.databricks.clusterUsageTags.clusterOwnerOrgId")

# AWS
aws_real_url = f"https://dbc-dp-{org_id}.cloud.databricks.com/driver-proxy/o/{org_id}/{cluster_id}/{server_port}/"
print(f"try this uri if AWS: {aws_real_url}")

# Azure
azure_real_uri = f"https://adb-dp-{org_id}.11.azuredatabricks.net/driver-proxy/o/{org_id}/{cluster_id}/{server_port}"
print(f"try this uri if Azure: {azure_real_uri}")

# COMMAND ----------

# We can start a streamlit this way
#!streamlit run db_app.py #--server.baseUrlPath=driver-proxy/o/2556758628403379/0115-034704-alj89d3k/8501

# COMMAND ----------

# we can start gradio this way
os.environ['GRADIO_SERVER_NAME'] = "0.0.0.0"
os.environ["GRADIO_SERVER_PORT"] = f'{server_port}'

# Create a secret first with the utils notebook then use that here
os.environ['DATABRICKS_HOST'] = f'https://{workspace_url}'
os.environ['DATABRICKS_TOKEN'] = dbutils.secrets.get(scope='qldllmpoc', key='rag_sp_token')

# choose the right path_format
os.environ['GRADIO_ROOT_PATH'] = f"https://dbc-dp-{org_id}.cloud.databricks.com/driver-proxy/o/{org_id}/{cluster_id}/{server_port}/"

# COMMAND ----------

# DBTITLE 1,Basic Application
!python3 app/basic_test_app.py

# COMMAND ----------

# DBTITLE 1,Chat Application
!python3 app/chat_app.py

# COMMAND ----------

# DBTITLE 1,Chat To Docs App
!python3 app/chat_to_docs_app.py

# COMMAND ----------

# DBTITLE 1,Adv Chat To Docs App
!python3 app/adv_chat_to_docs_app.py

# COMMAND ----------

# DBTITLE 1,Enhanced RAG
!python3 app/expansion_and_rerank_app.py
