# Databricks notebook source
# MAGIC %md
# MAGIC # Automating Secrets
# MAGIC As a part of building RAG products we need to work with and create secrets \
# MAGIC This will be for connecting orchestrators to models and setting up connectivity \
# MAGIC In Prod we probably want a service principal secret rather than one we use here
# COMMAND ----------

# MAGIC %pip install --upgrade databricks-sdk
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

from databricks.sdk import WorkspaceClient

w = WorkspaceClient()

# check what scopes we have
scopes = w.secrets.list_scopes()

# COMMAND ----------

scopes 

# COMMAND ----------

scope_name = 'dev_work'
# Optional
# w.secrets.create_scope(scope=scope_name)

# COMMAND ----------

token_name = 'rag_tutorial'
token = w.tokens.create(comment=token_name, lifetime_seconds=90*24*60*60)

key_name = 'rag_tutorial'
w.secrets.put_secret(scope=scope_name, key=key_name, string_value=token.token_value)

