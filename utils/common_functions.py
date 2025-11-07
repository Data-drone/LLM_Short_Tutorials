# Databricks notebook source
# MAGIC %md
# MAGIC # Common Utility Functions
# MAGIC
# MAGIC Shared functions used across multiple notebooks

# COMMAND ----------

def get_databricks_connection_info(spark, dbutils):
    """
    Get Databricks workspace connection information
    
    Returns:
        tuple: (workspace_url, api_token, base_url)
    """
    databricks_workspace_uri = spark.conf.get("spark.databricks.workspaceUrl")
    base_url = f'https://{databricks_workspace_uri}/serving-endpoints'
    db_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
    
    return databricks_workspace_uri, db_token, base_url

# COMMAND ----------

def get_current_username(spark):
    """
    Get the current Databricks username
    
    Returns:
        str: Current username
    """
    return spark.sql("SELECT current_user()").first()['current_user()']

# COMMAND ----------

def setup_mlflow_experiment(spark, experiment_name):
    """
    Setup MLflow experiment for current user
    
    Args:
        spark: Spark session
        experiment_name: Name of the experiment
        
    Returns:
        str: Experiment path
    """
    import mlflow
    
    username = get_current_username(spark)
    experiment_path = f'/Users/{username}/{experiment_name}'
    mlflow.set_experiment(experiment_path)
    
    return experiment_path

# COMMAND ----------

def create_uc_resources(spark, catalog, schema):
    """
    Create Unity Catalog resources if they don't exist
    
    Args:
        spark: Spark session
        catalog: Catalog name
        schema: Schema name
    """
    spark.sql(f"CREATE CATALOG IF NOT EXISTS {catalog}")
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}")

# COMMAND ----------

def format_docs(docs):
    """
    Format documents for context injection in RAG chains
    
    Args:
        docs: List of document objects
        
    Returns:
        str: Formatted document content
    """
    return "\n\n".join(doc.page_content for doc in docs)

# COMMAND ----------

def create_databricks_token(workspace_client, lifetime_days=30):
    """
    Create a Databricks token using workspace client
    
    Args:
        workspace_client: Databricks WorkspaceClient instance
        lifetime_days: Token lifetime in days (default 30)
        
    Returns:
        str: Token value
    """
    token_response = workspace_client.tokens.create(
        comment="llm_tutorial_token",
        lifetime_seconds=3600*24*lifetime_days
    )
    return token_response.token_value

