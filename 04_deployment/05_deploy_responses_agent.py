# Databricks notebook source
# MAGIC %md
# MAGIC # Deploy ResponsesAgent (Model as Code)
# MAGIC
# MAGIC This notebook deploys a ResponsesAgent that was defined in `03_agents/04_responses_agent.py` using the **model-as-code** approach.
# MAGIC
# MAGIC **Model as Code Benefits:**
# MAGIC - Agent code stays in the development notebook
# MAGIC - Deployment is just configuration and MLflow commands
# MAGIC - Easy to iterate on agent logic without touching deployment
# MAGIC - Follows Databricks best practices
# MAGIC
# MAGIC **Prerequisites:**
# MAGIC 1. Complete `03_agents/04_responses_agent.py` notebook
# MAGIC 2. Configure your tools in that notebook
# MAGIC 3. Run `mlflow.models.set_model(AGENT)` in the agent notebook
# MAGIC
# MAGIC **Based on**: [OpenAI MCP Tool-Calling Agent](https://docs.databricks.com/aws/en/notebooks/source/generative-ai/openai-mcp-tool-calling-agent.html)

# COMMAND ----------

# DBTITLE 1,Install Required Libraries
# MAGIC %pip install -U mlflow>=3.0 databricks-langchain openai databricks-sdk databricks-openai unitycatalog-ai databricks_mcp nest_asyncio pydantic>=2.0 databricks-agents
# MAGIC %restart_python

# COMMAND ----------

# DBTITLE 1,Configuration
import mlflow
import os
from databricks.sdk import WorkspaceClient

# Set registry to Unity Catalog
mlflow.set_registry_uri('databricks-uc')

# Get current user for experiment naming
username = spark.sql("SELECT current_user()").first()['current_user()']
parsed_name = username.split("@")[0].replace(".", "_")

# Configure deployment settings
DEPLOYMENT_CONFIG = {
    "catalog": "brian_ml_dev",
    "schema": "gen_ai_tutorials", 
    "model_name": "responses_agent_mcp",
    "experiment_name": "responses_agent_deployment",
    "agent_notebook_path": "../03_agents/04_responses_agent"  # Path to agent definition
}

# Set MLflow experiment
mlflow.set_experiment(f'/Users/{username}/{DEPLOYMENT_CONFIG["experiment_name"]}')

print(f"✅ Configuration set")
print(f"  Catalog: {DEPLOYMENT_CONFIG['catalog']}")
print(f"  Schema: {DEPLOYMENT_CONFIG['schema']}")
print(f"  Model: {DEPLOYMENT_CONFIG['model_name']}")
print(f"  Agent Code: {DEPLOYMENT_CONFIG['agent_notebook_path']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configure Agent Resources
# MAGIC
# MAGIC Specify the resources your agent uses. These should match what you configured in the agent notebook.

# COMMAND ----------

# DBTITLE 1,Define Agent Resources
from mlflow.models.resources import (
    DatabricksServingEndpoint,
    DatabricksVectorSearchIndex,
    DatabricksGenieSpace,
    DatabricksFunction
)

# LLM endpoint (required)
LLM_ENDPOINT = "databricks-gpt-oss-120b"

# Optional: Unity Catalog Functions
# List the UC function names you configured in the agent notebook
UC_FUNCTIONS = []
# Example: UC_FUNCTIONS = ["catalog.schema.my_function"]

# Optional: Vector Search indexes
# List the vector search indexes you configured in the agent notebook
VECTOR_SEARCH_INDEXES = []
# Example: VECTOR_SEARCH_INDEXES = ["catalog.schema.my_index"]

# Optional: Genie Space IDs
# List the Genie Space IDs you configured in the agent notebook
GENIE_SPACES = []
# Example: GENIE_SPACES = ["01234567-89ab-cdef-0123-456789abcdef"]

# Build resource list for MLflow
resources = [DatabricksServingEndpoint(endpoint_name=LLM_ENDPOINT)]

for uc_func in UC_FUNCTIONS:
    resources.append(DatabricksFunction(function_name=uc_func))

for vs_index in VECTOR_SEARCH_INDEXES:
    resources.append(DatabricksVectorSearchIndex(index_name=vs_index))

for genie_id in GENIE_SPACES:
    resources.append(DatabricksGenieSpace(genie_space_id=genie_id))

print(f"✅ Resources configured: {len(resources)} total")
print(f"  - LLM Endpoint: 1")
print(f"  - UC Functions: {len(UC_FUNCTIONS)}")
print(f"  - Vector Search: {len(VECTOR_SEARCH_INDEXES)}")
print(f"  - Genie Spaces: {len(GENIE_SPACES)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Log Model as Code
# MAGIC
# MAGIC Use MLflow to log the agent notebook as code, including all dependencies.

# COMMAND ----------

# DBTITLE 1,Log Model from Agent Notebook
# Get the path to the agent notebook
agent_notebook_path = os.path.join(os.getcwd(), DEPLOYMENT_CONFIG["agent_notebook_path"])

# Create input example for model signature
input_example = {
    "input": [
        {
            "role": "user",
            "content": "Hello! What can you help me with?"
        }
    ]
}

with mlflow.start_run(run_name='responses_agent_model_as_code') as run:
    
    # Set tags for tracking
    mlflow.set_tag("type", "responses_agent")
    mlflow.set_tag("deployment_type", "model_as_code")
    mlflow.set_tag("agent_notebook", DEPLOYMENT_CONFIG["agent_notebook_path"])
    mlflow.set_tag("resources_count", len(resources))
    
    # Log the model from the agent notebook
    logged_agent_info = mlflow.pyfunc.log_model(
        artifact_path="agent",
        python_model=agent_notebook_path,  # Path to agent notebook
        resources=resources,
        input_example=input_example,  # Example input for model signature
        pip_requirements=[
            "mlflow>=3.0",
            "databricks-langchain",
            "openai",
            "databricks-sdk",
            "databricks-openai",
            "unitycatalog-ai",
            "databricks_mcp",
            "nest_asyncio",
            "pydantic>=2.0"
        ]
    )
    
    run_id = run.info.run_id
    model_uri = logged_agent_info.model_uri

print(f"✅ Model logged successfully using model-as-code")
print(f"  Run ID: {run_id}")
print(f"  Model URI: {model_uri}")
print(f"  Agent Code: {agent_notebook_path}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test the Logged Model
# MAGIC
# MAGIC Load and test the model before deployment

# COMMAND ----------

# DBTITLE 1,Load and Test Model
loaded_agent = mlflow.pyfunc.load_model(model_uri)

# Test with a simple query
test_input = {
    "input": [
        {
            "role": "user",
            "content": "Hello! Please briefly tell me what you can do."
        }
    ]
}

print("Testing loaded model...")
result = loaded_agent.predict(test_input)

print("\n✅ Model test successful!")
print(f"Response received from agent")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Register to Unity Catalog
# MAGIC
# MAGIC Register the model to Unity Catalog for governance and versioning

# COMMAND ----------

# DBTITLE 1,Register Model
catalog = DEPLOYMENT_CONFIG["catalog"]
schema = DEPLOYMENT_CONFIG["schema"]
model_name = DEPLOYMENT_CONFIG["model_name"]

uc_model_name = f"{catalog}.{schema}.{model_name}"

uc_registered_model_info = mlflow.register_model(
    model_uri=model_uri,
    name=uc_model_name
)

print(f"✅ Model registered to Unity Catalog")
print(f"  Name: {uc_model_name}")
print(f"  Version: {uc_registered_model_info.version}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Deploy to Model Serving
# MAGIC
# MAGIC Deploy the agent to Databricks Model Serving with Review App

# COMMAND ----------

# DBTITLE 1,Deploy Agent
from databricks import agents

# Deploy the agent with scale-to-zero enabled
deployment_info = agents.deploy(
    model_name=uc_model_name,
    model_version=uc_registered_model_info.version,
    scale_to_zero=True  # Cost optimization
)

# Get deployment URL
browser_url = mlflow.utils.databricks_utils.get_browser_hostname()

print(f"✅ Agent deployed successfully!")
print(f"\n📊 Deployment Information:")
print(f"  Model: {uc_model_name} v{uc_registered_model_info.version}")
print(f"  Endpoint: {deployment_info.endpoint_name}")
print(f"  Status URL: https://{browser_url}/ml/endpoints/{deployment_info.endpoint_name}")
print(f"\n🧪 Review App:")
print(f"  Access the Review App from the Model Serving endpoint page to test your agent")
print(f"\n📝 Next Steps:")
print(f"  1. Test agent in Review App UI")
print(f"  2. Run MLflow Agent Evaluations")
print(f"  3. Monitor endpoint metrics")
print(f"  4. Integrate REST API into applications")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test Deployed Endpoint (Optional)
# MAGIC
# MAGIC Programmatically test the deployed endpoint

# COMMAND ----------

# DBTITLE 1,Test Deployment via API
import time

# Wait for deployment to be ready
print("Waiting for deployment to be ready...")
time.sleep(30)

# Test the deployed endpoint
from databricks import agents

try:
    # Note: agents.chat() uses "messages" parameter, which is different from
    # the model's direct input format that uses "input"
    response = agents.chat(
        model_name=uc_model_name,
        model_version=uc_registered_model_info.version,
        messages=[{"role": "user", "content": "Hello! What tools do you have?"}]
    )
    
    print("\n✅ Deployment API test successful!")
    print("Agent is responding correctly")
except Exception as e:
    print(f"\n⚠️ API test error (deployment may still be initializing): {e}")
    print("Try testing in the Review App UI instead")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC ### Deployment Complete! 🎉
# MAGIC
# MAGIC Your ResponsesAgent has been deployed using the **model-as-code** approach:
# MAGIC
# MAGIC ✅ **Agent Code**: Located in `03_agents/04_responses_agent.py`  
# MAGIC ✅ **Logged with MLflow**: All dependencies tracked  
# MAGIC ✅ **Registered**: Available in Unity Catalog  
# MAGIC ✅ **Deployed**: Live on Model Serving  
# MAGIC ✅ **Review App**: Ready for testing  
# MAGIC
# MAGIC ### Model as Code Benefits:
# MAGIC - 🔄 **Easy Updates**: Modify agent logic in the agent notebook
# MAGIC - 📦 **Clean Separation**: Development vs deployment concerns
# MAGIC - 🔍 **Version Control**: Track agent code changes
# MAGIC - 🚀 **Quick Iteration**: Redeploy by re-running this notebook
# MAGIC
# MAGIC ### Deployed Resources:
# MAGIC - **LLM Endpoint**: `{LLM_ENDPOINT}`
# MAGIC - **UC Functions**: {len(UC_FUNCTIONS)} functions
# MAGIC - **Vector Search**: {len(VECTOR_SEARCH_INDEXES)} indexes
# MAGIC - **Genie Spaces**: {len(GENIE_SPACES)} spaces
# MAGIC
# MAGIC ### To Update Your Agent:
# MAGIC 1. Modify code in `03_agents/04_responses_agent.py`
# MAGIC 2. Test changes in that notebook
# MAGIC 3. Re-run this deployment notebook
# MAGIC 4. MLflow will create a new version
# MAGIC
# MAGIC ### References:
# MAGIC - [Databricks Agent Framework](https://docs.databricks.com/generative-ai/agent-framework/)
# MAGIC - [Model as Code](https://docs.databricks.com/mlflow/models.html)
# MAGIC - [MCP Documentation](https://docs.databricks.com/generative-ai/mcp/)
