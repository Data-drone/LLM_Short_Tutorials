# Databricks notebook source
# MAGIC %md
# MAGIC # Prompt-Optimized ResponsesAgent with GEPA
# MAGIC
# MAGIC This notebook demonstrates how to optimize agent system prompts using MLflow's GEPA (Generative Evaluation and Prompt Augmentation).
# MAGIC
# MAGIC ## Key Features:
# MAGIC - **Prompt Registry**: Load system prompts from MLflow Prompt Registry
# MAGIC - **GEPA Optimization**: Automatically improve prompts using training examples
# MAGIC - **ResponsesAgent**: Databricks-native agent with tool calling
# MAGIC - **Production Ready**: Reload prompts without redeploying agent
# MAGIC
# MAGIC ## Flow:
# MAGIC 1. Define agent with prompt registry integration
# MAGIC 2. Register initial seed prompt
# MAGIC 3. Run GEPA optimization with training data
# MAGIC 4. Deploy agent using optimized prompt

# COMMAND ----------

# DBTITLE 1,Install Required Libraries
# MAGIC %pip install -U mlflow>=3.0 openai databricks-sdk pydantic>=2.0 gepa litellm
# MAGIC %restart_python

# COMMAND ----------

# DBTITLE 1,Imports and Setup
import json
import warnings
from typing import Any, Callable, Generator, Optional
from uuid import uuid4

import mlflow
import openai
from databricks.sdk import WorkspaceClient
from mlflow.entities import SpanType
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
    output_to_responses_items_stream,
    to_chat_completions_input,
)
from openai import OpenAI
from pydantic import BaseModel

# Import GEPA optimization tools
from mlflow.genai.optimize import optimize_prompts, GepaPromptOptimizer
from mlflow.genai.scorers import Correctness

# Mlflow Experiment
username = spark.sql("SELECT current_user()").first()['current_user()']
experiment_name = f'/Users/{username}/responses_agent_deployment'

mlflow.set_experiment(experiment_name)

# Enable MLflow autologging
mlflow.openai.autolog()

# Configuration
CATALOG = 'brian_ml_dev'
SCHEMA = 'gen_ai_tutorials'
LLM_ENDPOINT_NAME = "databricks-gpt-oss-120b"
PROMPT_NAME = f"{CATALOG}.{SCHEMA}.agent_system_prompt"

print("✅ Imports complete")

# COMMAND ----------

# DBTITLE 1,Define Prompt-Optimized ResponsesAgent
# MAGIC %md
# MAGIC ## Adding Tools to the Agent
# MAGIC
# MAGIC The agent supports multiple tool types. Here's how to add them:
# MAGIC
# MAGIC ### 1. Unity Catalog Functions
# MAGIC ```python
# MAGIC from databricks_openai import UCFunctionToolkit
# MAGIC from unitycatalog.ai.core.base import get_uc_function_client
# MAGIC
# MAGIC # Define UC function names
# MAGIC UC_TOOL_NAMES = ["catalog.schema.function_name"]
# MAGIC
# MAGIC # Create toolkit and function client
# MAGIC uc_toolkit = UCFunctionToolkit(function_names=UC_TOOL_NAMES)
# MAGIC uc_function_client = get_uc_function_client()
# MAGIC
# MAGIC # Create ToolInfo objects
# MAGIC uc_tools = []
# MAGIC for tool_spec in uc_toolkit.tools:
# MAGIC     tool_spec["function"].pop("strict", None)
# MAGIC     tool_name = tool_spec["function"]["name"]
# MAGIC     udf_name = tool_name.replace("__", ".")
# MAGIC     
# MAGIC     def exec_fn(**kwargs):
# MAGIC         result = uc_function_client.execute_function(udf_name, kwargs)
# MAGIC         return result.error if result.error else result.value
# MAGIC     
# MAGIC     uc_tools.append(ToolInfo(name=tool_name, spec=tool_spec, exec_fn=exec_fn))
# MAGIC ```
# MAGIC
# MAGIC ### 2. Vector Search Tools
# MAGIC ```python
# MAGIC from databricks_openai import VectorSearchRetrieverTool
# MAGIC
# MAGIC # Create Vector Search tool
# MAGIC vs_tool = VectorSearchRetrieverTool(
# MAGIC     index_name="catalog.schema.index_name",
# MAGIC     tool_description="Search through documents. Use this to find relevant information."
# MAGIC )
# MAGIC
# MAGIC # Convert to ToolInfo
# MAGIC vs_tool_info = ToolInfo(
# MAGIC     name=vs_tool.tool["function"]["name"],
# MAGIC     spec=vs_tool.tool,
# MAGIC     exec_fn=vs_tool.execute
# MAGIC )
# MAGIC ```
# MAGIC
# MAGIC ### 3. Genie Space (via MCP)
# MAGIC ```python
# MAGIC from databricks_mcp import DatabricksMCPClient
# MAGIC from databricks.sdk import WorkspaceClient
# MAGIC import nest_asyncio
# MAGIC nest_asyncio.apply()
# MAGIC
# MAGIC # Connect to Genie Space
# MAGIC workspace_client = WorkspaceClient()
# MAGIC genie_space_id = "01234567-89ab-cdef-0123-456789abcdef"  # Get from Genie > Settings
# MAGIC genie_mcp_url = f"{workspace_client.config.host}/api/2.0/mcp/genie/{genie_space_id}"
# MAGIC
# MAGIC genie_client = DatabricksMCPClient(
# MAGIC     server_url=genie_mcp_url,
# MAGIC     workspace_client=workspace_client
# MAGIC )
# MAGIC
# MAGIC # Get Genie tools
# MAGIC genie_tools = []
# MAGIC for tool in genie_client.list_tools():
# MAGIC     tool_spec = {
# MAGIC         "type": "function",
# MAGIC         "function": {
# MAGIC             "name": tool.name,
# MAGIC             "description": tool.description or "Genie tool for data analysis",
# MAGIC             "parameters": tool.inputSchema or {"type": "object", "properties": {}}
# MAGIC         }
# MAGIC     }
# MAGIC     
# MAGIC     def create_exec_fn(client, tool_name):
# MAGIC         def exec_fn(**kwargs):
# MAGIC             response = client.call_tool(tool_name, kwargs)
# MAGIC             return response.content[0].text if response and response.content else ""
# MAGIC         return exec_fn
# MAGIC     
# MAGIC     genie_tools.append(ToolInfo(
# MAGIC         name=tool.name,
# MAGIC         spec=tool_spec,
# MAGIC         exec_fn=create_exec_fn(genie_client, tool.name)
# MAGIC     ))
# MAGIC ```
# MAGIC
# MAGIC ### 4. Pass Tools to Agent
# MAGIC ```python
# MAGIC # Combine all tools
# MAGIC all_tools = uc_tools + [vs_tool_info] + genie_tools + [CALCULATOR_TOOL]
# MAGIC
# MAGIC # Create agent with tools
# MAGIC agent = PromptOptimizedAgent(
# MAGIC     llm_endpoint=LLM_ENDPOINT_NAME,
# MAGIC     prompt_uri=f"prompts:/{PROMPT_NAME}/1",
# MAGIC     tools=all_tools,
# MAGIC     max_iter=10
# MAGIC )
# MAGIC ```
# MAGIC
# MAGIC For detailed examples, see `03_agents/04_responses_agent.py`

# COMMAND ----------

# DBTITLE 1,Tool Definition Classes
class ToolInfo(BaseModel):
    """Tool specification for the agent"""
    name: str
    spec: dict
    exec_fn: Callable
    
    class Config:
        arbitrary_types_allowed = True


class PromptOptimizedAgent(ResponsesAgent):
    """
    Tool-calling Agent with MLflow Prompt Registry integration.
    
    Combines full ToolCallingAgent capabilities with prompt optimization:
    - Multi-step tool execution with streaming
    - Unity Catalog Functions, Vector Search, MCP tools support
    - MLflow tracing for tool calls
    - Dynamic prompt loading from registry
    - Runtime prompt updates without redeployment
    - A/B testing different prompts
    """
    
    def __init__(
        self, 
        llm_endpoint: str, 
        prompt_uri: str,
        tools: Optional[list[ToolInfo]] = None, 
        max_iter: int = 10
    ):
        """
        Initialize agent with prompt from registry.
        
        Args:
            llm_endpoint: Databricks model serving endpoint name
            prompt_uri: MLflow prompt URI (e.g., 'prompts:/name/1')
            tools: List of ToolInfo objects (UC functions, Vector Search, MCP, etc.)
            max_iter: Maximum reasoning iterations
        """
        self.llm_endpoint = llm_endpoint
        self.prompt_uri = prompt_uri
        self.max_iter = max_iter
        
        # Load system prompt from registry (cached for performance)
        self._system_prompt = self._load_prompt()
        
        # Initialize OpenAI client
        self.workspace_client = WorkspaceClient()
        self.model_serving_client: OpenAI = (
            self.workspace_client.serving_endpoints.get_open_ai_client()
        )
        
        # Setup tools
        self._tools_dict = {tool.name: tool for tool in tools} if tools else {}
        
        print(f"✅ Agent initialized with prompt: {prompt_uri}")
        print(f"   Tools: {len(self._tools_dict)}, Max iterations: {max_iter}")
    
    def _load_prompt(self) -> str:
        """Load prompt from MLflow registry"""
        try:
            prompt = mlflow.genai.load_prompt(self.prompt_uri)
            return prompt.template if hasattr(prompt, 'template') else str(prompt)
        except Exception as e:
            print(f"⚠️ Error loading prompt: {e}")
            return "You are a helpful AI assistant that can use tools when needed."
    
    def reload_prompt(self):
        """
        Reload prompt from registry without restarting agent.
        Useful for production updates and A/B testing.
        """
        old_prompt = self._system_prompt
        self._system_prompt = self._load_prompt()
        print(f"🔄 Prompt reloaded. Changed: {old_prompt != self._system_prompt}")
        return self._system_prompt
    
    def get_tool_specs(self) -> list[dict]:
        """Returns tool specifications in OpenAI format"""
        return [tool_info.spec for tool_info in self._tools_dict.values()]
    
    @mlflow.trace(span_type=SpanType.TOOL)
    def execute_tool(self, tool_name: str, args: dict) -> Any:
        """Execute specified tool with MLflow tracing"""
        return self._tools_dict[tool_name].exec_fn(**args)
    
    def call_llm(self, messages: list[dict[str, Any]]) -> Generator[dict[str, Any], None, None]:
        """Call LLM with streaming"""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="PydanticSerializationUnexpectedValue")
            for chunk in self.model_serving_client.chat.completions.create(
                model=self.llm_endpoint,
                messages=to_chat_completions_input(messages),
                tools=self.get_tool_specs() if self._tools_dict else None,
                stream=True,
            ):
                chunk_dict = chunk.to_dict()
                if len(chunk_dict.get("choices", [])) > 0:
                    yield chunk_dict
    
    def handle_tool_call(
        self,
        tool_call: dict[str, Any],
        messages: list[dict[str, Any]],
    ) -> ResponsesAgentStreamEvent:
        """Execute tool call, add result to message history, return stream event"""
        args = json.loads(tool_call["arguments"])
        result = str(self.execute_tool(tool_name=tool_call["name"], args=args))
        
        tool_call_output = self.create_function_call_output_item(tool_call["call_id"], result)
        messages.append(tool_call_output)
        return ResponsesAgentStreamEvent(type="response.output_item.done", item=tool_call_output)
    
    def call_and_run_tools(
        self,
        messages: list[dict[str, Any]],
    ) -> Generator[ResponsesAgentStreamEvent, None, None]:
        """Main agent loop: call LLM, execute tools, iterate until completion"""
        for _ in range(self.max_iter):
            last_msg = messages[-1]
            
            if last_msg.get("role", None) == "assistant":
                # LLM provided final answer
                return
            elif last_msg.get("type", None) == "function_call":
                # Execute tool
                yield self.handle_tool_call(last_msg, messages)
            else:
                # Call LLM
                yield from output_to_responses_items_stream(
                    chunks=self.call_llm(messages), aggregator=messages
                )
        
        # Max iterations reached
        yield ResponsesAgentStreamEvent(
            type="response.output_item.done",
            item=self.create_text_output_item("Max iterations reached.", str(uuid4())),
        )
    
    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        """Non-streaming prediction"""
        outputs = [
            event.item
            for event in self.predict_stream(request)
            if event.type == "response.output_item.done"
        ]
        return ResponsesAgentResponse(output=outputs, custom_outputs=request.custom_inputs)
    
    def predict_stream(
        self, request: ResponsesAgentRequest
    ) -> Generator[ResponsesAgentStreamEvent, None, None]:
        """Streaming prediction with prompt from registry"""
        # Convert input to messages
        if request.input and hasattr(request.input[0], 'model_dump'):
            messages = to_chat_completions_input([i.model_dump() for i in request.input])
        else:
            messages = to_chat_completions_input(request.input)
        
        # Use prompt from registry
        if self._system_prompt:
            messages.insert(0, {"role": "system", "content": self._system_prompt})
        
        yield from self.call_and_run_tools(messages=messages)


# Simple calculator tool for demonstration
def calculator(operation: str, x: float, y: float) -> float:
    """
    Perform basic math operations.
    
    Args:
        operation: One of 'add', 'subtract', 'multiply', 'divide'
        x: First number
        y: Second number
    
    Returns:
        Result of operation
    """
    ops = {
        'add': lambda a, b: a + b,
        'subtract': lambda a, b: a - b,
        'multiply': lambda a, b: a * b,
        'divide': lambda a, b: a / b if b != 0 else "Error: Division by zero"
    }
    return ops.get(operation, lambda a, b: "Error: Unknown operation")(x, y)


# Define calculator tool spec
CALCULATOR_TOOL = ToolInfo(
    name="calculator",
    spec={
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Perform basic math operations: add, subtract, multiply, divide",
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["add", "subtract", "multiply", "divide"],
                        "description": "The math operation to perform"
                    },
                    "x": {
                        "type": "number",
                        "description": "First number"
                    },
                    "y": {
                        "type": "number",
                        "description": "Second number"
                    }
                },
                "required": ["operation", "x", "y"]
            }
        }
    },
    exec_fn=calculator
)

print("✅ PromptOptimizedAgent class defined")
print("✅ Calculator tool configured")

# COMMAND ----------

# DBTITLE 1,Initialize Prompt in Registry
# Check if prompt exists, create if not
from mlflow import MlflowClient

client = MlflowClient()

try:
    # Try to find existing prompt and version
    prompt_versions = client.search_prompt_versions(name=f"{PROMPT_NAME}")
    latest_version = sorted(prompt_versions.prompt_versions, key=lambda pv: pv.version, reverse=True)[0].version
    # Try to get existing prompt
    existing_prompt = mlflow.genai.load_prompt(f"prompts:/{PROMPT_NAME}/{latest_version}")
    print(f"✅ Found existing prompt: {PROMPT_NAME}")
    prompt_text = existing_prompt.template if hasattr(existing_prompt, 'template') else str(existing_prompt)
    print(f"   Content: {prompt_text[:100]}...")
    
except Exception:
    # Prompt doesn't exist, create seed prompt
    print(f"Creating new prompt: {PROMPT_NAME}")
    
    seed_prompt = """You are a helpful AI assistant that can answer questions and use tools when needed.

When a user asks a mathematical question:
1. Use the calculator tool for all arithmetic operations
2. Show your work clearly
3. Provide the final answer in a friendly way

Be concise but accurate."""
    
    # Register the initial prompt
    mlflow.genai.register_prompt(
        name=PROMPT_NAME,
        template=seed_prompt
    )
    
    print(f"✅ Created prompt: {PROMPT_NAME}")
    print(f"   Content: {seed_prompt[:100]}...")

# COMMAND ----------

# DBTITLE 1,Define Training Data for GEPA
# Training examples showing desired agent behavior
# Format: The 'inputs' dict must have keys matching predict_fn parameter names
# Since predict_fn has parameter 'inputs', we need: {"inputs": {"inputs": {...}}}

training_data = [
    # Math calculations - should use calculator tool
    {
        "inputs": {
            "inputs": {"messages": [{"role": "user", "content": "What is 15 plus 27?"}]}
        }, 
        "expectations": {"expected_response": "42"}
    },
    {
        "inputs": {
            "inputs": {"messages": [{"role": "user", "content": "Calculate 144 divided by 12"}]}
        }, 
        "expectations": {"expected_response": "12"}
    },
    {
        "inputs": {
            "inputs": {"messages": [{"role": "user", "content": "What's 8 times 9?"}]}
        }, 
        "expectations": {"expected_response": "72"}
    },
    
    # Multi-step problems - should use calculator multiple times
    {
        "inputs": {
            "inputs": {"messages": [{"role": "user", "content": "If I have 100 dollars and spend 37, then earn 50, how much do I have?"}]}
        }, 
        "expectations": {"expected_response": "113 dollars"}
    },
    
    # General questions (should not use calculator)
    {
        "inputs": {
            "inputs": {"messages": [{"role": "user", "content": "What is the capital of France?"}]}
        }, 
        "expectations": {"expected_response": "Paris"}
    },
    {
        "inputs": {
            "inputs": {"messages": [{"role": "user", "content": "Explain what AI is in simple terms"}]}
        }, 
        "expectations": {"expected_response": "AI is technology that enables computers to perform tasks that typically require human intelligence, such as learning, reasoning, and problem-solving."}
    },
    
    # Edge cases - should handle decimals correctly
    {
        "inputs": {
            "inputs": {"messages": [{"role": "user", "content": "What's 25.5 times 4?"}]}
        }, 
        "expectations": {"expected_response": "102"}
    },
    {
        "inputs": {
            "inputs": {"messages": [{"role": "user", "content": "Subtract 99 from 1000"}]}
        }, 
        "expectations": {"expected_response": "901"}
    },
]

print(f"✅ Training data defined:")
print(f"   {len(training_data)} training examples")
print(f"   Format: inputs dict with 'inputs' key matching predict_fn parameter")

# COMMAND ----------

# DBTITLE 1,Run GEPA Prompt Optimization
# Define predict function that GEPA will use to evaluate prompts
def predict_fn(inputs: dict, **kwargs) -> str:
    """
    Prediction function for GEPA optimization.
    GEPA will pass the prompt being tested via kwargs.
    """
    messages = inputs["messages"]
    
    # GEPA passes the prompt being tested in kwargs
    prompt_to_use = kwargs.get(PROMPT_NAME)
    
    if prompt_to_use:
        system_prompt = prompt_to_use.template if hasattr(prompt_to_use, 'template') else str(prompt_to_use)
    else:
        # Fallback: load latest from registry
        prompt_versions = client.search_prompt_versions(name=f"{PROMPT_NAME}")
        if prompt_versions.prompt_versions:
            latest_version = sorted(prompt_versions.prompt_versions, key=lambda pv: pv.version, reverse=True)[0].version
            loaded_prompt = mlflow.genai.load_prompt(f"prompts:/{PROMPT_NAME}/{latest_version}")
        else:
            loaded_prompt = mlflow.genai.load_prompt(f"prompts:/{PROMPT_NAME}/1")
        system_prompt = loaded_prompt.template if hasattr(loaded_prompt, 'template') else str(loaded_prompt)
    
    # Call LLM with the prompt
    workspace_client = WorkspaceClient()
    model_serving_client = workspace_client.serving_endpoints.get_open_ai_client()
    
    chat_messages = [{"role": "system", "content": system_prompt}] + messages
    tools_spec = [CALCULATOR_TOOL.spec] if CALCULATOR_TOOL else None
    
    response = model_serving_client.chat.completions.create(
        model=LLM_ENDPOINT_NAME,
        messages=chat_messages,
        tools=tools_spec,
        temperature=0.1
    )
    
    output_text = response.choices[0].message.content or ""
    
    # Handle tool calls
    if response.choices[0].message.tool_calls:
        for tool_call in response.choices[0].message.tool_calls:
            tool_args = json.loads(tool_call.function.arguments)
            if tool_call.function.name == "calculator":
                result = CALCULATOR_TOOL.exec_fn(**tool_args)
                output_text += f" {result}"
    
    return output_text


# Run GEPA optimization
print("🚀 Starting GEPA optimization...")
print("   This will test different prompt variations to find the best one\n")

# Configure logging
import logging
logging.getLogger("py4j").setLevel(logging.ERROR)
logging.getLogger("databricks.sdk.config").setLevel(logging.ERROR)
warnings.filterwarnings('ignore', message='.*must be real number, not str.*')

try:
    optimization_results = optimize_prompts(
        predict_fn=predict_fn,
        train_data=training_data,
        prompt_uris=[PROMPT_NAME],
        optimizer=GepaPromptOptimizer(
            reflection_model=f"databricks:/{LLM_ENDPOINT_NAME}",
            max_metric_calls=50
        ),
        scorers=[
            Correctness(model=f"databricks:/{LLM_ENDPOINT_NAME}")
        ]
    )
    
    print("\n" + "="*60)
    print("✅ GEPA Optimization Complete!")
    print("="*60)
    
    # Display optimized prompt
    if hasattr(optimization_results, 'optimized_prompts') and optimization_results.optimized_prompts:
        optimized_prompt_version = optimization_results.optimized_prompts[0]
        prompt_name = optimized_prompt_version.name
        prompt_version = optimized_prompt_version.version
        optimized_prompt_uri = f"prompts:/{prompt_name}/{prompt_version}"
        
        print(f"\nOptimized prompt: {optimized_prompt_uri}")
        
        # Load and display the optimized prompt
        optimized_prompt = mlflow.genai.load_prompt(optimized_prompt_uri)
        prompt_text = optimized_prompt.template if hasattr(optimized_prompt, 'template') else str(optimized_prompt)
        print(f"\nOptimized prompt text:\n{prompt_text[:500]}...")
    else:
        print("\n⚠️ No optimized prompts returned")
    
except Exception as e:
    print(f"\n❌ Optimization failed: {e}")
    import traceback
    traceback.print_exc()
    print("\nThe agent will still work with the seed prompt.")

# COMMAND ----------

# DBTITLE 1,Production Usage Example
# MAGIC %md
# MAGIC ## Using the Optimized Agent in Production
# MAGIC 
# MAGIC **Two Usage Patterns:**
# MAGIC 
# MAGIC 1. **Optimization Time** (above cells):
# MAGIC    - Use `mlflow.genai.optimize_prompts()` with training data
# MAGIC    - Creates new optimized prompt versions in registry
# MAGIC    - Happens offline, not during inference
# MAGIC 
# MAGIC 2. **Production Time** (this cell):
# MAGIC    - Create agent pointing to specific prompt version
# MAGIC    - Use `/Champion` for best validated prompt
# MAGIC    - Use `/latest` for most recent optimization
# MAGIC    - Agent loads prompt once at initialization (cached)
# MAGIC    - Call `reload_prompt()` to pick up changes without redeployment

# COMMAND ----------

# DBTITLE 1,Test Optimized Agent
# Get the latest prompt version
prompt_versions = client.search_prompt_versions(name=f"{PROMPT_NAME}")
if prompt_versions.prompt_versions:
    latest_version = sorted(prompt_versions.prompt_versions, key=lambda pv: pv.version, reverse=True)[0].version
    production_prompt_uri = f"prompts:/{PROMPT_NAME}/{latest_version}"
else:
    production_prompt_uri = f"prompts:/{PROMPT_NAME}/1"

print(f"Using prompt: {production_prompt_uri}\n")

# Create agent with optimized prompt
production_agent = PromptOptimizedAgent(
    llm_endpoint=LLM_ENDPOINT_NAME,
    prompt_uri=production_prompt_uri,
    tools=[CALCULATOR_TOOL],
    max_iter=10
)

# Test agent
print("💬 User: What is 456 plus 789?")

test_request = ResponsesAgentRequest(
    input=[{"role": "user", "content": "What is 456 plus 789?"}]
)
response = production_agent.predict(test_request)

# Print response
for item in response.output:
    item_dict = item.model_dump() if hasattr(item, 'model_dump') else item
    if item_dict.get("type") == "message":
        content = item_dict.get('content', '')
        if isinstance(content, list):
            content = ''.join([p.get('text', '') for p in content if p.get('type') == 'output_text'])
        print(f"🤖 Assistant: {content}")
    elif item_dict.get("type") == "function_call":
        print(f"🔧 Tool: {item_dict.get('name')}({item_dict.get('arguments')})")

print("\n✅ Agent ready! You can reload prompts anytime with: agent.reload_prompt()")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC 
# MAGIC ### What We Built:
# MAGIC - ✅ **PromptOptimizedAgent**: ResponsesAgent with prompt registry integration
# MAGIC - ✅ **Prompt Registry**: Centralized prompt management in MLflow
# MAGIC - ✅ **GEPA Optimization**: Automated prompt improvement using training data
# MAGIC - ✅ **Tool Integration**: Calculator tool with proper OpenAI format
# MAGIC - ✅ **Production Pattern**: Runtime prompt updates without redeployment
# MAGIC 
# MAGIC ### Key Concepts:
# MAGIC 
# MAGIC **Optimization vs Production:**
# MAGIC - **Optimization** (Cells 4-5): Run `optimize_prompts()` with training data to improve prompts offline
# MAGIC - **Production** (Cell 6): Create agent with `prompt_uri` pointing to optimized version
# MAGIC 
# MAGIC **Prompt Versioning:**
# MAGIC - `/latest` - Most recent version
# MAGIC - `/Champion` - Best validated version (set manually or via evaluation)
# MAGIC - `/1`, `/2`, etc. - Specific version numbers
# MAGIC 
# MAGIC **Runtime Updates:**
# MAGIC - Call `agent.reload_prompt()` to pick up new prompt versions
# MAGIC - No need to redeploy the model serving endpoint
# MAGIC - Enables A/B testing and rapid iteration
# MAGIC 
# MAGIC ### Next Steps:
# MAGIC 1. Add more training examples to improve optimization
# MAGIC 2. Use real tools (UC functions, Vector Search) instead of calculator
# MAGIC 3. Deploy to Model Serving with `mlflow.models.set_model(production_agent)`
# MAGIC 4. Set up automated prompt evaluation pipeline
# MAGIC 5. Use Champion/Challenger pattern for prompt A/B testing

