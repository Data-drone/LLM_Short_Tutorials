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
# MAGIC %pip install -U mlflow>=3.0 openai databricks-sdk pydantic>=2.0
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
class ToolInfo(BaseModel):
    """Tool specification for the agent"""
    name: str
    spec: dict
    exec_fn: Callable
    
    class Config:
        arbitrary_types_allowed = True


class PromptOptimizedAgent(ResponsesAgent):
    """
    ResponsesAgent that loads system prompt from MLflow Prompt Registry.
    
    This enables:
    - Centralized prompt management
    - Version control for prompts
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
            prompt_uri: MLflow prompt URI (e.g., 'prompts:/agent_system_prompt/1' or 'prompts:/agent_system_prompt/Champion')
            tools: List of tools available to agent
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
        print(f"   System prompt preview: {self._system_prompt[:100]}...")
    
    def _load_prompt(self) -> str:
        """Load prompt from MLflow registry"""
        try:
            prompt = mlflow.genai.load_prompt(self.prompt_uri)
            # The loaded prompt object has a 'template' attribute
            return prompt.template if hasattr(prompt, 'template') else str(prompt)
        except Exception as e:
            print(f"⚠️ Error loading prompt: {e}")
            return "You are a helpful AI assistant."
    
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
        """Execute specified tool with arguments"""
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
        """Execute tool call and return result"""
        args = json.loads(tool_call["arguments"])
        result = str(self.execute_tool(tool_name=tool_call["name"], args=args))
        
        tool_call_output = self.create_function_call_output_item(tool_call["call_id"], result)
        messages.append(tool_call_output)
        return ResponsesAgentStreamEvent(type="response.output_item.done", item=tool_call_output)
    
    def call_and_run_tools(
        self,
        messages: list[dict[str, Any]],
    ) -> Generator[ResponsesAgentStreamEvent, None, None]:
        """Main agent loop: call LLM, execute tools, iterate"""
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
        """Streaming prediction with system prompt from registry"""
        # Convert input to messages
        if request.input and hasattr(request.input[0], 'model_dump'):
            messages = to_chat_completions_input([i.model_dump() for i in request.input])
        else:
            messages = to_chat_completions_input(request.input)
        
        # Inject system prompt from registry
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
def predict_fn(inputs: dict, **kwargs) -> dict:
    """
    Prediction function for GEPA optimization.
    GEPA will pass the prompt being tested via kwargs.
    """
    messages = inputs["messages"]
    
    # GEPA passes the prompt name in kwargs
    # Load the specific prompt version being tested
    prompt_to_use = kwargs.get(PROMPT_NAME)
    
    if prompt_to_use:
        # GEPA is testing a specific prompt - use it directly as system message
        system_prompt = prompt_to_use.template if hasattr(prompt_to_use, 'template') else str(prompt_to_use)
    else:
        # Fallback: load from registry
        try:
            prompt_versions = client.search_prompt_versions(name=f"{PROMPT_NAME}")
            if prompt_versions.prompt_versions:
                latest_version = sorted(prompt_versions.prompt_versions, key=lambda pv: pv.version, reverse=True)[0].version
                prompt_uri = f"prompts:/{PROMPT_NAME}/{latest_version}"
            else:
                prompt_uri = f"prompts:/{PROMPT_NAME}/1"
            loaded_prompt = mlflow.genai.load_prompt(prompt_uri)
            system_prompt = loaded_prompt.template if hasattr(loaded_prompt, 'template') else str(loaded_prompt)
        except Exception:
            system_prompt = "You are a helpful AI assistant."
    
    # Create a simplified agent that uses the prompt directly
    # Instead of using PromptOptimizedAgent, use OpenAI client directly
    workspace_client = WorkspaceClient()
    model_serving_client = workspace_client.serving_endpoints.get_open_ai_client()
    
    # Prepare messages with the system prompt
    chat_messages = [{"role": "system", "content": system_prompt}] + messages
    
    # Get tools specs
    tools_spec = [CALCULATOR_TOOL.spec] if CALCULATOR_TOOL else None
    
    # Call LLM
    response = model_serving_client.chat.completions.create(
        model=LLM_ENDPOINT_NAME,
        messages=chat_messages,
        tools=tools_spec,
        temperature=0.1
    )
    
    # Extract response text
    output_text = response.choices[0].message.content or ""
    
    # Handle tool calls if present
    if response.choices[0].message.tool_calls:
        for tool_call in response.choices[0].message.tool_calls:
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments)
            if tool_name == "calculator":
                result = CALCULATOR_TOOL.exec_fn(**tool_args)
                output_text += f" (Calculated: {result})"
    
    return {"response": output_text}


# Run GEPA optimization
print("🚀 Starting GEPA optimization...")
print("   This will test different prompt variations to find the best one")
print("")

try:
    # For GEPA, pass just the prompt name (not full URI)
    # GEPA will create and test new versions automatically
    print(f"   Optimizing prompt: {PROMPT_NAME}")
    
    optimization_results = optimize_prompts(
        predict_fn=predict_fn,
        train_data=training_data,
        prompt_uris=[PROMPT_NAME],  # Just the prompt name, not full URI
        optimizer=GepaPromptOptimizer(
            reflection_model=f"databricks:/{LLM_ENDPOINT_NAME}",
            max_metric_calls=50  # Number of optimization iterations
        ),
        scorers=[
            Correctness(
                model=f"databricks:/{LLM_ENDPOINT_NAME}"
            )
        ]
    )
    
    print("\n" + "="*60)
    print("✅ GEPA Optimization Complete!")
    print("="*60)
    
    # Access optimized prompts from results
    if hasattr(optimization_results, 'optimized_prompts') and optimization_results.optimized_prompts:
        optimized_prompt_info = optimization_results.optimized_prompts[0]
        print(f"\nOptimized prompt URI: {optimized_prompt_info}")
        
        # Load and display the optimized prompt
        optimized_prompt = mlflow.genai.load_prompt(optimized_prompt_info)
        prompt_text = optimized_prompt.template if hasattr(optimized_prompt, 'template') else str(optimized_prompt)
        print(f"\nOptimized prompt:\n{prompt_text}")
    else:
        print("\n⚠️ No optimized prompts returned")
    
    # Display metrics if available
    if hasattr(optimization_results, 'metrics'):
        print(f"\nOptimization metrics: {optimization_results.metrics}")
    
except Exception as e:
    print(f"\n⚠️ Optimization error: {e}")
    print("This may happen if the model endpoint is busy or if there are API issues.")
    print("The agent will still work with the seed prompt.")

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
# Create agent with optimized prompt (or seed if optimization failed)
# Get the latest version dynamically

try:
    prompt_versions = client.search_prompt_versions(name=f"{PROMPT_NAME}")
    if prompt_versions.prompt_versions:
        latest_version = sorted(prompt_versions.prompt_versions, key=lambda pv: pv.version, reverse=True)[0].version
        production_prompt_uri = f"prompts:/{PROMPT_NAME}/{latest_version}"
    else:
        production_prompt_uri = f"prompts:/{PROMPT_NAME}/1"
except Exception:
    # Fallback if we can't query versions
    production_prompt_uri = f"prompts:/{PROMPT_NAME}/1"

print(f"Using prompt: {production_prompt_uri}")

production_agent = PromptOptimizedAgent(
    llm_endpoint=LLM_ENDPOINT_NAME,
    prompt_uri=production_prompt_uri,  # Use specific version
    tools=[CALCULATOR_TOOL],
    max_iter=10
)

print("\n" + "="*60)
print("Testing production agent with optimized prompt")
print("="*60)

# Test with math question
test_request = ResponsesAgentRequest(
    input=[{"role": "user", "content": "What is 456 plus 789?"}]
)

response = production_agent.predict(test_request)

# Print response
print("\n💬 User: What is 456 plus 789?")
for item in response.output:
    if isinstance(item, dict):
        item_dict = item
    elif hasattr(item, 'model_dump'):
        item_dict = item.model_dump()
    else:
        item_dict = vars(item)
    
    if item_dict.get("type") == "message":
        content = item_dict.get('content', '')
        if isinstance(content, list):
            text_parts = [
                part.get('text', '') 
                for part in content 
                if isinstance(part, dict) and part.get('type') == 'output_text'
            ]
            content = ''.join(text_parts)
        print(f"🤖 Assistant: {content}")
    elif item_dict.get("type") == "function_call":
        print(f"🔧 Tool Call: {item_dict.get('name')}({item_dict.get('arguments')})")

# Demonstrate runtime prompt reload
print("\n" + "="*60)
print("Prompt Reload Capability")
print("="*60)
print("In production, you can update the prompt without redeploying:")
print("1. Optimize new prompt version with GEPA")
print("2. Call agent.reload_prompt() to pick up changes")
print("3. No service restart needed!")

# Example: reload prompt (will be same unless new version created)
production_agent.reload_prompt()

print("\n✅ Production agent ready for deployment!")

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

