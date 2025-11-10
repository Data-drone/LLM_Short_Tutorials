# Databricks notebook source
# MAGIC %md
# MAGIC # ResponsesAgent with Unity Catalog Tools
# MAGIC
# MAGIC This notebook demonstrates the Databricks-native ResponsesAgent pattern with Unity Catalog integration.
# MAGIC
# MAGIC ## Key Features:
# MAGIC - **Unity Catalog Functions**: Automatic tool discovery from UC
# MAGIC - **Vector Search Tools**: One-liner retrieval tool integration
# MAGIC - **OpenAI Responses API**: Industry-standard format
# MAGIC - **Production Ready**: Built-in error handling and tracing
# MAGIC
# MAGIC ## Architecture:
# MAGIC 1. Define tools (UC functions + Vector Search)
# MAGIC 2. Create ResponsesAgent with tool execution logic
# MAGIC 3. Log to MLflow for deployment

# COMMAND ----------

# DBTITLE 1,Install Required Libraries
# MAGIC %pip install -U mlflow>=3.0 databricks-langchain openai databricks-sdk databricks-openai unitycatalog-ai databricks_mcp nest_asyncio pydantic>=2.0
# MAGIC %restart_python

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration
# MAGIC
# MAGIC Define your LLM endpoint and system prompt

# COMMAND ----------

# DBTITLE 1,Import Libraries and Configure
import json
from typing import Any, Callable, Generator, Optional
from uuid import uuid4
import warnings

# Enable nested event loops for Databricks notebooks
import nest_asyncio
nest_asyncio.apply()

import mlflow
import openai
from databricks.sdk import WorkspaceClient
from databricks_openai import UCFunctionToolkit, VectorSearchRetrieverTool
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
from unitycatalog.ai.core.base import get_uc_function_client

# Enable MLflow autologging
mlflow.openai.autolog()

# COMMAND ----------

# DBTITLE 1,Agent Configuration
# Define your LLM endpoint
LLM_ENDPOINT_NAME = "databricks-gpt-oss-120b"

# System prompt for the agent (optional)
SYSTEM_PROMPT = """You are a helpful AI assistant with access to tools.
Use the available tools when needed to answer questions accurately."""

# COMMAND ----------

# MAGIC %md
# MAGIC ## Tool Setup
# MAGIC
# MAGIC Define tools for your agent to retrieve data or take actions beyond text generation.
# MAGIC
# MAGIC **Tool Types:**
# MAGIC - **Unity Catalog Functions**: Python/SQL UDFs registered in UC
# MAGIC - **Vector Search Tools**: Retrieval from vector search indexes
# MAGIC
# MAGIC See [documentation](https://docs.databricks.com/generative-ai/agent-framework/agent-tool.html) for more tool examples

# COMMAND ----------

# DBTITLE 1,Define Tool Info Structure
class ToolInfo(BaseModel):
    """
    Class representing a tool for the agent.
    
    Attributes:
        name (str): The name of the tool
        spec (dict): JSON description of the tool (OpenAI Responses format)
        exec_fn (Callable): Function that implements the tool logic
    """
    name: str
    spec: dict
    exec_fn: Callable
    
    class Config:
        arbitrary_types_allowed = True


def create_tool_info(tool_spec: dict, exec_fn_param: Optional[Callable] = None) -> ToolInfo:
    """
    Create a ToolInfo object from a tool specification.
    
    For UC functions, automatically creates an execution wrapper.
    For other tools, uses the provided execution function.
    """
    # Clean up the spec
    tool_spec["function"].pop("strict", None)
    tool_name = tool_spec["function"]["name"]
    udf_name = tool_name.replace("__", ".")
    
    # Define a wrapper for UC function execution
    def exec_fn(**kwargs):
        function_result = uc_function_client.execute_function(udf_name, kwargs)
        if function_result.error is not None:
            return function_result.error
        else:
            return function_result.value
    
    return ToolInfo(
        name=tool_name,
        spec=tool_spec,
        exec_fn=exec_fn_param or exec_fn
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Unity Catalog Function Tools
# MAGIC
# MAGIC Add Unity Catalog functions as agent tools for structured operations

# COMMAND ----------

# DBTITLE 1,Configure UC Function Tools
TOOL_INFOS = []

# Add Unity Catalog function names here
# Example: ["catalog.schema.function_name"]
UC_TOOL_NAMES = []

# Initialize UC toolkit and function client
if UC_TOOL_NAMES:
    uc_toolkit = UCFunctionToolkit(function_names=UC_TOOL_NAMES)
    uc_function_client = get_uc_function_client()
    
    for tool_spec in uc_toolkit.tools:
        TOOL_INFOS.append(create_tool_info(tool_spec))
    
    print(f"✅ Added {len(UC_TOOL_NAMES)} UC function tools")
else:
    uc_function_client = None
    print("⚠️ No UC function tools configured")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Vector Search Retrieval Tools
# MAGIC
# MAGIC Add Databricks Vector Search indexes as retrieval tools for RAG capabilities.
# MAGIC
# MAGIC See [Vector Search Tools Documentation](https://docs.databricks.com/generative-ai/agent-framework/unstructured-retrieval-tools.html) for details

# COMMAND ----------

# DBTITLE 1,Configure Vector Search Tools
VECTOR_SEARCH_TOOLS = []

# Add your vector search indexes here
# Update index_name with your actual index
# VECTOR_SEARCH_TOOLS.append(
#     VectorSearchRetrieverTool(
#         index_name="atrivedi.kaglemovielens.movies_metadata_credits_consolidated_vs_index",
#         tool_description="""Search through movie metadata and credits information.
#         Use this tool to find information about movies, cast, crew, and production details."""
#     )
# )

# Add vector search tools to the tool list
for vs_tool in VECTOR_SEARCH_TOOLS:
    TOOL_INFOS.append(create_tool_info(vs_tool.tool, vs_tool.execute))

print(f"✅ Added {len(VECTOR_SEARCH_TOOLS)} vector search tools")
print(f"📊 Total tools available: {len(TOOL_INFOS)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## MCP (Model Context Protocol) Server Tools
# MAGIC
# MAGIC Connect to external MCP servers to access third-party tools and integrations.
# MAGIC
# MAGIC **What are MCP Servers?**
# MAGIC - MCP enables standardized connections between LLMs and external data sources/tools
# MAGIC - Can access external services (GitHub, Slack, databases, etc.)
# MAGIC - Databricks provides managed proxies for secure authentication
# MAGIC
# MAGIC **Setup Options:**
# MAGIC 1. **Marketplace MCP Servers**: Pre-configured servers from Databricks Marketplace
# MAGIC 2. **Custom MCP Servers**: Self-hosted or third-party MCP servers
# MAGIC
# MAGIC **Note**: MCP clients use async operations. `nest_asyncio` is required for Databricks notebooks (already imported above).
# MAGIC
# MAGIC See [MCP documentation](https://docs.databricks.com/en/generative-ai/mcp/external-mcp.html) for setup instructions

# COMMAND ----------

# DBTITLE 1,Configure MCP Server Tools (Optional)
# Add MCP server connections here
# Format: Connection name that you created in Unity Catalog
MCP_CONNECTIONS = []

# Example: Uncomment to use GitHub MCP server
# MCP_CONNECTIONS = ["github_connection"]

# Initialize MCP tools if connections are configured
if MCP_CONNECTIONS:
    try:
        from databricks_mcp import DatabricksMCPClient
        
        workspace_client = WorkspaceClient()
        host = workspace_client.config.host
        
        for connection_name in MCP_CONNECTIONS:
            # Build the proxy URL for the external MCP server
            mcp_server_url = f"{host}/api/2.0/mcp/external/{connection_name}"
            
            print(f"Connecting to MCP server: {connection_name}")
            
            # Create MCP client
            mcp_client = DatabricksMCPClient(
                server_url=mcp_server_url,
                workspace_client=workspace_client
            )
            
            # List available tools from the MCP server
            mcp_tools = mcp_client.list_tools()
            print(f"  Found {len(mcp_tools)} tools from {connection_name}")
            
            # Convert MCP tools to ToolInfo format
            for tool in mcp_tools:
                # Create tool specification in OpenAI format
                tool_spec = {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description or f"Tool from {connection_name}",
                        "parameters": tool.inputSchema or {"type": "object", "properties": {}}
                    }
                }
                
                # Create execution function for this MCP tool
                def create_mcp_exec_fn(client, tool_name):
                    """Factory to create tool execution function"""
                    def exec_fn(**kwargs):
                        response = client.call_tool(tool_name, kwargs)
                        # Extract text from response
                        if response and response.content:
                            return response.content[0].text
                        return str(response) if response else "No response from tool"
                    return exec_fn
                
                # Add to tool list
                TOOL_INFOS.append(
                    create_tool_info(
                        tool_spec,
                        create_mcp_exec_fn(mcp_client, tool.name)
                    )
                )
        
        print(f"✅ Added {len(MCP_CONNECTIONS)} MCP server(s)")
        
    except ImportError:
        print("⚠️ databricks_mcp not installed. Run: %pip install databricks_mcp")
        print("Skipping MCP tools...")
    except Exception as e:
        print(f"⚠️ Error loading MCP tools: {str(e)}")
        print("Continuing without MCP tools...")
else:
    print("⚠️ No MCP connections configured. Skipping MCP tools.")
    print("To add MCP tools:")
    print("  1. Install an MCP server from Marketplace or create a Unity Catalog connection")
    print("  2. Add the connection name to MCP_CONNECTIONS list")
    print("  3. See: https://docs.databricks.com/en/generative-ai/mcp/external-mcp.html")

print(f"\n📊 Total tools available: {len(TOOL_INFOS)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Managed MCP Servers (Genie and System Tools)
# MAGIC
# MAGIC Connect to Databricks-managed MCP servers for native integrations.
# MAGIC
# MAGIC **Managed MCP Server Types:**
# MAGIC - **Genie Spaces**: Natural language queries over your data tables
# MAGIC - **System AI Tools**: Built-in Databricks AI capabilities
# MAGIC
# MAGIC **How to Get Genie Space ID:**
# MAGIC 1. Open your Genie Room in Databricks workspace
# MAGIC 2. Click the Settings tab (gear icon)
# MAGIC 3. Copy the Space ID (UUID format)
# MAGIC
# MAGIC **Note**: `nest_asyncio` handles async event loops (already imported above)

# COMMAND ----------

# DBTITLE 1,Configure Managed MCP Servers (Optional)
# Add Genie Space IDs here
# Format: List of Genie Space IDs (UUIDs)
GENIE_SPACE_IDS = []

# Example: Uncomment to use your Genie room
# GENIE_SPACE_IDS = ["01234567-89ab-cdef-0123-456789abcdef"]

# Add other managed MCP server URLs (optional)
# System AI tools are available by default
OTHER_MANAGED_MCP_URLS = []

# Example: Include system AI tools
# OTHER_MANAGED_MCP_URLS = [f"{workspace_client.config.host}/api/2.0/mcp/functions/system/ai"]

# Initialize Managed MCP tools
managed_mcp_count = 0

if GENIE_SPACE_IDS or OTHER_MANAGED_MCP_URLS:
    try:
        from databricks_mcp import DatabricksMCPClient
        
        workspace_client = WorkspaceClient()
        host = workspace_client.config.host
        
        # Process Genie spaces
        for genie_space_id in GENIE_SPACE_IDS:
            # Genie uses managed MCP server URL pattern
            genie_mcp_url = f"{host}/api/2.0/mcp/genie/{genie_space_id}"
            
            print(f"Connecting to Genie Space: {genie_space_id[:8]}...")
            
            try:
                # Create MCP client for Genie
                genie_client = DatabricksMCPClient(
                    server_url=genie_mcp_url,
                    workspace_client=workspace_client
                )
                
                # List available tools from Genie
                genie_tools = genie_client.list_tools()
                print(f"  Found {len(genie_tools)} tools from Genie space")
                
                # Convert Genie tools to ToolInfo format
                for tool in genie_tools:
                    # Create tool specification in OpenAI format
                    tool_spec = {
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "description": tool.description or f"Genie tool for data analysis",
                            "parameters": tool.inputSchema or {"type": "object", "properties": {}}
                        }
                    }
                    
                    # Create execution function for this Genie tool
                    def create_genie_exec_fn(client, tool_name):
                        """Factory to create tool execution function"""
                        def exec_fn(**kwargs):
                            response = client.call_tool(tool_name, kwargs)
                            # Extract text from response
                            if response and response.content:
                                return response.content[0].text
                            return str(response) if response else "No response from tool"
                        return exec_fn
                    
                    # Add to tool list
                    TOOL_INFOS.append(
                        create_tool_info(
                            tool_spec,
                            create_genie_exec_fn(genie_client, tool.name)
                        )
                    )
                
                managed_mcp_count += 1
                
            except Exception as e:
                print(f"  ⚠️ Error connecting to Genie space: {str(e)}")
                continue
        
        # Process other managed MCP servers
        for managed_url in OTHER_MANAGED_MCP_URLS:
            server_name = managed_url.split('/')[-1]
            print(f"Connecting to managed MCP server: {server_name}")
            
            try:
                # Create MCP client
                managed_client = DatabricksMCPClient(
                    server_url=managed_url,
                    workspace_client=workspace_client
                )
                
                # List available tools
                managed_tools = managed_client.list_tools()
                print(f"  Found {len(managed_tools)} tools from {server_name}")
                
                # Convert tools to ToolInfo format
                for tool in managed_tools:
                    tool_spec = {
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "description": tool.description or f"Tool from {server_name}",
                            "parameters": tool.inputSchema or {"type": "object", "properties": {}}
                        }
                    }
                    
                    def create_managed_exec_fn(client, tool_name):
                        """Factory to create tool execution function"""
                        def exec_fn(**kwargs):
                            response = client.call_tool(tool_name, kwargs)
                            if response and response.content:
                                return response.content[0].text
                            return str(response) if response else "No response from tool"
                        return exec_fn
                    
                    TOOL_INFOS.append(
                        create_tool_info(
                            tool_spec,
                            create_managed_exec_fn(managed_client, tool.name)
                        )
                    )
                
                managed_mcp_count += 1
                
            except Exception as e:
                print(f"  ⚠️ Error connecting to {server_name}: {str(e)}")
                continue
        
        if managed_mcp_count > 0:
            print(f"✅ Added {managed_mcp_count} managed MCP server(s)")
        
    except ImportError:
        print("⚠️ databricks_mcp not installed. Run: %pip install databricks_mcp")
        print("Skipping managed MCP tools...")
    except Exception as e:
        print(f"⚠️ Error loading managed MCP tools: {str(e)}")
        print("Continuing without managed MCP tools...")
else:
    print("⚠️ No managed MCP servers configured. Skipping.")
    print("To add Genie tools:")
    print("  1. Create a Genie room: Workspace > Genie > + New")
    print("  2. Open room > Settings tab > Copy Space ID")
    print("  3. Add Space ID to GENIE_SPACE_IDS list")

print(f"\n📊 Total tools available: {len(TOOL_INFOS)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## ResponsesAgent Implementation
# MAGIC
# MAGIC Define the agent class with tool-calling capabilities

# COMMAND ----------

# DBTITLE 1,Define Tool-Calling Agent
class ToolCallingAgent(ResponsesAgent):
    """
    Tool-calling Agent using the OpenAI Responses API format.
    
    This agent can:
    - Call multiple tools in sequence
    - Handle streaming responses
    - Automatically trace tool calls with MLflow
    - Iterate up to max_iter times to solve complex queries
    """
    
    def __init__(self, llm_endpoint: str, tools: list[ToolInfo], max_iter: int = 10):
        """Initialize the ToolCallingAgent with tools."""
        self.llm_endpoint = llm_endpoint
        self.max_iter = max_iter
        self.workspace_client = WorkspaceClient()
        self.model_serving_client: OpenAI = (
            self.workspace_client.serving_endpoints.get_open_ai_client()
        )
        self._tools_dict = {tool.name: tool for tool in tools}
    
    def get_tool_specs(self) -> list[dict]:
        """Returns tool specifications in the format OpenAI expects."""
        return [tool_info.spec for tool_info in self._tools_dict.values()]
    
    @mlflow.trace(span_type=SpanType.TOOL)
    def execute_tool(self, tool_name: str, args: dict) -> Any:
        """Executes the specified tool with the given arguments."""
        return self._tools_dict[tool_name].exec_fn(**args)
    
    def call_llm(self, messages: list[dict[str, Any]]) -> Generator[dict[str, Any], None, None]:
        """Call the LLM with streaming enabled."""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="PydanticSerializationUnexpectedValue")
            for chunk in self.model_serving_client.chat.completions.create(
                model=self.llm_endpoint,
                messages=to_chat_completions_input(messages),
                tools=self.get_tool_specs(),
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
        """
        Execute tool call, add result to message history, and return stream event.
        """
        args = json.loads(tool_call["arguments"])
        result = str(self.execute_tool(tool_name=tool_call["name"], args=args))
        
        tool_call_output = self.create_function_call_output_item(tool_call["call_id"], result)
        messages.append(tool_call_output)
        return ResponsesAgentStreamEvent(type="response.output_item.done", item=tool_call_output)
    
    def call_and_run_tools(
        self,
        messages: list[dict[str, Any]],
    ) -> Generator[ResponsesAgentStreamEvent, None, None]:
        """
        Main agent loop: call LLM, execute tools, iterate until completion.
        """
        for _ in range(self.max_iter):
            last_msg = messages[-1]
            
            # Check termination conditions
            if last_msg.get("role", None) == "assistant":
                # LLM provided final answer
                return
            elif last_msg.get("type", None) == "function_call":
                # Execute the tool
                yield self.handle_tool_call(last_msg, messages)
            else:
                # Call LLM
                yield from output_to_responses_items_stream(
                    chunks=self.call_llm(messages), aggregator=messages
                )
        
        # Max iterations reached
        yield ResponsesAgentStreamEvent(
            type="response.output_item.done",
            item=self.create_text_output_item("Max iterations reached. Stopping.", str(uuid4())),
        )
    
    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        """Non-streaming prediction."""
        outputs = [
            event.item
            for event in self.predict_stream(request)
            if event.type == "response.output_item.done"
        ]
        return ResponsesAgentResponse(output=outputs, custom_outputs=request.custom_inputs)
    
    def predict_stream(
        self, request: ResponsesAgentRequest
    ) -> Generator[ResponsesAgentStreamEvent, None, None]:
        """Streaming prediction."""
        # Handle both dict and Pydantic model inputs
        if request.input and hasattr(request.input[0], 'model_dump'):
            messages = to_chat_completions_input([i.model_dump() for i in request.input])
        else:
            messages = to_chat_completions_input(request.input)
        
        if SYSTEM_PROMPT:
            messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})
        yield from self.call_and_run_tools(messages=messages)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create and Test Agent
# MAGIC
# MAGIC Instantiate the agent and run a test query.
# MAGIC
# MAGIC **Note on MLflow 3.x Input Format:**
# MAGIC - Input uses standard message dict format: `[{"role": "user", "content": "..."}]`
# MAGIC - No need for `InputItem` wrapper classes

# COMMAND ----------

# DBTITLE 1,Create Agent
# Create the agent
AGENT = ToolCallingAgent(
    llm_endpoint=LLM_ENDPOINT_NAME,
    tools=TOOL_INFOS,
    max_iter=10
)

print("✅ Agent created successfully!")
print(f"  - LLM Endpoint: {LLM_ENDPOINT_NAME}")
print(f"  - Number of Tools: {len(TOOL_INFOS)}")
print(f"  - Max Iterations: {AGENT.max_iter}")

# COMMAND ----------

# DBTITLE 1,Test Agent - Basic Response
# Test basic agent functionality (works with or without tools)
# In MLflow 3.x, input is a list of message dicts

def print_agent_response(response):
    """Helper function to print agent response output items"""
    for item in response.output:
        # Convert Pydantic model to dict if needed
        if isinstance(item, dict):
            item_dict = item
        elif hasattr(item, 'model_dump'):
            item_dict = item.model_dump()
        elif hasattr(item, 'to_dict'):
            item_dict = item.to_dict()
        else:
            item_dict = vars(item)
        
        if item_dict.get("type") == "message":
            content = item_dict.get('content', '')
            # Handle structured content (list of content items)
            if isinstance(content, list):
                # Extract text from output_text items
                text_parts = [
                    part.get('text', '') 
                    for part in content 
                    if isinstance(part, dict) and part.get('type') == 'output_text'
                ]
                content = ''.join(text_parts)
            print(f"💬 Assistant: {content}")
        elif item_dict.get("type") == "function_call":
            print(f"🔧 Tool Call: {item_dict.get('name')}({item_dict.get('arguments')})")

print("Testing agent with basic query...")
print("=" * 60)

basic_request = ResponsesAgentRequest(
    input=[{"role": "user", "content": "What is 2+2? Just answer briefly."}]
)

response = AGENT.predict(basic_request)
print_agent_response(response)

print("\n✅ Basic agent test completed!")

# COMMAND ----------

# DBTITLE 1,Test Agent - With Tools (Optional)
# Test agent with a query that should use tools (if configured)
# Note: Update the query based on your configured tools

if len(TOOL_INFOS) > 0:
    print("\nTesting agent with tool-requiring query...")
    print("=" * 60)
    
    tool_request = ResponsesAgentRequest(
        input=[{"role": "user", "content": "Search for information about action movies"}]
    )
    
    response = AGENT.predict(tool_request)
    
    # Check if tools were used
    tool_used = False
    for item in response.output:
        if isinstance(item, dict):
            item_dict = item
        elif hasattr(item, 'model_dump'):
            item_dict = item.model_dump()
        else:
            item_dict = vars(item)
        
        if item_dict.get("type") == "function_call":
            tool_used = True
            break
    
    # Print the response
    print_agent_response(response)
    
    if tool_used:
        print("\n✅ Agent successfully used tools!")
    else:
        print("\n⚠️ Agent did not use tools (may need to adjust query or tool descriptions)")
else:
    print("\n⚠️ No tools configured. Skipping tool-based test.")
    print("Add UC functions or vector search tools to test tool calling functionality.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Set MLflow Model
# MAGIC
# MAGIC Set the Model Object for MLflow log as code

# COMMAND ----------

# DBTITLE 1,Log Agent to MLflow
mlflow.models.set_model(AGENT)

print("✅ Agent setup for Model As Code Logging!")
print("\nNext steps:")
print("1. Use Deployment notebook to log agent and register to model registry")
print("2. Deploy to Databricks Model Serving")
print("3. Test with MLflow Agent Evaluations")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC ### What We Built:
# MAGIC - ✅ ResponsesAgent with Unity Catalog tool integration
# MAGIC - ✅ Vector Search retrieval capabilities
# MAGIC - ✅ External MCP server integration (GitHub, Slack, etc.)
# MAGIC - ✅ Managed MCP servers (Genie Spaces, System AI)
# MAGIC - ✅ Automatic tool discovery and execution
# MAGIC - ✅ MLflow tracing and logging
# MAGIC - ✅ Production-ready with error handling
# MAGIC
# MAGIC ### Key Configuration Points:
# MAGIC 1. **LLM_ENDPOINT_NAME** - Your model serving endpoint
# MAGIC 2. **UC_TOOL_NAMES** - List of UC function names (optional)
# MAGIC 3. **VECTOR_SEARCH_TOOLS** - List of vector search indexes (optional)
# MAGIC 4. **MCP_CONNECTIONS** - External MCP server connection names (optional)
# MAGIC 5. **GENIE_SPACE_IDS** - Genie room Space IDs for data analysis (optional)
# MAGIC 6. **OTHER_MANAGED_MCP_URLS** - Additional managed MCP servers (optional)
# MAGIC 7. **SYSTEM_PROMPT** - Agent behavior guidance
# MAGIC
# MAGIC ### Tool Options:
# MAGIC - **UC Functions**: Python/SQL UDFs for custom logic
# MAGIC - **Vector Search**: RAG capabilities for document retrieval
# MAGIC - **External MCP**: Third-party integrations (GitHub, Slack, databases, etc.)
# MAGIC - **Genie Spaces**: Natural language queries over your data tables
# MAGIC - **System MCP**: Built-in Databricks AI tools
# MAGIC
# MAGIC ### MCP Server Types:
# MAGIC - **External MCP**: `/api/2.0/mcp/external/{connection_name}` - Requires Unity Catalog connection
# MAGIC - **Genie MCP**: `/api/2.0/mcp/genie/{space_id}` - Direct Space ID
# MAGIC - **System MCP**: `/api/2.0/mcp/functions/system/ai` - Built-in tools
# MAGIC
# MAGIC ### Compare with LangGraph:
# MAGIC - **ResponsesAgent** (this notebook): Simpler, Databricks-native, fewer dependencies
# MAGIC - **LangGraph** (`00_basic_agent.py`): More flexible, visual debugging, framework-agnostic
