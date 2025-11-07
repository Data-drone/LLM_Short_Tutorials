# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Building and Logging A Parameterised Agent Model File Chain
# MAGIC
# MAGIC We will do a model file chain with more advanced configs \
# MAGIC We will also look at structured input \
# MAGIC We will also use langgraph

# COMMAND ----------

# DBTITLE 1,Parameterise Pip installs
mlflow_version = 'mlflow==2.16.2'
langchain_base_version = 'langchain'
langchain_community_version = 'langchain_community==0.2.13'

# COMMAND ----------

# DBTITLE 1,Run Pip Install
# MAGIC %pip install -U pydantic>2.0.0 {mlflow_version} {langchain_base_version} {langchain_community_version} langchain_core langgraph langchain-databricks
# MAGIC %restart_python

# COMMAND ----------

# DBTITLE 1,Config Management
import mlflow

# We will use ModelConfig for config management
# The structure of these yamls is totally up to you

# Load the chain's configuration
model_config = mlflow.models.ModelConfig(development_config="rag_chain_config.yaml")

model_resources = model_config.get("resources")
uc_config = model_config.get("uc_config")

#### endpoints and models
llm_model_name = model_resources.get("llm_model_name")
embedding_model_name = model_resources.get("embedding_model")
vs_endpoint_name = model_resources.get("vs_endpoint")

#### uc_config
catalog = uc_config.get("catalog")
schema = uc_config.get("schema")
index_name = uc_config.get("index_name")

# COMMAND ----------

# DBTITLE 1,Loading Chat Model
from langchain_community.chat_models import ChatDatabricks

tool_model = ChatDatabricks(
            target_uri='databricks',
            endpoint=llm_model_name,
            temperature=0.1
        )

tool_model.invoke("Hello World! Are you around?")

# COMMAND ----------

# DBTITLE 1,Setting Up Vector Search Retriever
from langchain_community.embeddings import DatabricksEmbeddings
from databricks.vector_search.client import VectorSearchClient
from langchain_community.vectorstores import DatabricksVectorSearch

embeddings = DatabricksEmbeddings(endpoint=embedding_model_name)

vsc = VectorSearchClient()
index = vsc.get_index(endpoint_name=vs_endpoint_name,
                              index_name=f"{catalog}.{schema}.{index_name}")

retriever = DatabricksVectorSearch(
    index, text_column="page_content", 
    embedding=embeddings, columns=["source_doc"]
).as_retriever()

retriever.invoke("What can you tell me about RAG?")

# COMMAND ----------

# DBTITLE 1,Setup Tools
from langchain.agents.agent_toolkits import create_retriever_tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode
from typing import Annotated, Literal, TypedDict

arxiv_tool = create_retriever_tool(retriever,
        'Vector Search Tool for task tuning Large language models',
        """This tool encompasses research on adapting and applying large language models to perform specific tasks or follow instructions in various domains. 
        It includes work on aligning language models with human intent, grounding language in real-world capabilities, using language models for control and decision-making, 
        and evaluating their performance on diverse NLP tasks.
         
         Key aspects of papers in this category:
         Instruction Following: Research on training language models to better understand and execute user instructions or commands.
         Embodied AI: Studies exploring the integration of language models with robotic systems or physical world interactions.
         Task Generalization: Investigations into the ability of language models to adapt to and solve a wide range of NLP tasks without specific fine-tuning.
         Human Alignment: Methods for improving language model outputs to better match human preferences and intentions.
         Practical Applications: Focus on using language models for real-world problem-solving, decision-making, or control tasks.
        """)

tools = [arxiv_tool]

tool_node = ToolNode(tools)

tool_model = tool_model.bind_tools(tools)

# Define the function that determines whether to continue or not
def should_continue(state: MessagesState) -> Literal["tools", END]:
    messages = state['messages']
    last_message = messages[-1]
    # If the LLM makes a tool call, then we route to the "tools" node
    if last_message.tool_calls:
        return "tools"
    # Otherwise, we stop (reply to the user)
    return END

# Define the function that calls the model
def call_model(state: MessagesState):
    messages = state['messages']
    response = tool_model.invoke(messages)
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}

# COMMAND ----------

# DBTITLE 1,Define Initial Graph
workflow = StateGraph(MessagesState)

# Define the two nodes we will cycle between
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)

# Set the entrypoint as `agent`
# This means that this node is the first one called
workflow.add_edge(START, "agent")

# We now add a conditional edge
workflow.add_conditional_edges(
    # First, we define the start node. We use `agent`.
    # This means these are the edges taken after the `agent` node is called.
    "agent",
    # Next, we pass in the function that will determine which node is called next.
    should_continue,
)

# We now add a normal edge from `tools` to `agent`.
# This means that after `tools` is called, `agent` node is called next.
workflow.add_edge("tools", 'agent')

# Initialize memory to persist state between graph runs
#checkpointer = MemorySaver()

# Finally, we compile it!
# This compiles it into a LangChain Runnable,
# meaning you can use it as you would any other runnable.
# Note that we're (optionally) passing the memory when compiling the graph
app = workflow.compile()

# COMMAND ----------

# DBTITLE 1,Testing the Agent
from langchain_core.messages import HumanMessage

final_state = app.invoke(
    {"messages": [HumanMessage(content="what is the a RAG?")]},
    config={"configurable": {"thread_id": 42}}
)
final_state["messages"][-1].content

# COMMAND ----------

# MAGIC %md
# MAGIC # Building an Adv Graph
# MAGIC
# MAGIC We don't have to stick to just a single graph in a notebook \
# MAGIC We can continue development and decide what to log and store

# COMMAND ----------

# DBTITLE 1,Define adv chain
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

class StructuredResponse(BaseModel):
    response: str = Field(description="The model end response")
    tool_use: bool = Field(description="flag to show if tool was used or not")

parser = JsonOutputParser(pydantic_object=StructuredResponse)

prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system", 
            "Answer the user query. as per the format guidance\n{format_instructions}",
        ),
        ("human", "{input}"),
    ]
).partial(format_instructions=parser.get_format_instructions())

# | parser 
structured_output = prompt_template | app 

# COMMAND ----------

## TODO still need to make sure that we are returning correct format as string not as class

# we can see that things are working but we are returning a full chain of messages
## We can write a function to handle this and add it to the logic
structured_output.invoke("What is a RAG?")
structured_output.invoke("how I add extra functionality to a large language model?")

# COMMAND ----------

from langchain_core.runnables import RunnableLambda

def extract_final_answer(chat_message_array):
    return chat_message_array["messages"][-1].content

final_formatted_response = structured_output | RunnableLambda(extract_final_answer) | parser

# COMMAND ----------

result = final_formatted_response \
  .invoke({"input": "how I add extra functionality to a large language model?"})

# COMMAND ----------

# DBTITLE 1,Define Graph For Mlflow logging etc
mlflow.models.set_model(model=final_formatted_response)
