# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Lets build an Advanced Agent with user feedback
# MAGIC
# MAGIC Often with business applications, we need to perform complex tasks based on specific inofmrmation \
# MAGIC with LLM technology when there is missing information it makes it harder for us to get the task done correctly \
# MAGIC Lets look at how we can build a bot to do this

# COMMAND ----------

# DBTITLE 1,Parameterise Pip installs
mlflow_version = 'mlflow==2.17.0rc0'
langchain_base_version = 'langchain'
langchain_community_version = 'langchain_community==0.2.17'
langgraph_version = '0.2.35'

# COMMAND ----------

# DBTITLE 1,Setup Libs
# MAGIC %pip install -U pydantic>2.0.0 {mlflow_version} langchain {langchain_community_version} langchain_core langgraph>={langgraph_version} langchain-databricks
# MAGIC %restart_python

# COMMAND ----------

# DBTITLE 1,Configuration
model_to_use = 'databricks-meta-llama-3-1-405b-instruct'


# COMMAND ----------

# DBTITLE 1,Base Core bot
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.chat_models import ChatDatabricks

from langchain_core.runnables import RunnableLambda
from operator import itemgetter

def extract_user_query_string(chat_messages_array):
    return chat_messages_array[-1].content

llm_model = ChatDatabricks(
            target_uri='databricks',
            endpoint=model_to_use,
            temperature=0.1
        )

basic_template = ChatPromptTemplate.from_messages(
        [
            ("system", """
            You are a customer support bot intended to help a customer choose banking products.
             You first need to work out from the user request whether they are:
             - a current customer
             - a business or personal customer
             - where they are located

            In your response highlight and rephrase and ask for clarification on these 

            """),
            ("human", "{user_input}"),
         ] 
        )

chain = {"user_input": itemgetter("messages") | RunnableLambda(extract_user_query_string)} | basic_template | llm_model

# COMMAND ----------

# DBTITLE 1,Test Base Core Bot
# We can see that the interaction stops here and there is no logic for continuing the discussion
chain.invoke({"messages": [
    HumanMessage(content="Hello")
]})

# COMMAND ----------

# MAGIC %md
# MAGIC # Migrate to Langgraph
# MAGIC Langchain fo a while had a direct agent 

# COMMAND ----------

# DBTITLE 1,Move Corelogic to Langgraph
# We will need langgraph for complex logic
# The core part of Langgraph is the state.
from typing import TypedDict, Annotated, List, Union, Sequence
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.messages import BaseMessage
from langchain_core.prompts import MessagesPlaceholder
from langgraph.graph import StateGraph, START, END
from IPython.display import Image, display
import operator

# The state will be passed around the nodes
# it will be updated by each node
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    continue_or_not: str

# COMMAND ----------

# DBTITLE 1,Define Core Nodes and Prompts
agent_template = ChatPromptTemplate.from_messages(
    [
        ("system", """
            You are a customer support bot intended to help a customer choose banking products.
             You first need to work out from the user request whether they are:
             - a current customer
             - a business or personal customer
             - where they are located

            In your response highlight and rephrase and ask for clarification on these 

            """),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            #MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

def extract_chat_history(chat_message_array):
    return chat_message_array[:-1]


agent_chain = {"input": itemgetter("messages") | RunnableLambda(extract_user_query_string),
               "chat_history": itemgetter("messages") | RunnableLambda(extract_chat_history) } | agent_template | llm_model

# this is the core agent call
def core_agent(state):

    output = agent_chain.invoke(state)

    testing = agent_chain.invoke({"messages": [HumanMessage(content=f"""Did the response indicate that the agent knows the customer's:
                                                            - location
                                                            - current customer or not
                                                            - personal vs business customer
                                                            
                                                            <ai_response>
                                                            {output}
                                                            
                                                            <analysis of ai_response>
                                                            reply single word Yes / No only""")]})


    return {"messages": [output], "continue_or_not": testing.content}


#@tool
def seek_human_clarification(state):
    """If there isn't enough information then the bot can ask the user for more information"""
    #messages = state["messages"]
    #last_message = messages[-1]

    output = input(f"We need more information, can you please clarify and provide extra information on your circumstances:")
    return {"messages": [HumanMessage(content=output)]}

# should continue determines what to do next
def should_continue(state):    
    messages = state["messages"]
    last_message = messages[-1]

    last_state = state["continue_or_not"] 
    if last_state == "Yes":
        return "end"
    else:
        return "continue"

# COMMAND ----------

# DBTITLE 1,Define Workflow
workflow = StateGraph(AgentState)
workflow.add_node("core_agent", core_agent)
workflow.add_node("seek_input", seek_human_clarification)


workflow.add_edge(START, "core_agent")

workflow.add_conditional_edges(
    "core_agent",
    should_continue,
    {
        "continue": "seek_input",
        "end": END
    },
)

workflow.add_edge("seek_input", "core_agent")


graph = workflow.compile()

display(Image(graph.get_graph().draw_mermaid_png()))

# COMMAND ----------

# DBTITLE 1,Run workflow till end
state = {'messages': [HumanMessage(content="harro")]}

for s in graph.stream(state):
  print(list(s.values())[0])
  print('----')

