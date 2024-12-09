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
mlflow_version = 'mlflow==2.17.0'
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

# DBTITLE 1,Library Imports
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.chat_models import ChatDatabricks

from langchain_core.runnables import RunnableLambda
from operator import itemgetter

# COMMAND ----------

# DBTITLE 1,Build Helper Functions
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
# MAGIC Whilst Langchain has a direct agent we usually want to customise it a bit more \
# MAGIC Langgraph will give us the power to customise agents a bit more and provide distinct code execution paths \
# MAGIC
# MAGIC Langgraph consist of nodes and edges. Edges join nodes which can be thought of as python functions \
# MAGIC states and schemas are what helps us to ensure consistency of inputs and outputs for transfer of data between nodes

# COMMAND ----------

# DBTITLE 1,Setup Schemas
# We will need langgraph for complex logic
# The core part of Langgraph is the state / schema.
from typing import TypedDict, Annotated, List, Union, Sequence, Literal
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.messages import BaseMessage
from langchain_core.prompts import MessagesPlaceholder
from langgraph.graph import StateGraph, START, END
from IPython.display import Image, display
import operator

# The state will be passed around the nodes

# Pydantic is better the classes and TypedDict due to validation options
from langchain_core.pydantic_v1 import BaseModel, validator, ValidationError
from langchain_core.output_parsers import PydanticOutputParser

# The input schema is:
class AgentState(BaseModel):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    continue_or_not: Literal["Yes", "No"]

# We want a json structured output which will be:
class GraphOutput(BaseModel):
    customer_address: str
    current_customer_or_not: bool
    customer_type: Literal["business", "personal"]

    # PyDantic allows us to validation of our schemas
    # Otherwise with TypedDict and other alternatives it is suggested to the LLM but not enforced
    @validator('customer_type')
    def validate_customer_type(cls, value):
        if value not in ["business", "personal"]:
            raise ValueError("Each customer must be business or personal")
        return value
    
# Whilst we can have multiple schemas and enforce them for input and output
# The graph needs an internal state that is aware of all possible formatting structures
class InternatState(AgentState, GraphOutput):
    pass

# COMMAND ----------

# DBTITLE 1,Call Model Node

# This is the logic of the core call model node 
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
# You can see that we can do whatever we want inside
def core_agent(state: AgentState):

    state_dict = dict(state)

    output = agent_chain.invoke(state_dict)

    testing = agent_chain.invoke({"messages": [HumanMessage(content=f"""Did the response indicate that the agent knows the customer's:
                                                            - location
                                                            - current customer or not
                                                            - personal vs business customer
                                                            
                                                            <ai_response>
                                                            {output}
                                                            
                                                            <analysis of ai_response>
                                                            reply single word Yes / No only""")]})


    return {"messages": [output], "continue_or_not": testing.content}

# COMMAND ----------

# DBTITLE 1,Define Output Node
output_parser = PydanticOutputParser(pydantic_object=GraphOutput)

formatter_template = ChatPromptTemplate.from_messages(
    [
        (
            "system", 
            """Parse the user input and  as per the format guidance\n{format_instructions}
            respond with the output json only
            """,
        ),
        ("human", "{input}"),
    ]
).partial(format_instructions=output_parser.get_format_instructions())

llm_w_structured_output = formatter_template | llm_model | output_parser

def respond(state: AgentState) -> GraphOutput:
    last_message = state.messages[-1]
    response = llm_w_structured_output.invoke(last_message.content)

    print(response)
    return response


# COMMAND ----------

# DBTITLE 1,Define Query Human and conditional node

#@tool
def seek_human_clarification(state: AgentState):
    """If there isn't enough information then the bot can ask the user for more information"""
    #messages = state["messages"]
    #last_message = messages[-1]

    output = input(f"We need more information, can you please clarify and provide extra information on your circumstances:")
    return {"messages": [HumanMessage(content=output)]}

# should continue determines what to do next
def should_continue(state: AgentState):    
    #messages = state["messages"]
    state_dict = dict(state)
    print(state_dict)

    last_state = state_dict["continue_or_not"]
    if last_state == "Yes":
        return "respond"
    else:
        return "continue"

# COMMAND ----------

# DBTITLE 1,Define Workflow
workflow = StateGraph(InternatState, input=AgentState, output=GraphOutput)

# Nodes
workflow.add_node("core_agent", core_agent)
workflow.add_node("respond", respond)
workflow.add_node("seek_input", seek_human_clarification)

# Edges
workflow.add_edge(START, "core_agent")
workflow.add_conditional_edges(
    "core_agent",
    should_continue,
    {
        "continue": "seek_input",
        "respond": "respond"
    },
)

workflow.add_edge("seek_input", "core_agent")
workflow.add_edge("respond", END)


graph = workflow.compile()

display(Image(graph.get_graph().draw_mermaid_png()))

# COMMAND ----------

# DBTITLE 1,Run workflow till end
state = {'messages': [HumanMessage(content="harro")], "continue_or_not": "Yes"}

for s in graph.stream(state):
  print(list(s.values())[0])
  print('----')

# COMMAND ----------

# MAGIC %md
# MAGIC # Discussion
# MAGIC We now have a bot that will quiz the user until it gets the responses it needs \
# MAGIC It will then output a valid pydantic object