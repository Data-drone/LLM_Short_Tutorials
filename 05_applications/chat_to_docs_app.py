# This variant of chat to docs has the langchain baked into the app
# this is faster to test but we can't use inference tables

import gradio as gr

# imports fo the different components
from langchain_community.chat_models import ChatDatabricks
from langchain_community.embeddings import DatabricksEmbeddings
from databricks.vector_search.client import VectorSearchClient
from langchain_community.vectorstores import DatabricksVectorSearch

from langchain.schema import AIMessage, HumanMessage
from langchain import hub
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser

chat_model = 'mistral_7b_model'
embedding_model_name = 'embedding-bge-large-en'

endpoint_name = 'qld-treasury-llm-poc'
vs_index_fullname = 'qld_treasury_poc.rag_chatbot.embed_self_managed_vs_index_v2'

llm = ChatDatabricks(
    target_uri="databricks",
    endpoint=chat_model,
    temperature=0.1,
)
embeddings = DatabricksEmbeddings(endpoint=embedding_model_name)

# vector search configuration
vsc = VectorSearchClient()
index = vsc.get_index(endpoint_name=endpoint_name,
                      index_name=vs_index_fullname)

retriever = DatabricksVectorSearch(
    index, text_column="chunked_results", 
    embedding=embeddings, columns=["filename"]
).as_retriever()

# formatting for context
def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])


# To be replaced
rag_prompt = hub.pull("rlm/rag-prompt-mistral")

#rag_runnable = rag_prompt | llm

chain = (
    {'context': retriever | format_docs, "question": RunnablePassthrough()}
    | rag_prompt
    | llm 
    | StrOutputParser()
)


def predict(message, history):
    #history_langchain_format = []
    #for human, ai in history:
    #    history_langchain_format.append(HumanMessage(content=human))
    #    history_langchain_format.append(AIMessage(content=ai))
    #history_langchain_format.append(HumanMessage(content=message))
    gpt_response = chain.invoke(message)
    return gpt_response #.content

app = gr.ChatInterface(fn=predict,
                       examples=["Tell me about the budget"],
                       title="Chat to Docs Bot")

if __name__ == '__main__':

    app.launch()