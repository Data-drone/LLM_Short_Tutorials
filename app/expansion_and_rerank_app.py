# This variant of chat to docs has the langchain baked into the app
# this is faster to test but we can't use inference tables

import gradio as gr

# imports fo the different components
from langchain_community.chat_models import ChatDatabricks
from langchain_community.embeddings import DatabricksEmbeddings
from databricks.vector_search.client import VectorSearchClient
from langchain_community.vectorstores import DatabricksVectorSearch
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank

from langchain.schema import AIMessage, HumanMessage
from langchain import hub
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser

from langchain_core.prompts.prompt import PromptTemplate

# to do message history we need a history aware retriever 
# basically we receive the question and the history
# - Then we ask an LLM to reformulate
# - Then we send updated llm generated question to retriever
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from databricks import sql
import os


chat_model = 'mistral_7b_model'
embedding_model_name = 'embedding-bge-large-en'

endpoint_name = 'qld-treasury-llm-poc'
catalog = 'qld_treasury_poc'
schema = 'rag_chatbot'

vs_index_fullname = f'{catalog}.{schema}.embed_self_managed_vs_index_v2'

# should move this to the triggering notebook
connection = sql.connect(
    server_hostname = 'dbc-d26740e0-b34d.cloud.databricks.com',
    http_path       = '/sql/1.0/warehouses/8783329496652f0f',
    access_token    = os.getenv("DATABRICKS_TOKEN"))

with connection.cursor() as cursor:
    cursor.execute(f"SELECT DISTINCT filename FROM {catalog}.{schema}.sample_embedded_dataset_v2")
    list_of_docs = cursor.fetchall()

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
).as_retriever(search_kwargs={"k": 10})

compressor = FlashrankRerank()
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=retriever
)

# formatting for context
def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])

## adding history
# we don't use chat history like in langchain docs because it plays funny with mistral model
# and expected formatin Databricks
contextualize_q_prompt = PromptTemplate(
    input_variables=["input", "chat_history"],
    template="<s> [INST] Your job is to reformulate a question given a user question and the prior conversational history. DO NOT answer the question. If there is no chat history pass through the question [/INST] </s> \n [INST] Question: {input} \nHistory: {chat_history} \nAnswer: [/INST]"
)

history_aware_retriever = create_history_aware_retriever(
    llm, compression_retriever, contextualize_q_prompt
)

# To be replaced
#rag_prompt = hub.pull("rlm/rag-prompt-mistral")
rag_prompt = PromptTemplate(input_variables=['context', 'input', 'chat_history'],
                                      template="<s> [INST] You are a helpful personal assistant who helps users find what they need from documents. Be conversational, polite and use the following pieces of retrieved context and the conversational history to help answer the question. <unbreakable-instruction> ANSWER ONLY FROM THE CONTEXT </unbreakable-instruction> <unbreakable-instruction> If you don't know the answer, just say that you don't know. </unbreakable-instruction> Keep the answer concise. [/INST] </s> \n[INST] Question: {input} \nContext: {context} \nHistory: {chat_history} \nAnswer: [/INST]")

#rag_runnable = rag_prompt | llm

chain = (
    {'context': history_aware_retriever | format_docs, "question": RunnablePassthrough(), "history": RunnablePassthrough()}
    | rag_prompt
    | llm 
    | StrOutputParser()
)


def predict(message, history):
    history_langchain_format = []
    for human, ai in history:
        history_langchain_format.append(HumanMessage(content=human))
        history_langchain_format.append(AIMessage(content=ai))
    history_langchain_format.append(HumanMessage(content=message))
    gpt_response = chain.invoke({"question": message, "history": history_langchain_format})
    return gpt_response #.content


with gr.Blocks(title='Chat to Docs App',
               fill_height=True,
               css=".contain { display: flex !important; flex-direction: column !important; }"
               "#chatcol { height: 90vh !important; }"
               "#sidebar { height: 90vh !important; }"
               ) as app:
    with gr.Row():
        gr.Markdown("# Chat To docs App")
    with gr.Row():
        with gr.Column(scale=1, elem_id='sidebar'):
            gr.Markdown("## Source Docs")
            gr.Dataframe(list_of_docs, wrap=True)
        with gr.Column(scale=3, elem_id='chatcol'):
            gr.ChatInterface(fn=predict,
                examples=["Tell me about the budget", "tell me about tourism"],
                # chatbot = gr.Chatbot(
                #     show_copy_button = True,
                #     avatar_images = (
                #         'https://banner2.cleanpng.com/20180329/zue/kisspng-computer-icons-user-profile-person-5abd85306ff7f7.0592226715223698404586.jpg', 
                #         'https://banner2.cleanpng.com/20180207/bre/kisspng-robot-head-android-clip-art-cartoon-robot-pictures-5a7b4ce95ee663.0818532615180300573887.jpg'
                #     )
                # )
            )

if __name__ == '__main__':

    app.launch()