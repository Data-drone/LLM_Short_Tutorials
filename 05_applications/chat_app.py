import gradio as gr
from langchain_community.chat_models import ChatDatabricks
from langchain.schema import AIMessage, HumanMessage

chat_model = 'mistral_7b_model'

llm = ChatDatabricks(
    target_uri="databricks",
    endpoint=chat_model,
    temperature=0.1,
)

def predict(message, history):
    history_langchain_format = []
    for human, ai in history:
        history_langchain_format.append(HumanMessage(content=human))
        history_langchain_format.append(AIMessage(content=ai))
    history_langchain_format.append(HumanMessage(content=message))
    gpt_response = llm(history_langchain_format)
    return gpt_response.content

app = gr.ChatInterface(predict)

if __name__ == '__main__':

    app.launch()