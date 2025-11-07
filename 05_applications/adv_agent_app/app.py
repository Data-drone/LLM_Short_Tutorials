import gradio as gr
from databricks_langchain import ChatDatabricks
from langchain.schema import AIMessage, HumanMessage
import os
import json
import requests

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

db_token = os.getenv('ACCESS_TOKEN')
chat_model = os.getenv('endpoint')

logger.info(f"Connected to: {chat_model}")

endpoint = f"https://adb-984752964297111.11.azuredatabricks.net/serving-endpoints/{chat_model}/invocations"

def predict(message, history):
    history_langchain_format = []
    for human, ai in history:
        history_langchain_format.append({"role":"user", "content":human})
        history_langchain_format.append({"role":"agent", "content": ai})
    history_langchain_format.append({"role":"user", "content":message})
    
    headers = {'Authorization': f'Bearer {db_token}', 'Content-Type': 'application/json'}
    ds_dict = {"messages": history_langchain_format}
    

    data_json = json.dumps(ds_dict, allow_nan=True)
    logger.info(f"Request is: {data_json}")
    
    response = requests.request(method='POST', headers=headers, url=endpoint, data=data_json)
    logger.info(f"Model Response is: {response.content}")
    
    response_dict = json.loads(response.content.decode('utf-8'))
    
    response_format = response_dict['messages']
    
    return response_format

#app = gr.ChatInterface(predict)

main_heading = """
h1 {
    text-align: center;
    display:block;
}
"""

with gr.Blocks(title='Chat to Docs App',
               fill_height=True,
               css=".contain { display: flex !important; flex-direction: column !important; }"
               "#chatcol { height: 90vh !important; }"
               "#sidebar { height: 90vh !important; }"
               ".justified-text {text-align: justify !important; }"
               ) as app:
    with gr.Row():
        with gr.Column(scale=1, min_width=100):
            gr.Image("assets/logo.png", show_label=False, container=False, show_download_button=False)
        with gr.Column(scale=4, min_width=300):
            gr.Markdown("<h1 style='text-align: center;'> Intelligent Agents </h1>", elem_classes="justified-text")
    with gr.Group():
        with gr.Row():
            with gr.Column(scale=4, min_width=300):
                chatbot = gr.Chatbot(
                    show_copy_button = True,
                    avatar_images = (
                        'https://banner2.cleanpng.com/20180329/zue/kisspng-computer-icons-user-profile-person-5abd85306ff7f7.0592226715223698404586.jpg', 
                        'https://banner2.cleanpng.com/20180207/bre/kisspng-robot-head-android-clip-art-cartoon-robot-pictures-5a7b4ce95ee663.0818532615180300573887.jpg'
                        )
                    )
                gr.ChatInterface(fn=predict,
                                examples=["Help me research LLMs tell me about how to run finetuning",
                                        "Tell me my db cluster usage over the past day",
                                        "I wanna chat about the weather"],
                                chatbot=chatbot)
        

if __name__ == '__main__':

    app.launch()