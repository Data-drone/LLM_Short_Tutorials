import gradio as gr

io = gr.Interface(lambda x: "Hello, " + x + "!", "textbox", "textbox")

if __name__ == '__main__':

    io.launch()