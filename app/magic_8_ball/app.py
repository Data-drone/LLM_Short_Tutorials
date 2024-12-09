import io
from threading import Thread
import os

import numpy as np
import spaces
import gradio as gr
import torch
import pandas as pd

from parler_tts import ParlerTTSForConditionalGeneration
from pydub import AudioSegment
from transformers import AutoTokenizer, AutoFeatureExtractor, set_seed
from huggingface_hub import InferenceClient
from streamer import ParlerTTSStreamer
import time
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()
openai_client = w.serving_endpoints.get_open_ai_client()

#######

import wget
import tarfile
import glob
import shutil

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Create directories
#home = str(Path.home())
current_directory = os.getcwd()
logger.info(current_directory)

ffmpeg_load_path = 'ffmpeg_bin'
unzip_path = 'bin'

ffmpeg_bin = os.path.join(current_directory, ffmpeg_load_path)
bin_dir = os.path.join(current_directory, unzip_path)

os.makedirs(ffmpeg_bin, exist_ok=True)
os.makedirs(bin_dir, exist_ok=True)

# Change to ffmpeg_bin directory
os.chdir(ffmpeg_bin)

# Download the file
url = 'https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz'
wget.download(url)

# Extract the archive
with tarfile.open('ffmpeg-release-amd64-static.tar.xz') as tar:
    tar.extractall()

# Find the extracted directory
extracted_dir = glob.glob('ffmpeg-*-amd64-static')[0]
logger.info(os.listdir(extracted_dir))

# Move ffmpeg and ffprobe to bin directory
shutil.move(extracted_dir, bin_dir)
os.chdir(current_directory)

logger.info(f'Post move bin dir: {os.listdir(bin_dir)}' )
logger.info(f"Post Move extract folder in bin dir: {os.listdir(os.path.join(bin_dir, 'ffmpeg-7.0.2-amd64-static'))}")

ffmpeg_path = os.path.join(bin_dir, 'ffmpeg-7.0.2-amd64-static')
os.environ['PATH'] += os.pathsep + ffmpeg_path

#######

device = "cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
torch_dtype = torch.float16 if device != "cpu" else torch.float32

repo_id = "parler-tts/parler_tts_mini_v0.1"
jenny_repo_id = "ylacombe/parler-tts-mini-jenny-30H"

db_token = os.getenv('ACCESS_TOKEN')

model = ParlerTTSForConditionalGeneration.from_pretrained(
    jenny_repo_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True
).to(device)

client = InferenceClient(token=os.getenv("HF_TOKEN"))

tokenizer = AutoTokenizer.from_pretrained(repo_id)
feature_extractor = AutoFeatureExtractor.from_pretrained(repo_id)

SAMPLE_RATE = feature_extractor.sampling_rate
SEED = 42


def numpy_to_mp3(audio_array, sampling_rate):
    # Normalize audio_array if it's floating-point
    if np.issubdtype(audio_array.dtype, np.floating):
        max_val = np.max(np.abs(audio_array))
        audio_array = (audio_array / max_val) * 32767 # Normalize to 16-bit range
        audio_array = audio_array.astype(np.int16)

    # Create an audio segment from the numpy array
    audio_segment = AudioSegment(
        audio_array.tobytes(),
        frame_rate=sampling_rate,
        sample_width=audio_array.dtype.itemsize,
        channels=1
    )

    # Export the audio segment to MP3 bytes - use a high bitrate to maximise quality
    mp3_io = io.BytesIO()
    audio_segment.export(mp3_io, format="mp3", bitrate="320k")

    # Get the MP3 bytes
    mp3_bytes = mp3_io.getvalue()
    mp3_io.close()

    return mp3_bytes

sampling_rate = model.audio_encoder.config.sampling_rate
frame_rate = model.audio_encoder.config.frame_rate

import base64
import json
import requests

def get_mp3_as_base64_string(file_path):
    with open(file_path, 'rb') as file:
        binary_data = file.read()
        base64_string = base64.b64encode(binary_data).decode('utf-8')
    return base64_string


def score_model(dataset):
    url = 'https://adb-984752964297111.11.azuredatabricks.net/serving-endpoints/whisper_large_v3-2/invocations'
    headers = {'Authorization': f'Bearer {db_token}', 'Content-Type': 'application/json'}
    ds_dict = {'dataframe_split': dataset.to_dict(orient='split')}
    data_json = json.dumps(ds_dict, allow_nan=True)
    response = requests.request(method='POST', headers=headers, url=url, data=data_json)
    if response.status_code != 200:
        raise Exception(f'Request failed with status {response.status_code}, {response.text}')
    return response.json()

def generate_response(audio):
    gr.Info("Transcribing Audio", duration=5)
    
    #question = client.automatic_speech_recognition(audio).text
    audio_string = get_mp3_as_base64_string(audio)
    serving_input = pd.DataFrame({0: [audio_string]})
    question = score_model(serving_input)['predictions'][0]
    
    messages = [{"role": "system", "content": ("You are a magic 8 ball."
                                              "Someone will present to you a situation or question and your job "
                                              "is to answer with a cryptic addage or proverb such as "
                                              "'curiosity killed the cat' or 'The early bird gets the worm'."
                                              "Keep your answers short and do not include the phrase 'Magic 8 Ball' in your response. If the question does not make sense or is off-topic, say 'Foolish questions get foolish answers.'"
                                              "For example, 'Magic 8 Ball, should I get a dog?', 'A dog is ready for you but are you ready for the dog?'")},
                {"role": "user", "content": f"Magic 8 Ball please answer this question -  {question}"}]
    
    response = openai_client.chat.completions.create(messages=messages,  
                                             model="databricks-meta-llama-3-1-70b-instruct")
    response = response.choices[0].message.content.replace("Magic 8 Ball", "")
    return response, None, None

@spaces.GPU
def read_response(answer):

    play_steps_in_s = 2.0
    play_steps = int(frame_rate * play_steps_in_s)

    description = "Jenny speaks at an average pace with a calm delivery in a very confined sounding environment with clear audio quality."
    description_tokens = tokenizer(description, return_tensors="pt").to(device)

    streamer = ParlerTTSStreamer(model, device=device, play_steps=play_steps)
    prompt = tokenizer(answer, return_tensors="pt").to(device)

    generation_kwargs = dict(
        input_ids=description_tokens.input_ids,
        prompt_input_ids=prompt.input_ids,
        streamer=streamer,
        do_sample=True,
        temperature=1.0,
        min_new_tokens=10,
    )

    set_seed(SEED)
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    start = time.time()
    for new_audio in streamer:
        print(f"Sample of length: {round(new_audio.shape[0] / sampling_rate, 2)} seconds after {time.time() - start} seconds")
        yield answer, numpy_to_mp3(new_audio, sampling_rate=sampling_rate)


with gr.Blocks() as block:
    gr.HTML(
        f"""
        <h1 style='text-align: center;'> Magic 8 Ball 🎱 </h1>
        <h3 style='text-align: center;'> Ask a question and receive wisdom </h3>
        <p style='text-align: center;'> Powered by <a href="https://github.com/huggingface/parler-tts"> Parler-TTS</a>
        """
    )
    with gr.Group():
        with gr.Row():
            audio_out = gr.Audio(label="Spoken Answer", streaming=True, autoplay=True, loop=False)
            answer = gr.Textbox(label="Answer")
            state = gr.State()
        with gr.Row():
            audio_in = gr.Audio(label="Speak you question", sources="microphone", type="filepath")
    with gr.Row():
        gr.HTML("""<h3 style='text-align: center;'> Examples: 'What is the meaning of life?', 'Should I get a dog?' </h3>""")
    audio_in.stop_recording(generate_response, audio_in, [state, answer, audio_out]).then(fn=read_response, inputs=state, outputs=[answer, audio_out])

block.launch()