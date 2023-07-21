from fastapi import FastAPI, Depends
import gradio as gr
from dotenv import load_dotenv
import os
import requests
from fastapi.security import HTTPBasic

load_dotenv()

SUPERVISOR_API = os.environ.get("SUPERVISOR_API")

app = FastAPI()

security = HTTPBasic()


def login(username, password):
    response = requests.post(
        f'{SUPERVISOR_API}/token', data={'username': username, 'password': password})

    token = response.json().get('access_token')


username = gr.Textbox(label="Username")
password = gr.Textbox(label="Password", type="password")

login_interface = gr.Interface(
    fn=login, inputs=[username, password], outputs=None)

app = gr.mount_gradio_app(app, interface, "/")
