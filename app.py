import gradio as gr
from dotenv import load_dotenv
import os
import requests

load_dotenv()

SUPERVISOR_API = os.environ.get("SUPERVISOR_API")


with gr.Blocks() as demo:

    token = gr.State("")

    with gr.Column():
        token_box = gr.Markdown(f"Current Token: ")

    with gr.Row() as container_row:
        username = gr.Textbox(label="Username")
        button = gr.Button(value="Create new Container", variant="primary")

        def create_container(username):
            # create token
            token_response = requests.post(
                f'{SUPERVISOR_API}/token', data={'username': username, 'password': username})

            token = token_response.json().get('access_token')

            # create container
            container_response = requests.post(
                f'{SUPERVISOR_API}/container', headers={
                    'Authorization': f'Bearer {token}'
                }
            )

            return {
                token_box: f"Current Token: {token}",
                container_row: gr.update(visible=False)
            }

        button.click(create_container, username, [
                     token_box, container_row])

    # chat interface


if __name__ == "__main__":
    demo.launch()
