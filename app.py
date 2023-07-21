import gradio as gr
from dotenv import load_dotenv
import os
import requests
import io

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

            token_text = token_response.json().get('access_token')

            # create container
            container_response = requests.post(
                f'{SUPERVISOR_API}/container', headers={
                    'Authorization': f'Bearer {token_text}'
                }
            )

            return {
                token_box: f"Current Token: {token_text}",
                token: token_text,
                container_row: gr.update(visible=False)
            }

        button.click(create_container, username, [
                     token_box, token, container_row])

    def upload_file(file):
        print("token", token)
        file_response = requests.post(
            f'{SUPERVISOR_API}/uploadfile', headers={
                'Authorization': f'Bearer {token}'
            }, files={
                'file': io.BytesIO(file)
            }
        )
        print(file_response.json())

    # chat interface
    with gr.Column() as chatbot_column:
        chatbot = gr.Chatbot()
        with gr.Row() as chatbot_input:
            file_upload = gr.UploadButton(label="Upload File", type="bytes")
            msg = gr.Textbox(placeholder="Type your message here", lines=8)
            send = gr.Button(value="Send", variant="primary")

        file_upload.upload(upload_file, [file_upload])


if __name__ == "__main__":
    demo.launch()
