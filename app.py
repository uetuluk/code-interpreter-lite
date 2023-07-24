import gradio as gr
from dotenv import load_dotenv
import os
import requests
import io

load_dotenv()

SUPERVISOR_API = os.environ.get("SUPERVISOR_API")


with gr.Blocks() as demo:
    # interface
    token = gr.State(value="")
    file_list = gr.State(value={})

    with gr.Column():
        token_box = gr.Markdown(f"Current Token: ")

    with gr.Row() as container_row:
        # username = gr.Textbox(label="Username")
        username = gr.State(value="USER")
        button = gr.Button(value="Create new Container", variant="primary")

    # chat interface
    with gr.Column(visible=False) as chatbot_column:
        chatbot = gr.Chatbot()
        with gr.Row() as chatbot_input:
            with gr.Column():
                file_output = gr.JSON(label="File List")
                file_upload = gr.UploadButton(label="Upload File")
            msg = gr.Textbox(placeholder="Type your message here", lines=8)
            send = gr.Button(value="Send", variant="primary")

    # interactions

    def create_container(username_instance):
        # create token
        token_response = requests.post(
            f'{SUPERVISOR_API}/token', data={'username': username_instance, 'password': username_instance})

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
            container_row: gr.update(visible=False),
            chatbot_column: gr.update(visible=True)
        }

    def upload_file(file, token_instance, file_list_instance):
        file_name = os.path.basename(file.name)

        file_response = requests.post(
            f'{SUPERVISOR_API}/uploadfile', headers={
                'Authorization': f'Bearer {token_instance}'
            }, files={
                'file': open(file.name, 'rb')
            }
        )

        if file_response.status_code != 200:
            raise gr.Error("No container created yet")
            return {
                file_list: file_list_instance,
                file_output: file_list_instance
            }

        file_list_instance[file_name] = f'/mnt/data/{file_name}'

        return {
            file_list: file_list_instance,
            file_output: file_list_instance
        }

    button.click(create_container, username, [
        token_box, token, container_row, chatbot_column])
    file_upload.upload(upload_file, [file_upload, token, file_list], [
        file_list, file_output])


if __name__ == "__main__":
    demo.launch()
