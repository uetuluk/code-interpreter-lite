import gradio as gr
from dotenv import load_dotenv
import os
import requests
import io
from langchain.chat_models import ChatOpenAI
from langchain.llms import TextGen
from langchain.chat_models import ChatAnthropic
from langchain.agents import initialize_agent, AgentType, AgentExecutor
from langchain.tools import StructuredTool
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import AgentAction, LLMResult, AgentFinish


from typing import Any, Dict, List


load_dotenv()

SUPERVISOR_API = os.environ.get("SUPERVISOR_API")
TEXTGEN_MODEL_URL = os.environ.get("TEXTGEN_MODEL_URL")

DEFAULT_ASSISTANT = "local"
ASSISTANT_LIST = ["gpt3.5", "gpt4", "claude", "local"]

with gr.Blocks() as demo:
    # interface
    token = gr.State(value="")
    file_list = gr.State(value={})
    assistant = gr.State(
        value={"selected": DEFAULT_ASSISTANT, "options": ASSISTANT_LIST})

    with gr.Column():
        token_box = gr.Markdown(f"Current Token: ")
        assistant_box = gr.Markdown(f"Current Assistant: {DEFAULT_ASSISTANT}")

    with gr.Row() as container_row:
        # username = gr.Textbox(label="Username")
        username = gr.State(value="USER")
        button = gr.Button(value="Create new Container", variant="primary")

    with gr.Row() as assistant_row:
        assistant_selection = gr.Dropdown(
            label="Assistant", choices=ASSISTANT_LIST, value=DEFAULT_ASSISTANT)

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
            f'{SUPERVISOR_API}/token', data={'username': username_instance, 'password': username_instance}, timeout=600)

        token_text = token_response.json().get('access_token')

        # create container
        container_response = requests.post(
            f'{SUPERVISOR_API}/container', headers={
                'Authorization': f'Bearer {token_text}'
            }, timeout=600
        )

        if container_response.status_code != 200:
            raise gr.Error("Problem creating container")

        return {
            token_box: f"Current Token: {token_text}",
            token: token_text,
            container_row: gr.update(visible=False),
            chatbot_column: gr.update(visible=True)
        }

    def select_assistant(assistant_instance, assistant_selection_instance):
        assistant_instance["selected"] = assistant_selection_instance
        return {
            assistant: assistant_instance,
            assistant_selection: assistant_selection_instance,
            assistant_box: f"Current Assistant: {assistant_selection_instance}"
        }

    def upload_file(file, token_instance, file_list_instance):
        file_name = os.path.basename(file.name)

        file_response = requests.post(
            f'{SUPERVISOR_API}/uploadfile', headers={
                'Authorization': f'Bearer {token_instance}'
            }, files={
                'file': open(file.name, 'rb')
            }, timeout=600
        )

        if file_response.status_code != 200:
            raise gr.Error("No container created yet")
            # return {
            #     file_list: file_list_instance,
            #     file_output: file_list_instance
            # }

        file_list_instance[file_name] = f'/mnt/data/{file_name}'

        return {
            file_list: file_list_instance,
            file_output: file_list_instance
        }

    def chatbot_handle(chatbot_instance, msg_instance, token_instance, assistant_instance, file_list_instance):
        # get assistant
        assistant_name = assistant_instance["selected"]

        match assistant_name:
            case "gpt3.5":
                llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
                agent_type = AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION
            case "gpt4":
                llm = ChatOpenAI(temperature=0, model="gpt-4")
                agent_type = AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION
            case "local":
                llm = TextGen(model_url=TEXTGEN_MODEL_URL)
                agent_type = AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION
            case _:
                raise gr.Error("Assistant not supported yet")
                # return {
                #     chatbot: chatbot_instance,
                #     msg: msg_instance,
                # }

        # get tool
        def code_interpreter_lite(code: str) -> str:
            """Execute the python code and return the result."""
            code_response = requests.post(
                f'{SUPERVISOR_API}/run', headers={
                    'Authorization': f'Bearer {token_instance}'
                }, json={
                    'code_string': code
                }, timeout=600)
            if code_response.status_code != 200:
                raise gr.Error("No container created yet")
                # return {
                #     chatbot: chatbot_instance,
                #     msg: msg_instance,
                # }
            result = code_response.json()

            return result

        tool = StructuredTool.from_function(
            func=code_interpreter_lite, name="Code Interpreter Lite", description="useful for running python code")

        agent_executor = initialize_agent(
            [tool],
            llm,
            agent=agent_type,
        )

        # add files to the prompt
        files_prompt = ""
        for file_name, file_path in file_list_instance.items():
            files_prompt += f"I have uploaded a file named {file_name}. The file is located at {file_path}.\n"

        # format input
        chatbot_prompt = msg_instance + "\n" + files_prompt

        # format chatbot response
        # chatbot_response = tool_input + "\n" + tool_result + "\n" + agent_response
        # chatbot_response = agent_response

        class ChatbotHandler(BaseCallbackHandler):
            def __init__(self):
                self.chatbot_response = ""
                super().__init__()

            def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> Any:
                self.chatbot_response += outputs.get("output", "") + '\n'

            def on_tool_end(self, output: str, **kwargs: Any) -> Any:
                self.chatbot_response += f'```\n{output}\n```\n'

            def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
                chatbot_thought = action.log.split("\n")[0]
                chatbot_thought = chatbot_thought.replace("Thought: ", "")
                chatbot_tool_input_code_string = action.tool_input.get("code")
                self.chatbot_response += f"{chatbot_thought}\n"
                self.chatbot_response += f'```\n{chatbot_tool_input_code_string}\n```\n'

            def get_chatbot_response(self):
                return self.chatbot_response

        chatbotHandler = ChatbotHandler()
        agent_executor(
            chatbot_prompt, callbacks=[chatbotHandler])
        chatbot_response = chatbotHandler.get_chatbot_response()

        chatbot_instance.append((msg_instance, chatbot_response))

        return {
            chatbot: chatbot_instance,
            msg: "",
        }

    button.click(create_container, username, [
        token_box, token, container_row, chatbot_column])
    assistant_selection.input(
        select_assistant, [assistant, assistant_selection], [assistant, assistant_selection, assistant_box])
    file_upload.upload(upload_file, [file_upload, token, file_list], [
        file_list, file_output])
    send.click(chatbot_handle, [chatbot, msg, token,
               assistant, file_list], [chatbot, msg])


if __name__ == "__main__":
    demo.launch()
