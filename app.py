import gradio as gr
from dotenv import load_dotenv
import os
import requests
import io
from langchain import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.llms import TextGen
from langchain.chat_models import ChatAnthropic
from langchain.agents import ZeroShotAgent, initialize_agent, AgentType, AgentExecutor
from langchain.tools import StructuredTool
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import AgentAction, LLMResult, AgentFinish
from langchain.prompts import MessagesPlaceholder
from langchain.memory import ConversationBufferMemory

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
    agent_executor = gr.State()

    with gr.Column():
        token_box = gr.Markdown("Current Token: ")
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

        with gr.Column():
            gr.Markdown("Manual Download from Container")
            manual_download_text = gr.Textbox(
                placeholder="Add the path here. You can only download files created inside /mnt/data")
            with gr.Row():
                manual_download_file = gr.File(
                    label="Files", interactive=False, type="binary")
                manual_download_send = gr.Button(value="Download")

    # interactions

    def create_container(username_instance, assistant_instance):
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

        agent_executor_instance = setup_assistant(
            assistant_instance, token_text)

        return {
            token_box: f"Current Token: {token_text}",
            token: token_text,
            container_row: gr.update(visible=False),
            chatbot_column: gr.update(visible=True),
            agent_executor: agent_executor_instance
        }

    def setup_assistant(assistant_instance, token_instance):
        assistant_name = assistant_instance["selected"]

        match assistant_name:
            case "gpt3.5":
                llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
                max_iterations = 15
            case "gpt4":
                llm = ChatOpenAI(temperature=0, model="gpt-4")
                max_iterations = 15
            case "claude":
                llm = ChatAnthropic(temperature=0, model="claude-2")
                max_iterations = 5
            case "local":
                llm = TextGen(model_url=TEXTGEN_MODEL_URL)
                max_iterations = 30
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
        tools = [tool]

        memory = ConversationBufferMemory(
            memory_key="chat_history")

        prompt_prefix = 'You are an agent designed to write and execute python code to answer questions.\nYou have access to a Code Interpreter Lite tool, which you can use to execute python code.\nIf you get an error, debug your code and try again.\nOnly use the output of your code to answer the question. \nYou might know the answer without running any code, but you should still run the code to get the answer.\nIf there is a need to create or save a file, save it in the /mnt/data directory.\nYou cannot install new packages.\n'
        prompt_suffix = """Begin!"

{chat_history}
Question: {input}
{agent_scratchpad}"""

        prompt = ZeroShotAgent.create_prompt(
            tools,
            prefix=prompt_prefix,
            suffix=prompt_suffix,
            input_variables=["input", "chat_history", "agent_scratchpad"],
        )

        llm_chain = LLMChain(llm=llm, prompt=prompt)

        agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools,
                              verbose=True, max_iterations=max_iterations)

        agent_executor_instance = AgentExecutor.from_agent_and_tools(
            agent=agent, tools=tools, verbose=True, memory=memory
        )

        return agent_executor_instance

    def select_assistant(assistant_instance, assistant_selection_instance, token_instance):
        assistant_instance["selected"] = assistant_selection_instance

        agent_executor_instance = setup_assistant(
            assistant_instance, token_instance)

        return {
            assistant: assistant_instance,
            assistant_selection: assistant_selection_instance,
            assistant_box: f"Current Assistant: {assistant_selection_instance}",
            agent_executor: agent_executor_instance
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

    def download_file(file_name, token_instance):
        file_response = requests.get(
            f'{SUPERVISOR_API}/downloadfile?file_name={file_name}', headers={
                'Authorization': f'Bearer {token_instance}'
            }, timeout=600
        )

        if file_response.status_code != 200:
            raise gr.Error("No container created yet")

        file_content = file_response.content

        os.makedirs(f"/tmp/{token_instance}", exist_ok=True)
        temp_file_name = f"/tmp/{token_instance}/{file_name}"
        with open(temp_file_name, 'wb+') as out_file:
            out_file.write(file_content)

        return {
            manual_download_file: temp_file_name
        }

    def chatbot_handle(chatbot_instance, msg_instance, file_list_instance, agent_executor_instance):

        # add files to the prompt
        files_prompt = ""
        for file_name, file_path in file_list_instance.items():
            files_prompt += f"I have uploaded a file named {file_name}. The file is located at {file_path}.\n"

        # format input
        if len(files_prompt) > 0:
            chatbot_prompt = files_prompt + "\n" + msg_instance
        else:
            chatbot_prompt = msg_instance
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

                if isinstance(action.tool_input, str):
                    chatbot_tool_input_code_string = action.tool_input
                else:
                    chatbot_tool_input_code_string = action.tool_input.get(
                        "code")
                self.chatbot_response += f"{chatbot_thought}\n"
                self.chatbot_response += f'```\n{chatbot_tool_input_code_string}\n```\n'

            def get_chatbot_response(self):
                return self.chatbot_response

        chatbotHandler = ChatbotHandler()
        agent_executor_instance(
            chatbot_prompt, callbacks=[chatbotHandler])
        chatbot_response = chatbotHandler.get_chatbot_response()

        chatbot_instance.append((msg_instance, chatbot_response))

        return {
            chatbot: chatbot_instance,
            msg: "",
            agent_executor: agent_executor_instance
        }

    button.click(create_container, [username, assistant], [
        token_box, token, container_row, chatbot_column, agent_executor])
    assistant_selection.input(
        select_assistant, [assistant, assistant_selection, token], [assistant, assistant_selection, assistant_box, agent_executor])
    file_upload.upload(upload_file, [file_upload, token, file_list], [
        file_list, file_output])
    send.click(chatbot_handle, [chatbot, msg, file_list, agent_executor], [
               chatbot, msg, agent_executor])

    manual_download_send.click(download_file, [
                               manual_download_text, token], manual_download_file)


if __name__ == "__main__":
    demo.launch()
