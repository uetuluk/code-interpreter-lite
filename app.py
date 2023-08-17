from markdown import Markdown
import gradio as gr
from dotenv import load_dotenv
import os
import requests
import io
import re
import base64

import langchain
from langchain import PromptTemplate, LLMChain
from langchain.llms import TextGen
from typing import Any, Dict, List, Optional, Iterator, Tuple
import json
from langchain.schema.output import GenerationChunk
from langchain.callbacks.manager import CallbackManagerForLLMRun
import websocket
from langchain.tools import StructuredTool
from langchain.agents import ZeroShotAgent
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.base import BaseCallbackHandler
from langchain.agents import AgentExecutor
from langchain.schema import AgentAction, LLMResult, AgentFinish, OutputParserException
from threading import Thread
from queue import Queue, Empty

from langchain.chat_models import ChatOpenAI
from langchain.chat_models import ChatAnthropic
from langchain.agents import initialize_agent, AgentType
from langchain.tools import StructuredTool
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import AgentAction, LLMResult, AgentFinish, OutputParserException
from langchain.prompts import MessagesPlaceholder
from langchain.memory import ConversationBufferMemory

load_dotenv()

SUPERVISOR_API = os.environ.get("SUPERVISOR_API")
TEXTGEN_MODEL_URL = os.environ.get("TEXTGEN_MODEL_URL")
DEFAULT_ASSISTANT = os.environ.get("DEFAULT_ASSISTANT", "gpt3.5")

ASSISTANT_LIST = ["gpt3.5", "gpt4", "claude", "local"]

# chatbot style

ALREADY_CONVERTED_MARK = "<!-- ALREADY CONVERTED BY PARSER. -->"


def insert_newline_before_triple_backtick(text):
    modified_text = text.replace(" ```", "\n```")

    return modified_text


def insert_summary_block(text):
    pattern = r"(Action: CIL)(.*?)(Observation:|$)"
    replacement = r"<details><summary>CIL Code</summary>\n\1\n\2\n</details>\n"

    return re.sub(pattern, replacement, text, flags=re.DOTALL)


def postprocess(
    self, history: List[Tuple[str | None, str | None]]
) -> List[Tuple[str | None, str | None]]:
    markdown_converter = Markdown(
        extensions=["nl2br", "fenced_code"])

    if history is None or history == []:
        return []

    print(history)

    formatted_history = []
    for conversation in history:
        user, bot = conversation

        formatted_user = user
        # if user == None or user.endswith(ALREADY_CONVERTED_MARK):
        #     formatted_user = user
        # else:
        #     formatted_user = markdown_converter.convert(
        #         user) + ALREADY_CONVERTED_MARK

        if bot == None or bot.endswith(ALREADY_CONVERTED_MARK):
            formatted_bot = bot
        else:
            preformatted_bot = insert_newline_before_triple_backtick(bot)
            summary_bot = insert_summary_block(preformatted_bot)
            print(summary_bot)

            formatted_bot = markdown_converter.convert(
                summary_bot) + ALREADY_CONVERTED_MARK

        formatted_history.append((formatted_user, formatted_bot))

    return formatted_history


gr.Chatbot.postprocess = postprocess

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

    class CustomTextGen(TextGen):
        def _get_parameters(self, stop: Optional[List[str]] = None) -> Dict[str, Any]:
            """
            Performs sanity check, preparing parameters in format needed by textgen.

            Args:
                stop (Optional[List[str]]): List of stop sequences for textgen.

            Returns:
                Dictionary containing the combined parameters.
            """

            # Raise error if stop sequences are in both input and default params
            # if self.stop and stop is not None:
            combined_stop = []
            if self.stopping_strings and stop is not None:
                # combine
                combined_stop = self.stopping_strings + stop
                # raise ValueError("`stop` found in both the input and default params.")

            if self.preset is None:
                params = self._default_params
            else:
                params = {"preset": self.preset}

            # then sets it as configured, or default to an empty list:
            params["stop"] = combined_stop or self.stopping_strings or stop or []

            return params

        def _call(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
        ) -> str:
            """Call the textgen web API and return the output.

            Args:
                prompt: The prompt to use for generation.
                stop: A list of strings to stop generation when encountered.

            Returns:
                The generated text.

            Example:
                .. code-block:: python

                    from langchain.llms import TextGen
                    llm = TextGen(model_url="http://localhost:5000")
                    llm("Write a story about llamas.")
            """
            if self.streaming:
                combined_text_output = ""
                for chunk in self._stream(
                    prompt=prompt, stop=stop, run_manager=run_manager, **kwargs
                ):
                    combined_text_output += chunk.text
                print(prompt + combined_text_output)
                result = combined_text_output

            else:
                url = f"{self.model_url}/api/v1/generate"
                params = self._get_parameters(stop)
                request = params.copy()
                request["prompt"] = prompt
                response = requests.post(url, json=request)

                if response.status_code == 200:
                    result = response.json()["results"][0]["text"]
                    print(prompt + result)
                else:
                    print(f"ERROR: response: {response}")
                    result = ""

            return result

        def _stream(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
        ) -> Iterator[GenerationChunk]:
            """Yields results objects as they are generated in real time.

            It also calls the callback manager's on_llm_new_token event with
            similar parameters to the OpenAI LLM class method of the same name.

            Args:
                prompt: The prompts to pass into the model.
                stop: Optional list of stop words to use when generating.

            Returns:
                A generator representing the stream of tokens being generated.

            Yields:
                A dictionary like objects containing a string token and metadata.
                See text-generation-webui docs and below for more.

            Example:
                .. code-block:: python

                    from langchain.llms import TextGen
                    llm = TextGen(
                        model_url = "ws://localhost:5005"
                        streaming=True
                    )
                    for chunk in llm.stream("Ask 'Hi, how are you?' like a pirate:'",
                            stop=["'","\n"]):
                        print(chunk, end='', flush=True)

            """
            params = {**self._get_parameters(stop), **kwargs}

            url = f"{self.model_url}/api/v1/stream"

            request = params.copy()
            request["prompt"] = prompt

            websocket_client = websocket.WebSocket()

            websocket_client.connect(url)

            websocket_client.send(json.dumps(request))

            while True:
                result = websocket_client.recv()
                result = json.loads(result)

                if result["event"] == "text_stream":
                    chunk = GenerationChunk(
                        text=result["text"],
                        generation_info=None,
                    )
                    yield chunk
                elif result["event"] == "stream_end":
                    websocket_client.close()
                    return

                if run_manager:
                    run_manager.on_llm_new_token(token=chunk.text)

    def setup_assistant(assistant_instance, token_instance):
        assistant_name = assistant_instance["selected"]

        match assistant_name:
            case "gpt3.5":
                llm = ChatOpenAI(
                    temperature=0, model="gpt-3.5-turbo", streaming=True)
                max_iterations = 15
                approach = 1
            case "gpt4":
                llm = ChatOpenAI(temperature=0, model="gpt-4", streaming=True)
                max_iterations = 15
                approach = 1
            case "claude":
                # raise gr.Error("This assistant is not working properly yet.")
                llm = ChatAnthropic(
                    temperature=0, model="claude-2", streaming=True)
                max_iterations = 5
                approach = 2
            case "local":
                # raise gr.Error("This assistant is not working properly yet.")
                llm = CustomTextGen(model_url=TEXTGEN_MODEL_URL, temperature=0.1, max_new_tokens=1024, streaming=True, callbacks=[
                    StreamingStdOutCallbackHandler()], stopping_strings=["<|im_end|>", "<|im_sep|>", "Observation:"])
                max_iterations = 30
                approach = 2
            case _:
                raise gr.Error("Assistant not supported yet")
                # return {
                #     chatbot: chatbot_instance,
                #     msg: msg_instance,
                # }

        # get tool
        def code_interpreter_lite(code: str) -> str:
            """Execute the python code and return the result."""
            # handle markdown
            def extract_code_from_markdown(md_text):
                # Using regex to extract text between ```
                pattern = r"```[\w]*\n(.*?)```"
                match = re.search(pattern, md_text, re.DOTALL)
                if match:
                    return match.group(1).strip()
                else:
                    # might not be markdown
                    return md_text
            code = extract_code_from_markdown(code)

            code_response = requests.post(
                f'{SUPERVISOR_API}/run', headers={
                    'Authorization': f'Bearer {token_instance}'
                }, json={
                    'code_string': code
                }, timeout=600)
            if code_response.status_code != 200:
                raise Exception("No container created yet", code_response.text)
                # return {
                #     chatbot: chatbot_instance,
                #     msg: msg_instance,
                # }
            result = code_response.json()

            def is_base64(string):
                try:
                    # Try to decode the string as base64
                    base64.b64decode(string, validate=True)
                    return True
                except:
                    return False

            # handle base64 results - ie images
            if len(result) > 256:
                base64_string = result.split("\n")[1]
                # check if it base64
                if is_base64(base64_string):
                    decoded_data = base64.b64decode(base64_string)

                    # Generate the file name for the image
                    filename = "result.png"

                    # Save the image
                    with open(filename, "wb") as f:
                        f.write(decoded_data)

                    result = filename

                else:
                    result = "The result is too long to display."

            return result

        tool = StructuredTool.from_function(
            func=code_interpreter_lite, name="CIL", description="useful for running python code. The input should be a string of python code.")
        tools = [tool]

        memory = ConversationBufferMemory(
            memory_key="chat_history")

        if approach == 1:
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

        elif approach == 2:
            prefix = """<|im_start|>system
You are an assistant to a user who is trying to solve a question. You can write and execute Python code to find the solution.
You should only use the name of the tool to call it.
If you need to output any kind of graph to the user, you should save it in a file and return the file location.
You have access to the following tools:"""
            suffix = """Begin! Remember to use the tools with the correct format which is:
Action: CIL
Action Input: ```python
your code
```<|im_end|>
<|im_start|>user
Question: {input}<|im_end|>
<|im_start|>assistant
Thought: {agent_scratchpad}"""

            prompt = ZeroShotAgent.create_prompt(
                tools=[tool], prefix=prefix, suffix=suffix, input_variables=[
                    "input", "agent_scratchpad"]
            )

            model_url = TEXTGEN_MODEL_URL

            llm = CustomTextGen(model_url=model_url, temperature=0.1, max_new_tokens=1024,
                                streaming=True, stopping_strings=["<|im_end|>", "<|im_sep|>", "Observation:"])

            llm_chain = LLMChain(llm=llm, prompt=prompt)

            tool_names = [tool.name for tool in tools]
            agent = ZeroShotAgent(llm_chain=llm_chain, allowed_tools=tool_names,
                                  verbose=True, max_iterations=max_iterations)

            agent_executor_instance = AgentExecutor.from_agent_and_tools(
                agent=agent, tools=tools, memory=memory
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
            agent_executor: agent_executor_instance,
            chatbot: gr.update(value=[]),
            msg: gr.update(value="")
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

    def message_handle(chatbot_instance, msg_instance, file_list_instance):

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

        return {
            chatbot: chatbot_instance + [[chatbot_prompt, None]],
            msg: "",
            file_list: {},
        }

    def chatbot_handle(chatbot_instance, agent_executor_instance):

        # class ChatbotHandler(BaseCallbackHandler):
        #     def __init__(self):
        #         self.chatbot_response = ""
        #         super().__init__()

        #     def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> Any:
        #         self.chatbot_response += outputs.get("output", "") + '\n'

        #     def on_tool_end(self, output: str, **kwargs: Any) -> Any:
        #         self.chatbot_response += f'```\n{output}\n```\n'

        #     def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
        #         chatbot_thought = action.log.split("\n")[0]
        #         chatbot_thought = chatbot_thought.replace("Thought: ", "")

        #         if isinstance(action.tool_input, str):
        #             chatbot_tool_input_code_string = action.tool_input
        #         else:
        #             chatbot_tool_input_code_string = action.tool_input.get(
        #                 "code")
        #         self.chatbot_response += f"{chatbot_thought}\n"
        #         self.chatbot_response += f'```\n{chatbot_tool_input_code_string}\n```\n'

        #     def get_chatbot_response(self):
        #         return self.chatbot_response

        class QueueCallback(BaseCallbackHandler):
            """Callback handler for streaming LLM responses to a queue."""

            def __init__(self, queue):
                self.queue = queue

            def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
                self.queue.put(token)

            def on_tool_end(self, output: str, **kwargs: Any) -> None:
                self.queue.put(f'Observation: \n```\n{output}\n```\n')

            def on_llm_end(self, *args, **kwargs: Any) -> None:
                return self.queue.empty()

        streaming_queue = Queue()
        job_done = object()

        user_message = chatbot_instance[-1][0]

        def task():
            try:
                agent_executor_instance(
                    user_message, callbacks=[QueueCallback(streaming_queue)])
                streaming_queue.put(job_done)
            except OutputParserException as error:
                streaming_queue.put(job_done)
                raise gr.Error(
                    "Assistant could not handle the request. Error: " + str(error))
            except Exception as error:
                streaming_queue.put(job_done)
                raise gr.Error(
                    "Server could not handle the request. Error: " + str(error))

        streaming_thread = Thread(target=task)
        streaming_thread.start()

        chatbot_instance[-1][1] = ""

        while True:
            try:
                next_token = streaming_queue.get(True, timeout=1)
                if next_token is job_done:
                    break
                chatbot_instance[-1][1] += next_token
                yield chatbot_instance
            except Empty:
                continue

        # try:
        #     chatbotHandler = ChatbotHandler()
        #     agent_executor_instance(
        #         chatbot_prompt, callbacks=[chatbotHandler])
        #     chatbot_response = chatbotHandler.get_chatbot_response()

        # except OutputParserException as e:
        #     raise gr.Error(
        #         "Assistant could not handle the request. Error: " + str(e))

        # chatbot_instance.append((msg_instance, chatbot_response))

        # return {
        #     chatbot: chatbot_instance,
        #     msg: "",
        #     agent_executor: agent_executor_instance
        # }

    button.click(create_container, [username, assistant], [
        token_box, token, container_row, chatbot_column, agent_executor])
    assistant_selection.input(
        select_assistant, [assistant, assistant_selection, token], [assistant, assistant_selection, assistant_box, agent_executor, chatbot, msg])
    file_upload.upload(upload_file, [file_upload, token, file_list], [
        file_list, file_output])
    send.click(message_handle, [chatbot, msg, file_list], [
               chatbot, msg, file_list], queue=False).then(
        chatbot_handle, [chatbot, agent_executor], [chatbot]
    )

    manual_download_send.click(download_file, [
                               manual_download_text, token], manual_download_file)

demo.queue()
demo.launch(server_name="0.0.0.0", server_port=7861)
