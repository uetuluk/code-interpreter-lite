import gradio as gr
from typing import List, Tuple
# from markdown2 import Markdown
from markdown import Markdown
import re

ALREADY_CONVERTED_MARK = "<!-- ALREADY CONVERTED BY PARSER. -->"


def insert_newline_before_triple_backtick(text):
    modified_text = text.replace(" ```", "\n```")

    return modified_text


def insert_summary_block(text):
    pattern = r"(Action: CIL)(.*?)(Observation:|$)"
    replacement = r"<details><summary>CIL Code</summary>\n\1\2</details>\n\3"

    return re.sub(pattern, replacement, text, flags=re.DOTALL)


def postprocess(
    self, history: List[Tuple[str | None, str | None]]
) -> List[Tuple[str | None, str | None]]:
    markdown_converter = Markdown(
        extensions=["nl2br", "fenced_code"])

    if history is None or history == []:
        return []

    formatted_history = []
    for conversation in history:
        user, bot = conversation

        if user.endswith(ALREADY_CONVERTED_MARK):
            formatted_user = user
        else:
            formatted_user = markdown_converter.convert(
                user) + ALREADY_CONVERTED_MARK

        if bot.endswith(ALREADY_CONVERTED_MARK):
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

    chatbot = gr.Chatbot([["""```
    a
    ```""",
                           """I will need to write a Python function that takes in the user's name and returns their age. To do this, I can create a dictionary of names and ages and then look up the user's name to find their age.
Action: CIL
Action Input: ```python
def get_age(name):
    ages = {
        'Alice': 25,
        'Bob': 30,
        'Charlie': 40
    }
    return ages[name] if name in ages else None
```
Observation: Code executed successfully.
Thought: The code should work for any name that is present in the dictionary. If the name is not found, it will return None.
Final Answer: The function returns the age of a user based on their name. For example, `get_age('Alice')` would return 25 and `get_age('Eve')`, which is not in the dictionary, would return None."""
                           ]])

# demo.queue()
demo.launch()
