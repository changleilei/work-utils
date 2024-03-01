
# Welcome to Cursor



# 1. Try generating with command K on a new line. Ask for a pytorch script of a feedforward neural network
# 2. Then, select the outputted code and hit chat. Ask if there's a bug. Ask how to improve.
# 3. Try selecting some code and hitting edit. Ask the bot to add residual layers.
# 4. To try out cursor on your own projects, go to the file menu (top left) and open a folder.

import gradio as gr
import openai

import os
os.environ['http_proxy'] = 'http://127.0.0.1:7890'
os.environ['https_proxy'] = 'http://127.0.0.1:7890'

# openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = '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'
history = []
pre_command = ''


def execute_command(command, text, option, clean):
    global history
    global pre_command
    if command != pre_command:
        pre_command = command
        history = []
    if clean:
        history = []
        return "History cleared."
    role_template = {'role': 'user', 'content': text}
    assistant_template = {'role': 'assistant', 'content': ''}
    if option == "ChatGPT":

        if history:
            history.append(role_template)
            messages = history
        else:
            role_template['content'] = command + '\n' + text
            messages = [role_template]


        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages
            )

        if response:
            print(response)

            message = response.choices[0].message['content']
            history.append(role_template)
            assistant_template['content'] = message
            history.append(assistant_template)
        else:
            message = 'Ops! Something was wrong. Please try again.'
        return message.strip()

    elif option == "Davinci-003":
        if history:
            prompt = '\n'.join([h['role'] + ':' + h['content'] for h in history])
            prompt = command + '\n' + prompt
            prompt += '\nuser: ' + text + '\n' + 'assistant: '
        else:
            prompt = command + '\n' + text + '\n'
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=1024,
            n=1,
            stop=None,
            temperature=0.95,
        )
    else:

        prompt = command + '\n' + text + '\n'
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=1024,
            n=1,
            stop=None,
            temperature=0.95,
        )
    print(response)
    if response:
        message = response.choices[0].text
        history.append(role_template)
        assistant_template['content'] = message
        history.append(assistant_template)
    else:
        message = 'Ops! Something was wrong. Please try again.'
    return message.strip()

command_input = gr.inputs.Textbox(label="指令")
text_input = gr.inputs.Textbox(label="输入你的问题")
option_input = gr.inputs.Radio(["ChatGPT", "Davinci-003"], label="Select a model")
# clear = gr.inputs.Radio(['Clear'],label="Clear")
clear = gr.components.Checkbox(label="Clear")
output_text = gr.outputs.Textbox(label="输出")

gr.Interface(fn=execute_command,
             inputs=[command_input, text_input, option_input, clear],
             outputs=output_text,
             title="智者").launch(share=True, auth=('chang', 'chang123'),server_name='0.0.0.0', server_port=7080)

