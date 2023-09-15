# coding=utf8
"""================================
@Author: Mr.Chang
@Date  : 2023/9/15 11:07
==================================="""
import gradio as gr
import mdtex2html

from chat_jojo import openai_chat, get_embedding, CustomMilvus,minimax_chat_chain, get_milvus_client
import os
from dotenv import load_dotenv, find_dotenv
from chat_jojo.consts import JOJO_KNOWLEDGE_NAME, EMBEDDING_MODEL_NAME
from chat_jojo.prompts.minimax_prompts import \
    (MiniMax_ChatMessage,
     NotContext_HumanMessage,
     SYSTEM_QUESTION_SPLIT_PROMPT,
     USER_QUESTION_SPLIT_PROMPT)
from utils import assemble_prompt,history2_str

from loguru import logger
from copy import deepcopy

load_dotenv(find_dotenv())
# os.environ['http_proxy'] = 'http://127.0.0.1:7890'
# os.environ['https_proxy'] = 'http://127.0.0.1:7890'

logger.add("logs/web_demo.log", rotation="500 MB", encoding="utf-8", enqueue=True, compression="zip", retention="10 days")
vector_db = get_milvus_client()
# openai_chat_chain, memory = openai_chat(milvus_vectordb=vector_db, verbose=True)


def postprocess(self, y):
    if y is None:
        return []
    for i, (message, response) in enumerate(y):
        y[i] = (
            None if message is None else mdtex2html.convert((message)),
            None if response is None else mdtex2html.convert(response),
        )
    return y


gr.Chatbot.postprocess = postprocess


def process_history(history):
    knowledge = ''
    for t in history[::-1]:
        content = t.content
        if "====" in content:
            splits = content.split("====")
            if len(splits) == 3:
                knowledge = splits[1]
                b = splits[2]
                t.content = b
                break


    return history, knowledge



pre_search_text = ''

def predict(input, chatbot, history, historys, search_text):
    if input == "":
        return chatbot, history,historys
    input_ = input.strip()

    logger.debug(f"input:{input_}")
    turbo_chain = minimax_chat_chain(model_name="abab5.5-chat", temperature=0.01, verbose=True)
    # question compress
    # if len(input_) > 15:
    #     messages = [MINIMAX_SYSTEM_COMPRESS_PROMPT] + [Compress_NotContext_HumanMessage.format(question=input_)]
    #     search_input_ = turbo_chain(messages).content
    #     if '简化后的问题：' in search_input_:
    #         input_ = search_input_.split('简化后的问题：')[1].strip()
    #         # search_input_ = input_
    #     logger.debug(f"compress input:{search_input_}")
    # else:
    #     search_input_ = input_
    search_result = vector_db.similarity_search_with_score(input_, k=1)
    search_result_temp = [(item, score) for item,score in search_result if score > 0.75]

    chatbot.append((input_, ""))
    # search the answer
    global pre_search_text
    if len(input_) < 45:
        search_text = ''
        for i, (item, score) in enumerate(search_result_temp):

            logger.debug(f"search text:{item.metadata['content_']}\nscore:{score}")

            search_text += f"{item.metadata['content_']}\n"
        # if talk about the same thing, then do not search again
        search_text = search_text.strip()

        if pre_search_text != search_text:
            pre_search_text = search_text
            historys = []
            system_messages = [MiniMax_ChatMessage.format(answer=search_text)]
        else:
            system_messages = [MiniMax_ChatMessage.format(answer=pre_search_text)]
    else:

        system_messages = [SYSTEM_QUESTION_SPLIT_PROMPT]+[USER_QUESTION_SPLIT_PROMPT.format(question=input_)]
        ai_message = turbo_chain(system_messages)
        prefix = "抱歉，我需要进一步了解您的问题，请您简明扼要的描述，例如：\n"
        response = prefix + ai_message.content
        logger.debug(f"response:{response}")

        ai_message.content = response
        historys.append(system_messages[-1])
        historys.append(ai_message)
        # memory.save_context({'input': input}, {'output': response})
        history.append((input_, response))
        chatbot[-1] = (input, response)
        logger.debug(f"final response:{response}")
        return chatbot, history, historys, search_text
    logger.debug(f"system message: {system_messages[0].content}")
    messages = [NotContext_HumanMessage.format(question=input_)]
    if len(historys) > 8:
        historys = historys[-8:]

    messages = system_messages +  historys + messages
    logger.debug(f"input text: {messages[-1].content}")

    # get response

    ai_message = turbo_chain(messages)
    response = ai_message.content
    logger.debug(f"response:{response}")
    if '根据“====”中的信息，' in response:
        response = response.replace('根据“====”中的信息，', '')
    if '答案：' in response:
        response = response.replace('答案：', '')
    if 'IOS 9.0' in response or 'iOS 9.0' in response:
        response = response.replace('IOS 9.0', 'IOS 10.0').replace('iOS 9.0', 'iOS 10.0')
        if 'iPod touch' in response:
            response = response.replace('iPod touch', '手机')
    if '根据您提供的信息，' in response:
        response = response.replace('根据您提供的信息，', '')
    ai_message.content = response
    historys.append(messages[-1])
    historys.append(ai_message)
    # memory.save_context({'input': input}, {'output': response})
    history.append((input_, response))
    chatbot[-1] = (input, response)
    logger.debug(f"final response:{response}")
    return chatbot, history, historys, search_text


def reset_user_input():
    return gr.update(value='')


def reset_state():
    # memory.clear()
    return [], [], [], ''


with gr.Blocks() as demo:
    gr.HTML("""<h1 align="center">叫叫小助手</h1>""")
    chatbot = gr.Chatbot()
    with gr.Row():
        with gr.Column(scale=4):
            with gr.Column(scale=12):
                user_input = gr.Textbox(show_label=False, placeholder="Input...", lines=10)
            with gr.Column(min_width=32, scale=1):
                submitBtn = gr.Button("Submit", variant="primary")
        with gr.Column(scale=1):
            emptyBtn = gr.Button("Clear History")
            # max_length = gr.Slider(0, 32768, value=8192, step=1.0, label="Maximum length", interactive=True)
            # top_p = gr.Slider(0, 1, value=0.8, step=0.01, label="Top P", interactive=True)
            # temperature = gr.Slider(0, 1, value=0.95, step=0.01, label="Temperature", interactive=True)

    history = gr.State([])
    historys = gr.State([])
    search_text = gr.State('')
    submitBtn.click(predict, [user_input, chatbot, history,historys, search_text],
                    [chatbot, history, historys, search_text], show_progress=True)
    submitBtn.click(reset_user_input, [], [user_input])

    emptyBtn.click(reset_state, outputs=[chatbot, history, historys,search_text], show_progress=True)

demo.queue(concurrency_count=8).launch(share=True, server_name='0.0.0.0', server_port=8093)
