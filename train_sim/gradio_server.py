# # coding=utf8
# """================================
# @Author: Mr.Chang
# @Date  : 2023/3/31 4:48 PM
# ==================================="""
# import gradio as gr
#
# # Load whisper model
# import whisper
#
# whisper_model = whisper.load_model("base")
#
# # Define route for speech recognition
#
# async def speech_recognition(file: UploadFile = File(...)):
#     # Save uploaded file
#     with open(file.filename, "wb") as buffer:
#         buffer.write(await file.read())
#     # Use whisper model for speech recognition
#     result = whisper_model.transcribe(file.filename)
#     # Return recognized text
#     return {"text": result['text']}
# gr.Interface(
#     ai_text,
#     inputs="textbox",
#     outputs=[
#         HighlightedText(
#             label="Output",
#             show_legend=True,
#         ),
#         JSON(
#             label="JSON Output"
#         )
#     ],
#     title="Chinese Spelling Correction",
#     description="Copy or input error Chinese text. Submit and the machine will correct text.",
#     article="Link to <a href='https://github.com/shibing624/pycorrector' style='color:blue;' target='_blank\'>Github REPO</a>",
#     examples=examples
# ).launch(share=True, server_name='0.0.0.0', server_port=8081)
