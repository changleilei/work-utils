# coding=utf8
"""================================
@Author: Mr.Chang
@Date  : 2023/3/31 4:31 PM
==================================="""
# Import necessary libraries
import uvicorn
from fastapi import FastAPI, File, UploadFile
from starlette.responses import FileResponse
from typing import List
import os
import subprocess

# Create FastAPI instance
app = FastAPI()

# Load whisper model
import whisper

whisper_model = whisper.load_model("base")

# Define route for speech recognition
@app.post("/speech_recognition/")
async def speech_recognition(file: UploadFile = File(...)):
    # Save uploaded file
    with open(file.filename, "wb") as buffer:
        buffer.write(await file.read())
    # Use whisper model for speech recognition
    result = whisper_model.transcribe(file.filename)
    # Return recognized text
    return {"text": result['text']}

# Run the app with uvicorn
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8086)
