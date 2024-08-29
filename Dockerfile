FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 -y

COPY requirements.txt .
RUN pip install -r requirements.txt
