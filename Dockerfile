FROM python:3.8-slim-buster
WORKDIR /app
COPY . /app

RUN apt update -y && apt install awscli -y

RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 unzip && pip install --upgrade pip && pip install -r requirements.txt --log pip-log.txt

#RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 unzip -y

#RUN pip install -r requirements.txt
#ffmpeg, libsm6, libxext6, and unzip are system-level dependencies that might be required for 
#the Python application or for handling media files (like images or videos).

CMD ["python3", "application.py"]