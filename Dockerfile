FROM python:3.10.10

COPY . /code
WORKDIR "/code"
# Install dependencies
RUN apt update
RUN apt install -y python3-pip

RUN pip install --upgrade pip
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN apt-get install build-essential python3-dev
# RUN python -m apk add build-base
# RUN python -m pip install numpy 

RUN pip install -r requirements.txt

USER root
RUN ["chmod","+x","runserver.sh"] 
CMD ./runserver.sh