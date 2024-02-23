from python:3.11.1-buster

WORKDIR /

RUN pip install runpod

ADD func.py .

CMD [ "python", "-u", "/func.py" ]