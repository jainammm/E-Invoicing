FROM python:3.6-slim

ENV PYTHONUNBUFFERED 1

RUN apt-get update
RUN apt-get install -y python-opencv
RUN apt-get install -y poppler-utils

RUN mkdir /code
WORKDIR /code
COPY requirements.txt /code/
RUN pip install -r requirements.txt
COPY . /code/

EXPOSE 8000

ENTRYPOINT [ "uvicorn", "app:app", "--host=0.0.0.0"]