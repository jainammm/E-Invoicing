FROM python:3.6-slim

ENV PYTHONUNBUFFERED 1

RUN apt-get update && apt-get install -y libenchant1c2a

RUN mkdir /code
WORKDIR /code
COPY requirements.txt /code/
RUN pip install -r requirements.txt
COPY . /code/

EXPOSE 8001

ENTRYPOINT [ "uvicorn", "app:app", "--port=8001", "--host=0.0.0.0" ]