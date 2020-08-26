FROM tensorflow/serving

RUN mkdir /code
WORKDIR /code
COPY . /code/

ENTRYPOINT [ "tensorflow_model_server", "--port=8500", "--model_name=CUTIE", "--model_base_path=/code/model_for_serving/CUTIE" ]