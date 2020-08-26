import os

class modelParams:
    def __init__(self):
        self.text_case = os.environ.get("TEXT_CASE", False)
        self.tokenize = os.environ.get("TOKENIZE")
        self.dict_path = os.environ.get("DICT_PATH")
        self.load_dict_from_path = os.environ.get("DICT_PATH")
        self.embedding_size = os.environ.get("embedding_size", 256)
        self.tensorflow_host = os.environ.get("TENSORFLOW_HOST", 'localhost')
        self.tensorflow_port = os.environ.get("TENSORFLOW_PORT", 8500)
        self.tensorflow_model = os.environ.get("TENSORFLOW_MODEL", 'CUTIE')
        self.tensorflow_signature_name = \
            os.environ.get("TENSORFLOW_SIGNATURE_NAME", 'serving_default')

        self.invoiceholder_host = os.environ.get("INVOICEHOLDER_HOST", 'localhost')
        self.invoiceholder_port = os.environ.get("INVOICEHOLDER_PORT", 8001)

model_params = modelParams()