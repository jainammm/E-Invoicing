import os

class modelParams:
    def __init__(self):
        self.ckpt_path = os.environ.get("CKPT_PATH")
        self.ckpt_file = os.environ.get("CKPT_FILE")
        self.text_case = os.environ.get("TEXT_CASE", False)
        self.tokenize = os.environ.get("TOKENIZE")
        self.dict_path = os.environ.get("DICT_PATH")
        self.load_dict_from_path = os.environ.get("DICT_PATH")
        self.embedding_size = os.environ.get("embedding_size", 256)

model_params = modelParams()