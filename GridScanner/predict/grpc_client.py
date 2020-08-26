from __future__ import print_function

import argparse
import time
import numpy as np
from scipy.misc import imread
import json

import grpc
import tensorflow as tf
from tensorflow.contrib.util import make_tensor_proto

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

from config.model_config import model_params


def get_model_output(data):
    '''
    Call tensorflow model server through grpc for model output
    '''

    host, port, model, signature_name = \
        model_params.tensorflow_host, model_params.tensorflow_port, \
            model_params.tensorflow_model, model_params.tensorflow_signature_name

    channel = grpc.insecure_channel('{host}:{port}'.format(host=host, port=port))
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

    start = time.time()

    # Call classification model to make prediction on the image
    request = predict_pb2.PredictRequest()
    request.model_spec.name = model
    request.model_spec.signature_name = signature_name
    request.inputs['input'].CopyFrom(make_tensor_proto(data['grid_table'], shape=data['grid_table'].shape))

    result = stub.Predict(request, 10.0)

    outputs_tensor_proto = result.outputs['output']
    shape = tf.TensorShape(outputs_tensor_proto.tensor_shape)

    outputs = np.array(outputs_tensor_proto.float_val).reshape(shape.as_list())

    return outputs
