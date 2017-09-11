#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
from grpc.beta import implementations
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2


def request_model_result(input_data, host, port, model_name, model_version,
                         request_timeout):
    input_tensor_proto = {}
    for key in input_data:
        input_tensor_proto[key] = tf.make_tensor_proto(
            input_data[key], dtype=tf.float32)

    # Create gRPC client and request
    channel = implementations.insecure_channel(host, port)
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
    request = predict_pb2.PredictRequest()
    request.model_spec.name = model_name
    if type(model_version) != int or model_version < 0:
        print('Invalid version number!')
    elif model_version > 0:
        request.model_spec.version.value = model_version
    for key in input_tensor_proto:
        request.inputs[key].CopyFrom(input_tensor_proto[key])

    # Send request
    raw_result = stub.Predict(request, request_timeout)
    return raw_result
