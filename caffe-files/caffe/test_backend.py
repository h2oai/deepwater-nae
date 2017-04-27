#!/usr/bin/env python3

import numpy as np

import backend
import deepwater_pb2

backend.size = 1
backend.gpus = []
res = deepwater_pb2.Cmd()

sizes = [784, 1024, 1024, 512, 10]
types = ['data', 'relu', 'relu', 'relu', 'loss']
drops = [.8, .5, .5, .5, 0]

create = deepwater_pb2.Cmd()
create.type = deepwater_pb2.Create
create.input_shape.extend([64, 1, 1, sizes[0]])
create.solver_type = 'SGD'
create.sizes.extend(sizes)
create.types.extend(types)
create.dropout_ratios.extend(drops)
create.learning_rate = .01
create.momentum = .9
backend.message(create, res)

for i in range(2):
    train = deepwater_pb2.Cmd()
    train.type = deepwater_pb2.Train
    train.input_shape.extend(create.input_shape)
    batch = np.zeros(train.input_shape, dtype = np.float32)
    label = np.zeros([train.input_shape[0], 1], dtype = np.float32)
    train.data.extend([
        batch.tobytes(),
        label.tobytes(),
    ])
    backend.message(train, res)

    pred = deepwater_pb2.Cmd()
    pred.type = deepwater_pb2.Predict
    pred.input_shape.extend(create.input_shape)
    batch = np.zeros(pred.input_shape, dtype = np.float32)
    pred.data.extend([
        batch.tobytes(),
    ])
    backend.message(pred, res)
