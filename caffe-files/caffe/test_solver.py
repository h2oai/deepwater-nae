#!/usr/bin/env python3

from multiprocessing import *
from solver import *
import numpy as np
import deepwater_pb2
cmd = deepwater_pb2.Cmd()

cmd.type = deepwater_pb2.Create
cmd.input_shape.extend([128, 1, 1, 784])
cmd.solver_type = 'SGD'
cmd.sizes.extend([784, 4096, 4096, 10])
cmd.types.extend(['data', 'relu', 'relu', 'loss'])
cmd.dropout_ratios.extend([.1, .1, .1, 0])
cmd.learning_rate = .01
cmd.momentum = .9

size = 2
uid = None
gpus = []

pool = Pool(size,
            initializer = create,
            initargs = (cmd, uid, size, gpus))

pool.map(start, list(range(size)))

for i in range(10):
    batch = np.zeros(cmd.input_shape, dtype = np.float32)
    label = np.zeros([cmd.input_shape[0], 1], dtype = np.float32)

    if i % 10 == 0:
        print('map', (batch.shape, label.shape), file = sys.stderr)
    tmp = list(zip(np.split(batch, size), np.split(label, size)))
    if i % 10 == 0:
        print([(x[0].shape, x[1].shape) for x in tmp])
    pool.map(train, tmp)

r = pool.map(predict, np.split(batch, size))
r = np.concatenate(r)
print(r.shape)
