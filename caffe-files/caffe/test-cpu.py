from __future__ import print_function

from multiprocessing import *
from solver import *
import numpy as np
import deepwater_pb2
cmd = deepwater_pb2.Cmd()

cmd.type = deepwater_pb2.Create
cmd.batch_size = 256
cmd.solver_type = 'SGD'
cmd.sizes.extend([784, 1024, 512, 10])
cmd.types.extend(['data', 'relu', 'relu', 'loss'])
cmd.dropout_ratios.extend([.8, .5, .5, 0])
cmd.learning_rate = .01
cmd.momentum = .9

size = 1
uid = None
gpus = []

pool = Pool(size,
            initializer = create,
            initargs = (cmd, uid, size, gpus))

pool.map(start, range(size))

for i in range(10):
    batch = np.zeros([cmd.batch_size, 1, 1, cmd.sizes[0]], dtype = np.float32)
    label = np.zeros([cmd.batch_size, 1], dtype = np.float32)

    print('map', (batch.shape, label.shape), file=sys.stderr)
    tmp = zip(np.split(batch, size), np.split(label, size))
    print([(x[0].shape, x[1].shape) for x in tmp])
    pool.map(train, tmp)

r = pool.map(predict, np.split(batch, size))
r = np.concatenate(r)
print(r.shape)
