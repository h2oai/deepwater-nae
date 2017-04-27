#!/usr/bin/env python3

import sys
import solver
import numpy as np
import deepwater_pb2
cmd = deepwater_pb2.Cmd()

cmd.type = deepwater_pb2.Create
cmd.graph = 'lenet'
cmd.input_shape.extend([128, 1, 28, 28])
cmd.solver_type = 'SGD'
cmd.learning_rate = .01
cmd.momentum = .9

size = 1
gpus = []

s = solver.Solver(cmd, None, size, gpus)
solver.solver = s
s.start(0)

for _ in range(0, 2):
    for i in range(2):
        batch = np.zeros(cmd.input_shape, dtype = np.float32)
        label = np.zeros([cmd.input_shape[0], 1], dtype = np.float32)
        s.buffs = [batch, label]
        s.caffe.step(1)

    batch = np.zeros(cmd.input_shape, dtype = np.float32)
    print(solver.predict(batch))
