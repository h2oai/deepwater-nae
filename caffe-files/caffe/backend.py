#!/usr/bin/env python3
"""
"""

import os, sys, struct
import numpy as np
from multiprocessing import *

try:
    print('PYTHONPATH: ' + os.environ['PYTHONPATH'], file = sys.stderr)
except:
    pass

import deepwater_pb2
from solver import *

gpus = None  # automatic
pool = None
size = 1

if gpus is None:
    # Separate process or CUDA fails for later forks
    def get_gpus(pipe):
        try:
            import pycuda.driver as drv
            drv.init()
            cnt = drv.Device.count()
            pipe.send(list(range(cnt)))
            pipe.close()
            print('Detected', cnt, 'GPUs', file = sys.stderr)
        except Exception as e:
            print(e, file = sys.stderr)
            print('No GPU found!', file = sys.stderr)
            pipe.send([])
            pipe.close()


    parent, child = Pipe()
    p = Process(target = get_gpus, args = (child,))
    p.start()
    gpus = parent.recv()
    p.join()


def read_cmd():
    cmd = deepwater_pb2.Cmd()
    size_bytes = sys.stdin.buffer.read(4)
    if len(size_bytes) == 0:
        return None
    size = struct.unpack('>I', size_bytes)[0]
    data = sys.stdin.buffer.read(size)
    cmd.ParseFromString(data)
    if debug:
        print('Received', size, file = sys.stderr)
    return cmd


def write_cmd(cmd):
    data = cmd.SerializeToString()
    sys.stdout.buffer.write(struct.pack('>I', len(data)))
    sys.stdout.buffer.write(data)
    sys.stdout.buffer.flush()
    if debug:
        print('Sent', len(data), file = sys.stderr)


def message(req, res):
    global gpus, pool, size

    if req.type == deepwater_pb2.Create:
        if debug:
            from google.protobuf import text_format
            print(text_format.MessageToString(req), file = sys.stderr)

        uid = None

        if len(gpus) > 1:
            uid = caffe.NCCL.new_uid()
            size = len(gpus)
        if req.input_shape[0] % size != 0:
            raise Exception('Batch size must be divisible by GPU count')

        pool = Pool(size,
                    initializer = create,
                    initargs = (req, uid, size, gpus))
        pool.map(start, list(range(size)))
        res.type = deepwater_pb2.Success

    if req.type == deepwater_pb2.Train:
        batch = np.frombuffer(req.data[0], dtype = np.float32)
        label = np.frombuffer(req.data[1], dtype = np.float32)
        assert batch.size == np.prod(np.array(req.input_shape)), \
            '%s %s' % (str(batch.shape), str(req.input_shape))
        batch = batch.reshape(req.input_shape)
        label = label.reshape([req.input_shape[0], 1])
        data = list(zip(np.split(batch, size), np.split(label, size)))
        pool.map(train, data)
        res.type = deepwater_pb2.Success

    if req.type == deepwater_pb2.Predict:
        batch = np.frombuffer(req.data[0], dtype = np.float32)
        batch = batch.reshape(req.input_shape)
        data = np.split(batch, size)
        data = pool.map(predict, data)
        data = np.concatenate(data)
        res.data.append(data.tobytes())
        res.type = deepwater_pb2.Success

    if req.type == deepwater_pb2.SaveGraph:
        print('Saving graph to', req.path, file = sys.stderr)
        pool.map(save_graph, [req.path])

    if req.type == deepwater_pb2.Save:
        print('Saving model to', req.path, file = sys.stderr)
        pool.map(save, [req.path])

    if req.type == deepwater_pb2.Load:
        print('Loading model to', req.path, file = sys.stderr)
        pool.map(load, [req.path])


if __name__ == '__main__':
    print('Started Caffe backend', file = sys.stderr)
    while True:
        req = read_cmd()
        if req is None:
            break
        res = deepwater_pb2.Cmd()
        message(req, res)
        write_cmd(res)
