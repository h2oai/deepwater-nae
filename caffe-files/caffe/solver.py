#!/usr/bin/env python3

import sys, os
import caffe
import numpy as np
import deepwater_pb2
from caffe.proto import caffe_pb2
from google.protobuf import text_format
from caffe import layers as L

debug = True


class Solver:
    def __init__(self, cmd, uid, size, gpus):
        self.cmd = cmd
        self.uid = uid
        self.size = size
        self.gpus = gpus
        self.rank = 0
        self.device = None
        self.caffe = None
        self.buffs = None
        self.graph = None

        log = INFO = 0
        # log = WARNING = 1
        # log = ERROR = 2
        # log = FATAL = 3
        caffe.init_log(log)

    def start(self, rank):
        self.rank = rank

        if len(self.gpus) > 0:
            self.device = self.gpus[rank]
            if debug:
                s = 'solver gpu %d' % self.gpus[self.rank] + \
                    ' pid %d' % os.getpid() + ' size %d' % self.size + \
                    ' rank %d' % self.rank
                print(s, file = sys.stderr)
            caffe.set_mode_gpu()
            caffe.set_device(self.device)
            caffe.set_solver_count(self.size)
            caffe.set_solver_rank(self.rank)
            caffe.set_multiprocess(True)
        else:
            print('solver cpu', file = sys.stderr)
            caffe.set_mode_cpu()

        if self.cmd.graph.endswith('.json'):
            with open(self.cmd.graph, mode = 'r') as f:
                graph = caffe_pb2.SolverParameter()
                text_format.Merge(f.read(), graph)
                self.graph = graph
        else:
            self.graph = self.solver_graph()

        import tempfile
        with tempfile.NamedTemporaryFile(mode = 'w+', delete = False) as f:
            text_format.PrintMessage(self.graph, f)
            tmp = f.name
        self.caffe = caffe.AdamSolver(tmp)

        if self.uid:
            self.nccl = caffe.NCCL(self.caffe, self.uid)
            self.nccl.bcast()
            self.caffe.add_callback(self.nccl)
            if self.caffe.param.layer_wise_reduce:
                self.caffe.net.after_backward(self.nccl)

    def solver_graph(self):
        proto = caffe_pb2.SolverParameter()
        proto.type = self.cmd.solver_type
        if self.device is not None:
            proto.solver_mode = caffe_pb2.SolverParameter.SolverMode.Value(
                'GPU')
            proto.device_id = self.device
        else:
            proto.solver_mode = caffe_pb2.SolverParameter.SolverMode.Value(
                'CPU')
        proto.lr_policy = 'fixed'
        proto.base_lr = self.cmd.learning_rate
        proto.momentum = self.cmd.momentum
        proto.max_iter = int(2e9)
        proto.random_seed = self.cmd.random_seed + self.rank
        print('Setting seed ', proto.random_seed, file = sys.stderr)
        proto.display = 1

        batch = int(solver.cmd.input_shape[0] / solver.size)
        if self.cmd.graph:
            dir = os.path.dirname(os.path.realpath(__file__))
            proto.net = dir + '/' + self.cmd.graph + '.prototxt'
        else:
            proto.train_net_param.MergeFrom(self.net_def(caffe.TRAIN))
            proto.test_net_param.add().MergeFrom(self.net_def(caffe.TEST))

        proto.test_iter.append(1)
        proto.test_interval = 999999999  # cannot disable or set to 0
        proto.test_initialization = False
        return proto

    def net_def(self, phase):
        print('sizes', self.cmd.sizes, file = sys.stderr)
        print('types', self.cmd.types, file = sys.stderr)
        if len(self.cmd.sizes) != len(self.cmd.types):
            raise Exception

        n = caffe.NetSpec()
        name = ''

        for i in range(len(self.cmd.types)):
            if self.cmd.types[i] == 'data':
                name = 'data'
                if phase == caffe.TRAIN:
                    n[name], n.label = L.Python(
                        module = 'solver',
                        layer = 'DataLayer',
                        ntop = 2,
                    )
                else:
                    n[name] = L.Python(
                        module = 'solver',
                        layer = 'DataLayer',
                    )

            else:
                fc = L.InnerProduct(
                    n[name],
                    inner_product_param = {'num_output': self.cmd.sizes[i],
                                           'weight_filler': {'type': 'xavier',
                                                             'std': 0.1},
                                           'bias_filler': {'type': 'constant',
                                                           'value': 0}})
                name = 'fc%d' % i
                n[name] = fc

                if self.cmd.types[i] == 'relu':
                    relu = L.ReLU(n[name], in_place = True)
                    name = 'relu%d' % i
                    n[name] = relu
                elif self.cmd.types[i] == 'loss':
                    if self.cmd.regression:
                        if phase == caffe.TRAIN:
                            n.loss = L.EuclideanLoss(n[name], n.label)
                    else:
                        if phase == caffe.TRAIN:
                            n.loss = L.SoftmaxWithLoss(n[name], n.label)
                        else:
                            n.output = L.Softmax(n[name])
                else:
                    raise Exception('TODO unsupported: ' + self.cmd.types[i])

        return n.to_proto()


solver = None


class DataLayer(caffe.Layer):
    def setup(self, bottom, top):
        pass

    def reshape(self, bottom, top):
        batch = int(solver.cmd.input_shape[0] / solver.size)
        input_shape = [batch,
                       solver.cmd.input_shape[1],
                       solver.cmd.input_shape[2],
                       solver.cmd.input_shape[3], ]
        top[0].reshape(*input_shape)
        print('top[0] shape ', list(top[0].shape), file = sys.stderr)
        if self.phase == caffe.TRAIN:
            top[1].reshape(batch, 1)
            print('top[1] shape ', list(top[1].shape), file = sys.stderr)

    def forward(self, bottom, top):
        print('forward ', list(top[0].shape), file = sys.stderr)
        assert len(solver.buffs) == len(top), \
            '%d %d' % (len(solver.buffs), len(top))
        for i in range(len(solver.buffs)):
            assert solver.buffs[i].size == top[i].data.size, \
                '%s %s' % (str(solver.buffs[i].shape), str(top[i].data.shape))
            batch = solver.buffs[i].reshape(top[i].data.shape)
            top[i].data[...] = batch

    def backward(self, top, propagate_down, bottom):
        pass


def create(cmd, uid, size, gpus):
    global solver
    solver = Solver(cmd, uid, size, gpus)


def start(rank):
    if debug:
        print('Starting rank', rank, file = sys.stderr)
    solver.start(rank)


def config(cmd):
    solver.config(cmd)


def train(batch):
    if debug and solver.rank == 0:
        print('Train', (batch[0].shape, batch[1].shape), file = sys.stderr)
    solver.buffs = batch
    loss = solver.caffe.step(1)
    print(loss, file = sys.stderr)


def predict(batch):
    if debug and solver.rank == 0:
        print('Predict', batch.shape, file = sys.stderr)
    solver.buffs = (batch,)
    net = solver.caffe.test_nets[0]
    solver.caffe.share_weights(net)
    res = net.forward()
    res = list(res.values())[0]
    print(res.shape, file = sys.stderr)
    print(res[0], file = sys.stderr)
    return res


def save_graph(path):
    if debug:
        print('Saving solver to', path, file = sys.stderr)
    with open(path, 'w') as f:
        f.write(text_format.MessageToString(solver.graph))


def save(path):
    if debug:
        print('Saving weights to', path, file = sys.stderr)
    solver.caffe.net.save_hdf5(str(path))


def load(path):
    if debug:
        print('Loading weights from', path, file = sys.stderr)
    solver.caffe.net.load_hdf5(str(path))
