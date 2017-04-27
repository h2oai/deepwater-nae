#!/usr/bin/env python2
"""
"""

import sys, struct, numpy as np
import deepwater_pb2

if __name__ == '__main__':
    print >> sys.stderr, 'Started pycaffe'

    while True:
        cmd = deepwater_pb2.Cmd()
        len = struct.unpack('>I', sys.stdin.read(4))[0]
        print >> sys.stderr, 'read len %d' % len
        data = sys.stdin.read(len)
        cmd.ParseFromString(data)
        array = np.frombuffer(cmd.batch, dtype = 'float32')
        print >> sys.stderr, array
        print >> sys.stderr, 'float %f %f %f' % (array[0], array[1], array[2])
        print >> sys.stderr, 'label %d %d %d' % (cmd.labels[0], cmd.labels[1], cmd.labels[2])

        cmd.type = 0
        data = cmd.SerializeToString()
        print >> sys.stderr, 'write len %d' % len(data)
        sys.stdout.write(struct.pack('>I', len(data)))
        sys.stdout.write(data)
        sys.stdout.flush()

        # cmd.labels = 43
        # data = cmd.SerializeToString()
        # print >> sys.stderr, 'write len %d' % len(data)
        # sys.stdout.write(struct.pack('>I', len(data)))
        # sys.stdout.write(data)
        # sys.stdout.flush()

