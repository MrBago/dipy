#!/usr/bin/env python
import numpy as np
from dipy.tracking.streamlinespeed import length, length_bago


sl_double = []
sl_float = []
for i in range(100000):
    n = np.random.randint(10, 100)
    sl = np.random.random((n, 3))
    sl_double.append(sl)
    sl_float.append(sl.astype('float32'))


def timer(func, arg):
    def helper():
        func(arg)
    return helper


if __name__ == "__main__":
    import timeit

    assert all(length(sl_double) == length_bago(sl_double))
    assert all(length(sl_float) == length_bago(sl_float))
    # assert np.allclose(length(streamlines), length_bago(streamlines))

    # setup = "from __main__ import time_length_bago, time_length"
    print "timing double"
    print "with list:", timeit.timeit(timer(length_bago, sl_double), number=10)
    print "with vector:", timeit.timeit(timer(length, sl_double), number=10)

    print ""
    print "timing float32"
    print "with list:", timeit.timeit(timer(length_bago, sl_float), number=10)
    print "with vector:", timeit.timeit(timer(length, sl_float), number=10)

