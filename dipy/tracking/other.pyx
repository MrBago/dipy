import numpy as np

cimport cython
cimport numpy as np

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void fixed_step(double direction[3], double point[3],
                     double new_point[3], double stepsize) nogil:
    # Size of point and new_point are not enforced, it's up to the developer to
    # ensure that arrays of size 3 are passed.
    for i in range(3):
        new_point[i] = point[i] + direction[i] * stepsize

class MarkovTracker(object):

    def __init__(self, get_direction, maxlen, stepsize):
        self.get_direction = get_direction
        self.stepsize = stepsize
        # Also creates _temp_array
        self.maxlen = maxlen

    @property
    def maxlen(self):
        return self._maxlen

    @maxlen.setter
    def maxlen(self, value):
        self._maxlen = value
        self._temp_array_F = np.empty((value + 1, 3))
        self._temp_array_B = np.empty((value + 1, 3))


"""
def _work(get_direction, double[::1] seed, np.ndarray first_step,
          np.ndarray[np.float_t, ndim=2, mode=c] streamline, double stepsize):
"""
from ..reconst.interpolate import OutsideImage, NearestNeighborInterpolator
@cython.boundscheck(False)
@cython.wraparound(False)
def _work(get_direction,
          np.ndarray[np.float_t, ndim=1, mode='c'] seed,
          np.ndarray[np.float_t, ndim=1, mode='c'] first_step,
          np.ndarray[np.float_t, ndim=2, mode='c'] streamline,
          double stepsize):

    cdef:
        np.ndarray[np.float_t, ndim=1, mode='c'] dir = first_step.copy()
        size_t i
        int status

    take_step = fixed_step
    streamline[0, :] = seed

    try:
        for i in range(1, streamline.shape[0]):
            take_step(&dir[0], &streamline[i-1, 0], &streamline[i, 0], stepsize)
            dir = get_direction(streamline[i], dir)
            if dir is None:
                break
        i += 1
    except OutsideImage:
        pass

    return streamline[:i]

