cdef extern from "dpy_math.h" nogil:
    int signbit(double x)
    double floor(double x)

cimport cython
cimport numpy as np

import numpy as np
from .markov import endstreamline, outsidearea

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void fixed_step(double *direction, double *point, double *new_point,
                     double stepsize) nogil:
    # Size of point and new_point are not enforced, it's up to the developer to
    # ensure that arrays of size 3 are passed.
    for i in range(3):
        new_point[i] = point[i] + direction[i] * stepsize


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.profile(False)
@cython.cdivision(True)
cdef void stepToBoundry(double *direction, double *point, double *new_point,
                        double overstep) nogil:

    cdef:
        double step_sizes[3], smallest_step

    for i in range(3):
        step_sizes[i] = point[i] + .5
        step_sizes[i] = floor(step_sizes[i]) - step_sizes[i]
        step_sizes[i] += not signbit(direction[i])
        step_sizes[i] /= direction[i]

    smallest_step = step_sizes[0]
    for i in range(1, 3):
        if step_sizes[i] < smallest_step:
            smallest_step = step_sizes[i]

    smallest_step += overstep
    for i in range(3):
        new_point[i] = point[i] + smallest_step * direction[i]

"""
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

from ..reconst.interpolate import OutsideImage, NearestNeighborInterpolator
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def _work(get_direction,
          np.ndarray[np.float_t, ndim=1] seed,
          np.ndarray[np.float_t, ndim=1] first_step,
          np.ndarray[np.float_t, ndim=1] voxel_size,
          np.ndarray[np.float_t, ndim=2] streamline,
          double stepsize,
          int fixedstep):

    if (seed.shape[0] != 3 or first_step.shape[0] != 3 or voxel_size.shape[0]
        != 3 or streamline.shape[1] != 3):
        raise ValueError()
    if (streamline.strides[1] != sizeof(double)):
        raise ValueError()

    cdef:
        np.ndarray[np.float_t, ndim=1, mode='c'] dir = first_step.copy()
        np.ndarray[np.float_t, ndim=1, mode='c'] point = seed.copy()
        size_t i
        double step_dir[3] #, vs[3]

    if fixedstep:
        take_step = fixed_step
    else:
        take_step = stepToBoundry

    """
    for i in range(3):
        point[i] = seed[i]
        # vs[i] = voxel_size[i]
    """

    for i in range(0, streamline.shape[0]):
        try:
            dir = get_direction(point, dir)
        except IndexError:
            i -= 1
            break
        except StopIteration:
            break
        # Only copy the point into streamline if get_direction does not raise
        for j in range(3):
            streamline[i, j] = point[j]
        if dir is None:
            break
        for j in range(3):
            step_dir[j] = dir[j] / voxel_size[j]
        # Compute the next point in the streamline
        take_step(step_dir, &streamline[i, 0], &point[0], stepsize)

    return i

