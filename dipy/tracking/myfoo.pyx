cimport cython

import numpy as np
cimport numpy as cnp

cdef extern from "dpy_math.h" nogil:
    int signbit(double x)
    double ceil(double x)
    double floor(double x)


@cython.profile(False)
@cython.cdivision(True)
cdef inline double python_mod(double a, double b) nogil:
    mod = a % b
    if mod < 0:
        mod += b
    return mod

@cython.profile(False)
@cython.cdivision(True)
cdef inline double other(double a, double b) nogil:
    mod = ((a % b) + b) % b
    return mod


@cython.boundscheck(False)
@cython.profile(False)
def test_pm(double[::1] a, double[::1] b, double value):

    with nogil:
        for i in range(a.shape[0]):
            b[i] = python_mod(a[i], value)

    return
@cython.boundscheck(False)
@cython.profile(False)
def test_ot(double[::1] a, double[::1] b, double value):

    with nogil:
        for i in range(a.shape[0]):
            b[i] = other(a[i], value)
    return

@cython.boundscheck(False)
@cython.profile(False)
def step_to_b(cnp.ndarray[cnp.float64_t, ndim=1, mode='c'] point,
              cnp.ndarray[cnp.float64_t, ndim=1, mode='c'] direction,
              cnp.ndarray[cnp.float64_t, ndim=1, mode='c'] voxel_size,
              double overstep):

    if (point.shape[0] != 3 or
        direction.shape[0] != 3 or
        voxel_size.shape[0] != 3):
        raise ValueError()

    cdef cnp.ndarray[cnp.float64_t] new_point = point.copy()
    stepToBoundry(&point[0], &direction[0], &voxel_size[0], &new_point[0],
                  overstep)

    return new_point

@cython.boundscheck(False)
@cython.profile(False)
def step_to_T(cnp.ndarray[cnp.float64_t, ndim=2, mode='c'] point,
              cnp.ndarray[cnp.float64_t, ndim=2, mode='c'] direction,
              cnp.ndarray[cnp.float64_t, ndim=2, mode='c'] voxel_size,
              double overstep,
              int mode):

    if (point.shape[1] != 3 or
        direction.shape[1] != 3 or
        voxel_size.shape[1] != 3):
        raise ValueError()

    if mode:
        fun = stepToBoundry1
    else:
        fun = stepToBoundry

    cdef cnp.ndarray[cnp.float64_t, ndim=2, mode='c'] new_point = point.copy()
    for i in range(point.shape[0]):
        fun(&point[i, 0], &direction[i, 0], &voxel_size[i, 0],
            &new_point[i, 0], overstep)
    return new_point



@cython.boundscheck(False)
@cython.profile(False)
def step_newp(cnp.ndarray[cnp.float64_t, ndim=1, mode='c'] point,
              cnp.ndarray[cnp.float64_t, ndim=1, mode='c'] direction,
              cnp.ndarray[cnp.float64_t, ndim=1, mode='c'] voxel_size,
              cnp.ndarray[cnp.float64_t, ndim=1, mode='c'] new_point,

              double overstep):

    if (point.shape[0] != 3 or direction.shape[0] != 3 or
        voxel_size.shape[0] != 3 or new_point.shape[0] != 3):
        raise ValueError()

    stepToBoundry(&point[0], &direction[0], &voxel_size[0], &new_point[0],
                  overstep)


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.profile(False)
@cython.cdivision(True)
cdef void stepToBoundry(double *point,
                        double *direction,
                        double *voxel_size,
                        double *new_point,
                        double overstep
                       ) nogil:

    cdef:
        double step_sizes[3]
        double smallest_step

    for i in range(3):
        step_sizes[i] = voxel_size[i] * (1 - signbit(direction[i]))
        step_sizes[i] -= python_mod(point[i], voxel_size[i])
        step_sizes[i] /= direction[i]

    smallest_step = step_sizes[0]
    for i in range(1, 3):
        if step_sizes[i] < smallest_step:
            smallest_step = step_sizes[i]

    smallest_step += overstep
    for i in range(3):
        new_point[i] = point[i] + smallest_step * direction[i]

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.profile(False)
@cython.cdivision(True)
cdef void stepToBoundry1(double *point,
                         double *direction,
                         double *voxel_size,
                         double *new_point,
                         double overstep
                        ) nogil:

    cdef:
        double step_sizes[3]
        double smallest_step

    for i in range(3):
        step_sizes[i] = not signbit(direction[i])
        step_sizes[i] -= point[i] - floor(point[i])
        step_sizes[i] /= direction[i]

    smallest_step = step_sizes[0]
    for i in range(1, 3):
        if step_sizes[i] < smallest_step:
            smallest_step = step_sizes[i]

    smallest_step += overstep
    for i in range(3):
        new_point[i] = point[i] + smallest_step * direction[i]
