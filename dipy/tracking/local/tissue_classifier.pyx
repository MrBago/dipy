cimport cython
cimport numpy as np

from libc.math cimport round

from .interpolation cimport trilinear_interpolate4d, _trilinear_interpolate_c_4d

cdef class TissueClassifier:
    cpdef TissueClass check_point(self, double[::1] point):
        pass


cdef class ThresholdTissueClassifier(TissueClassifier):
    """
    cdef:
        double threshold, interp_out_double[1]
        double[:]  interp_out_view = interp_out_view
        double[:, :, :] metric_map
    """

    def __cinit__(self, metric_map, threshold):
        self.interp_out_view = self.interp_out_double
        self.metric_map = metric_map
        self.threshold = threshold

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cpdef TissueClass check_point(self, double[::1] point):
        cdef:
            double result
            int err

        err = _trilinear_interpolate_c_4d(self.metric_map[..., None], point,
                                          self.interp_out_view)
        if err == -1:
            return OUTSIDEIMAGE
        elif err == -2:
            raise ValueError("Point has wrong shape")
        elif err != 0:
            # This should never happen
            raise RuntimeError("You seem to have found a bug in dipy")

        result = self.interp_out_view[0]

        if result > self.threshold:
            return TRACKPOINT
        else:
            return ENDPOINT

cdef class ActTissueClassifier(TissueClassifier):
    """
    Anatomically-Constained Tractography from Smith et al. (NeuroImage 2012).

    cdef:
        double interp_out_double[1]
        double[:]  interp_out_view = interp_out_view
        double[:, :, :] include_map, exclude_map
    """

    def __cinit__(self, include_map, exclude_map):
        self.interp_out_view = self.interp_out_double
        self.include_map = include_map
        self.exclude_map = exclude_map

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cpdef TissueClass check_point(self, double[::1] point):
        cdef:
            double include_result, exclude_result
            int err

        include_err = _trilinear_interpolate_c_4d(self.include_map[..., None], point,
                                          self.interp_out_view)
        include_result = self.interp_out_view[0]

        exclude_err = _trilinear_interpolate_c_4d(self.exclude_map[..., None], point,
                                          self.interp_out_view)        
        exclude_result = self.interp_out_view[0]

        if include_err == -1 or exclude_err == -1:
            return OUTSIDEIMAGE
        elif include_err == -2 or exclude_err == -2:
            raise ValueError("Point has wrong shape")
        elif include_err != 0 or exclude_err != 0:
            # This should never happen
            raise RuntimeError("You seem to have found a bug in dipy")

        if include_result > 0.5:
            return ENDPOINT
        elif exclude_result > 0.5:
            return INVALIDPOINT
        else:
            return TRACKPOINT


