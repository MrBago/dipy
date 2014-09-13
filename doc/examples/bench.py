import numpy as np
from reconst_csa import *
from dipy.reconst.interpolate import NearestNeighborInterpolator

from dipy.tracking.markov import (BoundaryStepper,
                                  FixedSizeStepper,
                                  ProbabilisticOdfWeightedTracker)

from dipy.tracking.utils import seeds_from_mask

from dipy.reconst.peaks import default_sphere as sphere

stepper = FixedSizeStepper(1)

"""
Read the voxel size from the image header:
"""

zooms = img.get_header().get_zooms()[:3]

seeds = seeds_from_mask(mask, [1, 1, 1], zooms)
seeds = seeds[:2000]

interpolator = NearestNeighborInterpolator(maskdata, zooms)

pwt = ProbabilisticOdfWeightedTracker(csamodel, interpolator, mask,
                                      stepper, 20, seeds, sphere)
# csa_streamlines = list(pwt)

from dipy.tracking.local import *

fit = csamodel.fit(maskdata, mask)
ttc = ThresholdTissueClassifier(fit.gfa, .2)
pdg = ProbabilisticDirectionGetter.fromShmFit(fit, sphere, 20)

affine = np.zeros([4, 4])
affine[[0, 1, 2], [0, 1, 2]] = zooms
affine[3, 3] = 1

streamlines = LocalTracking(pdg, ttc, seeds, affine, .5, max_cross=1)

