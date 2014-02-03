from dipy.reconst.shm import CsaOdfModel
from dipy.data import read_stanford_labels
import dipy.tracking.markov
from dipy.tracking.markov import *
from dipy.reconst.interpolate import NearestNeighborInterpolator
from dipy.tracking.utils import seeds_from_mask
from dipy.tracking.eudx import EuDX
from dipy.reconst.peaks import peaks_from_model
from scipy import ndimage

img,  gtab, Limg = read_stanford_labels()
labels = Limg.get_data()
model = CsaOdfModel(gtab, 6)
stepper = FixedSizeStepper(1.1)
data = img.get_data()
mask = (labels == 1) | (labels == 2)
mask = ndimage.binary_erosion(mask, iterations=2)

print(mask.sum())
affine = np.eye(4)

do = CDT_NNO._get_directions
do.sphere.vertices = do.sphere.vertices.copy(order='C')
seeds = seeds_from_mask(mask, [1, 1, 1], affine=affine)
# seeds = np.array([[ 39.,  40.,  38.]])
nni = NearestNeighborInterpolator(data, [2, 2, 2.])
idx = np.random.randint(0, len(seeds), 1000)
one = CDT_NNO(model, nni, mask, stepper, 45, seeds[idx], max_cross=None, maxlen=100, affine=affine)
two = CDT_NNO(model, nni, mask, stepper, 45, seeds[idx], max_cross=None, maxlen=100, affine=affine)
two.msfun = dipy.tracking.markov._markov_streamline

sphere = one._get_directions.sphere
pfm = peaks_from_model(model, data, sphere, .5, 45, mask=mask)
qa = (pfm.peak_indices >= 0) * mask[..., None].astype(float)

pfm.set_stop_conditions(qa, .5, 60, .5)
# dg = DG(pfm.peak_indices, sphere)
# dg._helper(qa, .5, 60., .5)

r = MyI(pfm.get_direction, seeds[idx], 100, .5, affine=affine)
eu = EuDX(qa, pfm.peak_indices, seeds[idx], sphere.vertices, a_low=.5,
          affine=affine, max_points=100)



