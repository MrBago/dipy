from dipy.reconst.msd import MultiShellDeconvModel
from dipy.sims.voxel import (multi_tensor, )

def _make_evecs(ev1):
    """ Makes a set of mutually perpendicular eigenvectors given ev1."""
    assert ev1.shape == (3,)
    dummy = np.zeros(3)
    dummy[abs(ev1).argmax()] = 1.
    ev2 = np.cross(ev1, dummy)
    ev2 /= vector_norm(ev2)
    ev3 = np.cross(ev1, ev2)
    return np.column_stack([ev1, ev2, ev3])

def test_MultiShellDeconvModel():

    evals = np.array([.992, .254, .254]) * 1e-3
    evecs = np.empty((3, 3))
    z = np.array([0, 0, 1.])
    evecs[:, 0] = z
    evecs[:2, 1:] = np.eye(2)
    wm_response = single_tensor(gtab, 100., evals, evecs snr=None)
    gm_response = single_tensor(gtab, 100., [.76e-3] * 3, np.eye(3), snr=None)
    csf_response = single_tensor(gtab, 100., [3.e-3] * 3, np.eye(3), snr=None)

    ev1 = np.array([0.36, 0.48, 0.8])
    ev2 = np.cross(ev1, z)
    ev2 /= vector_norm(ev2)
    ev3 = np.cross(ev1, ev2)
    evecs = np.column_stack([ev1, ev2, ev3])

    evecs = np.

    gtab = gradient_table(bvals, bvecs)
    mevals = np.array(([0.0015, 0.0003, 0.0003],
                       [0.0015, 0.0003, 0.0003]))

    angles = [(0, 0), (60, 0)]

    S, sticks = multi_tensor(gtab, mevals, S0, angles=angles,
                             fractions=[50, 50], snr=SNR)



