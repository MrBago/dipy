from dipy.data import default_sphere

from cvxopt import matrix
from cvxopt.solvers import qp

def foo(signals, B, iso_comp, m, n, bval):

    n_coeff = len(n)

    # Check assumptions about the basis:
    # Isotropic compartments are the beginning.
    assert np.all(B[:iso_comp + 1] == B[0, 0])
    # Only even SH degrees are present.
    assert np.all((n % 2) == 0)
    # B has the right shape.
    assert B.shape[0] == len(bval)
    assert (B.shape[1] - iso_comp) == len(n) == len(m)
    # SH degrees are in ascending order.
    assert np.all(n[m == 0] == np.arange(0, n[-1] + 1, 2))

    # TODO: round bval

    B_dwi = B[:, iso_comp:]
    B_ax_sym = B_dwi[:, m == 0]

    bvalues = np.unique(bval)
    R = np.empty((len(bvalues), n_coeff))

    for i, b in enumerate(bvalues):
        part = bval == b
        for c in range(iso_comp):
            R[i, c] = signals[c][part].mean() / B[0, 0]
        dwi_signal = signals[iso_comp]
        rh = np.linalg.lstsq(B_ax_sym[:, part], dwi_signal)
        R[i, iso_comp:] = rh[n // 2]
    return bvalues, R


def closest(haystack, needle):
    diff = abs(haystack[None, :] - needle)
    return diff.argmin(axis=0)


class MultiShellDeconvModel(ConstrainedSphericalDeconvModel):

    def __init__(self, gtab, responses, reg_sphere=default_sphere, sh_order=8,
                 tissue_classes=3, *args, **kwargs):
        """
        """
        SphHarmModel.__init__(self, gtab)
        B, m, n = multi_tissue_basis(gtab, sh_order, tissue_classes)
        uniq_bval, R = foo(responses, B, tissue_classes, m, n, gtab.bval)
        multiplier_matrix = R[closest(uniq_bval, gtab.bvals)]

        r, theta, phi = cart2sphere(reg_sphere.x, self.sphere.y, self.sphere.z)
        B_reg = real_sph_harm(m, n, theta[:, None], phi[:, None])
        X = B * multiplier_matrix

        self.fitter = QpFitter(X, B_reg)
        self._X = X
        self.sphere = reg_sphere
        self.B_dwi = B
        self.R = R
        self.shells = uniq_bval

    @multi_voxel_fit
    def fit(self, data):
        coeff = self.fitter(data)
        return MSDecovFit(self, coeff, None)

def _rank(A, tol=1e-5):
    s = la.svd(A, False, False)
    threshold = (s[0] * tol)
    rnk = (s > threshold).sum()
    return rnk


class QpFitter(object):

    def _lstsq_initial(self, z):
        fodf_sh = _solve_cholesky(self._P, z)
        s = np.dot(self._reg, fodf_sh)
        init = {'x':matrix(fodf_sh),
                's':matrix(s.clip(1e-10))}
        return init

    def __init__(self, X, reg):
        self._P = P = np.dot(X.T, X)

        # No super res for now.
        assert _rank(P) == P.shape[0]

        self._reg = reg
        self._P_init = np.dot(X[:, :N].T, X[:, :N])

        # Make cvxopt matrix types for later re-use.
        self._P_mat = matrix(P)
        self._reg_mat = matrix(-reg)
        self._h_mat = matrix(0., (reg.shape[0], 1))

    def __call__(self, signal):
        z = np.dot(self._X.T, signal)
        init = self._lstsq_initial(z)

        z_mat = matrix(-z)
        r = qp(self._P_mat, z_mat, self._reg_mat, self._h_mat, initvals=init)
        fodf_sh = r['x']
        fodf_sh = np.array(fodf_sh)[:, 0]
        return fodf_sh

