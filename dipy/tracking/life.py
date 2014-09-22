"""
This is an implementation of the Linear Fascicle Evaluation (LiFE) algorithm
described in:

Pestilli, F., Yeatman, J, Rokem, A. Kay, K. and Wandell B.A. (2014). Validation
and statistical inference in living connectomes. Nature Methods

"""
import numpy as np
import scipy.sparse as sps
import scipy.linalg as la

from dipy.reconst.base import ReconstModel, ReconstFit
from dipy.core.onetime import ResetMixin
from dipy.core.onetime import auto_attr
import dipy.core.sphere as dps
from dipy.tracking.utils import unique_rows, move_streamlines
import dipy.reconst.dti as dti
import dipy.sims.voxel as sims
from dipy.tracking.vox2track import voxel2fiber, transform_sl


def spdot(A, B):
  """The same as np.dot(A, B), except it works even if A or B or both
  are sparse matrices.

  Parameters
  ----------
  A, B : arrays of shape (m, n), (n, k)

  Returns
  -------
  The matrix product A @ B.

  See discussion here:
  http://mail.scipy.org/pipermail/scipy-user/2010-November/027700.html
  """
  if sps.issparse(A) and sps.issparse(B):
      return A * B
  elif sps.issparse(A) and not sps.issparse(B):
      return (A * B).view(type=B.__class__)
  elif not sps.issparse(A) and sps.issparse(B):
      return (B.T * A.T).T.view(type=A.__class__)
  else:
      return np.dot(A, B)


def rsq(ss_residuals, ss_residuals_to_mean):
    """
    Calculate: $R^2 = \frac{1-SSE}{\sigma^2}$

    Parameters
    ----------
    ss_residuals : array
        Model fit errors relative to the data
    ss_residuals_to_mean : array
        Residuals of the data relative to the mean of the data (variance)

    Returns
    -------
    rsq : the variance explained.
    """
    return 100 * (1 - ss_residuals/ss_residuals_to_mean)


def sparse_sgd(y, X, momentum=0,
               prop_select=0.01,
               step_size=0.01,
               non_neg=True,
               prop_bad_checks=0.1,
               check_error_iter=10,
               max_error_checks=10,
               converge_on_r=0.2,
               verbose=False,
               plot=False,
               lamda=0,
               alpha=0):
    """

    Solve y=Xh for h, using a stochastic gradient descent, with X a sparse
    matrix

    Parameters
    ----------

    y: 1-d array of shape (N)
        The data

    X: ndarray. May be either sparse or dense. Shape (N, M)
       The regressors

    prop_select: float (0-1, default 0.01)
        What proportion of the samples to evaluate in each iteration of the
        algorithm.

    step_size: float, optional (default: 0.05).
        The increment of parameter update in each iteration

    non_neg: Boolean, optional (default: True)
        Whether to enforce non-negativity of the solution.

    prop_bad_checks: float (default: 0.1)
       If this proportion of error checks so far has not yielded an improvement
       in r squared, we halt the optimization.

    check_error_iter: int (default:10)
        How many rounds to run between error evaluation for
        convergence-checking.

    max_error_checks: int (default: 10)
        Don't check errors more than this number of times if no improvement in
        r-squared is seen.

    converge_on_r: float (default: 1)
      a percentage improvement in rsquared that is required each time to say
      that things are still going well.

    verbose: Boolean (default: True).
       Whether to display information in each iteration

    plot: whether to generate a plot of the progression of the optimization

    lamda, alpha: float (0-1)
        ElasticNet penalty parameters

    Returns
    -------
    h_best: The best estimate of the parameters.

    Notes
    -----
    This includes an implementation of a stochastic Elastic Net penalty
    parameters, alpha and lamda. For details, see [ZouHastie2005]_

    .. [ZouHastie2005] Zou, H., & Hastie, T. (2005). Regularization and
       variable selection via the elastic net. J. R. Statist. Soc. B, 301-320.
    """

    num_data = y.shape[0]
    num_regressors = X.shape[1]
    n_select = np.round(prop_select * num_data)

    # Initialize the parameters at the origin:
    h = np.zeros(num_regressors)

    # If nothing good happens, we'll return that in the end:
    h_best = np.zeros(num_regressors)
    gradient = np.zeros(num_regressors)

    iteration = 1
    count = 1
    ss_residuals = []  # This will hold the residuals in each iteration
    ss_residuals_min = np.inf  # This will keep track of the best solution so far
    ss_residuals_to_mean = np.sum((y - np.mean(y))**2) # The variance of y
    rsq_max = -np.inf   # This will keep track of the best r squared so far
    count_bad = 0  # Number of times estimation error has gone up.
    error_checks = 0  # How many error checks have we done so far

    while 1:
        # indices of data points to select
        idx = np.floor(np.random.rand(n_select) * num_data).astype(int);

        # Select for this round
        y0 = y[idx]
        X0 = X[idx]

        if iteration>1:
            # The sum of squared error given the current parameter setting:
            sse = np.sum((y - spdot(X,h))**2)
            # The gradient is (Kay 2008 supplemental page 27):
            gradient = ((spdot(X0.T, spdot(X0,h) - y0))
                        +
                        # This is Elastic Net:
                        lamda *((1-alpha) + alpha * h)
                       )
            gradient += momentum*gradient
            # Normalize to unit-length
            unit_length_gradient = gradient / np.sqrt(np.dot(gradient, gradient))
            # Update the parameters in the direction of the gradient:
            h -= step_size * unit_length_gradient

            if non_neg:
                # Set negative values to 0:
                h[h<0] = 0

        # Every once in a while check whether it's converged:
        if np.mod(iteration, check_error_iter):
            # This calculates the sum of squared residuals at this point:
            ss_residuals.append(np.sum(np.power(y - spdot(X,h), 2)) +
                                lamda * (alpha*np.sum(h**2) +
                                         (1-alpha)*np.sum(h)))
            rsq_est = rsq(ss_residuals[-1], ss_residuals_to_mean)
            if verbose:
                print("Itn #:%03d | SSE: %.1f | R2=%.1f "%
                      (iteration,
                       ss_residuals[-1],
                          rsq_est))

            # Did we do better this time around?
            if  ss_residuals[-1]<ss_residuals_min:
                # Update your expectations about the minimum error:
                ss_residuals_min = ss_residuals[-1]
                n_iterations = iteration # This holds the number of iterations
                                        # for the best solution so far.
                h_best = h # This holds the best params we have so far

                # Are we generally (over iterations) converging on
                # improvement in r-squared?
                if rsq_est>rsq_max*(1+converge_on_r/100):
                    rsq_max = rsq_est
                    count_bad = 0 # We're doing good. Null this count for now
                else:
                    count_bad += 1
            else:
                count_bad += 1

            if count_bad >= np.max([max_error_checks,
                                    np.round(prop_bad_checks*error_checks)]):
                if verbose:
                    print("\nOptimization terminated after %s iterations"%
                          iteration)
                    print("R2= %.1f "%rsq_max)
                    print("Sum of squared residuals= %.1f"%ss_residuals_min)

                if plot:
                    import matplotlib.pyplot as plt
                    fig, ax = plt.subplots()
                    ax.plot(ss_residuals)
                    ax.set_xlabel("Iteration #")
                    ax.set_ylabel(r"$\sum{(\hat{y} - y)^2}$")
                # Break out here, because this means that not enough
                # improvement has happened recently
                return h_best
            error_checks += 1
        iteration += 1


def sl_gradients(sl):
    """
    Calculate the gradients of the stream-line along the spatial dimension
    """
    return np.array(np.gradient(np.asarray(sl))[0])


def grad_tensor(grad, evals):
    """
    Calculate the 3 by 3 tensor for a given spatial gradient, given a canonical
    tensor shape (also as a 3 by 3), pointing at [1,0,0]

    Parameters
    ----------
    grad : 1d array of shape (3,)
        The spatial gradient (e.g between two nodes of a streamline).

    evals: 1d array of shape (3,)
        The eigenvalues of a canonical tensor to be used as a response function.

    """
    # This is the rotation matrix from [1, 0, 0] to this gradient of the sl:
    R = la.svd(np.matrix(grad), overwrite_a=True)[2]
    # This is the 3 by 3 tensor after rotation:
    T = np.dot(np.dot(R, np.diag(evals)), R.T)
    return T

def sl_tensors(sl, evals=[0.0015, 0.0005, 0.0005]):
    """

    The tensors generated by this fiber.

    Parameters
    ----------
    sl :

    evals : iterable with three entries
        The estimated eigenvalues of a single fiber tensor.
        (default: [0.0015, 0.0005, 0.0005]).

    Returns
    -------
    An n_nodes by 3 by 3 array with the tensor for each node in the fiber.

    Note
    ----
    Estimates of the radial/axial diffusivities may rely on
    empirical measurements (for example, the AD in the Corpus Callosum), or
    may be based on a biophysical model of some kind.
    """

    grad = sl_gradients(sl)

    # Preallocate:
    tensors = np.empty((grad.shape[0], 3, 3))

    for grad_idx, this_grad in enumerate(grad):
        tensors[grad_idx] = grad_tensor(this_grad, evals)
    return tensors


def sl_signal(sl, gtab, evals=[0.0015, 0.0005, 0.0005]):
    """
    The signal from a single streamline estimate along each of its nodes.

    Parameters
    ----------
    sl : a single streamline

    gtab : GradientTable class instance

    evals : list of length 3 (optional. Default: [0.0015, 0.0005, 0.0005])
        The eigenvalues of the canonical tensor used as an estimate of the
        signal generated by each node of the streamline.
    """
    # Gotta have those tensors:
    tensors = sl_tensors(sl, evals)
    sig = np.empty((len(sl), np.sum(~gtab.b0s_mask)))
    # Extract them once:
    bvecs = gtab.bvecs[~gtab.b0s_mask]
    bvals = gtab.bvals[~gtab.b0s_mask]
    for ii, tensor in enumerate(tensors):
        ADC = np.diag(np.dot(np.dot(bvecs, tensor), bvecs.T))
        # Use the Stejskal-Tanner equation with the ADC as input, and S0 = 1:
        sig[ii] = np.exp(-bvals * ADC)
    return sig


class FiberModel(ReconstModel):
    """
    A class for representing and solving predictive models based on
    tractography solutions.

    Notes
    -----

    [1] Pestilli, F., Yeatman, J, Rokem, A. Kay, K. and Wandell
        B.A. (2014). Validation and statistical inference in living
        connectomes. Nature Methods.
    """
    def __init__(self, gtab):
        """
        Parameters
        ----------
        gtab : a GradientTable class instance

        """
        # Initialize the super-class:
        ReconstModel.__init__(self, gtab)


    def setup(self, sl, affine, evals=[0.0015, 0.0005, 0.0005]):
        """
        Set up the necessary components for the LiFE model: the matrix of
        fiber-contributions to the DWI signal, and the coordinates of voxels
        for which the equations will be solved
        """
        sl = transform_sl(sl, affine)
        # Assign some local variables, for shorthand:
        all_coords = np.concatenate(sl)
        vox_coords = unique_rows(all_coords.astype(int))
        n_vox = vox_coords.shape[0]
        # We only consider the diffusion-weighted signals:
        n_bvecs = self.gtab.bvals[~self.gtab.b0s_mask].shape[0]

        v2f, v2fn = voxel2fiber(sl, transformed=True, affine=affine,
                                unique_idx=vox_coords)

        # How many fibers in each voxel (this will determine how many
        # components are in the fiber part of the matrix):
        n_unique_f = np.sum(v2f)

        # Preallocate these, which will be used to generate the two sparse
        # matrices:

        # This one will hold the fiber-predicted signal
        f_matrix_sig = np.zeros(n_unique_f * n_bvecs)
        f_matrix_row = np.zeros(n_unique_f * n_bvecs)
        f_matrix_col = np.zeros(n_unique_f * n_bvecs)

        keep_ct = 0

        fiber_signal = [sl_signal(s, self.gtab, evals) for s in sl]

        # In each voxel:
        for v_idx, vox in enumerate(vox_coords):
            # dbg:
            # print(100*float(v_idx)/n_vox)
            # For each fiber:
            for f_idx in np.where(v2f[v_idx])[0]:
                # Sum the signal from each node of the fiber in that voxel:
                vox_fiber_sig = np.zeros(n_bvecs)
                for node_idx in np.where(v2fn[f_idx]==v_idx)[0]:
                    this_signal = fiber_signal[f_idx][node_idx]
                    vox_fiber_sig += (this_signal - np.mean(this_signal))
                # For each fiber-voxel combination, we now store the row/column
                # indices and the signal in the pre-allocated linear arrays
                f_matrix_row[keep_ct:keep_ct+n_bvecs] =\
                                        np.arange(n_bvecs) + v_idx * n_bvecs
                f_matrix_col[keep_ct:keep_ct+n_bvecs] =\
                                        np.ones(n_bvecs) * f_idx
                f_matrix_sig[keep_ct:keep_ct+n_bvecs] = vox_fiber_sig
                keep_ct += n_bvecs


        # Allocate the sparse matrix, using the more memory-efficient 'csr'
        # format (converted from the coo format, which we rely on for the
        # initial allocation):
        life_matrix = sps.coo_matrix((f_matrix_sig,
                                       [f_matrix_row, f_matrix_col])).tocsr()

        return life_matrix, vox_coords


    def _signals(self, data, vox_coords):
        """
        Helper function to extract and separate all the signals we need to fit
        and evaluate a fit of this model

        Parameters
        ----------
        data : 4D array

        vox_coords: n by 3 array
            The coordinates into the data array of the fiber nodes.
        """
        # Fitting is done on the S0-normalized-and-demeaned diffusion-weighted
        # signal:
        idx_tuple = (vox_coords[:, 0], vox_coords[:, 1], vox_coords[:, 2])
        # We'll look at a 2D array, extracting the data from the voxels:
        vox_data =  data[idx_tuple]
        weighted_signal = vox_data[:, ~self.gtab.b0s_mask]
        b0_signal = np.mean(vox_data[:, self.gtab.b0s_mask], -1)
        relative_signal = (weighted_signal/b0_signal[:,None])

        # The mean of the relative signal across directions in each voxel:
        mean_sig =  np.mean(relative_signal, -1)
        to_fit = (relative_signal - mean_sig[:,None]).ravel()

        return (to_fit, weighted_signal, b0_signal, relative_signal, mean_sig,
                vox_data)


    def fit(self, data, sl, affine=None, evals=[0.0015, 0.0005, 0.0005]):
        """
        Fit the LiFE FiberModel for data and a set of streamlines associated
        with this data

        Parameters
        ----------
        data : 4D array
           Diffusion-weighted data

        sl : list
           A bunch of streamlines

        affine: 4 by 4 array (optional)
           The affine to go from the streamline coordinates to the data
           coordinates. Defaults to use `np.eye(4)`

        evals : list (optional)
           The eigenvalues of the tensor response function used in constructing
           the model signal. Default: [0.0015, 0.0005, 0.0005]

        Returns
        -------
        FiberFit class instance

        """
        life_matrix, vox_coords = \
            self.setup(sl, affine, evals=evals)

        to_fit, weighted_signal, b0_signal, relative_signal, mean_sig, vox_data=\
            self._signals(data, vox_coords)

        # Use a heuristic for selecting the number of rows in the matrix in
        # each iteration of SGD:
        prop_select = np.max([1.0, life_matrix.shape[0]/2048.])
        beta = sparse_sgd(to_fit, life_matrix, prop_select=prop_select)

        return FiberFit(self, life_matrix, vox_coords, to_fit, beta,
                        weighted_signal, b0_signal, relative_signal, mean_sig,
                        vox_data, sl, affine, evals)


class FiberFit(ReconstFit):
    """
    A fit of the LiFE model to diffusion data
    """
    def __init__(self, fiber_model, life_matrix, vox_coords, to_fit, beta,
                 weighted_signal, b0_signal, relative_signal, mean_sig,
                 vox_data, sl, affine, evals):
        """
        Parameters
        ----------
        fiber_model : A FiberModel class instance

        params : the parameters derived from a fit of the model to the data.

        """
        ReconstFit.__init__(self, fiber_model, vox_data)

        self.life_matrix = life_matrix
        self.vox_coords = vox_coords
        self.fit_data = to_fit
        self.beta = beta
        self.weighted_signal = weighted_signal
        self.b0_signal = b0_signal
        self.relative_signal = relative_signal
        self.mean_signal = mean_sig
        self.sl = sl
        self.affine = affine
        self.evals = evals


    def predict(self, gtab=None, S0=None):
        """
        Predict the signal

        Parameters
        ----------
        """
        # We generate the prediction and in each voxel, we add the
        # offset, according to the isotropic part of the signal, which was
        # removed prior to fitting:

        if gtab is None:
            _matrix = self.life_matrix
            gtab = self.model.gtab
        else:
            _model = FiberModel(gtab)
            _matrix, _ = model.setup(self.sl,
                                     self.affine,
                                     self.evals)

        pred = np.reshape(spdot(_matrix, self.beta),
                          (self.vox_coords.shape[0],
                          np.sum(~gtab.b0s_mask)))

        if S0 is None:
            S0 =  self.b0_signal

        return (pred + self.mean_signal[:, None] ) * S0[:, None]
