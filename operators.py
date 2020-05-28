import numpy as np
import scipy as sp
import scipy.linalg as sla
import scipy.sparse

from . import states
from .config import default_cutoff
from .functions import dens_matr, tensor


def destroy(n=default_cutoff, sparse=None):
    """Annihilation operator.

    Parameters
    ----------
    n : int
        The highest number state that is included in the finite number state
        representation of the operator.

    sparse : bool, optional
        Whether to return a sparse matrix.

    Returns
    -------
    oper : np.array or sp.sparse.csr_matrix
    """
    if not isinstance(n, (int, np.integer)):
        raise ValueError("Hilbert space dimension must be integer value")

    if sparse is None:
        sparse = (True if n > 3 else False)

    data = np.sqrt(np.arange(1, n + 1, dtype=np.complex64))
    ind = np.arange(1, n + 1, dtype=np.int32)
    ptr = np.arange(n + 2, dtype=np.int32)
    ptr[-1] = n

    a = sp.sparse.csr_matrix((data, ind, ptr), shape=(n + 1, n + 1), dtype=np.complex64)

    if sparse:
        return a

    return np.asarray(a.todense(), dtype=np.complex64)


def create(n=default_cutoff):
    """Creation operator.

    Parameters
    ----------
    n : int
        The highest number state that is included in the finite number state
        representation of the operator.

    Returns
    -------
    oper : np.array or sp.sparse.csc_matrix
    """
    return destroy(n).conj().T


def identity(a=destroy()):
    """Identity operator.
    Parameters
    ----------
    a : np.array or sp.sparse.csr_matrix
        Annihilation operator.
    Returns
    -------
    oper : np.array or sp.sparse.csr_matrix
    """
    if sp.sparse.issparse(a):
        return sp.sparse.eye(a.shape[0], dtype=np.complex64)

    return np.eye(a.shape[0], dtype=np.complex64)


def beamsplitter(theta, phi=0, mode1=0, mode2=1, mode_n=2, a=destroy(), decimals=8):
    """Beam splitter operator.

    Parameters
    ----------
    theta : float
        Beam splitter angle.

    phi : float
        Beam splitter phase.

    mode1 : int
        Which mode enters the first port of the beam splitter.

    mode2 : int
        Which mode enters the second port of the beam splitter.

    mode_n : int
        Number of modes.

    a : np.matrix
        Annihilation operator.

    decimals : int, optional
        Number of decimal places to round to.

    Returns
    -------
    oper : np.array
    """
    sparse = sp.sparse.issparse(a)

    id1 = identity(a)

    op1 = tensor(*(a if idx == mode1 else id1 for idx in range(mode_n)), sparse=sparse)
    op2 = tensor(*(a if idx == mode2 else id1 for idx in range(mode_n)), sparse=sparse)

    arg = theta * (np.exp(1j * phi) * op1 @ op2.conj().T - np.exp(-1j * phi) * op1.conj().T @ op2)

    if sparse:
        arg = arg.tocsr()

    return np.around(sla.expm(arg), decimals)


def displace(alpha, a=destroy()):
    """Single-mode displacement operator.

    Parameters
    ----------
    alpha : float/complex
        Displacement amplitude.

    a : np.matrix
        Annihilation operator.

    Returns
    -------
    oper : np.array
    """
    return sla.expm(alpha * a.conj().T - np.conj(alpha) * a)


def squeeze_sm(z, a=destroy()):
    """Single-mode Squeezing operator.

    Parameters
    ----------
    z : float/complex
        Squeezing parameter.

    a : np.matrix
        Annihilation operator.

    Returns
    -------
    oper : np.array
    """
    if sp.sparse.issparse(a):
        return sla.expm(0.5 * np.conj(z) * a ** 2 - 0.5 * z * a.conj().T ** 2)
    return sla.expm(0.5 * np.conj(z) * np.linalg.matrix_power(a, 2) - 0.5 * z * np.linalg.matrix_power(a.conj().T, 2))


def squeeze_tm(z, a1=destroy(), a2=destroy()):
    """Two-mode Squeezing operator.

    Parameters
    ----------
    z : float/complex
        Squeezing parameter.

    a1 : np.matrix
        Annihilation operator for mode 1.

    a2 : np.matrix
        Annihilation operator for mode 2.

    Returns
    -------
    oper : np.array
    """
    sparse = (sp.sparse.issparse(a1) or sp.sparse.issparse(a2))

    id1 = identity(a1)
    id2 = identity(a2)

    return sla.expm(0.5 * np.conj(z) * tensor(*(a1, id2), sparse=sparse) @ tensor(*(id1, a2), sparse=sparse) -
                    0.5 * z * tensor(*(a1.conj().T, id2), sparse=sparse) @ tensor(*(id1, a2.conj().T), sparse=sparse))


def rotate(theta, a=destroy()):
    """Phase-delay (rotation) operator.

    Parameters
    ----------
    theta : float
        Rotation angle/delay phase.

    a : np.matrix
        Annihilation operator.

    Returns
    -------
    oper : np.array
    """
    return sla.expm(1j * theta * a.conj().T @ a)


def no_click(eta, a=destroy()):
    """Projector on "no click" event of an SPD.

    Parameters
    ----------
    eta : float
        Detector quantum efficiency.

    a : np.matrix
        Annihilation operator.

    Returns
    -------
    oper : np.array
    """

    def coef(n: int):
        if n == 0:
            return 1.
        else:
            return (1. - eta) ** n

    diag = np.vectorize(coef)(np.arange(a.shape[0]))

    if sp.sparse.issparse(a):
        return sp.sparse.diags(diag, 0)

    return np.diag(diag)


def click(eta, a=destroy()):
    """Projector on "click" event of an SPD.

    Parameters
    ----------
    eta : float
        Detector quantum efficiency.

    a : np.matrix
        Annihilation operator.

    Returns
    -------
    oper : np.array
    """
    return identity(a) - no_click(eta, a)


def fock_projector(n, a=destroy()):
    """Projector on Fock state.

    Parameters
    ----------
    n : int
        Number of photons to project on.

    a : np.array
        Annihilation operator.

    Returns
    -------
    proj : np.array
    """
    proj = np.diag([0] * a.shape[0])
    proj[n, n] = 1
    if sp.sparse.issparse(a):
        return sp.sparse.csr_matrix(proj)
    return proj


def homodyne_projector(phi=0, eta=1., max_gamma=1.6, a=destroy()):
    """Projector on (cos(phi/2)*x + sin(phi/2)*p)=0.

    Parameters
    ----------
    phi : float
        Squeezing phase.

    eta : float
        Squeezing efficiency.

    max_gamma : float
        Max squeezing parameter that can be simulated with squeeze_sm.

    a : np.array
        Annihilation operator.

    Returns
    -------
    proj : np.array
    """
    if eta == 1.:
        return dens_matr(states.inf_squeezed(phi, a))

    gamma = -0.5 * np.log(1. - eta)

    if gamma > max_gamma:
        return dens_matr(states.inf_squeezed(phi, a))

    return dens_matr(states.squeezed_and_displaced(gamma * np.exp(1j * phi), 0, a))
