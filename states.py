import numpy as np
import scipy as sp
import scipy.sparse
import scipy.sparse.linalg
import scipy.linalg as sla
from scipy.special import factorial as fac

from .functions import to_ket
from .operators import destroy, displace, squeeze_sm


def fock(n, a=destroy()):
    """Fock state.

    Parameters
    ----------
    n : int
        Number of photons.

    a : np.array
        Annihilation operator.

    Returns
    -------
    ket : np.array
    """
    vec = [0] * a.shape[0]
    vec[n] = 1
    return to_ket(vec)


def squeezed_and_displaced(z, alpha, a=destroy()):
    """Single-mode squeezed and displaced vacuum state.

    Parameters
    ----------
    z : float/complex
        Squeezing parameter.

    alpha : float/complex
        Displacement amplitude.

    a : np.matrix
        Annihilation operator.

    Returns
    -------
    ket : np.array
    """
    arr = np.zeros(a.shape[0])
    arr[0] = 1
    vac = to_ket(arr)

    return displace(alpha, a) @ squeeze_sm(z, a) @ vac


def inf_squeezed(phi=0, a=destroy()):
    """Infinitely squeezed vacuum state.

    Parameters
    ----------
    phi : float
        Squeezing angle.
        phi = 0 => |x = 0>
        phi = pi => |p = 0>

    a : np.matrix
        Annihilation operator.

    Returns
    -------
    ket : np.array
    """
    ket = to_ket([(-0.5 * np.exp(1j * phi)) ** (n // 2) * np.sqrt(fac(n)) / fac(n // 2) if n % 2 == 0 else 0.
                  for n in range(a.shape[0])])
    if sp.sparse.issparse(a):
        return ket / sp.sparse.linalg.norm(ket)
    return ket / sla.norm(ket)
