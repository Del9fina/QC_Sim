import numpy as np
import scipy as sp
import scipy.sparse
import matplotlib.pyplot as plt


def wigner_iterative(rho, xvec, yvec, g=np.sqrt(2)):
    """Wigner function for a state vector or density matrix at points
    `xvec + i * yvec`.

    Parameters
    ----------

    rho : np.array
        A state vector or density matrix.

    xvec : array_like
        x-coordinates at which to calculate the Wigner function.

    yvec : array_like
        y-coordinates at which to calculate the Wigner function.  Does not
        apply to the 'fft' method.

    g : float
        Scaling factor for `a = 0.5 * g * (x + iy)`, default `g = sqrt(2)`.

    Returns
    -------

    w : array
        Values representing the Wigner function calculated over the specified
        range [xvec,yvec].

    Notes
    -----
    Using an iterative method to evaluate the wigner functions for the Fock
    state :math:`|m><n|`.

    The Wigner function is calculated as
    :math:`w = \sum_{mn} \\rho_{mn} w_{mn}` where :math:`w_{mn}` is the Wigner
    function for the density matrix :math:`|m><n|`.

    In this implementation, for each row m, w_list contains the Wigner functions
    w_list = [0, ..., w_mm, ..., w_mN]. As soon as one w_mn Wigner function is
    calculated, the corresponding contribution is added to the total Wigner
    function, weighted by the corresponding element in the density matrix
    :math:`rho_{mn}`.
    """

    m_max = int(np.prod(rho.shape[0]))
    x, y = sp.meshgrid(xvec, yvec)
    a = 0.5 * g * (x + 1.0j * y)

    w_list = np.array([np.zeros(np.shape(a), dtype=complex) for k in range(m_max)])
    w_list[0] = np.exp(-2.0 * abs(a) ** 2) / np.pi

    w = np.real(rho[0, 0]) * np.real(w_list[0])
    for n in range(1, m_max):
        w_list[n] = (2.0 * a * w_list[n - 1]) / np.sqrt(n)
        w += 2 * np.real(rho[0, n] * w_list[n])

    for m in range(1, m_max):
        temp = sp.copy(w_list[m])
        w_list[m] = (2 * np.conj(a) * temp - np.sqrt(m) * w_list[m - 1]) / np.sqrt(m)

        # w_list[m] = Wigner function for |m><m|
        w += sp.real(rho[m, m] * w_list[m])

        for n in range(m + 1, m_max):
            temp2 = (2 * a * w_list[n - 1] - np.sqrt(m) * temp) / np.sqrt(n)
            temp = sp.copy(w_list[n])
            w_list[n] = temp2

            # w_list[n] = Wigner function for |m><n|
            w += 2 * sp.real(rho[m, n] * w_list[n])

    return 0.5 * w * g ** 2


def plot_wigner(state, x=(-3, 3), p=(-3, 3), div=0.1):
    x_vec = np.arange(*x, div)
    p_vec = np.arange(*p, div)

    if sp.sparse.issparse(state):
        wig = wigner_iterative(state.todense(), x_vec, p_vec)
    else:
        wig = wigner_iterative(state, x_vec, p_vec)

    plt.close()
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(1, 1, 1, aspect=1)

    pcm = ax.pcolormesh(x_vec, x_vec, wig, cmap='RdBu', shading='gouraud')
    fig.colorbar(pcm, ax=ax)

    ax.grid()
    plt.show()


def to_ket(vec, sparse=None):
    """Makes ket-vector from 1D array.

    Parameters
    ----------
    vec : list or np.array
    sparse : bool, optional

    Returns
    -------
    ket : np.array
    """
    if sparse is None:
        sparse = (True if len(vec) > 4 else False)

    ket = np.array([vec], dtype=np.complex64).T

    if sparse:
        return sp.sparse.csr_matrix(ket)

    return ket


def dens_matr(psi):
    """Makes density matrix from ket-vector.

    Parameters
    ----------
    psi : np.array
        Ket-vector.

    Returns
    -------
    rho : np.array
        Density matrix.
    """
    if sp.sparse.issparse(psi):
        return sp.sparse.kron(psi, psi.conj().T).astype(dtype=np.complex64)
    return np.kron(psi, psi.conj().T).astype(dtype=np.complex64)


def fidelity(rho, rho_target):
    """Computes fidelity between two states represented by density matrices.
    Parameters
    ----------
    rho, rho_target : np.array
        Density matrices.
    Returns
    -------
    fidelity : float
    """
    if sp.sparse.issparse(rho_target):
        sqrt_target = sp.linalg.sqrtm(rho_target.todense())
    else:
        sqrt_target = sp.linalg.sqrtm(rho_target)
        
    if sp.sparse.issparse(rho):
        return np.abs(np.einsum('ii->', sp.linalg.sqrtm(sqrt_target @ rho.todense() @ sqrt_target)) ** 2)
    else:
        return np.abs(np.einsum('ii->', sp.linalg.sqrtm(sqrt_target @ rho @ sqrt_target)) ** 2)


def tensor(*qlist, sparse=None):
    """Calculates the tensor product of input operators.
    Parameters
    ----------
    qlist : array_like
        ``list`` or ``array`` of quantum objects for tensor product.
    sparse : bool, optional
    Returns
    -------
    obj : A composite quantum object.
    """
    out = 1
    for q in qlist:
        if sp.sparse.issparse(q) and sparse is None:
            sparse = True

        if sparse:
            out = sp.sparse.kron(out, q)
        else:
            out = np.kron(out, q)
    return out.astype(np.complex64)


def ptrace(rho, modes, mode_n):
    """Partial trace of the density matrix with selected components remaining.
    Parameters
    ----------
    rho : Density matrix of a multimode quantum state.

    modes : int/list
        An ``int`` or ``list`` of components to keep after partial trace.

    mode_n : int
        Initial number of modes.
    Returns
    -------
    prho : Density matrix representing partial trace with selected components
        remaining.
    """
    if isinstance(modes, int):
        modes = [modes]
    idxs = sorted(list(set(range(mode_n)) - set(modes)), reverse=True)
    dim = int(round(rho.shape[0] ** (1 / mode_n)))

    if sp.sparse.issparse(rho):
        # prho = rho.todense().reshape([dim] * mode_n * 2)
        raise NotImplementedError
        #
        # prho = sparse.COO.from_scipy_sparse(rho).reshape(tuple([dim] * mode_n * 2))
        #
        # return prho
        #
        # new_dim = dim ** len(modes)
        # prho = sp.sparse.csr_matrix((new_dim, new_dim))
        # for i in range(new_dim):
        #     for j in range(new_dim):
        #         val = 0

    else:
        prho = rho.reshape([dim] * mode_n * 2)
    n = mode_n
    for idx in idxs:
        prho = np.trace(prho, axis1=idx, axis2=idx+n)
        n -= 1

    return prho
