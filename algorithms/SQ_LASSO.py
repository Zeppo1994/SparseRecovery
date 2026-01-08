import math
import torch
from pykeops.torch import Vi, Vj
import algorithms.PD_algorithms as PD_algorithms
from deepinv.optim.utils import least_squares


def Reconstruction_Fourier(
    samples,
    values,
    frequencies,
    restarts=9,
    tol=1e-5,
    beta=2.0,
    alpha=1.0,
    lam_est=1.0,
    lstsq_rec=False,
):
    N = samples.size(0)  # number of sample points
    dtype = samples.dtype
    device = samples.device

    A_op = aNUDFT(samples, frequencies)
    AT_op = NUDFT(samples, frequencies)

    def A(x):
        return A_op(x) / math.sqrt(N)

    def AT(y):
        return AT_op(y) / math.sqrt(N)

    b = values / math.sqrt(N)

    if lstsq_rec:
        # compute least squares reconstruction, 1/gamma is Tikhonov regularization parameter
        lsr_rec = least_squares(
            A,
            AT,
            b,
            solver="lsqr",
            gamma=1e4,
            tol=1e-5,
            max_iter=2000,
            parallel_dim=-1,
        )
    else:
        lsr_rec = None

    lam_est = lam_est / math.sqrt(N)
    s0 = torch.zeros_like(AT(b), requires_grad=False)

    # run prim dual algorithm with restarts
    rec, vals, vals_dual = PD_algorithms.restart_pd_rLASSO(
        s0, A, AT, b, lam=lam_est, tol=tol, restarts=restarts, beta=beta, alpha=alpha
    )
    residuals = torch.zeros((2,), dtype=dtype, device=device)
    residuals[0] = torch.linalg.norm(A(rec) - b)
    if lstsq_rec:
        residuals[1] = torch.linalg.norm(A(lsr_rec) - b)
    return rec, vals, vals_dual, lsr_rec, residuals


def Reconstruction_Tchebychev(
    samples,
    values,
    indices,
    restarts=9,
    tol=1e-5,
    beta=3.0,
    alpha=0.3,
    lam_est=1.0,
    tensor=False,
    lstsq_rec=False,
):
    D = samples.size(1)  # dimension of sample points
    N = samples.size(0)  # number of sample points
    dtype = samples.dtype
    device = samples.device

    # precompute normalization coefficients for computational efficiency
    norm_coeffs = math.sqrt(2) ** (indices.clamp(0, 1).sum(-1, keepdim=True)).to(
        dtype=dtype
    )

    if tensor:
        A_Mat = norm_coeffs * torch.cos(
            indices[:, None, :] * torch.acos(samples[None, :, :])
        ).prod(-1)
    else:
        A_op = Tchebychev_eval(samples, indices, norm_coeffs, D)
        AT_op = aTchebychev_eval(samples, indices, norm_coeffs, D)

    def A(x):
        if tensor:
            return A_Mat.transpose(1, 0) @ x / math.sqrt(N)
        else:
            return A_op(x) / math.sqrt(N)

    def AT(y):
        if tensor:
            return A_Mat @ y / math.sqrt(N)
        else:
            return AT_op(y) / math.sqrt(N)

    b = values / math.sqrt(N)

    if lstsq_rec:
        # compute least squares reconstruction, 1/gamma is Tikhonov regularization parameter
        lsr_rec = least_squares(
            A,
            AT,
            b,
            solver="lsqr",
            gamma=1e4,
            tol=1e-5,
            max_iter=2000,
            parallel_dim=-1,
        )
    else:
        lsr_rec = None

    lam_est = lam_est / math.sqrt(N)
    s0 = torch.zeros_like(AT(b), requires_grad=False)

    # run prim dual algorithm with restarts
    rec, vals, vals_dual = PD_algorithms.restart_pd_rLASSO(
        s0, A, AT, b, lam=lam_est, tol=tol, restarts=restarts, beta=beta, alpha=alpha
    )
    residuals = torch.zeros((2,), dtype=dtype, device=device)
    residuals[0] = torch.linalg.norm(A(rec) - b)
    if lstsq_rec:
        residuals[1] = torch.linalg.norm(A(lsr_rec) - b)
    return rec, vals, vals_dual, lsr_rec, residuals


def Reconstruction_Cosine(
    samples,
    values,
    indices,
    restarts=9,
    tol=1e-5,
    beta=3.0,
    alpha=0.3,
    lam_est=1.0,
    lstsq_rec=False,
):
    D = samples.size(1)  # dimension of sample points
    N = samples.size(0)  # number of sample points
    dtype = samples.dtype
    device = samples.device

    # precompute normalization coefficients for computational efficiency
    norm_coeffs = math.sqrt(2) ** ((indices.clamp(0, 1) - 1).sum(-1, keepdim=True)).to(
        dtype=dtype
    )

    A_op = Cosine_eval(samples, indices, norm_coeffs, D)
    AT_op = aCosine_eval(samples, indices, norm_coeffs, D)

    def A(x):
        return A_op(x) / math.sqrt(N)

    def AT(y):
        return AT_op(y) / math.sqrt(N)

    b = values / math.sqrt(N)

    if lstsq_rec:
        # compute least squares reconstruction, 1/gamma is Tikhonov regularization parameter
        lsr_rec = least_squares(
            A,
            AT,
            b,
            solver="lsqr",
            gamma=1e4,
            tol=1e-5,
            max_iter=2000,
            parallel_dim=-1,
        )
    else:
        lsr_rec = None

    lam_est = lam_est / math.sqrt(N)
    s0 = torch.zeros_like(AT(b), requires_grad=False)

    # run prim dual algorithm with restarts
    rec, vals, vals_dual = PD_algorithms.restart_pd_rLASSO(
        s0, A, AT, b, lam=lam_est, tol=tol, restarts=restarts, beta=beta, alpha=alpha
    )
    residuals = torch.zeros((2,), dtype=dtype, device=device)
    residuals[0] = torch.linalg.norm(A(rec) - b)
    if lstsq_rec:
        residuals[1] = torch.linalg.norm(A(lsr_rec) - b)
    return rec, vals, vals_dual, lsr_rec, residuals


def aTchebychev_eval(p, k, pre, D):
    # Adjoint Techebychev transform (multidimensional)
    # x : tensor of type torch.Tensor and shape (N,1), real valued
    # p, k : tensors of type torch.Tensor and shapes (N,D), (M,D)
    k_i = Vi(k)  # (M, 1, D) LazyTensor
    pre_i = Vi(pre)  # (M, 1, 1) LazyTensor
    p_acos_j = Vj(torch.acos(p))  # (1, N, D) LazyTensor
    x_j = Vj(0, 1)  # (1, N, 1) LazyTensor

    tmp = (k_i[:, :, 0] * p_acos_j[:, :, 0]).cos()
    for d in range(D - 1):
        tmp *= (k_i[:, :, d + 1] * p_acos_j[:, :, d + 1]).cos()
    return (pre_i * tmp * x_j).sum_reduction(dim=1, use_double_acc=True)


def Tchebychev_eval(p, k, pre, D):
    # Techebychev transform (multidimensional)
    # x : tensor of type torch.Tensor and shape (N,1), real valued
    # p, k : tensors of type torch.Tensor and shapes (N,D), (M,D)
    k_j = Vj(k)  # (1, M, D) LazyTensor
    pre_j = Vj(pre)  # (1, M, 1) LazyTensor
    p_acos_i = Vi(torch.acos(p))  # (N, 1, D) LazyTensor
    x_j = Vj(0, 1)  # (1, M, 1) LazyTensor

    tmp = (k_j[:, :, 0] * p_acos_i[:, :, 0]).cos()
    for d in range(D - 1):
        tmp *= (k_j[:, :, d + 1] * p_acos_i[:, :, d + 1]).cos()
    return (pre_j * tmp * x_j).sum_reduction(dim=1, use_double_acc=True)


def aCosine_eval(p, k, pre, D):
    # Adjoint Cosine transform (multidimensional)
    # x : tensor of type torch.Tensor and shape (N,1), real valued
    # p, k : tensors of type torch.Tensor and shapes (N,D), (M,D)
    k_i = Vi(math.pi * k)  # (M, 1, D) LazyTensor
    pre_i = Vi(pre)  # (M, 1, 1) LazyTensor
    p_j = Vj(p)  # (N, 1, D) LazyTensor
    x_j = Vj(0, 1)  # (1, N, 1) LazyTensor

    tmp = (k_i[:, :, 0] * (p_j[:, :, 0] + 1) / 2).cos()
    for d in range(D - 1):
        tmp *= (k_i[:, :, d + 1] * (p_j[:, :, d + 1] + 1) / 2).cos()
    return (pre_i * tmp * x_j).sum_reduction(dim=1, use_double_acc=True)


def Cosine_eval(p, k, pre, D):
    # Cosine transform (multidimensional)
    # x : tensor of type torch.Tensor and shape (N,1), real valued
    # p, k : tensors of type torch.Tensor and shapes (N,D), (M,D)
    k_j = Vj(math.pi * k)  # (1, M, D) LazyTensor
    pre_j = Vj(pre)  # (1, M, 1) LazyTensor
    p_i = Vi(p)  # (N, 1, D) LazyTensor
    x_j = Vj(0, 1)  # (1, M, 1) LazyTensor

    tmp = (k_j[:, :, 0] * (p_i[:, :, 0] + 1) / 2).cos()
    for d in range(D - 1):
        tmp *= (k_j[:, :, d + 1] * (p_i[:, :, d + 1] + 1) / 2).cos()
    return (pre_j * tmp * x_j).sum_reduction(dim=1, use_double_acc=True)


def NUDFT(p, f):
    # Non-Uniform Discrete Fourier Transform (multidimensional)
    # x : tensor of type torch.Tensor and shape (N,1), real valued
    # p, f : tensors of type torch.Tensor and shapes (N,D), (M,D)
    f_i = Vi(f)  # (M, 1, D) LazyTensor
    p_j = Vj(p)  # (1, N, D) LazyTensor
    x_j = Vj(0, 1)  # (1, N, 1) LazyTensor
    return ((p_j | f_i).unary("ComplexExp1j", dimres=2) * x_j).sum_reduction(
        dim=1, use_double_acc=True
    )


def aNUDFT(p, f):
    # Adjoint Non-Uniform Discrete Fourier Transform (multidimensional)
    # x : tensor of type torch.Tensor and shape (N,2), real-valued
    # p, f : tensors of type torch.Tensor and shapes (N,D), (M,D)
    f_j = Vj(f)  # (1, M, D) LazyTensor
    p_i = Vi(p)  # (N, 1, D) LazyTensor
    x_j = Vj(0, 2)  # (1, M, 2) LazyTensor
    return ((f_j | p_i).unary("ComplexExp1j", dimres=2) | x_j).sum_reduction(
        dim=1, use_double_acc=True
    )
