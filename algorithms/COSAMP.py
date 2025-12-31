import math
import torch
from pykeops.torch import Vi, Vj
from deepinv.optim.utils import least_squares
from functools import partial


def Tchebychev(
    samples, values, indices, sparsity=5000, num_iters=100, tol=1e-5, lstsq_rec=False
):
    D = samples.size(1)  # dimension of sample points
    N = samples.size(0)  # number of sample points
    dtype = samples.dtype
    device = samples.device

    # precompute normalization coefficients for computational efficiency
    norm_coeffs = math.sqrt(2) ** (indices.clamp(0, 1).sum(-1, keepdim=True))

    # store samples in acos format
    samples_acos = torch.acos(samples)

    normalization = torch.sqrt(
        normalization_Techebychev(samples_acos, indices, norm_coeffs, D) / N
    )

    A_handle = Tchebychev_eval(samples_acos, D)
    AT_handle = aTchebychev_eval(samples_acos, D)

    def A(x, mask_indices=None):
        if mask_indices == None:
            f = indices
            pre = norm_coeffs
        else:
            f = indices[mask_indices]
            pre = norm_coeffs[mask_indices]
        return A_handle(x, f, pre) / math.sqrt(N)

    def AT(x, mask_indices=None):
        if mask_indices == None:
            f = indices
            pre = norm_coeffs
        else:
            f = indices[mask_indices]
            pre = norm_coeffs[mask_indices]
        return AT_handle(x, f, pre) / math.sqrt(N)

    b = values / math.sqrt(N)

    if lstsq_rec:
        # compute least squares reconstruction, 1/gamma is Tikhonov regularization parameter
        lsr_rec = least_squares(
            A,
            AT,
            b,
            solver="lsqr",
            gamma=2e5,
            tol=1e-5,
            max_iter=2000,
            parallel_dim=-1,
        )
    else:
        lsr_rec = None

    # run OMP reconstruction
    rec = COSAMP(
        A,
        AT,
        normalization,
        b,
        indices,
        samples_acos,
        sparsity=sparsity,
        num_iters=num_iters,
        tol=tol,
    )
    residuals = torch.zeros((2,), dtype=dtype, device=device)
    residuals[0] = torch.linalg.norm(A(rec) - b)
    if lstsq_rec:
        residuals[1] = torch.linalg.norm(A(lsr_rec) - b)
    return rec, lsr_rec, residuals


def COSAMP(A, AT, normalization, b, f, p, sparsity=5000, num_iters=100, tol=1e-4):
    device = p.device
    dtype = p.dtype
    sparsity = min(sparsity, f.size(0))
    two_sparsity = min(2 * sparsity, f.size(0))

    x = torch.zeros(normalization.numel(), device=device, dtype=dtype)
    z_2 = torch.zeros_like(x)
    support = torch.empty(0, device=device, dtype=torch.long)
    inv_diag = 1.0 / (normalization.flatten() + 1e-10)

    res = b.clone()
    res_norm_old = 0

    for j in range(num_iters):
        torch.abs(AT(res).flatten() * inv_diag, out=z_2)
        _, ind = torch.topk(z_2, k=two_sparsity)
        support_cand = torch.cat((support, ind))
        support_cand = torch.unique(support_cand)
        A_small = partial(A, mask_indices=support_cand)
        AT_small = partial(AT, mask_indices=support_cand)

        lsr_rec = least_squares(
            A_small,
            AT_small,
            b,
            z=0,
            init=x[support_cand].view(-1, 1),
            solver="minres",
            gamma=1e5,
            tol=1e-6,
            max_iter=500,
            parallel_dim=-1,
        )

        lsr_flat = lsr_rec.flatten()
        _, max_ind = torch.topk(torch.abs(lsr_flat), k=sparsity)
        vals = lsr_flat[max_ind]
        support = support_cand[max_ind]
        res = b - A(vals.view(-1, 1), mask_indices=support)

        x.zero_()
        x[support] = vals

        res_norm = torch.linalg.vector_norm(res)
        if torch.abs(res_norm - res_norm_old) / (torch.abs(res_norm) + 1e-4) < tol:
            break
        if j % 5 == 0:
            print("Iteration:", j + 1, " Residual:", res_norm.item())
        res_norm_old = res_norm
    return x.view(-1, 1)


def aTchebychev_eval(p_acos, D):
    # Adjoint Techebychev Transform (multidimensional)
    # x : tensor of type torch.Tensor and shape (N,1), real valued
    # p, k : tensors of type torch.Tensor and shapes (N,D), (M,D)
    k_i = Vi(1, D)  # (M, 1, D) LazyTensor
    pre_i = Vi(2, 1)  # (M, 1, 1) LazyTensor
    p_acos_j = Vj(p_acos)  # (1, N, D) LazyTensor
    x_j = Vj(0, 1)  # (1, N, 1) LazyTensor

    tmp = (k_i[:, :, 0] * p_acos_j[:, :, 0]).cos()
    for d in range(D - 1):
        tmp *= (k_i[:, :, d + 1] * p_acos_j[:, :, d + 1]).cos()
    return (pre_i * tmp * x_j).sum_reduction(dim=1, use_double_acc=True)


def Tchebychev_eval(p_acos, D):
    # Techebychev Transform (multidimensional)
    # x : tensor of type torch.Tensor and shape (N,1), real valued
    # p, k : tensors of type torch.Tensor and shapes (N,D), (M,D)
    k_j = Vj(1, D)  # (1, M, D) LazyTensor
    pre_j = Vj(2, 1)  # (1, M, 1) LazyTensor
    p_acos_i = Vi(p_acos)  # (N, 1, D) LazyTensor
    x_j = Vj(0, 1)  # # (1, M, 1) LazyTensor

    tmp = (k_j[:, :, 0] * p_acos_i[:, :, 0]).cos()
    for d in range(D - 1):
        tmp *= (k_j[:, :, d + 1] * p_acos_i[:, :, d + 1]).cos()
    return (pre_j * tmp * x_j).sum_reduction(dim=1, use_double_acc=True)


def normalization_Techebychev(p_acos, k, pre, D):
    # normalization of matrix columns
    k_i = Vi(k)  # (M, 1, D) LazyTensor
    pre_i = Vi(pre)  # (M, 1, 1) LazyTensor
    p_acos_j = Vj(p_acos)  # (1, N, D) LazyTensor

    tmp = (k_i[:, :, 0] * p_acos_j[:, :, 0]).cos()
    for d in range(D - 1):
        tmp *= (k_i[:, :, d + 1] * p_acos_j[:, :, d + 1]).cos()
    return ((pre_i * tmp) ** 2).sum_reduction(dim=1, use_double_acc=True)
