import math
import torch
from pykeops.torch import Vi, Vj
from deepinv.optim.utils import least_squares


def Fourier(samples, values, frequencies, num_iters=5000, tol=1e-4, lstsq_rec=False):
    D = samples.size(1)  # dimension of sample points
    N = samples.size(0)  # number of sample points
    dtype = samples.dtype
    device = samples.device

    A_handle = aNUDFT(samples, D)
    AT_handle = NUDFT(samples, D)

    def A(x, mask_indices=None):
        if mask_indices is None:
            f = frequencies
            return A_handle(x, f)
        else:
            # only evaluate selected frequencies
            ind_freq = mask_indices >> 1
            f = frequencies[ind_freq, :]
            # handle real and imaginary parts
            ind_y = mask_indices & 1
            tmp = torch.zeros((x.size(0), 2), dtype=dtype, device=device)
            tmp.scatter_(1, ind_y.unsqueeze(1), x)
            return A_handle(tmp, f)

    def AT(x, mask_indices=None):
        if mask_indices is None:
            f = frequencies
            return AT_handle(x, f)
        else:
            # only evaluate selected frequencies
            ind_freq = mask_indices >> 1
            f = frequencies[ind_freq, :]
            result = AT_handle(x, f)
            # extract correct real or imaginary part
            ind_y = mask_indices & 1
            return torch.gather(result, 1, ind_y.unsqueeze(1))

    def col_extractor(samples, frequencies, max_ind):
        # build complex matrix for least squares problem explicitly
        ind_x, ind_y = max_ind >> 1, max_ind & 1
        mat = (samples.to(torch.double) * frequencies[ind_x, :].to(torch.double)).sum(
            -1
        )
        if ind_y == 0:
            return torch.cos(mat)[:, None].to(dtype)
        else:
            return torch.sin(mat)[:, None].to(dtype)

    b = values
    normalization = torch.sqrt(normalization_fourier(samples, frequencies))

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
    rec = OMP(
        col_extractor,
        A,
        AT,
        normalization,
        b,
        frequencies,
        samples,
        num_iters=num_iters,
        tol=tol,
    )
    residuals = torch.zeros((2,), dtype=dtype, device=device)
    residuals[0] = torch.linalg.norm(A(rec) - b) / math.sqrt(N)
    if lstsq_rec:
        residuals[1] = torch.linalg.norm(A(lsr_rec) - b) / math.sqrt(N)
    return rec, lsr_rec, residuals


def Tchebychev(samples, values, indices, num_iters=5000, tol=1e-4, lstsq_rec=False):
    D = samples.size(1)  # dimension of sample points
    N = samples.size(0)  # number of sample points
    dtype = samples.dtype
    device = samples.device

    # store acos of samples
    samples_acos = torch.acos(samples)

    # precompute normalization coefficients for computational efficiency
    norm_coeffs = math.sqrt(2) ** (indices.clamp(0, 1).sum(-1, keepdim=True))

    A_handle = Tchebychev_eval(samples_acos, D)
    AT_handle = aTchebychev_eval(samples_acos, D)

    def A(x, mask_indices=None):
        if mask_indices == None:
            f = indices
            pre = norm_coeffs
        else:
            f = indices[mask_indices]
            pre = norm_coeffs[mask_indices]
        return A_handle(x, f, pre)

    def AT(x, mask_indices=None):
        if mask_indices == None:
            f = indices
            pre = norm_coeffs
        else:
            f = indices[mask_indices]
            pre = norm_coeffs[mask_indices]
        return AT_handle(x, f, pre)

    def col_extractor(samples_acos, indices, max_ind):
        # build complex matrix for least squares problem explicitly
        mat = norm_coeffs[max_ind, 0] * torch.cos(
            indices[max_ind, :].to(torch.double) * samples_acos.to(torch.double)
        ).prod(-1)
        return mat[:, None].to(dtype)

    b = values
    normalization = torch.sqrt(
        normalization_Techebychev(samples_acos, indices, norm_coeffs, D)
    )

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
    rec = OMP(
        col_extractor,
        A,
        AT,
        normalization,
        b,
        indices,
        samples_acos,
        num_iters=num_iters,
        tol=tol,
    )
    residuals = torch.zeros((2,), dtype=dtype, device=device)
    residuals[0] = torch.linalg.norm(A(rec) - b) / math.sqrt(N)
    if lstsq_rec:
        residuals[1] = torch.linalg.norm(A(lsr_rec) - b) / math.sqrt(N)
    return rec, lsr_rec, residuals


def OMP(col_extractor, A, AT, normalization, b, f, p, num_iters=1500, tol=1e-5):
    device = p.device
    dtype = p.dtype
    eps = torch.finfo(dtype).eps
    N = b.size(0)
    corr = AT(b)
    x = torch.zeros_like(corr, device=device, dtype=dtype, requires_grad=False)

    selected_indices = torch.zeros(
        num_iters, device=device, dtype=torch.long, requires_grad=False
    )

    L = torch.zeros((num_iters, num_iters), device=device, dtype=dtype)
    rhs = torch.zeros((num_iters, 1), device=device, dtype=dtype)
    diag = torch.clamp(normalization.flatten(), min=eps)
    z_2 = torch.empty_like(diag)
    res = b
    num_iters = min(num_iters, x.numel())
    num_iters = math.floor(min(num_iters, b.size(0)))

    for j in range(num_iters):
        # Compute and mask correlations
        torch.abs(AT(res).flatten() / diag, out=z_2)
        if j > 0:
            z_2[selected_indices[:j]] = -1

        # Find maximum correlation
        max_ind = torch.argmax(z_2)
        selected_indices[j] = max_ind

        # Recursive construction of Cholesky decomposition
        # See https://ieeexplore.ieee.org/document/6333943/
        new_col = col_extractor(p, f, max_ind)

        if j == 0:
            L[0, 0] = torch.linalg.vector_norm(new_col)
        else:
            corr = AT(new_col, mask_indices=selected_indices[:j])
            v = torch.linalg.solve_triangular(L[:j, :j], corr, upper=False)
            L[j, :j] = v.T
            L[j, j] = torch.sqrt(
                torch.clamp(diag[max_ind] ** 2 - torch.sum(v**2), min=eps**2)
            )

        rhs[j, :] = torch.dot(new_col.flatten(), b.flatten())
        out = torch.cholesky_solve(rhs[: j + 1, :], L[: j + 1, : j + 1])
        res = b - A(out[: j + 1], mask_indices=selected_indices[: j + 1])

        residual = torch.linalg.vector_norm(res) / math.sqrt(N)
        if j % 100 == 0:
            print("Iteration:", j + 1, " Residual:", residual.item())
        if residual < tol:
            num_iters = j + 1
            break

    # Update solution using selected indices
    x.flatten()[selected_indices[:num_iters]] = out[:num_iters].flatten()

    return x


def NUDFT(p, D):
    # Non-Uniform Discrete Fourier Transform (multidimensional)
    # x : tensor of type torch.Tensor and shape (N,1), real valued
    # p, f : tensors of type torch.Tensor and shapes (N,D), (M,D)
    f_i = Vi(1, D)  # (M, 1, D) LazyTensor
    p_j = Vj(p)  # (1, N, D) LazyTensor
    x_j = Vj(0, 1)  # (1, N, 1) LazyTensor
    return ((p_j | f_i).unary("ComplexExp1j", dimres=2) * x_j).sum_reduction(
        dim=1, use_double_acc=True
    )


def aNUDFT(p, D):
    # Non-Uniform Discrete Fourier Transform (multidimensional)
    # x : tensor of type torch.Tensor and shape (N,2), real-valued
    # p, f : tensors of type torch.Tensor and shapes (N,D), (M,D)
    f_j = Vj(1, D)  # (1, M, D) LazyTensor
    p_i = Vi(p)  # (N, 1, D) LazyTensor
    x_j = Vj(0, 2)  # (1, M, 2) LazyTensor
    return ((f_j | p_i).unary("ComplexExp1j", dimres=2) | x_j).sum_reduction(
        dim=1, use_double_acc=True
    )


def normalization_fourier(p, f):
    # normalization of matrix columns
    f_i = Vi(f)
    p_j = Vj(p)
    return ((p_j | f_i).unary("ComplexExp1j", dimres=2) ** 2).sum_reduction(
        dim=1, use_double_acc=True
    )


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
    x_j = Vj(0, 1)  # (1, M, 1) LazyTensor

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
