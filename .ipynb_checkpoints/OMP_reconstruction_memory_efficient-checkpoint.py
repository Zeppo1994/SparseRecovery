import math
import torch
import numpy as np
from pykeops.torch import Genred, Vi, Vj
from deepinv.optim.utils import least_squares

def Fourier(samples, values, frequencies, num_iters=5000, tol=1e-5, lstsq_rec=False):
    D = samples.size(1)  # dimension of sample points
    N = samples.size(0)  # number of sample points
    dtype = samples.dtype
    device = samples.device

    A_op = aNUDFT(samples, frequencies)
    AT_op = NUDFT(samples, frequencies)

    def A(x):
        return A_op(x) / math.sqrt(N)


    def AT(y):
        return AT_op(y) / math.sqrt(N)

    def col_extractor(samples, frequencies, max_ind):
        # build complex matrix for least squares problem explicitly
        ind_x, ind_y = torch.floor(max_ind/2).to(int), max_ind % 2
        mat = (samples[:,None,:].to(torch.double) * frequencies[None,ind_x,:].to(torch.double)).sum(-1)
        if ind_y==0:
            mat = torch.cos(mat)
        else:
            mat = torch.sin(mat)     
        return mat.to(dtype)/np.sqrt(N)

    b = values / math.sqrt(N)
    normalization = torch.sqrt(normalization_fourier(samples, frequencies)/N)

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

    # run OMP reconstruction
    rec = OMP(col_extractor, AT, normalization, b, frequencies, samples, num_iters=num_iters)
    residuals = torch.zeros((2,), dtype=dtype, device = device)
    residuals[0] = torch.linalg.norm(A(rec) - b)
    if lstsq_rec:
        residuals[1] = torch.linalg.norm(A(lsr_rec) - b)
    return rec, lsr_rec, residuals


def Tchebychev(samples, values, indices, num_iters=5000, tol=1e-5, tensor=False, lstsq_rec=False):
    D = samples.size(1)  # dimension of sample points
    N = samples.size(0)  # number of sample points
    dtype = samples.dtype
    device = samples.device

    # precompute normalization coefficients for computational efficiency
    norm_coeffs = math.sqrt(2) ** (indices.clamp(0, 1).sum(-1, keepdim=True)).to(
    dtype=dtype
    )

    if tensor:
        A_Mat = norm_coeffs * torch.cos(k[:, None, :] * torch.acos(p[None, :, :])).prod(-1)
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

    def col_extractor(samples, indices, max_ind):
        # build complex matrix for least squares problem explicitly
        mat =  norm_coeffs[max_ind] * torch.cos(indices[None, max_ind, :] * torch.acos(samples[:,None, :])).prod(-1)
        return mat.to(dtype)/np.sqrt(N)

    b = values / math.sqrt(N)
    normalization = torch.sqrt(normalization_Techebychev(samples, indices, norm_coeffs, D)/N)

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

    # run OMP reconstruction
    rec = OMP(col_extractor, AT, normalization, b, indices, samples, num_iters=num_iters)
    residuals = torch.zeros((2,), dtype=dtype, device = device)
    residuals[0] = torch.linalg.norm(A(rec) - b)
    if lstsq_rec:
        residuals[1] = torch.linalg.norm(A(lsr_rec) - b)
    return rec, lsr_rec, residuals


def OMP(col_extractor, AT, normalization, b, f, p, num_iters=1500, tol=1e-5):
    device = p.device
    dtype = p.dtype
    
    x = torch.zeros_like(AT(b), device=device, dtype=dtype, requires_grad=False)
    indices = torch.zeros_like(AT(b), device=device, dtype=bool,  requires_grad=False).flatten()
    out_ind = torch.zeros(num_iters, device=device, dtype=int, requires_grad=False)
    L = torch.zeros((num_iters, num_iters), device=device, dtype=dtype)
    diag = normalization.flatten()
    res = b
    
    for j in range(num_iters):
        if j%25==0:
            print("Iteration:", j+1, " Residual:",torch.linalg.vector_norm(res).item())
        if (torch.linalg.vector_norm(res)) < tol:
            num_iters = j
            break
        
        z_2 = AT(res).flatten()/(diag+1e-4)
        z_2[indices] = 0
        max_ind = torch.argmax(torch.abs(z_2))
        out_ind[j] = max_ind
        indices[max_ind] = True

        # Recursive construction of Cholesky decomposition
        # See https://ieeexplore.ieee.org/document/6333943/
        if j == 0:
            A_mat = col_extractor(p, f, max_ind)
            L[0,0] = torch.sqrt( A_mat.transpose(0,1) @ A_mat)
        else:
            new_col = col_extractor(p, f, max_ind)
            v = torch.linalg.solve_triangular(L[:j,:j], A_mat.transpose(0,1) @ new_col, upper=False)
            L[j, :j] = v.T
            L[j, j] = torch.sqrt(
                torch.clamp(diag[max_ind]**2 - torch.sum(v**2), min=1e-8)
            )
            
            A_mat = torch.cat((A_mat, new_col), dim=1)
        rhs = A_mat.transpose(0,1) @ b
        out = torch.cholesky_solve(rhs, L[:j+1,:j+1])
        res = b - A_mat @ out
    x_flat = x.flatten()
    out = out.squeeze()
    for j in range(num_iters):
        x_flat[out_ind[j]] = out[j]
    x = x_flat.view_as(x)
    return x


def NUDFT(p, f):
    # Non-Uniform Discrete Fourier Transform (multidimensional)
    #   x : tensor of type torch.Tensor and shape (N,1), real valued
    #   p, f : tensors of type torch.Tensor and shapes (N,D)
    f_i = Vi(f)
    p_j = Vj(p)
    x_j = Vj(0, 1)
    return ((p_j | f_i).unary("ComplexExp1j", dimres=2) * x_j).sum_reduction(
        dim=1, use_double_acc=True
    )


def aNUDFT(p, f):
    # Non-Uniform Discrete Fourier Transform (multidimensional)
    #   x : tensor of type torch.Tensor and shape (N,2), real-valued
    #   p, f : tensors of type torch.Tensor and shapes (N,D)
    f_j = Vj(f)
    p_i = Vi(p)
    x_j = Vj(0, 2)
    return ((f_j | p_i).unary("ComplexExp1j", dimres=2) | x_j).sum_reduction(
        dim=1, use_double_acc=True
    )


def normalization_fourier(p, f):
    # Non-Uniform Discrete Fourier Transform (multidimensional)
    #   x : tensor of type torch.Tensor and shape (N,1), real valued
    #   p, f : tensors of type torch.Tensor and shapes (N,D)
    f_i = Vi(f)
    p_j = Vj(p)
    return ((p_j | f_i).unary("ComplexExp1j", dimres=2)**2).sum_reduction(
        dim=1, use_double_acc=True
    )
    

def aTchebychev_eval(p, k, pre, D):
    # Non-Uniform Discrete Fourier Transform (multidimensional)
    #   x : tensor of type torch.Tensor and shape (N,1), real valued
    #   p, k : tensors of type torch.Tensor and shapes (N,D)
    k_i = Vi(k)  # (M, 1, D) LazyTensor
    pre_i = Vi(pre)  # (M, 1, 1) LazyTensor
    p_acos_j = Vj(torch.acos(p))  # (1, N, D) LazyTensor
    x_j = Vj(0, 1)

    tmp = (k_i[:, :, 0] * p_acos_j[:, :, 0]).cos()
    for d in range(D - 1):
        tmp *= (k_i[:, :, d + 1] * p_acos_j[:, :, d + 1]).cos()
    return (pre_i * tmp * x_j).sum_reduction(dim=1, use_double_acc=True)


def Tchebychev_eval(p, k, pre, D):
    # Non-Uniform Discrete Fourier Transform (multidimensional)
    #   x : tensor of type torch.Tensor and shape (N,2), real-valued
    #   p, f : tensors of type torch.Tensor and shapes (N,D)
    k_j = Vj(k)  # (M, 1, D) LazyTensor
    pre_j = Vj(pre)  # (M, 1, 1) LazyTensor
    p_acos_i = Vi(torch.acos(p))  # (1, N, D) LazyTensor
    x_j = Vj(0, 1)

    tmp = (k_j[:, :, 0] * p_acos_i[:, :, 0]).cos()
    for d in range(D - 1):
        tmp *= (k_j[:, :, d + 1] * p_acos_i[:, :, d + 1]).cos()
    return (pre_j * tmp * x_j).sum_reduction(dim=1, use_double_acc=True)


def normalization_Techebychev(p, k, pre, D):
    # Non-Uniform Discrete Fourier Transform (multidimensional)
    #   x : tensor of type torch.Tensor and shape (N,1), real valued
    #   p, k : tensors of type torch.Tensor and shapes (N,D)
    k_i = Vi(k)  # (M, 1, D) LazyTensor
    pre_i = Vi(pre)  # (M, 1, 1) LazyTensor
    p_acos_j = Vj(torch.acos(p))  # (1, N, D) LazyTensor

    tmp = (k_i[:, :, 0] * p_acos_j[:, :, 0]).cos()
    for d in range(D - 1):
        tmp *= (k_i[:, :, d + 1] * p_acos_j[:, :, d + 1]).cos()
    return ((pre_i * tmp)**2).sum_reduction(dim=1, use_double_acc=True)