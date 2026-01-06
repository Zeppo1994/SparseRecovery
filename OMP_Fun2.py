import math
import torch
import numpy as np
import algorithms.OMP as OMP

dtype = torch.float
device_id = "cuda"


def bspline_test_7d(x):
    """
    7-dimensional test function consisting of tensor-products of B-splines.
    """
    if x.shape[1] != 7:
        raise ValueError("input must be M x 7 matrix")

    valout = bspline_o2(x[:, [0, 2, 3]]) + bspline_o4(x[:, [1, 4, 5, 6]])

    return valout


def bspline_o2(x):
    """Quadratic B-spline"""
    x = x - torch.floor(x)

    val = torch.ones(x.shape[0], device=device_id, dtype=dtype)

    for t in range(x.shape[1]):
        ind = torch.where((0 <= x[:, t]) & (x[:, t] < 1 / 2))[0]
        if len(ind) > 0:
            val[ind] = val[ind] * 4.0 * x[ind, t]

        ind = torch.where((1 / 2 <= x[:, t]) & (x[:, t] < 1))[0]
        if len(ind) > 0:
            val[ind] = val[ind] * 4.0 * (1 - x[ind, t])

        val = math.sqrt(3 / 4) * val

    return val


def bspline_o4(x):
    """Quartic B-spline"""
    x = x - torch.floor(x)

    val = torch.ones(x.shape[0], device=device_id, dtype=dtype)

    for t in range(x.shape[1]):
        ind = torch.where((0 <= x[:, t]) & (x[:, t] < 1 / 4))[0]
        if len(ind) > 0:
            val[ind] = val[ind] * 128 / 3 * x[ind, t] ** 3

        ind = torch.where((1 / 4 <= x[:, t]) & (x[:, t] < 2 / 4))[0]
        if len(ind) > 0:
            val[ind] = val[ind] * (
                8 / 3 - 32 * x[ind, t] + 128 * x[ind, t] ** 2 - 128 * x[ind, t] ** 3
            )

        ind = torch.where((2 / 4 <= x[:, t]) & (x[:, t] < 3 / 4))[0]
        if len(ind) > 0:
            val[ind] = val[ind] * (
                -88 / 3 - 256 * x[ind, t] ** 2 + 160 * x[ind, t] + 128 * x[ind, t] ** 3
            )

        ind = torch.where((3 / 4 <= x[:, t]) & (x[:, t] < 1))[0]
        if len(ind) > 0:
            val[ind] = val[ind] * (
                128 / 3
                - 128 * x[ind, t]
                + 128 * x[ind, t] ** 2
                - (128 / 3) * x[ind, t] ** 3
            )

        val = math.sqrt(315 / 604) * val

    return val


def bspline_test_7d_fouriercoeff(freq_out):

    fhat = torch.zeros(freq_out.shape[0], device=device_id, dtype=freq_out.dtype)

    ind = torch.where(torch.sum(torch.abs(freq_out[:, [1, 4, 5, 6]]), axis=1) <= 1e-8)[
        0
    ]
    if len(ind) > 0:
        fhat[ind] = fhat[ind] + bspline_o2_hat(freq_out[ind][:, [0, 2, 3]])

    ind = torch.where(torch.sum(torch.abs(freq_out[:, [0, 2, 3]]), axis=1) <= 1e-8)[0]
    if len(ind) > 0:
        fhat[ind] = fhat[ind] + bspline_o4_hat(freq_out[ind][:, [1, 4, 5, 6]])

    norm_fct_square = (
        2
        + 2
        * bspline_o2_hat(torch.zeros((1, 3), device=device_id, dtype=freq_out.dtype))[0]
        * bspline_o4_hat(torch.zeros((1, 4), device=device_id, dtype=freq_out.dtype))[0]
    )

    return fhat, norm_fct_square


def bspline_o2_hat(k):
    """Fourier coefficients of quadratic B-spline"""
    val = torch.ones(k.shape[0], device=device_id, dtype=k.dtype)

    for t in range(k.shape[1]):
        ind = torch.where(k[:, t] != 0)[0]
        if len(ind) > 0:
            val[ind] = (
                val[ind]
                * bspline_sinc(torch.pi / 2 * k[ind, t]) ** 2
                * (-1) ** k[ind, t]
            )

        val = math.sqrt(3 / 4) * val

    return val


def bspline_o4_hat(k):
    """Fourier coefficients of quartic B-spline"""
    val = torch.ones(k.shape[0], device=device_id, dtype=k.dtype)

    for t in range(k.shape[1]):
        ind = torch.where(k[:, t] != 0)[0]
        if len(ind) > 0:
            val[ind] = (
                val[ind]
                * bspline_sinc(torch.pi / 4 * k[ind, t]) ** 4
                * (-1) ** k[ind, t]
            )

        val = math.sqrt(315 / 604) * val

    return val


def bspline_sinc(x):
    """Sinc function: sin(x)/x"""
    return torch.sin(x) / x


def generate_data(N, M, D, dtype=torch.float, device="cuda"):
    # sample D-dimensional array of random points in [0,1]^D
    samples = torch.rand(N, D, dtype=dtype, device=device, requires_grad=False)

    # define D-dimensional array of frequencies in [-M, M]^D based on hyperbolic cross density
    def hyp_cross(d, M):
        if d == 1:
            return [[k] for k in range(-M, M + 1)]
        out = []
        for k in range(-M, M + 1):
            for temp in hyp_cross(d - 1, int(M / max(1, abs(k)))):
                out.append([k] + temp)
        return out

    # frequencies in [-M, M]^D scaled by -2π
    frequencies = torch.FloatTensor(hyp_cross(D, M)).to(
        dtype=torch.double, device=device_id
    )
    M_f = frequencies.size(0)

    # compute truncation error
    coeffs_gt = torch.zeros((M_f, 2), dtype=torch.double)
    coeffs_gt[:, 0], norm_sq = bspline_test_7d_fouriercoeff(frequencies)
    coeffs_gt = coeffs_gt / math.sqrt(norm_sq)
    trunc_error = torch.sqrt(1 - torch.sum(coeffs_gt[:, 0] ** 2))

    # cast to chosen dtype/device
    frequencies = frequencies.to(dtype=dtype)
    coeffs_gt = coeffs_gt.to(dtype=dtype, device=device_id)

    frequencies = -2 * math.pi * torch.round(frequencies)

    # since we approximate real functions, we can drop half of the Fourier coeffcients
    frequencies_half = frequencies[: math.ceil(frequencies.size(0) / 2), :]

    # create vector with normalized function values
    values = bspline_test_7d(samples)[:, None] / math.sqrt(norm_sq)

    # Print info
    print("Number of Fourier frequencies in hyperbolic cross:", M_f)
    print("Truncation error computed with double precision:", trunc_error.item())

    return frequencies, frequencies_half, samples, values, coeffs_gt, trunc_error


m_values = [1000, 5000, 10000, 50000, 100000, 300000, 600000]  #
J_values = [5, 8, 20, 30, 60]  #
D = 7

results = []

for m in m_values:
    for J in J_values:
        frequencies, frequencies_half, samples, values, coeffs_gt, trunc_error = (
            generate_data(m, J, D, device=device_id, dtype=dtype)
        )

        coeffs_rec, coeffs_lsr_rec, residuals = OMP.Fourier(
            samples,
            values,
            frequencies_half,
            num_iters=20000,
            tol=1e-4,
        )

        # add missing Fourier coeffcients
        coeffs_rec = torch.cat(
            (
                coeffs_rec[:-1, :] / 2,
                coeffs_rec[-1, :][None],
                torch.flipud(coeffs_rec[:-1, :]) / 2,
            ),
            0,
        )
        coeffs_rec[-math.ceil(frequencies.size(0) / 2) :, 1] = -coeffs_rec[
            -math.ceil(frequencies.size(0) / 2) :, 1
        ]

        error = (trunc_error + torch.linalg.norm(coeffs_gt - coeffs_rec)).item()
        results.append([m, J, error])

        # Convert to numpy array and save
        results_array = np.array(results)
        np.savetxt(
            "results/results_omp_fun2.txt",
            results_array,
            header="m J error",
            fmt=["%d", "%d", "%.6e"],
        )

        print(f"Completed m={m}, J={J}, error={error:.6e}")
