import math
import torch
import numpy as np
import algorithms.COSAMP as COSAMP

dtype = torch.float
device_id = "cuda"


def generate_data(N, M, D, dtype=torch.double, device="cuda"):
    # sample D-dimensional array of random Tchebychev points in [-1,1]^D
    samples = math.pi * (
        torch.rand(N, D, dtype=dtype, device=device, requires_grad=False) - 0.5
    )
    samples = samples.tan()
    samples = samples / torch.sqrt(samples**2 + 1)

    # define D-dimensional array of indices in [0, R]^D based on hyperbolic cross density
    def hyp_cross(dim, R):
        if dim == 1:
            return [[k] for k in range(0, R + 1)]
        out = []
        for kk in hyp_cross(dim - 1, R):
            prod_val = 1
            for x in kk:
                prod_val *= max([1, abs(x)])
            rk = R // prod_val
            for r in range(0, rk + 1):
                out.append(kk + [r])
        return out

    # create hyperbolic cross indices
    indices = torch.round(
        torch.FloatTensor(hyp_cross(D, M)).to(dtype=torch.double, device=device)
    )
    indices.requires_grad = False

    # function in H^3/2 with known Fourier coefficients
    def fun2(x):
        neg_mask = x <= 0
        out = torch.where(
            neg_mask,
            -(x**2) / 4 - x / 2 + 0.5,
            x**2 / 8 - x / 2 + 0.5,
        )
        out *= math.sqrt(1024 / (367 - 256 / math.pi))
        return out.prod(dim=1, keepdim=True)

    # Tchebychev coefficients
    def fun2_tchebychev_coeffs(indices):
        # Initialize output
        out = torch.zeros_like(indices, dtype=torch.double)

        # k = 0
        mask0 = indices == 0
        out = out + 15 / math.sqrt((367 - 256 / math.pi)) * mask0.double()

        # k = 1
        mask1 = indices == 1
        out = (
            out
            - 8
            / math.pi
            * math.sqrt(2 / (367 - 256 / math.pi))
            * (math.pi - 1)
            * mask1.double()
        )

        # k = 2
        mask2 = indices == 2
        out = out - 1 / math.sqrt((2 * 367 - 512 / math.pi)) * mask2.double()

        # k >= 3: Use exact formula without clamp
        mask_large = indices >= 3
        k_large = indices[mask_large]

        # Compute sin(π*k/2) exactly: alternates 0, 1, 0, -1, 0, 1, ...
        k_mod_4 = k_large % 4
        sin_vals = torch.where(
            k_mod_4 == 1,
            torch.ones_like(k_large),
            torch.where(
                k_mod_4 == 3, -torch.ones_like(k_large), torch.zeros_like(k_large)
            ),
        )

        coeff_large = (
            -24
            / math.pi
            * math.sqrt(2 / (367 - 256 / math.pi))
            * sin_vals
            / (k_large * (k_large**2 - 4))
        )

        out[mask_large] = coeff_large

        return torch.prod(out, dim=1, keepdim=True)

    # estimate norm and truncation error
    coeffs_gt = fun2_tchebychev_coeffs(indices)
    trunc_error = torch.sqrt(1 - torch.sum((coeffs_gt) ** 2))

    # convert k and x_gt to given dtype
    indices = indices.to(dtype=dtype)
    coeffs_gt = coeffs_gt.to(dtype=dtype)

    # create vector with normalized function values
    values = fun2(samples)

    # Optionally print info
    print("Number of indices in hyperbolic cross:", indices.size(0))
    print("Truncation error (double precision):", trunc_error.item())

    return indices, samples, values, coeffs_gt, trunc_error


m_values = [
    1000,
    5000,
    10000,
    50000,
    100000,
    200000,
]  # [4000, 8000, 16000, 32000, 64000, 128000, 256000]
J_values = [5, 10, 20, 30, 50, 100]  #

results = []

for m in m_values:
    s = int(min(m / 4, 20000))  # sparsity level
    for J in J_values:
        indices, samples, values, coeffs_gt, trunc_error = generate_data(
            m, J, 6, device=device_id, dtype=dtype
        )

        coeffs_rec, coeffs_lsr_rec, residuals = COSAMP.Tchebychev(
            samples,
            values,
            indices,
            sparsity=s,
            tol=1e-4,
            num_iters=200,
        )

        error = (trunc_error + torch.linalg.norm(coeffs_gt - coeffs_rec)).item()
        results.append([m, J, error])

        # Convert to numpy array and save
        results_array = np.array(results)
        np.savetxt(
            "results/results_cosamp_fun3.txt",
            results_array,
            header="m J error",
            fmt=["%d", "%d", "%.6e"],
        )

        print(f"Completed m={m}, J={J}, error={error:.6e}")
