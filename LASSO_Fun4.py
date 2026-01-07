import math
import torch
import numpy as np
import algorithms.SQ_LASSO as SQ_LASSO

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
            return np.arange(0, R + 1, dtype=np.int32).reshape(-1, 1)
        out = []
        for k in range(0, R + 1):
            prod_val = max(1, abs(k))
            R_reduced = R // prod_val
            sub_result = hyp_cross(dim - 1, R_reduced)
            extended = np.empty((len(sub_result), dim), dtype=np.int32)
            extended[:, 0] = k  # First dimension value (broadcast to all rows)
            extended[:, 1:] = sub_result  # Remaining dimensions from recursion
            out.append(extended)
        return np.vstack(out)

    # create hyperbolic cross indices
    indices= torch.from_numpy(hyp_cross(D, M)).to(dtype=torch.double, device=device)

    # function in H^3/2 with known Fourier coefficients
    def fun3(x):
        out = (torch.abs(8 * x - 6.4) + 1) / 2
        return out.prod(dim=1, keepdim=True)

    # estimate norm and truncation error
    norm_f_sq = ((4.8 + 6.4 * math.asin(0.8)) / math.pi + 18.49) ** D
    norm_f = math.sqrt(norm_f_sq)

    # convert k and x_gt to given dtype
    indices = indices.to(dtype=dtype)

    # create vector with normalized function values
    values = fun3(samples) / norm_f

    # Optionally print info
    print("Number of indices in hyperbolic cross:", indices.size(0))

    return indices, samples, values


def estimate_error(coeffs, M, D, N_MC=1_000_000):
    device_id = coeffs.device
    dtype = coeffs.dtype
    indices, samples_MC, values_MC = generate_data(
        N_MC, M, D, device=device_id, dtype=dtype
    )

    # precompute normalization coefficients for computational efficiency
    pre_comp = math.sqrt(2) ** (indices.clamp(0, 1).sum(-1, keepdim=True)).to(
        dtype=dtype
    )

    A_op = SQ_LASSO.Tchebychev_eval(samples_MC, indices, pre_comp, D)
    return torch.sqrt(torch.sum((A_op(coeffs) - values_MC) ** 2) / N_MC)


m_values = [1000, 5000, 10000, 20000, 50000, 100000, 300000, 600000]  #
J_values = [5, 8, 20, 30, 60, 135]  #

results = []

for m in m_values:
    for J in J_values:
        indices, samples, values = generate_data(m, J, 6, device=device_id, dtype=dtype)

        coeffs_rec, vals, vals_dual, coeffs_lsr_rec, residuals = (
            SQ_LASSO.Reconstruction_Tchebychev(
                samples,
                values,
                indices,
                lstsq_rec=False,
                tol=1e-5,
                restarts=11,
            )
        )
        duality_gap = vals[-1] - vals_dual[-1]

        # Run 5 times and collect errors
        errors = []
        for run in range(5):
            error = estimate_error(coeffs_rec, J, 6).item()
            errors.append(error)

        # Compute mean and std
        error_mean = np.mean(errors)
        error_std = np.std(errors, ddof=1)  # ddof=1 for sample std

        results.append([m, J, error_mean, error_std, duality_gap])
        # Convert to numpy array and save
        results_array = np.array(results)
        np.savetxt(
            "results/results_lasso_fun4.txt",
            results_array,
            header="m J error_mean error_std duality_gap",
            fmt=["%d", "%d", "%.6e", "%.6e", "%.6e"],
        )
        print(
            f"Completed m={m}, J={J}, error_mean={error_mean:.6e}, error_std={error_std:.6e}, duality_gap={duality_gap:.6e}"
        )
