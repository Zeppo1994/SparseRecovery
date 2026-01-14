import math
import torch
import numpy as np
import algorithms.OMP as OMP

dtype = torch.float
device_id = "cuda"


def generate_data(N, M, D, dtype=torch.float, device="cuda"):
    # sample D-dimensional array of random points in [0,1]^D
    samples = torch.rand(N, D, dtype=dtype, device=device, requires_grad=False)

    # define D-dimensional array of frequencies in [-M, M]^D based on hyperbolic cross density
    def hyp_cross(d, M):
        if d == 1:
            return np.arange(-M, M + 1, dtype=np.int32).reshape(-1, 1)

        results = []
        for k in range(-M, M + 1):
            sub_result = hyp_cross(d - 1, int(M / max(1, abs(k))))
            extended = np.empty((len(sub_result), d), dtype=np.int32)
            extended[:, 0] = k
            extended[:, 1:] = sub_result
            results.append(extended)

        return np.vstack(results)

    # frequencies in [-M, M]^D scaled by -2π
    frequencies = (
        -2
        * math.pi
        * torch.from_numpy(hyp_cross(D, M)).to(dtype=torch.double, device=device)
    )
    M_f = frequencies.size(0)

    # function in H^3/2 with known Fourier coefficients
    def fun1(x, D):
        out = torch.ones_like(x[:, 0], requires_grad=False)
        for i in range(D):
            out = out * torch.clip(
                0.2 - (x[:, i] - 0.5) ** 2,
                min=0,
                max=None,
            )
        return (15 / (4 * np.sqrt(3)) * 5 ** (3 / 4)) ** D * out[:, None]

    def fun1_fourier_coeffs(f, D):
        out = torch.ones_like(f[:, 0], requires_grad=False)
        for i in range(D):
            tmp = (
                5 ** (5 / 4)
                * np.sqrt(3)
                * (-1) ** torch.round(f[:, i] / (2 * np.pi))
                * (
                    np.sqrt(5) * torch.sin(f[:, i] / np.sqrt(5))
                    - f[:, i] * torch.cos(f[:, i] / np.sqrt(5))
                )
                / (f[:, i] ** 3)
            )
            tmp[torch.isnan(tmp)] = 5 ** (1 / 4) / np.sqrt(3)
            out = out * tmp
        return out

    # compute truncation error
    coeffs_gt_d = torch.zeros((M_f, 2), dtype=torch.double)
    coeffs_gt_d[:, 0] = fun1_fourier_coeffs(frequencies.to(dtype=torch.double), D)
    trunc_error = torch.sqrt(1 - torch.sum(coeffs_gt_d[:, 0] ** 2))

    # cast to chosen dtype/device
    frequencies = frequencies.to(dtype=dtype)
    coeffs_gt = torch.zeros((M_f, 2), dtype=dtype, device=device_id)
    coeffs_gt[:, 0] = fun1_fourier_coeffs(frequencies, D)

    # since we approximate real functions, we can drop half of the Fourier coeffcients
    frequencies_half = frequencies[: math.ceil(frequencies.size(0) / 2), :]

    # create vector with normalized function values
    values = fun1(samples, D)

    # Optionally print info
    print("Number of Fourier frequencies in Hyperbolic cross:", M_f)
    print("Truncation error computed with double precision:", trunc_error.item())

    return frequencies, frequencies_half, samples, values, coeffs_gt, trunc_error


m_values = [1000, 5000, 10000, 50000, 100000, 300000, 600000]
# m_values = [1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000, 256000, 512000, 1024000, 2048000]
J_values = [5, 8, 20, 30, 60, 135]  # 145
D = 5

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
            num_iters=20000,  # 30k plot, 20k table
            lstsq_rec=False,
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
            "results/results_omp_fun1.txt",
            results_array,
            header="m J error",
            fmt=["%d", "%d", "%.6e"],
        )

        print(f"Completed m={m}, J={J}, error={error:.6e}")
