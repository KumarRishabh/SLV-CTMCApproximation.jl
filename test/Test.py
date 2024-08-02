import numpy as np
import time

def HestonDiscretizationKahlJackel(S0, V0, T, N, params, Δt=1e-7):
    # Implement the Heston Discretization using the Kahl-Jackel method
    V = np.zeros((N, T+1))
    logS = np.zeros((N, T+1))
    V[:, 0] = V0
    logS[:, 0] = np.log(S0)
    
    for i in range(N):
        for j in range(1, T+1):
            Δβ = np.sqrt(Δt) * np.random.randn()
            ΔB = np.sqrt(Δt) * np.random.randn()  # independent brownian motions
            V[i, j] = V[i, j-1] + (params["nu"] - params["mean_reversion_coeff"] * V[i, j-1]) * Δt + params["kappa"] * np.sqrt(V[i, j-1]) * Δβ + 0.25 * params["kappa"]**2 * (Δβ**2 - Δt)
            Δβ = 0.04
            logS[i, j] = logS[i, j-1] + params["mu"] * Δt - 0.25 * (V[i, j-1] + V[i, j]) * Δt + params["rho"] * np.sqrt(V[i, j-1]) * Δβ + 0.5 * (np.sqrt(V[i, j-1]) + np.sqrt(V[i, j])) * (ΔB - params["rho"] * Δβ) + 0.25 * params["rho"] * (Δβ - Δt)
    
    return np.exp(logS), V

S0, V0, T, N = 100, 0.04, 100000, 1000
PS_1 = {"nu": 0.04, "mean_reversion_coeff": 0.1, "kappa": 0.5, "mu": 0.05, "rho": -0.75}

start_time = time.time()
S, V = HestonDiscretizationKahlJackel(S0, V0, T, N, PS_1)
end_time = time.time()

execution_time = end_time - start_time
print("Execution time:", execution_time, "seconds")
