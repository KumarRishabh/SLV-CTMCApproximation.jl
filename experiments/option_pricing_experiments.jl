using Plots
using LinearAlgebra
using SparseArrays

include("../src/CuiSDEApproximation.jl")

# Example usage
S0 = 10.0         # Initial stock price
V0 = 0.04    
# S0 = 10, v0 = 0.04, T = 1, K = 4, ρ = −0.75, σv = 0.15, η = 4, θ = 0.035, r = 0      # Initial variance
params = Dict(
    "r" => 0.0,        # Risk-free rate
    "mu" => 0.05,        # Expected return
    "kappa" => 4,      # Mean reversion rate
    "theta" => 0.035,     # Long-term variance
    "sigma" => 0.15,      # Volatility of variance
    "rho" => -0.75        # Correlation between asset and variance
)
T = 1.0          # Time horizon (in years)
M = 40 # Number of variance levels (states)
N = 40        # Number of asset price levels (states)
# Choose the mapping function
mapping_function_S = arcsinh_mapping
mapping_function_V = arcsinh_mapping # or any other mapping function


# Simulate the Heston model65
Strike = 4.0
times_asset, S, V_path, times_variance = simulate_heston_ctmc_general(S0, V0, params, T, M, mapping_function_V)

plot(times_asset, S, label="Asset Price", xlabel="Time", ylabel="Price", legend=:topleft)

plot(times_variance, V_path, label="Variance", xlabel="Time", ylabel="Variance", legend=:topleft)


European_call_price = European_call_price_krylov(S0, V0, params, T, M, N, mapping_function_S, mapping_function_V, Strike, "call"; risk_free_rate=0.0)

monitoring_times = collect(0.0:0.05:T)
option_price = price_american_option_ctmc(S0, V0, params, T, M, N, linear_mapping, linear_mapping, Strike, "call", monitoring_times)

println("American Call option price using CTMC: $option_price")
