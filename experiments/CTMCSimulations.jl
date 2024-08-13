include("../src/SLVCTMCApproximation.jl")
include("../src/DiscreteTimeApproximation.jl")
using .SLVCTMCApproximation, .DiscreteTimeApproximation
using Revise, Random, Plots

PS_1 = Dict(
    "mu" => 0.02,
    "nu" => 0.085,
    "mean_reversion_coeff" => 6.21, # ϱ
    "rho" => -0.7,
    "kappa" => 0.2
)

PS_2 = Dict(
    "mu" => 0.02,
    "nu" => 0.424,
    "mean_reversion_coeff" => 6.00, # ϱ
    "rho" => -0.75,
    "kappa" => 0.8
)

PS_3 = Dict(
    "mu" => 0.02,
    "nu" => 0.225,
    "mean_reversion_coeff" => 2.86, # ϱ
    "rho" => -0.96,
    "kappa" => 0.6
)

explicit_params = Dict(
    "mu" => 0.0319,
    "nu" => 0.093025,
    "mean_reversion_coeff" => 6.21, # ϱ
    "rho" => -0.7,
    "kappa" => 0.61
)

S_0, V_0, n, N, r, T = 100, 0.501, 200, 10, 10, 200
@time S, V = DiscreteTimeApproximation.weighted_heston(S_0, V_0, n, N, 6, T, PS_1)
# μ, ν, ϱ, κ, ρ = 0.02, 0.085, 6.21, 0.2, -0.7
# Use PS1 parameters
μ, ν, ρ, κ, ϱ = PS_1["mu"], PS_1["nu"], PS_1["rho"], PS_1["kappa"], PS_1["mean_reversion_coeff"]
@time S, V, transitions = SLVCTMCApproximation.multiple_price_volatility_simulations(T, μ, ν, ρ, κ, ϱ, S_0, V_0)
for i in length(transitions)
    println(transitions[i])
end