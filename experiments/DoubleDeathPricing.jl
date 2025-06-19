using Plots
using LinearAlgebra
using SparseArrays
using Parameters
using Test
using Revise

include("../src/DoubleDeathApproximation.jl")


dummy_b1(S,V) = 0.05 * S
dummy_b2(V) = 0.1 * (0.04 - V)
dummy_sigma11(S,V) = sqrt(1 - (-0.7)^2) * sqrt(V) * S
dummy_sigma12(S,V) = -0.7 * sqrt(V) * S
dummy_sigma22(V) = 6.21 * sqrt(V)

function linear_mapping(ξ, V₀, V_min, V_max, M, γ)
    return V_min .+ ξ * (V_max - V_min)
end
# Benchmark parameter sets from Table 1
PS_american = DoubleDeathApproximation.HestonParams(
    S0 = 100.0,
    μ  = 0.05 - 0.03,
    ν   = 1.0 * 0.09,    # κ * θ
    κ = 0.40,
    ϱ = 1.0,
    ρ   = -0.7,
    v0  = 0.09
)
PS_european = DoubleDeathApproximation.HestonParams(
    S0 = 100.0,
    μ = 0.05 - 0.03,
    ν   = 1.0 * 0.09,    # κ * θ
    κ = 0.40,
    ϱ = 1.0,
    ρ   = -0.7,
    v0  = 0.09
)

T_american, K_american = 1.0, 100.0
T_european, K_european = 0.25, 100.0

function adaptive_kappa(s, κ, ρ)
    absρ = abs(ρ)
    if 0 < s <= κ * absρ
        return s / absρ
    elseif s <= κ / absρ
        return κ
    else
        return absρ * s
    end
end
heston_b1(S, V; params=PS_american) = params.μ * S
heston_b2(V; params=PS_american) = params.ν - params.ϱ * V
heston_sigma11(S, V; params=PS_american) = sqrt(1 - params.ρ^2) * sqrt(V) * S
heston_sigma12(S, V; params=PS_american) = params.ρ * sqrt(V) * S
heston_sigma22(S, V; params=PS_american) = adaptive_kappa(S, params.κ, params.ρ) * sqrt(V)
# Test the mapping functions

@test length(arcsinh_mapping(range(0,1,length=10), 0.04, 0.01, 0.1, 10, 5)) == 10
# Removed invalid mapping_function assignment and call below:
mapped_vals = linear_mapping(range(0,1,length=10), 0.04, 0.01, 0.1, 10, 5)
@test length(mapped_vals) == 10
# Test generator matrix construction
N = 5; ℓ = 0.01
Q = DoubleDeathApproximation.construct_Q(N, ℓ; b1=heston_b1, b2=heston_b2,
                  sigma11=heston_sigma11, sigma12=heston_sigma12, sigma22=heston_sigma22)
2@test size(Q) == (N*N, N*N)
 
# Price European call option using PS1 parameters for the heston model
K = 16.0
T = 1
r = 0.0
N = 50
ℓ = 1.0 / N
S = range(0.001, 200, length=N)
V = range(0.01, 10, length=N)
dt = 0.01
monitoring_times = (0.0+dt):dt:T

# Benchmark explicit CTMC pricing for both parameter sets
param_list = [
    ("American call", PS_american, T_american, K_american, false),
    ("European put", PS_european, T_european, K_european, true)
]

for (name, PS, T, K, is_european) in param_list
    println("=== Benchmarking $name ===")
    # Construct the generator matrix
    heston_b1(S, V; params=PS) = PS.μ * S
    heston_b2(V; params=PS) = PS.ν - PS.ϱ * V
    heston_sigma11(S, V; params=PS) = sqrt(1 - PS.ρ^2) * sqrt(V) * S
    heston_sigma12(S, V; params=PS) = PS.ρ * sqrt(V) * S
    heston_sigma22(S, V; params=PS) = adaptive_kappa(S, PS.κ, PS.ρ) * sqrt(V)

    Q = DoubleDeathApproximation.construct_Q(N, ℓ;
        b1 = heston_b1, b2 = heston_b2,
        sigma11 = heston_sigma11, sigma12 = heston_sigma12,
        sigma22 = heston_sigma22, reduced = true)

    # Time the explicit method
    t_explicit = @elapsed price_explicit = if is_european
        DoubleDeathApproximation.price_european_option_double_death(
            PS.S0, PS.v0, T_european, Q, N, N, K, "put", PS;
            risk_free_rate = PS.μ
        )
        # DoubleDeathApproximation.European_call_price_krylov(
        #     PS.S0, PS.v0, T_european, Q, N, N, K, "put", PS;
        #     risk_free_rate = PS.μ
        # )
    else
        # convert PS to the required format
        DoubleDeathApproximation.price_american_option_ctmc(
            PS.S0, PS.v0, T_american, N, N, linear_mapping, linear_mapping, K, "call", monitoring_times, PS; risk_free_rate = PS.μ
        )
    end

    println("$name -> Explicit price: $price_explicit, time: $(round(t_explicit, digits=4)) seconds\n")
end