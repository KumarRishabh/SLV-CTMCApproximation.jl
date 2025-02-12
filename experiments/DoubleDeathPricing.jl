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


# PS1 = DoubleDeathApproximation.HestonParams(
#     S0 = 100,
#     μ_h = 0.02,
#     ν = 0.085,
#     θ_h = 0.04,    # or another desired value for the long-run variance
#     κ_h = 0.2,     # override the default mean reversion rate
#     ρ = -0.7,
#     v0 = 0.501
# )
PS1 = DoubleDeathApproximation.HestonParams(S0=10.0, μ_h=0.0, ν=0.14, θ_h=4.0, κ_h=0.15, ρ=-0.75, v0=0.04)
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
heston_b1(S, V; params=PS1) = params.μ_h * S
heston_b2(V; params=PS1) = params.ν - params.θ_h * V
heston_sigma11(S, V; params=PS1) = sqrt(1 - params.ρ^2) * sqrt(V) * S
heston_sigma12(S, V; params=PS1) = params.ρ * sqrt(V) * S
heston_sigma22(S, V; params=PS1) = adaptive_kappa(S, params.κ_h, params.ρ) * sqrt(V)
# Test the mapping functions

@test length(arcsinh_mapping(range(0,1,length=10), 0.04, 0.01, 0.1, 10, 5)) == 10

# Test generator matrix construction
N = 5; ℓ = 0.01
Q = DoubleDeathApproximation.construct_Q(N, ℓ; b1=heston_b1, b2=heston_b2,
                  sigma11=heston_sigma11, sigma12=heston_sigma12, sigma22=heston_sigma22)
@test size(Q) == (N*N, N*N)

# Price European call option using PS1 parameters for the heston model
K = 16.0
T = 1
r = 0.0
N = 50
ℓ = 1.0 / N
S = range(0.001, 200, length=N)
V = range(0.01, 10, length=N)
@time Q = DoubleDeathApproximation.construct_Q(N, ℓ; b1=heston_b1, b2=heston_b2,
                  sigma11=heston_sigma11, sigma12=heston_sigma12, sigma22=heston_sigma22, reduced=true)
# Price the European call option

# Check if the non-diagonal elements of Q are negative

negative_off_diagonal_indices = [(i, j) for i in 1:size(Q, 1), j in 1:size(Q, 2) if i != j && Q[i, j] < 0]

# Display the indices and their corresponding values
for (i, j) in negative_off_diagonal_indices
    if i != j
        println("Q[$i, $j] = ", Q[i, j])
    end
end
option_price = DoubleDeathApproximation.price_european_option_double_death(PS1.S0, PS1.v0, T, Q, N, N, K, "call", PS1; risk_free_rate=r)
println("Option price: $option_price")

# Krylov method
option_price_krylov = DoubleDeathApproximation.European_call_price_krylov(PS1.S0, PS1.v0, T, Q, N, N, K, "call", PS1; risk_free_rate=r)

# American option 
option_price_american = DoubleDeathApproximation.price_american_option_double_death(PS1.S0, PS1.v0, T, Q, N, N, K, "put", PS1; risk_free_rate=r)