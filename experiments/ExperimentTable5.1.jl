"""
Need to price European call options and American put options using the DoubleDeathApproximation, DOA method and Cui's method
"""

"""
For European options, use the following numbers for the parameters: 
Option Type     &  κ       θ     σ     ρ   rd  rf  T    K
European Put         1.15 0.0348 0.39 -0.64 0.04 0 0.25 100
American Call     1.15 0.0348 0.39 -0.64 0.04 0 1    100
"""

# Set up the parameters for the model 
include("../src/CuiSDEApproximation.jl")
include("../src/DoubleDeathApproximation.jl")
using Main.CuiSDEApproximation
using .DoubleDeathApproximation
using Plots
using LinearAlgebra
using SparseArrays
using Parameters
using Test
using Revise
using ProgressMeter

function linear_mapping(ξ, V₀, V_min, V_max, M, γ)
    return V_min .+ ξ * (V_max - V_min)
end

function arcsinh_mapping(ξ, V₀, V_min, V_max, M, γ)
    return V₀ + (V_max - V₀) * sinh(ξ * (asinh(V_min / V₀) - asinh(V_max / V₀)) / (M - 1))
end

# Define the parameters for the Heston model
PS_european = DoubleDeathApproximation.HestonParams(
    S0 = 100.0,
    μ = 0.04,
    ν   = 1.0 * 0.09,    # κ * θ
    κ = 1.15,
    ϱ = 1.0,        # κ in the alternate formulation of Heston model   
    ρ   = -0.64,
    v0  = 0.09
)
PS_american = DoubleDeathApproximation.HestonParams(
    S0 = 100.0,
    μ = 0.04,
    ν   = 1.0 * 0.09,    # κ * θ
    κ = 1.15,
    ϱ = 1.0,            # κ in the alternate formulation of the Heston model
    ρ   = -0.64,
    v0  = 0.09
)


T_american, K_american = 0.25, 100.0
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

# Initial option prices 
S_1 = 90.0
S_2 = 100.0
S_3 = 110.0

# Array of grid sizes

M = [8, 16, 32]
# N = [32, 64, 72, 80]
N = [10,15,20,25,30,35,40,45,50]

# Price the European call option using the Double Death Approximation
function GetEuropeanCallPrice(N, M; use_krylov=false)
    # Dictionary to store the prices
    European_call_prices = Dict{Tuple{Int, Int}, Float64}()
    # use a progress bar to track the progress of the loop

    @showprogress for n in N
            variance_grid_size = n 
            asset_grid_size = n
            Q = DoubleDeathApproximation.construct_Q(asset_grid_size, variance_grid_size; b1=heston_b1, b2=heston_b2,
                       sigma11=heston_sigma11, sigma12=heston_sigma12, sigma22=heston_sigma22)
            if use_krylov==true
                price = DoubleDeathApproximation.European_call_price_krylov(S_2, 0.09, T_european, Q, variance_grid_size, asset_grid_size, K_european, "call", PS_european; risk_free_rate=0.04)
            else
                # Use the Double Death Approximation method
                price = DoubleDeathApproximation.price_european_option_double_death(S_2, 0.09, T_european, Q, variance_grid_size, asset_grid_size, K_european, "call", PS_european; risk_free_rate=0.04)
            end
            European_call_prices[(n, n)] = price
    end
    return European_call_prices
end

Prices = GetEuropeanCallPrice(N, M; use_krylov=true)
println("European Call option prices using Double Death Approximation:", Prices)

# Price the American call option using the Double Death Approximation
function GetAmericanCallPrice(N, M)
    # Dictionary to store the prices
    American_call_prices = Dict{Tuple{Int, Int}, Float64}()
    for n in N
        for m in M
            variance_grid_size = m
            asset_grid_size = n
            Q = DoubleDeathApproximation.construct_Q(n, m; b1=heston_b1, b2=heston_b2,
                       sigma11=heston_sigma11, sigma12=heston_sigma12, sigma22=heston_sigma22)
            price = DoubleDeathApproximation.price_american_option_double_death(S_2, 0.09, T_american, Q, variance_grid_size, asset_grid_size, K_american, "call", PS_american; risk_free_rate=0.04)
            American_call_prices[(n, m)] = price
        end
    end
    return American_call_prices
end