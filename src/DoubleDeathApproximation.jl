module DoubleDeathApproximation
using Random
using Distributions
using LinearAlgebra
using SparseArrays
using StatsBase
using ExponentialUtilities: expv  # For efficient matrix exponential computation
using FastExpm: fastExpm
using Expokit: expmv
using Parameters
using ProgressMeter
using Revise
using Test
using Printf
using Plots

# Exported functions and types
export arcsinh_mapping, linear_mapping, construct_variance_levels, construct_asset_price_levels,
       simulate_variance_process, simulate_asset_price, compute_transition_matrix, compute_transition_matrices,
       construct_Q, construct_generator_matrix_general, price_european_option_double_death, construct_combined_generator_matrix,
       construct_payoff_vector, price_european_option_exponentiation, European_call_price_krylov,
       price_american_option_ctmc, initialize_option_values, backward_induction,
       HestonParams, SABRParams, ThreeTwoParams

#############
# Type Definitions
#############

@with_kw mutable struct HestonParams
    S0::Float64 = 100.0
    μ::Float64 = 0.05
    ν::Float64 = 0.10
    κ::Float64 = 0.3
    ϱ::Float64 = 2.0
    ρ::Float64 = -0.5
    v0::Float64 = 0.05
end

@with_kw mutable struct SABRParams
    S0::Float64 = 100.0
    β::Float64 = 0.5
    κ_s::Float64 = 6.00
    ρ::Float64 = -0.75
    v0::Float64 = 0.11
end

@with_kw mutable struct ThreeTwoParams
    S0::Float64 = 100.0
    μ_32::Float64 = 0.02
    ν_32::Float64 = 0.424
    θ_32::Float64 = 0.04
    κ_32::Float64 = 6.00
    ρ::Float64 = -0.7
    v0::Float64 = 0.501
end

#############
# Mapping Functions
#############

const DEFAULT_H_PARAMS = HestonParams()

#############
# Mapping / Coefficient Functions
#############

# asset‐variance covariance terms
function sigma11(S, V; params::HestonParams = DEFAULT_H_PARAMS)
    sqrt(1 - params.ρ^2) * sqrt(V) * S
end

function sigma12(S, V; params::HestonParams = DEFAULT_H_PARAMS)
    params.ρ * sqrt(V) * S
end

function sigma21(S, V; params::HestonParams = DEFAULT_H_PARAMS)
    0.0
end

function sigma22(S, V; params::HestonParams = DEFAULT_H_PARAMS)
    params.κ * sqrt(V)
end

# drift terms
function b11(S, V; params::HestonParams = DEFAULT_H_PARAMS)
    params.μ * S
end

function b22(V; params::HestonParams = DEFAULT_H_PARAMS)
    params.ν - params.ϱ * V
end


"""
    arcsinh_mapping(ξ, V₀, V_min, V_max, M, γ)

Maps a grid `ξ` in [0,1] to variance levels using the arcsinh transformation.
"""
function arcsinh_mapping(ξ, V₀, V_min, V_max, M, γ)
    c1 = asinh((V_min - V₀) / γ)
    c2 = asinh((V_max - V₀) / γ)
    return V₀ .+ γ * sinh.(c1 .+ ξ * (c2 - c1))
end

"""
    linear_mapping(ξ, V₀, V_min, V_max, M, γ)

A simple linear mapping from [0,1] to [V_min,V_max].
"""
function linear_mapping(ξ, V₀, V_min, V_max, M, γ)
    return V_min .+ ξ * (V_max - V_min)
end

"""
    construct_variance_levels(V_min, V_max, M, mapping_function, V₀; γ=5)

Constructs a grid of variance levels.
"""
function construct_variance_levels(V_min, V_max, M, mapping_function, V₀; γ=5)
    ξ = range(0.0, 1.0, length=M)
    return mapping_function(ξ, V₀, V_min, V_max, M, γ)
end

"""
    construct_asset_price_levels(S_min, S_max, N, mapping_function, S₀; γ=5)

Constructs a grid of asset price levels.
"""
function construct_asset_price_levels(S_min, S_max, N, mapping_function, S₀; γ=5)
    ξ = range(0.0, 1.0, length=N)
    return mapping_function(ξ, S₀, S_min, S_max, N, γ)
end

#############
# Simulation Functions
#############

"""
    simulate_variance_process(Q, V_levels, V₀, T)

Simulates the variance process over time using a CTMC with generator matrix `Q`.
"""
function simulate_variance_process(Q, V_levels, V₀, T)
    M = length(V_levels)
    idx_v = findmin(abs.(V_levels .- V₀))[2]
    V_adj = V_levels[idx_v]  # Adjust V₀ to nearest grid point

    t = 0.0
    times = [0.0]
    V_path = [V_adj]
    idx_v_path = [idx_v]

    while t < T
        rate = -1 / Q[idx_v, idx_v]
        τ = rate <= 0 ? (T - t) : rand(Exponential(rate))
        t += τ
        if t > T
            t = T
            push!(times, t)
            push!(V_path, V_levels[idx_v])
            push!(idx_v_path, idx_v)
            break
        end

        push!(times, t)
        push!(V_path, V_levels[idx_v])
        push!(idx_v_path, idx_v)

        # Transition step
        probs = copy(Q[idx_v, :])
        probs[idx_v] = 0.0
        total_rate = sum(probs)
        if total_rate > 0
            probs = probs / total_rate
            idx_v = sample(1:M, Weights(probs))
        end
    end

    return times, V_path, idx_v_path
end

"""
    simulate_asset_price(S₀, times, V_path, μ, ρ)

Simulates the asset price process over time. The function uses the Cholesky
decomposition to incorporate the correlation `ρ` between the Brownian drivers.
"""
function simulate_asset_price(S₀, times, V_path, μ, ρ)
    L = [1.0 0.0; ρ sqrt(1 - ρ^2)]
    S = [S₀]
    times_asset = [times[1]]
    for i in 1:(length(times)-1)
        Δt = times[i+1] - times[i]
        v_current = V_path[i]
        drift = (μ - 0.5 * v_current) * Δt
        # Generate a correlated Brownian increment for the asset
        Z = L * randn(2)
        diffusion = sqrt(v_current) * sqrt(Δt) * Z[1]
        S_new = S[end] * exp(drift + diffusion)
        push!(S, S_new)
        push!(times_asset, times[i+1])
    end
    return times_asset, S
end

"""
    compute_transition_matrix(Q, T)

Computes the transition probability matrix using the matrix exponential.
"""
function compute_transition_matrix(Q, T)
    return fastExpm(Q * T)
end

#############
# Generator Matrix Construction
#############

"""
    transition_rates(S, V, N; b1, b2, sigma11, sigma12, sigma22)

Computes the transition rates for the joint asset price and variance process.
The functions `b1`, `b2`, `sigma11`, `sigma12`, and `sigma22` should be provided
by the user (or by a wrapper that specifies a particular model).
"""
function transition_rates_dd(S, V, N; b1, b2, sigma11, sigma12, sigma22) # Only works with linear mappings for now
    a10 = N * max(b1(S, V), 0) + (N^2 / 2) * (sigma11(S, V)^2 + sigma12(S, V)^2 - abs(sigma12(S, V)) * sigma22(S, V))
    s10 = N * max(-b1(S, V), 0) + (N^2 / 2) * (sigma11(S, V)^2 + sigma12(S, V)^2 - abs(sigma12(S, V)) * sigma22(S, V))
    a01 = N * max(b2(V), 0) + (N^2 / 2) * (sigma22(S, V)^2 - abs(sigma12(S, V)) * sigma22(S, V))
    s01 = N * max(-b2(V), 0) + (N^2 / 2) * (sigma22(S, V)^2 - abs(sigma12(S, V)) * sigma22(S, V))
    a11 = s11 = (N^2 / 2) * max(sigma12(S, V), 0) * sigma22(S, V)
    t12 = t21 = (N^2 / 2) * max(-sigma12(S, V), 0) * sigma22(S, V)
    return (a10, s10, a01, s01, a11, s11, t12, t21)
end

function transition_rates_reduced(S, V, N; b1, b2, sigma11, sigma12, sigma22)
    a10 = N * max(b1(S, V), 0) + (N^2 / 2) * (sigma11(S, V)^2 + sigma12(S, V)^2 - abs(sigma12(S, V)) * sigma22(S, V))
    s10 = N * max(-b1(S, V), 0) + (N^2 / 2) * (sigma11(S, V)^2 + sigma12(S, V)^2)
    a01 = N * max(b2(V), 0) + (N^2 / 2) * (sigma22(S, V)^2 - 2 * max(sigma12(S, V), 0) * sigma22(S, V))
    s01 = N * max(-b2(V), 0) + (N^2 / 2) * (sigma22(S, V)^2 - 2 * max(-sigma12(S, V), 0) * sigma22(S, V))
    a11 = (N^2) * max(sigma12(S, V), 0) * sigma22(S, V)
    t21 = (N^2) * max(-sigma12(S, V), 0) * sigma22(S, V)
    t12 = s11 = 0.0
    return (a10, s10, a01, s01, a11, s11, t12, t21)
end
"""
    construct_Q(N, ℓ; b1, b2, sigma11, sigma12, sigma22)

Constructs the generator matrix Q for the CTMC approximation over a grid of
N asset price levels and N variance levels. The step size is given by ℓ.
"""
function construct_Q(N, S_max, S_min, V_max, V_min; b1 = b11, b2 = b22, sigma11 = sigma11, sigma12 = sigma12, sigma22 = sigma22, reduced=false)
    
    Q = spzeros(N * N, N * N)
    l1 = (S_max - S_min)/N
    l2 = (V_max - V_min)/N
    ProgressMeter.@showprogress for i in 1:N, j in 1:N
        idx = (i - 1) * N + j
        S = i * l1
        V = j * l2
        if reduced==false
            a10, s10, a01, s01, a11, s11, t12, t21 = transition_rates_dd(S, V, N;
                b1=b1, b2=b2, sigma11=sigma11, sigma12=sigma12, sigma22=sigma22)
        else
            a10, s10, a01, s01, a11, s11, t12, t21 = transition_rates_reduced(S, V, N;
                b1=b1, b2=b2, sigma11=sigma11, sigma12=sigma12, sigma22=sigma22)
        end
        if i < N
            Q[idx, idx + N] = a01
        end
        if i > 1
            Q[idx, idx - N] = s01
        end
        if j < N
            Q[idx, idx + 1] = a10
        end
        if j > 1
            Q[idx, idx - 1] = s10
        end
        if i < N && j < N
            Q[idx, idx + N + 1] = a11
        end
        if i > 1 && j > 1
            Q[idx, idx - N - 1] = s11
        end
        if i < N && j > 1
            Q[idx, idx + N - 1] = t12
        end
        if i > 1 && j < N
            Q[idx, idx - N + 1] = t21
        end
        Q[idx, idx] = -sum(Q[idx, :])
    end
    return Q
end


#############
# Option Pricing Functions
#############

"""
    construct_payoff_vector(V_levels, S_levels, Strike, option_type)

Constructs a payoff vector over the grid for a European call or put option.
"""
function construct_payoff_vector(V_levels, S_levels, Strike, option_type)
    M = length(V_levels)
    N = length(S_levels)
    total_states = M * N
    G = zeros(total_states)
    for vi in 1:M
        for si in 1:N
            state_idx = (vi - 1) * N + si
            S_i = S_levels[si]
            if option_type == "call"
                G[state_idx] = max(S_i - Strike, 0.0)
            elseif option_type == "put"
                G[state_idx] = max(Strike - S_i, 0.0)
            else
                error("Invalid option type. Choose 'call' or 'put'.")
            end
        end
    end
    return G
end

"""
    price_european_option_exponentiation(S₀, V₀, params, T, M, N, mapping_function_S, mapping_function_V, K, option_type; risk_free_rate=0.05)

Prices a European option using the matrix exponentiation approach.
dV_t ;=; ν - ϱ V_t dt + κ sqrt{V_t} dW_t^{(2)},
Because the CIR process is ergodic, its unconditional law converges to a Gamma distribution with mean ν/ϱ and variance frac{κ^2 ν^2}{2 ϱ}. 
"""
function price_european_option_double_death(S_0, V_0, T, variance_grid_size, asset_grid_size, strike_price, option_type, params::HestonParams; risk_free_rate=0.05)
    r = risk_free_rate
    V_min = max(0.0, params.ν / params.ϱ - 3 * params.κ * params.ν / sqrt(2 * params.ϱ)) # Lower bound for variance
    V_max = params.ν / params.ϱ + 3 * params.κ * params.ν / sqrt(2 * params.ϱ) # Upper bound for variance
    V_levels = construct_variance_levels(V_min, V_max, variance_grid_size, linear_mapping, V_0)

    S_min = S_0 * 0.1 # Could be adjusted based on the model
    S_max = S_0 * 5.0  # Could be adjusted based on the model
    S_levels = construct_asset_price_levels(S_min, S_max, asset_grid_size, linear_mapping, S_0)
    Q = construct_Q(asset_grid_size, S_max, S_min, V_max, V_min)
    # Use the double death approximation to construct the generator matrix
    # Q, V_levels, S_levels = construct_combined_generator_matrix(V_levels, S_levels, params)
    
    P = compute_transition_matrix(Q, T)
    G = construct_payoff_vector(V_levels, S_levels, strike_price, option_type)
    # @test size(G) == (variance_grid_size * asset_grid_size,)
    idx_V0 = findmin(abs.(V_levels .- V_0))[2]
    idx_S0 = findmin(abs.(S_levels .- S_0))[2]
    initial_state = (idx_V0 - 1) * asset_grid_size + idx_S0
    
    # return P, G
    initial_distribution = zeros(length(G))
    initial_distribution[initial_state] = 1.0

    option_price = exp(-r * T) * (initial_distribution' * P * G)[1]
    return option_price
end

function price_european_option_exponentiation(S₀, V₀, params::HestonParams, T, M, N, mapping_function_S, mapping_function_V, K, option_type; risk_free_rate=0.05)
    r = risk_free_rate
    V_min = max(0.0, params.ν / params.ϱ - 3 * params.κ * params.ν / sqrt(2 * params.ϱ)) 
    V_max = params.ν / params.ϱ + 3 * params.κ * params.ν / sqrt(2 * params.ϱ) 
    V_levels = construct_variance_levels(V_min, V_max, M, mapping_function_V, V₀)

    S_min = S₀ * 0.1 # Could be adjusted based on the model
    S_max = S₀ * 5.0 # Could be adjusted based on the model
    S_levels = construct_asset_price_levels(S_min, S_max, N, mapping_function_S, S₀)

    # Q, V_levels, S_levels = construct_combined_generator_matrix(V_levels, S_levels, params)
    Q = construct_Q(N, S_max, S_min, V_max, V_min)
    P = compute_transition_matrix(Q, T)
    G = construct_payoff_vector(V_levels, S_levels, K, option_type)

    idx_V0 = findmin(abs.(V_levels .- V₀))[2]
    idx_S0 = findmin(abs.(S_levels .- S₀))[2]
    initial_state = (idx_V0 - 1) * N + idx_S0
    π₀ = zeros(length(G))
    π₀[initial_state] = 1.0

    option_price = exp(-r * T) * (π₀' * P * G)[1]
    return option_price
end

"""
    European_call_price_krylov(S₀, V₀, params, T, M, N, mapping_function_S, mapping_function_V, K, option_type; risk_free_rate=0.0)

Prices a European call option using a Krylov subspace method.
"""
function European_call_price_krylov(S_0, V_0, params::HestonParams, T, variance_grid_size, asset_grid_size, mapping_function_S, mapping_function_V, strike_price, option_type; risk_free_rate=0.0, subspace_dim = 10)
    r = risk_free_rate
    V_min = max(0.0, params.ν / params.ϱ - 3 * params.κ * params.ν / sqrt(2 * params.ϱ)) # Lower bound for variance
    V_max = params.ν / params.ϱ + 3 * params.κ * params.ν / sqrt(2 * params.ϱ) # Upper bound for variance
    V_levels = construct_variance_levels(V_min, V_max, variance_grid_size, linear_mapping, V_0)

    S_min = S_0 * 0.1
    S_max = S_0 * 5.0
    S_levels = construct_asset_price_levels(S_min, S_max, asset_grid_size, linear_mapping, S_0)

    # Use the double death approximation to construct the generator matrix
    # Q, V_levels, S_levels = construct_combined_generator_matrix(V_levels, S_levels, params)
    Q = construct_Q(asset_grid_size, S_max, S_min, V_max, V_min)
    
    # P = compute_transition_matrix(Q, T)
    G = construct_payoff_vector(V_levels, S_levels, strike_price, option_type)
    # @test size(G) == (variance_grid_size * asset_grid_size,)
    idx_V0 = findmin(abs.(V_levels .- V_0))[2]
    idx_S0 = findmin(abs.(S_levels .- S_0))[2]
    initial_state = (idx_V0 - 1) * asset_grid_size + idx_S0
    
    # return P, G
    π₀ = zeros(length(G))
    π₀[initial_state] = 1.0

    Q_transpose = transpose(Q)
    Q_sparse = sparse(Q)
    # w_tilde = expmv(T, Q_transpose, π₀; tol=1e-6, m=subspace_dim)
    # w_tilde = expv(T, Q_transpose, π₀; tol=1e-6, m=subspace_dim)
    # track progress with Progressmeter for expmv
    # println("Computing w_tilde using Krylov subspace method...")
    @info "Computing w_tilde using Krylov subspace method..."
    
    w_tilde = expv(T, Q_transpose, π₀; tol=1e-6, m=min(subspace_dim, size(Q, 1) - 1))
    option_price = exp(-r * T) * dot(w_tilde, G)
    return option_price
end
"""
    compute_transition_matrices(Q, Δt_array)

Computes a list of transition matrices corresponding to each time step in Δt_array.
"""
function compute_transition_matrices(Q, Δt_array)
    P_matrices = []
    for Δt in Δt_array
        push!(P_matrices, fastExpm(Q * Δt))
    end
    return P_matrices
end

"""
    initialize_option_values(V_levels, S_levels, K, option_type)

Initializes the option payoff vector at maturity.
"""
function initialize_option_values(V_levels, S_levels, K, option_type)
    M = length(V_levels)
    N = length(S_levels)
    total_states = M * N
    V_option = zeros(total_states)
    for vi in 1:M
        for si in 1:N
            state_idx = (vi - 1) * N + si
            S_i = S_levels[si]
            if option_type == "call"
                V_option[state_idx] = max(S_i - K, 0.0)
            elseif option_type == "put"
                V_option[state_idx] = max(K - S_i, 0.0)
            else
                error("Invalid option type. Choose 'call' or 'put'.")
            end
        end
    end
    return V_option
end

"""
    backward_induction(P_matrices, V_option, V_levels, S_levels, K, option_type, r, Δt_array)

Performs backward induction over the monitoring times for pricing an American option.
"""
function backward_induction(P_matrices, V_option, V_levels, S_levels, K, option_type, r, Δt_array)
    M = length(V_levels)
    N = length(S_levels)
    total_states = M * N
    # for n in length(P_matrices):-1:1
    #use ProgressMeter to show progress
    ProgressMeter.@showprogress for n in length(P_matrices):-1:1
        P = P_matrices[n]
        continuation_value = real.(P * V_option)
        exercise_value = zeros(total_states)
        for state_idx in 1:total_states
            si = (state_idx - 1) % N + 1
            S_i = S_levels[si]
            if option_type == "call"
                exercise_value[state_idx] = max(S_i - K, 0.0)
            elseif option_type == "put"
                exercise_value[state_idx] = max(K - S_i, 0.0)
            end
        end
        continuation_value *= exp(-r * Δt_array[n])
        # Use krylov method for efficient computation
        V_option = expmv(-r * Δt_array[n], P, V_option)
        # V_option = real.(P * V_option) * exp(-r * Δt_array[n])
        V_option = max.(exercise_value, continuation_value)
    end
    return V_option
end

"""
    price_american_option_ctmc(S₀, V₀, params, T, M, N, mapping_function_S, mapping_function_V, K, option_type, monitoring_times)

Prices an American option using CTMC backward induction.
"""
function price_american_option_ctmc(S₀, V₀, T, M, N, mapping_function_S, mapping_function_V, K, option_type, monitoring_times, params::HestonParams; risk_free_rate=0.05)
    r = risk_free_rate

    V_min = max(0.0, params.ν / params.ϱ - 3 * params.κ * params.ν / sqrt(2 * params.ϱ))
    V_max = params.ν / params.ϱ + 3 * params.κ * params.ν / sqrt(2 * params.ϱ)
    V_levels = construct_variance_levels(V_min, V_max, M, mapping_function_V, V₀)

    S_min = S₀ * 0.02
    S_max = S₀ * 2.5
    S_levels = construct_asset_price_levels(S_min, S_max, N, mapping_function_S, S₀)

    Q = construct_Q(N, S_max, S_min, V_max, V_min)
    Δt_array = diff(monitoring_times)
    P_matrices = compute_transition_matrices(Q, Δt_array)
    V_option = initialize_option_values(V_levels, S_levels, K, option_type)
    V_option = backward_induction(P_matrices, V_option, V_levels, S_levels, K, option_type, r, Δt_array)

    idx_V0 = findmin(abs.(V_levels .- V₀))[2]
    idx_S0 = findmin(abs.(S_levels .- S₀))[2]
    initial_state = (idx_V0 - 1) * N + idx_S0

    option_price = V_option[initial_state]
    return option_price
end

"""
    my_heston_generator(f, s, v, params)

Infinitesimal generator for the SDE system:
    dS = μ S dt + sqrt((1 - ρ^2) v) S dW1 + ρ sqrt(v) S dW2
    dv = (ν - ϱ v) dt + κ sqrt(v) dW2
"""
function my_heston_generator(f, s, v, M, N, params::HestonParams)
    # extract parameters so params is used
    μ = params.μ
    ν = params.ν
    ϱ = params.ϱ
    κ = params.κ
    ρ = params.ρ

    l1 = (params.S0 * 5.0 - params.S0 * 0.1) / N  # Step size for asset price
    l2 = (ν / ϱ + 3 * κ * ν / sqrt(2 * ϱ) - max(0.0, ν / ϱ - 3 * κ * ν / sqrt(2 * ϱ))) / N  # Step size for variance
    # Numerical derivatives
    df_ds = (f(s + l1, v) - f(s - l1, v)) / (2*l1)
    df_dv = (f(s, v + l2) - f(s, v - l2)) / (2*l2)
    d2f_ds2 = (f(s + l1, v) - 2 * f(s, v) + f(s - l1, v)) / (l1^2)
    d2f_dv2 = (f(s, v + l2) - 2 * f(s, v) + f(s, v - l2)) / (l2^2)
    d2f_dsdv = (f(s + l1, v + l2) - f(s + l1, v - l2) - f(s - l1, v + l2) + f(s - l1, v - l2)) / (4*l1 * l2)

    # Generator as derived above
    out = μ * s * df_ds +
          (ν - ϱ * v) * df_dv +
          0.5 * v * s^2 * d2f_ds2 +
          0.5 * κ^2 * v * d2f_dv2 +
          ρ * κ * v * s * d2f_dsdv

    return out
end

function test_func(s, v)
    return s^2 + v^2
end

function run_generator_convergence_test()
    # params = HestonParams(S0=100.0, ν=0.04, κ=0.1, ϱ=6.21, ρ=-0.7, v0=0.04)
    params = HestonParams()

    # Grids to test (increasing resolution)
    grid_sizes = [10, 15, 20, 25, 30, 40, 50, 70, 85, 100, 150]
    errors = zeros(length(grid_sizes))

    for (gi, N) in enumerate(grid_sizes)
        M = N
        S_levels = range(params.S0 * 0.1, params.S0 * 5.0, length=N) |> collect
        V_levels = range(max(0.0, params.ν / params.ϱ - 3 * params.κ * params.ν / sqrt(2 * params.ϱ)), params.ν / params.ϱ + 3 * params.κ * params.ν / sqrt(2 * params.ϱ), length=M) |> collect

        # Combined generator and grid
        # Q, V_grid, S_grid = construct_combined_generator_matrix(V_levels, S_levels, params)
        Q = construct_Q(N, params.S0 * 5.0, params.S0 * 0.1, params.ν / params.ϱ + 3 * params.κ * params.ν / sqrt(2 * params.ϱ), max(0.0, params.ν / params.ϱ - 3 * params.κ * params.ν / sqrt(2 * params.ϱ)); reduced=false)

        # Evaluate f(s, v) on the grid
        F_grid = zeros(M, N)
        for vi in 1:M, si in 1:N
            F_grid[vi, si] = test_func(S_levels[si], V_levels[vi])
        end
        plot(F_grid, title="Function Values on Grid", xlabel="Variance Levels", ylabel="Asset Price Levels",
             colorbar=true, aspect_ratio=:equal, legend=false)
        # Flatten (order: v varies slowest)
        F_vec = vec(F_grid') # Julia is column-major

        # Discrete generator action
        L_discrete = Q * F_vec

        # Analytic generator action at each grid point
        L_analytic = zeros(M, N)
        for vi in 1:M, si in 1:N
            s = S_levels[si]
            v = V_levels[vi]
            L_analytic[vi, si] = my_heston_generator(test_func, s, v, M, N, params)
        end
        L_analytic_vec = vec(L_analytic')

        # Compute grid error (max-norm or L2 norm)
        errors[gi] = norm(L_discrete - L_analytic_vec, 2)
        @printf("Grid %dx%d: max error = %g\n", M, N, errors[gi])
    end

    # Plot convergence
    plot(grid_sizes, errors, xscale=:log10, yscale=:log10,
         marker=:circle, xlabel="Grid size (N=M)", ylabel="Max error",
         title="Convergence of Discrete Generator to Heston Operator")
end
end

# using .DoubleDeathApproximation

# Dummy data for testing

# # Example usage
S0 = 100.0      # Initial stock price
V0 = 0.05    

params = DoubleDeathApproximation.HestonParams()
T = 1.0          # Time horizon (in years)
M = 10 # Number of variance levels (states)
N = 10     # Number of asset price levels (states)
# # Choose the mapping function
# # mapping_function_S = arcsinh_mapping
# # mapping_function_V = arcsinh_mapping # or any other mapping function
mapping_function_S = DoubleDeathApproximation.linear_mapping
mapping_function_V = DoubleDeathApproximation.linear_mapping

# # Simulate the Heston model65
Strike = 100.0


@time European_call_price_normal = DoubleDeathApproximation.price_european_option_exponentiation(S0, V0, params, T, M, N, mapping_function_S, mapping_function_V, Strike, "call", risk_free_rate=0.0)

@time European_call_price = DoubleDeathApproximation.European_call_price_krylov(S0, V0, params, T, M, N, mapping_function_S, mapping_function_V, Strike, "call"; risk_free_rate=0.0)

# Plot the convergence of the European call price as we increase the number of grid points
European_call_prices = []
for N in 20:10:80
    @time European_call_price = DoubleDeathApproximation.European_call_price_krylov(S0, V0, params, T, N, N, mapping_function_S, mapping_function_V, Strike, "call"; risk_free_rate=0.05)
    println("European Call Price with N=$N: $European_call_price")
    push!(European_call_prices, European_call_price)
end
using Plots
# plot the convergence of the European call price as we increase the number of grid points
plot(20:10:150, European_call_prices, marker=:circle, xlabel="Number of grid points (N)", ylabel="European Call Price",
     title="Convergence of European Call Price with Increasing Grid Points", legend=false)


# # Price the American option
# option_price = price_american_option_ctmc(S0, V0, params, T, M, N, linear_mapping, linear_mapping, Strike, "call", monitoring_times)


# println("American Call option price using CTMC: $option_price")
using .DoubleDeathApproximation
M = 20 # Number of variance levels (states)
N = 20     # Number of asset price levels (states)
T = 1
S0 = 100.0
V0 = 0.501
Strike = 100.0
GT_params = DoubleDeathApproximation.HestonParams(S0=S0, ν=0.502, κ=0.2, ϱ=6.21, ρ=-0.7, v0=V0)
Δt = 0.05
monitoring_times = collect(Δt:Δt:T)

@time price_european_option = DoubleDeathApproximation.price_european_option_exponentiation(S0, V0, GT_params, T, M, N, DoubleDeathApproximation.linear_mapping, DoubleDeathApproximation.linear_mapping, Strike, "call", risk_free_rate=0.0)
price_american_option_ctmc = DoubleDeathApproximation.price_american_option_ctmc(S0, V0, T, M, N, DoubleDeathApproximation.linear_mapping, DoubleDeathApproximation.linear_mapping, Strike, "call", monitoring_times, GT_params)

b11 = DoubleDeathApproximation.b11
b22 = DoubleDeathApproximation.b22
sigma11 = DoubleDeathApproximation.sigma11
sigma12 = DoubleDeathApproximation.sigma12
sigma22 = DoubleDeathApproximation.sigma22


Q_matrix = DoubleDeathApproximation.construct_Q(200, 500.0, 0.0, 1.0, 0.0; 
    b1=b11, b2=b22, sigma11=sigma11, sigma12=sigma12, sigma22=sigma22, reduced=false) # so the progress meter works


function plot_option_price_curve(S_levels, V_levels, option_prices)
    # Create a meshgrid for S and V
    S_grid, V_grid = meshgrid(S_levels, V_levels)

    # Plot the surface
    surface(S_grid, V_grid, option_prices, xlabel="Asset Price", ylabel="Variance", zlabel="Option Price",
            title="Option Price Surface", color=:viridis)
end