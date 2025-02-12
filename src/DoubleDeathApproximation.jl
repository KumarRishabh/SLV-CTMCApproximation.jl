module DoubleDeathApproximation

using Random
using Distributions
using LinearAlgebra
using SparseArrays
using StatsBase
using ExponentialUtilities  # For efficient matrix exponential computation
using FastExpm
using Expokit
using Parameters
using ProgressMeter
using Revise
using Test
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
    μ_h::Float64 = 0.02
    ν::Float64 = 0.085
    θ_h::Float64 = 0.04
    κ_h::Float64 = 6.21
    ρ::Float64 = -0.7
    v0::Float64 = 0.501
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
function construct_Q(N, ℓ; b1, b2, sigma11, sigma12, sigma22, reduced=false)
    Q = spzeros(N * N, N * N)
    for i in 1:N, j in 1:N
        idx = (i - 1) * N + j
        S = i * ℓ
        V = j * ℓ
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

"""
    construct_generator_matrix_general(V_levels, κ, θ, σ)

A dummy implementation to construct a generator matrix for the variance process.
Replace this with your actual method.
"""
function construct_generator_matrix_general(V_levels, κ, θ, σ)
    M = length(V_levels)
    Q = spzeros(M, M)
    # For illustration: a simple tridiagonal generator matrix
    for i in 2:(M - 1)
        Q[i, i - 1] = κ
        Q[i, i + 1] = κ
        Q[i, i] = -2 * κ
    end
    Q[1, 2] = κ
    Q[1, 1] = -κ
    Q[M, M - 1] = κ
    Q[M, M] = -κ
    return Q, V_levels
end

"""
    construct_combined_generator_matrix(V_levels, S_levels, params)

A dummy implementation that constructs the combined generator matrix for the joint
asset price and variance process. Replace with your proper implementation.
"""
function construct_combined_generator_matrix(V_levels, S_levels, params)
    M = length(V_levels)
    N = length(S_levels)
    total_states = M * N
    Q = spzeros(total_states, total_states)
    # Simple dummy construction: set off-diagonals to 1 and diagonal so that each row sums to zero.
    for i in 1:total_states
        Q[i, i] = -1.0
        if i < total_states
            Q[i, i + 1] = 1.0
        end
    end
    return Q, V_levels, S_levels
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
"""
function price_european_option_double_death(S_0, V_0, T, Q, variance_grid_size, asset_grid_size, strike_price, option_type, params::HestonParams; risk_free_rate=0.05)
    r = risk_free_rate
    V_min = max(0.0, params.θ_h - 3 * params.ν * sqrt(params.θ_h) / sqrt(2 * params.κ_h)) # Lower bound for variance
    V_max = params.θ_h + 3 * params.ν * sqrt(params.θ_h) / sqrt(2 * params.κ_h) # Upper bound for variance
    V_levels = construct_variance_levels(V_min, V_max, variance_grid_size, linear_mapping, V_0)

    S_min = S_0 * 0.1
    S_max = S_0 * 5.0
    S_levels = construct_asset_price_levels(S_min, S_max, asset_grid_size, linear_mapping, S_0)

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
    V_min = max(0.0, params["theta"] - 3 * params["sigma"] * sqrt(params["theta"]) / sqrt(2 * params["kappa"]))
    V_max = params["theta"] + 3 * params["sigma"] * sqrt(params["theta"]) / sqrt(2 * params["kappa"])
    V_levels = construct_variance_levels(V_min, V_max, M, mapping_function_V, V₀)

    S_min = S₀ * 0.5
    S_max = S₀ * 1.5
    S_levels = construct_asset_price_levels(S_min, S_max, N, mapping_function_S, S₀)

    Q, V_levels, S_levels = construct_combined_generator_matrix(V_levels, S_levels, params)
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
function European_call_price_krylov(S_0, V_0, T, Q, variance_grid_size, asset_grid_size, strike_price, option_type, params::HestonParams; risk_free_rate=0.0)
    r = risk_free_rate
    V_min = max(0.0, params.θ_h - 3 * params.ν * sqrt(params.θ_h) / sqrt(2 * params.κ_h)) # Lower bound for variance
    V_max = params.θ_h + 3 * params.ν * sqrt(params.θ_h) / sqrt(2 * params.κ_h) # Upper bound for variance
    V_levels = construct_variance_levels(V_min, V_max, variance_grid_size, linear_mapping, V_0)

    S_min = S_0 * 0.1
    S_max = S_0 * 5.0
    S_levels = construct_asset_price_levels(S_min, S_max, asset_grid_size, linear_mapping, S_0)

    # Use the double death approximation to construct the generator matrix
    # Q, V_levels, S_levels = construct_combined_generator_matrix(V_levels, S_levels, params)
    
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
    w_tilde = expmv(T, Q_transpose, π₀; tol=1e-6, m=20)
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
    for n in length(P_matrices):-1:1
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
        V_option = max.(exercise_value, continuation_value)
    end
    return V_option
end

"""
    price_american_option_ctmc(S₀, V₀, params, T, M, N, mapping_function_S, mapping_function_V, K, option_type, monitoring_times)

Prices an American option using CTMC backward induction.
"""
function price_american_option_ctmc(S₀, V₀, params::Dict, T, M, N, mapping_function_S, mapping_function_V, K, option_type, monitoring_times)
    r = params["r"]

    V_min = max(0.0, params["theta"] - 3 * params["sigma"] * sqrt(params["theta"]) / sqrt(2 * params["kappa"]))
    V_max = params["theta"] + 3 * params["sigma"] * sqrt(params["theta"]) / sqrt(2 * params["kappa"])
    V_levels = construct_variance_levels(V_min, V_max, M, mapping_function_V, V₀)

    S_min = S₀ * 0.02
    S_max = S₀ * 2.5
    S_levels = construct_asset_price_levels(S_min, S_max, N, mapping_function_S, S₀)

    Q, V_levels, S_levels = construct_combined_generator_matrix(V_levels, S_levels, params)
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

end  # module CTMCModels

using .DoubleDeathApproximation

# Dummy data for testing
