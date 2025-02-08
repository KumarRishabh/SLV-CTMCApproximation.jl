using Random
using Distributions
using LinearAlgebra
using Plots
using StatsBase
using ExponentialUtilities  # For efficient matrix exponential computation
using FastExpm
using Expokit
using Revise
using SparseArrays
using ProgressMeter
using Parameters
"""
Constructs the generator matrix Q for the CTMC approximation using the general format.

# Arguments
- `V_levels::Vector{Float64}`: Variance levels (grid points).
- `kappa::Float64`: Mean reversion rate.
- `theta::Float64`: Long-term variance.
- `sigma::Float64`: Volatility of variance.

# Returns
- `Q::Matrix{Float64}`: Generator matrix.
"""
# Set parameter values for the Heston model using Paramters.jl
@with_kw mutable struct HestonParams
    S0::Float64 = 100.0
    μ_h::Float64 = 0.02
    ν::Float64 = 0.085
    θ_h::Float64 = 0.04
    κ_h::Float64 = 6.21
    ρ::Float64 = -0.7
    v0::Float64 = 0.501
end

# Set parameter values for the SABR model using Paramters.jl
@with_kw mutable struct SABRParams
    S0::Float64 = 100.0
    β::Float64 = 0.5
    κ_s::Float64 = 6.00
    ρ::Float64 = -0.75
    v0::Float64 = 0.11
end

# Set parameter values for the 3/2 model using Paramters.jl
@with_kw mutable struct ThreeTwoParams
    S0::Float64 = 100.0
    μ_32::Float64 = 0.02
    ν_32::Float64 = 0.424
    θ_32::Float64 = 0.04
    κ_32::Float64 = 6.00
    ρ::Float64 = -0.7
    v0::Float64 = 0.501
end

function arcsinh_mapping(ξ, V_0, V_min, V_max, M, γ) # this is what Lo and Skindilias used
    c1 = asinh((V_min - V_0)/γ)
    c2 = asinh((V_max - V_0)/γ)
    return V_0 .+ γ * sinh.(c1 .+ ξ * (c2 - c1))
end

function linear_mapping(ξ, V_0, V_min, V_max, M, γ) # this is what we use
    return V_min .+ ξ * (V_max - V_min)
end

function construct_variance_levels(V_min, V_max, M, mapping_function, v_0; γ = 5)
    ξ = range(0.0, 1.0, length=M)
    V_levels = mapping_function(ξ, v_0, V_min, V_max, M, γ)
    return V_levels
end

function construct_asset_price_levels(S_min, S_max, N, mapping_function, S_0; γ = 5)
    ξ = range(0.0, 1.0, length=N)
    S_levels = mapping_function(ξ, S_0, S_min, S_max, N, γ)
    return S_levels
end


function simulate_variance_process(Q, V_levels, V0, T)
    M = length(V_levels)
    idx_v = findmin(abs.(V_levels .- V0))[2]
    V0 = V_levels[idx_v]  # Adjust V0 to the nearest grid point

    # Initialize time and state variables
    t = 0.0
    times = [0.0]
    V_path = [V0]
    idx_v_path = [idx_v]

    while t < T
        rate = -1/Q[idx_v, idx_v]
        if rate <= 0
            τ = T - t  # Stay in current state until T
            t = T
        else
            τ = rand(Exponential(rate))
            print("τ: ", τ)
            t = t + τ
            if t > T
                t = T  # Adjust if time exceeds T
                times = [times; t]
                V_path = [V_path; V_levels[idx_v]]
                idx_v_path = [idx_v_path; idx_v]
                break
            end
        end

        times = [times; t]
        V_path = [V_path; V_levels[idx_v]]
        idx_v_path = [idx_v_path; idx_v]

        # Transition to next state
        probs = Q[idx_v, :]
        probs[idx_v] = 0.0  # Exclude current state
        total_rate = sum(probs)
        if total_rate > 0
            probs = probs / total_rate
            idx_v = sample(1:M, Weights(probs))
        else
            # If no transitions are possible, stay in current state
            pass
        end
    end

    return times, V_path, idx_v_path
end

"""
The double operations approximation constructs the generator matrix Q for the CTMC approximation
for the joint asset price and variance process. It contains both the double births and double deaths
along with the transfer rates between asset price and variance levels. 

Consider the following class of models: 
$$
    d \begin{pmatrix} S_t \\ V_t \end{pmatrix} = \begin{pmatrix} b_1(S_t, V_t) \\ b_2(V_t) \end{pmatrix} dt + \begin{pmatrix} \sigma_{11}(S_t, V_t) & \sigma_{12}(S_t, V_t) \\ 0 & \sigma_{22}(V_t) \end{pmatrix} d \begin{pmatrix} B_t \\ \beta_t \end{pmatrix}
$$

Then the generator matrix Q is constructed as follows: 
$$
a_{1,0}^N &= N b_1^+\left(\frac{\ell}{N}\right) + \frac{N^2}{2} \left(\sigma_{11}^2\left(\frac{\ell}{N}\right) + \sigma_{12}^2\left(\frac{\ell}{N}\right) - |\sigma_{12}\left(\frac{\ell}{N}\right)| \sigma_{22}\left(\frac{\ell}{N}\right)\right), \\
a_{0,1}^N &= N b_2^+\left(\frac{\ell}{N}\right) + \frac{N^2}{2} \left(\sigma_{22}^2\left(\frac{\ell}{N}\right) - |\sigma_{12}\left(\frac{\ell}{N}\right)| \sigma_{22}\left(\frac{\ell}{N}\right)\right), \\
s_{1,0}^N &= N b_1^-\left(\frac{\ell}{N}\right) + \frac{N^2}{2} \left(\sigma_{11}^2\left(\frac{\ell}{N}\right) + \sigma_{12}^2\left(\frac{\ell}{N}\right) - |\sigma_{12}\left(\frac{\ell}{N}\right)| \sigma_{22}\left(\frac{\ell}{N}\right)\right), \\
s_{0,1}^N &= N b_2^-\left(\frac{\ell}{N}\right) + \frac{N^2}{2} \left(\sigma_{22}^2\left(\frac{\ell}{N}\right) - |\sigma_{12}\left(\frac{\ell}{N}\right)| \sigma_{22}\left(\frac{\ell}{N}\right)\right).
$$

Where a_{1, 0} is the transition from S_t \to S_t + \ell and a_{0, 1} is the transition from V_t \to V_t + \ell.
and s_{1, 0} is the transition from S_t \to S_t - \ell and s_{0, 1} is the transition from V_t \to V_t - \ell.


Simultaneous transitions between price and volatility are given by:
$$
    a_{1,1}^N &= s_{1,1}^N = \frac{N^2}{2} \sigma_{12}^+\left(\frac{l}{N}\right) \sigma_{22}\left(\frac{l}{N}\right), \\
    t_{1 \to 2}^N &= t_{2 \to 1}^N = \frac{N^2}{2} \sigma_{12}^-\left(\frac{l}{N}\right) \sigma_{22}\left(\frac{l}{N}\right).
$$
where \( a_{1, 1}^N \) is the transition for (S_t, V_t) \to (S_t + \ell, V_t + \ell), \( s_{1, 1}^N \) is the transition for (S_t, V_t) \to (S_t - \ell, V_t - \ell), \( t_{1 \to 2}^N \) is the transition for (S_t, V_t) \to (S_t + \ell, V_t - \ell), and \( t_{2 \to 1}^N \) is the transition for (S_t, V_t) \to (S_t - \ell, V_t + \ell).
"""

heston_b1(S, V; params = heston_params) = params.μ_h * S
heston_b2(V; params = heston_params) = params.ν_h * (params.θ_h - V)
heston_σ_11(S, V; params = heston_params) = sqrt(1 - params.ρ^2) * sqrt(V) * S
heston_σ_12(S, V; params = heston_params) = params.ρ * sqrt(V) * S
heston_σ_22(V; params = heston_params) = params.κ_h * sqrt(V)

b1_SABR(S, V; params = SABR_params) = 0
b2_SABR(V; params = SABR_params) = 0
σ11_SABR(S, V; params = SABR_params) = sqrt(1 - params.ρ^2) * sqrt(V) * S^params.β
σ12_SABR(S, V; params = SABR_params) = params.ρ * V * S^params.β
σ22_SABR(V; params = SABR_params) = params.κ_s * V

# since 3/2 model is similar to heston model, we can use the same functions
b1_32(S, V; params = threetwo_params) = params.μ_32 * S
b2_32(V; params = threetwo_params) = params.ν_32 * (params.θ_32 - V^2)
σ11_32(S, V; params = threetwo_params) = sqrt(1 - params.ρ^2) * sqrt(V) * S
σ12_32(S, V; params = threetwo_params) = params.ρ * sqrt(V) * S
σ22_32(V; params = threetwo_params) = params.κ_32 * V^(3/2)

pos(x) = max(x, 0)
neg(x) = max(-x, 0)

function transition_rates(S, V, N)
    a10 = N * pos(b1(S, V)) + (N^2 / 2) * (sigma11(S, V)^2 + sigma12(S, V)^2 - abs(sigma12(S, V)) * sigma22(V))
    s10 = N * neg(b1(S, V)) + (N^2 / 2) * (sigma11(S, V)^2 + sigma12(S, V)^2 - abs(sigma12(S, V)) * sigma22(V))
    a01 = N * pos(b2(V)) + (N^2 / 2) * (sigma22(V)^2 - abs(sigma12(S, V)) * sigma22(V))
    s01 = N * neg(b2(V)) + (N^2 / 2) * (sigma22(V)^2 - abs(sigma12(S, V)) * sigma22(V))
    a11 = s11 = (N^2 / 2) * pos(sigma12(S, V)) * sigma22(V)
    t12 = t21 = (N^2 / 2) * neg(sigma12(S, V)) * sigma22(V)
    return (a10, s10, a01, s01, a11, s11, t12, t21)
end

# Construct Q matrix functionally
function construct_Q(N, ℓ)
    Q = spzeros(N*N, N*N)
    for i in 1:N, j in 1:N
        idx = (i - 1) * N + j
        S, V = i * ℓ, j * ℓ
        a10, s10, a01, s01, a11, s11, t12, t21 = transition_rates(S, V, N)
        
        if i < N Q[idx, idx + N] = a10 end
        if i > 1 Q[idx, idx - N] = s10 end
        if j < N Q[idx, idx + 1] = a01 end
        if j > 1 Q[idx, idx - 1] = s01 end
        if i < N && j < N Q[idx, idx + N + 1] = a11 end
        if i > 1 && j > 1 Q[idx, idx - N - 1] = s11 end
        if i < N && j > 1 Q[idx, idx + N - 1] = t12 end
        if i > 1 && j < N Q[idx, idx - N + 1] = t21 end
        Q[idx, idx] = -sum(Q[idx, :])
    end
    return Q
end


"""
Main function to simulate Heston model paths using CTMC approximation with general generator matrix.

# Arguments
- `S0::Float64`: Initial stock price.
- `V0::Float64`: Initial variance.
- `params::Dict`: Dictionary of model parameters.
- `T::Float64`: Time horizon.
- `M::Int`: Number of variance levels (states).
- `mapping_function::Function`: Function that defines the mapping from [0,1] to [0,1].

# Returns
- `times_asset::Vector{Float64}`: Times of the asset price process.
- `S::Vector{Float64}`: Asset prices over time.
- `V_path::Vector{Float64}`: Variance levels over time.
- `times_variance::Vector{Float64}`: Times of the variance process.
"""

function compute_transition_matrix(Q, T)
    # P = exp(Q * T)
    P = fastExpm(Q * T)  # Matrix exponential
    return P
end

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

function simulate_heston_ctmc_general(S0, V0, params::Dict, T, M, mapping_function)
    # Unpack parameters
    mu = params["mu"]
    kappa = params["kappa"]
    theta = params["theta"]
    sigma = params["sigma"]
    rho = params["rho"]

    # Define V_min and V_max
    V_min = max(0.0001, theta - 5 * sigma * sqrt(theta) / sqrt(2 * kappa))
    V_max = theta + 5 * sigma * sqrt(theta) / sqrt(2 * kappa)

    # Construct variance levels using the provided mapping function
    V_levels = construct_variance_levels(V_min, V_max, M, mapping_function, V0)

    # Construct generator matrix with general format
    Q, V_levels_adjusted = construct_generator_matrix_general(V_levels, kappa, theta, sigma)

    # Simulate variance process
    times_variance, V_path, idx_v_path = simulate_variance_process(Q, V_levels_adjusted, V0, T)

    # Simulate asset price process
    println("Time steps for variance process: ", times_variance)
    times_asset, S = simulate_asset_price(S0, times_variance, V_path, mu, rho)

    return times_asset, S, V_path, times_variance
end

function simulate_asset_price(S0, times, V_path, mu, rho)
    # Precompute Cholesky decomposition for correlation
    L = [1.0 0.0; rho sqrt(1 - rho^2)]

    S = [S0]
    times_asset = [times[1]]

    for i in 1:length(times)-1
        t_current = times[i]
        t_next = times[i+1]
        Δt = t_next - t_current
        v_current = V_path[i]
        sqrt_v_current = sqrt(v_current)

        # Simulate asset price increment
        Z = randn(2)
        Z = L * Z
        dW_S = sqrt(Δt) * Z[1]
        drift = (mu - 0.5 * v_current) * Δt
        diffusion = sqrt_v_current * dW_S
        S_new = S[end] * exp(drift + diffusion)

        # Append results
        S = [S; S_new]
        times_asset = [times_asset; t_next]
    end

    return times_asset, S
end

function price_european_option_exponentiation(S0, V0, params::Dict, T, M, N, mapping_function_S, mapping_function_V, K, option_type; risk_free_rate=0.05)
    # Unpack parameters
    r = risk_free_rate
    # Construct variance levels and asset price levels
    V_min = max(0.0, params["theta"] - 3 * params["sigma"] * sqrt(params["theta"]) / sqrt(2 * params["kappa"]))
    V_max = params["theta"] + 3 * params["sigma"] * sqrt(params["theta"]) / sqrt(2 * params["kappa"])
    V_levels = construct_variance_levels(V_min, V_max, M, mapping_function_V, V0)

    S_min = S0 * 0.5  # Set based on expected range of asset prices
    S_max = S0 * 1.5
    S_levels = construct_asset_price_levels(S_min, S_max, N, mapping_function_S, S0)

    # Construct the combined generator matrix
    Q, V_levels, S_levels = construct_combined_generator_matrix(V_levels, S_levels, params)

    # Compute the transition probability matrix
    @time P = compute_transition_matrix(Q, T)

    # Construct the payoff vector
    G = construct_payoff_vector(V_levels, S_levels, K, option_type)
    println("Payoff vector: ", G)

    # Define initial state
    # Find the indices closest to initial V0 and S0
    idx_V0 = findmin(abs.(V_levels .- V0))[2]
    idx_S0 = findmin(abs.(S_levels .- S0))[2]
    initial_state = (idx_V0 - 1) * N + idx_S0

    # Initial distribution vector
    π0 = zeros(length(G))
    π0[initial_state] = 1.0

    # Compute the option price

    option_price = exp(-r * T) * (π0'*P*G)[1]

    return option_price
end

function European_call_price_krylov(S0, V0, params::Dict, T, M, N, mapping_function_S, mapping_function_V, K, option_type; risk_free_rate=0.0)
    # Unpack parameters
    r = risk_free_rate
    # Construct variance levels and asset price levels
    V_min = max(0.0, params["theta"] - 3 * params["sigma"] * sqrt(params["theta"]) / sqrt(2 * params["kappa"]))
    V_max = params["theta"] + 3 * params["sigma"] * sqrt(params["theta"]) / sqrt(2 * params["kappa"])
    V_levels = construct_variance_levels(V_min, V_max, M, mapping_function_V, V0)

    S_min = S0 * 0.5  # Set based on expected range of asset prices
    S_max = S0 * 1.5
    S_levels = construct_asset_price_levels(S_min, S_max, N, mapping_function_S, S0)

    # Construct the combined generator matrix
    Q, V_levels, S_levels = construct_combined_generator_matrix(V_levels, S_levels, params)

    # Compute the transition probability matrix
    @time P = compute_transition_matrix(Q, T)

    # Construct the payoff vector
    G = construct_payoff_vector(V_levels, S_levels, K, option_type)
    println("Payoff vector: ", G)

    # Define initial state
    # Find the indices closest to initial V0 and S0
    idx_V0 = findmin(abs.(V_levels .- V0))[2]
    idx_S0 = findmin(abs.(S_levels .- S0))[2]
    initial_state = (idx_V0 - 1) * N + idx_S0

    # Initial distribution vector
    π0 = zeros(length(G))
    π0[initial_state] = 1.0

    # Compute the option price
    # option_price = exp(-r * T) * (π0'*P*G)[1]
    # Compute w_tilde = exp(Q^T t) * v
    Q_transpose = transpose(Q)

    w_tilde = expmv(T, Q_transpose, π0; tol = 1e-6, m = 20)
    option_price = exp(-r * T) * dot(w_tilde, G)
    println("w_tilde: ")

    return option_price
end
# The rest of the functions remain the same:
# - construct_variance_levels
# - simulate_variance_process
# - simulate_asset_price

# Example mapping functions (as before)
# ...



# fast matrix exponentiation for tridiagonal such that row sum is zero
function compute_transition_matrices(Q, Δt_array)
    P_matrices = []
    for Δt in Δt_array
        P = fastExpm(Q * Δt)
        push!(P_matrices, P)
    end
    return P_matrices
end

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


function backward_induction(P_matrices, V_option, V_levels, S_levels, K, option_type, r, Δt_array)
    M = length(V_levels)
    N = length(S_levels)
    total_states = M * N
    N_steps = length(P_matrices)

    for n in N_steps:-1:1
        P = P_matrices[n]
        # Expected continuation value
        continuation_value = real.(P * V_option)
        # Immediate exercise value
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
        # Discount continuation value
        continuation_value *= exp(-r * Δt_array[n])
        # Update option value
        V_option = max.(exercise_value, continuation_value)
    end
    return V_option
end

function price_american_option_ctmc(S0, V0, params::Dict, T, M, N, mapping_function_S, mapping_function_V, K, option_type, monitoring_times)
    # Unpack parameters
    r = params["r"]

    # Construct variance levels and asset price levels
    V_min = max(0.0, params["theta"] - 3 * params["sigma"] * sqrt(params["theta"]) / sqrt(2 * params["kappa"]))
    V_max = params["theta"] + 3 * params["sigma"] * sqrt(params["theta"]) / sqrt(2 * params["kappa"])
    V_levels = construct_variance_levels(V_min, V_max, M, mapping_function_V, V0)

    S_min = S0 * 0.02  # Adjust as needed
    S_max = S0 * 2.5
    S_levels = construct_asset_price_levels(S_min, S_max, N, mapping_function_S, S0)

    println("Constructed asset and variance levels:", S_levels, V_levels)
    # Construct the combined generator matrix
    Q, V_levels, S_levels = construct_combined_generator_matrix(V_levels, S_levels, params)
    println("Constructed combined generator matrix")

    # Compute time intervals
    Δt_array = diff(monitoring_times)

    # Compute transition matrices
    P_matrices = compute_transition_matrices(Q, Δt_array)
    println("Computed transition matrices")
    # Initialize option values at maturity
    V_option = initialize_option_values(V_levels, S_levels, K, option_type)
    println("Initialized option values at maturity")
    # Perform backward induction
    V_option = backward_induction(P_matrices, V_option, V_levels, S_levels, K, option_type, r, Δt_array)

    # Find the initial state index
    idx_V0 = findmin(abs.(V_levels .- V0))[2]
    idx_S0 = findmin(abs.(S_levels .- S0))[2]
    initial_state = (idx_V0 - 1) * N + idx_S0

    # Option price is the option value at the initial state
    option_price = V_option[initial_state]

    return option_price
end

# Example usage
# S0 = 10.0         # Initial stock price
# V0 = 0.04    
# # S0 = 10, v0 = 0.04, T = 1, K = 4, ρ = −0.75, σv = 0.15, η = 4, θ = 0.035, r = 0      # Initial variance
# params = Dict(
#     "r" => 0.0,        # Risk-free rate
#     "mu" => 0.05,        # Expected return
#     "kappa" => 4,      # Mean reversion rate
#     "theta" => 0.035,     # Long-term variance
#     "sigma" => 0.15,      # Volatility of variance
#     "rho" => -0.75        # Correlation between asset and variance
# )
# T = 1.0          # Time horizon (in years)
# M = 20 # Number of variance levels (states)
# N = 20        # Number of asset price levels (states)
# # Choose the mapping function
# mapping_function_S = arcsinh_mapping
# mapping_function_V = arcsinh_mapping # or any other mapping function


# # Simulate the Heston model65
# Strike = 4.0
# times_asset, S, V_path, times_variance = simulate_heston_ctmc_general(S0, V0, params, T, M, mapping_function_V)

# plot(times_asset, S, label="Asset Price", xlabel="Time", ylabel="Price", legend=:topleft)
# # plot!(times_variance, V_path, label="Variance", xlabel="Time", ylabel="Variance", legend=:topleft)
# plot(times_variance, V_path, label="Variance", xlabel="Time", ylabel="Variance", legend=:topleft)

# # European_call_price = price_european_option_exponentiation(S0, V0, params, T, M, N, mapping_function_S, mapping_function_V, Strike, "call", risk_free_rate=0.0)
# European_call_price = European_call_price_krylov(S0, V0, params, T, M, N, mapping_function_S, mapping_function_V, Strike, "call"; risk_free_rate=0.0)

# monitoring_times = collect(0.0:0.05:T)

# # Price the American option
# option_price = price_american_option_ctmc(S0, V0, params, T, M, N, linear_mapping, linear_mapping, Strike, "call", monitoring_times)

# println("American Call option price using CTMC: $option_price")

