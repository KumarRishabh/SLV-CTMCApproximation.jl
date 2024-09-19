using Random
using Distributions
using LinearAlgebra
using Plots
using StatsBase
using ExponentialUtilities  # For efficient matrix exponential computation

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
function construct_variance_levels(V_min, V_max, M, mapping_function)
    ξ = range(0.0, 1.0, length=M)
    V_levels = V_min .+ (V_max - V_min) .* mapping_function.(ξ)
    return V_levels
end

function construct_asset_price_levels(S_min, S_max, N, mapping_function)
    ξ = range(0.0, 1.0, length=N)
    S_levels = S_min .+ (S_max - S_min) .* mapping_function.(ξ)
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


function construct_generator_matrix_general(V_levels, kappa, theta, sigma)
    M = length(V_levels)
    Q = zeros(M, M)

    # Calculate m(s_i) and s^2(s_i) at each grid point
    m_s = kappa .* (theta .- V_levels)
    s2_s = sigma^2 .* V_levels

    # Calculate m^+(s_i) and m^-(s_i)
    m_plus = max.(0.0, m_s)
    m_minus = max.(0.0, -m_s)

    # Calculate k_i (differences between grid points)
    k = diff(V_levels)  # Length M - 1

    # Enforce the constraints on k_i
    # for i in 1:M-1
    #     k_i = k[i]
    #     # Constraint: k_i ≤ s2_s[i] / |m_s[i]|
    #     if abs(m_s[i]) > 1e-8  # Avoid division by zero
    #         max_k_i = s2_s[i] / abs(m_s[i])
    #         if k_i > max_k_i
    #             k[i] = max_k_i
    #         end
    #     end
    # end

    # Recalculate V_levels based on adjusted k_i if necessary
    V_levels_adjusted = [V_levels[1]]
    for i in 1:M-1
        V_levels_adjusted = [V_levels_adjusted; V_levels_adjusted[end] + k[i]]
    end

    # Update m_s, s2_s, m_plus, m_minus with adjusted V_levels
    V_levels = V_levels_adjusted
    m_s = kappa .* (theta .- V_levels)
    s2_s = sigma^2 .* V_levels
    m_plus = max.(0.0, m_s)
    m_minus = max.(0.0, -m_s)

    # Construct Q matrix
    for i in 1:M
        # For interior points
        if i > 1 && i < M
            k_i_minus1 = k[i - 1]
            k_i = k[i]
            denominator = k_i_minus1 + k_i

            q_im1 = (m_minus[i] / k_i_minus1) + (s2_s[i] - (k_i_minus1 * m_minus[i] + k_i * m_plus[i])) / (k_i_minus1 * denominator)
            q_ip1 = (m_plus[i] / k_i) + (s2_s[i] - (k_i_minus1 * m_minus[i] + k_i * m_plus[i])) / (k_i * denominator)

            Q[i, i - 1] = max(q_im1, 0.0)
            Q[i, i + 1] = max(q_ip1, 0.0)
        elseif i == 1 && M > 1
            # First point (no i - 1)
            k_i = k[i]
            denominator = k_i
            q_ip1 = (m_plus[i] / k_i) + (s2_s[i] - (0.0 + k_i * m_plus[i])) / (k_i * denominator)
            Q[i, i + 1] = max(q_ip1, 0.0)
            println("Q[1, 2]: ", Q[1, 2])
        elseif i == M && M > 1
            # Last point (no i + 1)
            k_i_minus1 = k[i - 1]
            denominator = k_i_minus1
            q_im1 = (m_minus[i] / k_i_minus1) + (s2_s[i] - (k_i_minus1 * m_minus[i] + 0.0)) / (k_i_minus1 * denominator)
            Q[i, i - 1] = max(q_im1, 0.0)
            println("Q[M, M-1]: ", Q[M, M-1])
        end

        # Diagonal element
        Q[i, i] = -sum(Q[i, :])
    end

    return Q, V_levels
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

function construct_combined_generator_matrix(V_levels, S_levels, params)
    M = length(V_levels)
    N = length(S_levels)
    total_states = M * N
    Q = zeros(total_states, total_states)  # Sparse matrix for efficiency

    # Retrieve parameters
    kappa = params["kappa"]
    theta = params["theta"]
    sigma = params["sigma"]
    rho = params["rho"]
    r = params["r"]  # Risk-free rate

    # Construct generator for variance process (as before)
    Q_V, V_levels = construct_generator_matrix_general(V_levels, kappa, theta, sigma)

    # Construct generator for asset price process
    # For simplicity, we can use a finite difference approximation for the asset price generator
    Q_S = construct_asset_price_generator(S_levels, V_levels, params)

    # Combine the generators
    for vi in 1:M
        for si in 1:N
            state_idx = (vi - 1) * N + si

            # Variance transitions
            for vj in 1:M
                if Q_V[vi, vj] != 0
                    target_idx = (vj - 1) * N + si
                    Q[state_idx, target_idx] += Q_V[vi, vj]
                end
            end

            # Asset price transitions
            for sj in 1:N
                if Q_S[si, sj, vi] != 0
                    target_idx = (vi - 1) * N + sj
                    Q[state_idx, target_idx] += Q_S[si, sj, vi]
                end
            end
        end
    end

    return Q, V_levels, S_levels
end

function construct_asset_price_generator(S_levels, V_levels, params)
    N = length(S_levels)
    M = length(V_levels)
    Q_S = zeros(N, N, M)

    r = params["r"]
    rho = params["rho"]

    for vi in 1:M
        v_i = V_levels[vi]

        # Drift and diffusion coefficients for the asset price process
        m_s = r * S_levels  # Risk-neutral drift
        s2_s = (sqrt(v_i) * S_levels) .^ 2  # Diffusion term

        # Calculate k_i (differences between S_levels)
        k = diff(S_levels)

        # Transition rates for asset price process
        for si in 1:N
            s_i = S_levels[si]

            if si > 1 && si < N
                k_i_minus1 = k[si-1]
                k_i = k[si]
                denominator = k_i_minus1 + k_i

                q_im1 = ((s2_s[si]) / (k_i_minus1 * denominator)) - (m_s[si] / denominator)
                q_ip1 = ((s2_s[si]) / (k_i * denominator)) + (m_s[si] / denominator)

                Q_S[si, si-1, vi] = max(q_im1, 0.0)
                Q_S[si, si+1, vi] = max(q_ip1, 0.0)
            elseif si == 1 && N > 1
                k_i = k[si]
                denominator = k_i * k_i

                q_ip1 = ((s2_s[si]) / (k_i * denominator)) + (m_s[si] / denominator)
                Q_S[si, si+1, vi] = max(q_ip1, 0.0)
            elseif si == N && N > 1
                k_i_minus1 = k[si-1]
                denominator = k_i_minus1 * k_i_minus1

                q_im1 = ((s2_s[si]) / (k_i_minus1 * denominator)) - (m_s[si] / denominator)
                Q_S[si, si-1, vi] = max(q_im1, 0.0)
            end

            # Diagonal element
            Q_S[si, si, vi] = -sum(Q_S[si, :, vi])
        end
    end

    return Q_S
end

function compute_transition_matrix(Q, T)
    P = exp(Q * T)  # Matrix exponential
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
    V_levels = construct_variance_levels(V_min, V_max, M, mapping_function)

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
    V_levels = construct_variance_levels(V_min, V_max, M, mapping_function_V)

    S_min = S0 * 0.5  # Set based on expected range of asset prices
    S_max = S0 * 1.5
    S_levels = construct_asset_price_levels(S_min, S_max, N, mapping_function_S)

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
# The rest of the functions remain the same:
# - construct_variance_levels
# - simulate_variance_process
# - simulate_asset_price

# Example mapping functions (as before)
# ...

# Example usage
linear_mapping(x) = x
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
M = 75  # Number of variance levels (states)
N = 75        # Number of asset price levels (states)
# Choose the mapping function
mapping_function_S = linear_mapping
mapping_function_V = linear_mapping  # or any other mapping function

# Simulate the Heston model65
Strike = 4.0
times_asset, S, V_path, times_variance = simulate_heston_ctmc_general(S0, V0, params, T, M, mapping_function_V)

plot(times_asset, S, label="Asset Price", xlabel="Time", ylabel="Price", legend=:topleft)
# plot!(times_variance, V_path, label="Variance", xlabel="Time", ylabel="Variance", legend=:topleft)
plot(times_variance, V_path, label="Variance", xlabel="Time", ylabel="Variance", legend=:topleft)

European_call_price = price_european_option_exponentiation(S0, V0, params, T, M, N, mapping_function_S, mapping_function_V, Strike, "call", risk_free_rate=0.0)


# fast matrix exponentiation for tridiagonal such that row sum is zero

option_price = exp(-r * T) * (π0'*P*G)[1]