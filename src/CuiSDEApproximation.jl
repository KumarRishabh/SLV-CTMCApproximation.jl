using Random
using Distributions
using LinearAlgebra
using Plots
using StatsBase
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

function EuropeanOptionPricing()
# The rest of the functions remain the same:
# - construct_variance_levels
# - simulate_variance_process
# - simulate_asset_price

# Example mapping functions (as before)
# ...

# Example usage
S0 = 100.0         # Initial stock price
V0 = 0.04          # Initial variance
params = Dict(
    "mu" => 0.05,        # Expected return
    "kappa" => 1.5,      # Mean reversion rate
    "theta" => 0.04,     # Long-term variance
    "sigma" => 0.3,      # Volatility of variance
    "rho" => -0.7        # Correlation between asset and variance
)
T = 10.0           # Time horizon (in years)
M = 100            # Number of variance levels (states)

# Choose the mapping function
mapping_function = linear_mapping  # or any other mapping function

# Simulate the Heston model65

times_asset, S, V_path, times_variance = simulate_heston_ctmc_general(S0, V0, params, T, M, mapping_function)

plot(times_asset, S, label="Asset Price", xlabel="Time", ylabel="Price", legend=:topleft)
# plot!(times_variance, V_path, label="Variance", xlabel="Time", ylabel="Variance", legend=:topleft)
plot(times_variance, V_path, label="Variance", xlabel="Time", ylabel="Variance", legend=:topleft)