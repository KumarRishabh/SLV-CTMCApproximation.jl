using SparseArrays, Distributions, Random, StatsBase, LinearAlgebra
# Define drift and volatility functions
b1(S, V) = 0.05 * S  # Example drift for S
b2(V) = 0.02 * V      # Example drift for V
sigma11(S, V) = 0.2 * S  # Example volatility function
sigma12(S, V) = 0.1 * S  # Example volatility function
sigma22(V) = 0.3 * V     # Example volatility function

# Grid size and discretization parameter
N = 100  # Number of grid points per dimension
ℓ = 1.0 / N  # Step size

# Helper functions for positive/negative parts
pos(x) = max(x, 0)
neg(x) = max(-x, 0)

# Compute transition rates
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

# Generate Q matrix
Q = construct_Q(N, ℓ)
# Define Heston model parameters
mu = 0.05       # Drift of asset price
kappa = 2.0     # Mean reversion speed
theta = 0.04    # Long-term variance
sigma = 0.3     # Volatility of variance process
rho = -0.7      # Correlation between asset and variance processes

# Define drift and volatility functions
b1(S, V) = mu * S
b2(V) = kappa * (theta - V)

sigma11(S, V) = sqrt(V) * S
sigma12(S, V) = rho * sigma * sqrt(V) * S
sigma22(V) = sigma * sqrt(V)

# Grid size and discretization parameter
N = 100  # Number of grid points per dimension
ℓ = 1.0 / N  # Step size

define_grid(N, ℓ) = [(i * ℓ, j * ℓ) for i in 1:N, j in 1:N]

# Function to simulate paths
function simulate_paths_ctmc(Q, grid, S0, V0, T, num_paths)
    num_states = length(grid)
    paths = Vector{Vector{Tuple{Float64, Float64, Float64}}}(undef, num_paths)
    
    # Initialize each path
    for p in 1:num_paths
        paths[p] = [(0.0, S0, V0)]  # Initial state
    end

    # Get diagonal exit rates
    exit_rates = -diag(Q)  # λ_i = -Q_{ii}
    # exit_rates = -diagm(Q)

    # Run simulation for each path in a vectorized manner
    for p in 1:num_paths
        t = 0.0
        i = clamp(Int(round(S0 / ℓ)), 1, N)
        j = clamp(Int(round(V0 / ℓ)), 1, N)
        idx = (i - 1) * N + j

        while t < T
            λ = exit_rates[idx]
            if λ <= 0
                break  # No transitions possible
            end
            
            # Sample holding time
            t += rand(Exponential(1 / λ))

            if t >= T
                break  # Stop if exceeded total time
            end

            # Compute transition probabilities
            transition_probs = Q[idx, :]
            transition_probs[idx] = 0  # Remove self-transition
            transition_probs /= sum(transition_probs)  # Normalize

            # Sample next state
            next_state = sample(1:num_states, Weights(transition_probs))
            S_new, V_new = grid[next_state]

            push!(paths[p], (t, S_new, V_new))
            idx = next_state  # Update state index
        end
    end
    return paths
end

# Construct Q matrix
Q = construct_Q(N, ℓ)
grid = define_grid(N, ℓ)
# Simulate paths
Random.seed!(42) # For reproducibility
S0, V0 = 1.0, 0.04  # Initial conditions
T, dt = 1.0, 0.01  # Time horizon and step size
num_paths = 10
paths = simulate_paths_ctmc(Q, grid, S0, V0, T, num_paths)

all_times = Vector{Vector{Float64}}(undef, length(paths))
all_S = Vector{Vector{Float64}}(undef, length(paths))
all_V = Vector{Vector{Float64}}(undef, length(paths))

# Iterate over each path to extract t, S, and V
for i in 1:length(paths)
    current_path = paths[i]
    all_times[i] = [t for (t, _, _) in current_path]
    all_S[i] = [S for (_, S, _) in current_path]
    all_V[i] = [V for (_, _, V) in current_path]
end

# Plot the first few simulated paths
using Plots
pyplot()
# plot(all_times[1], all_S[1], label="S", xlabel="Time", ylabel="Value", title="Simulated Paths", lw=2)
for i in 1:num_paths
    plot!(all_times[i], all_S[i], label="S Path $i", lw=2)
end
plot!(legend=:topleft)