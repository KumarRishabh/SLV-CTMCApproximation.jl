using LinearAlgebra
using SparseArrays
using KrylovKit
using FastExpm
using Expokit
using BenchmarkTools
# Function to construct the generator matrix Q for a CTMC
function construct_generator_matrix(n)
    # Create a sparse tridiagonal matrix for simplicity
    # Rates are arbitrary for this example
    diagonals = [
        -2 * ones(n),     # Main diagonal
        ones(n - 1),      # Superdiagonal
        ones(n - 1)       # Subdiagonal
    ]
    Q = spdiagm(-1 => diagonals[3], 0 => diagonals[1], 1 => diagonals[2])

    # Adjust first and last rows for boundary conditions
    Q[1, 1] = -1.0
    Q[1, 2] = 1.0
    Q[end, end-1] = 1.0
    Q[end, end] = -1.0

    return Matrix(Q)
end

construct_generator_matrix(5)
# Function to construct the payoff vector G
function construct_payoff_vector(n, S_levels, K, option_type)
    G = zeros(n)
    for i in 1:n
        S = S_levels[i]
        if option_type == "call"
            G[i] = max(S - K, 0.0)
        elseif option_type == "put"
            G[i] = max(K - S, 0.0)
        else
            error("Invalid option type. Choose 'call' or 'put'.")
        end
    end
    return G
end

# Function for the normal exponentiation method
function price_option_normal_exponentiation(Q, G, r, t, initial_state_index)
    # Compute the transition matrix P(t) = exp(Q * t)
    P = fastExpm(Q * t)

    # Extract the row corresponding to the initial state
    pi0 = zeros(size(Q, 1))
    pi0[initial_state_index] = 1.0

    # Compute the option price
    option_price = exp(-r * t) * dot(pi0' * P, G)

    return option_price
end

# Function for the Krylov subspace method
function price_option_krylov(Q, G, r, t, initial_state_index)
    # Transpose of Q
    Q_transpose = transpose(Q)

    # Initial vector (sparse)
    v = zeros(size(Q, 1))
    v[initial_state_index] = 1.0

    # Compute w_tilde = exp(Q^T t) * v using Krylov subspace method
    w_tilde = expmv(t, Q_transpose, v; tol=1e-7, m=30)

    # Compute the option price

    option_price = exp(-r * t) * dot(w_tilde, G)

    return option_price
end

# Main function to perform the comparison
function compare_methods()
    # Parameters
    n = 10000                      # Size of the state space (adjustable)
    S0 = 100.0                    # Initial asset price
    V0 = 0.04                     # Initial variance (not used directly here)
    r = 0.05                      # Risk-free rate
    t = 1.0                       # Time to maturity
    K = 100.0                     # Strike price
    option_type = "call"          # Option type: "call" or "put"
    initial_state_index = n ÷ 2   # Starting from the middle state
    S_min = 50.0                  # Minimum asset price level
    S_max = 150.0                 # Maximum asset price level

    # Construct state space levels for S
    S_levels = range(S_min, S_max, length=n)
    S_levels = collect(S_levels)

    # Construct the generator matrix Q
    Q = construct_generator_matrix(n)

    # Construct the payoff vector G
    G = construct_payoff_vector(n, S_levels, K, option_type)

    # Time the normal exponentiation method
    println("Timing the normal exponentiation method...")
    @time option_price_normal = price_option_normal_exponentiation(Q, G, r, t, initial_state_index)

    # Time the Krylov subspace method
    println("\nTiming the Krylov subspace method...")
    @time option_price_krylov = price_option_krylov(Q, G, r, t, initial_state_index)

    # Alternatively, use BenchmarkTools for more accurate timing
    println("\nBenchmarking the normal exponentiation method...")
    normal_benchmark = @benchmark price_option_normal_exponentiation($Q, $G, $r, $t, $initial_state_index)

    println("\nBenchmarking the Krylov subspace method...")
    krylov_benchmark = @benchmark price_option_krylov($Q, $G, $r, $t, $initial_state_index)

    # Print the option prices
    println("\nOption price using normal exponentiation method: $option_price_normal")
    println("Option price using Krylov subspace method: $option_price_krylov")

    # Compare timings
    normal_time = median(normal_benchmark).time / 1e6  # Convert to milliseconds
    krylov_time = median(krylov_benchmark).time / 1e6  # Convert to milliseconds

    println("\nMedian execution time (normal exponentiation method): $normal_time ms")
    println("Median execution time (Krylov subspace method): $krylov_time ms")

    # Compute speedup
    speedup = normal_time / krylov_time
    println("\nSpeedup factor (normal / Krylov): $speedup")
end

# Run the comparison
compare_methods()