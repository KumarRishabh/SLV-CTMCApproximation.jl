# Check if the combined generator (1000 randomly generated) are diagonalisable


using LinearAlgebra
using Test
using PrettyTables
using Random

# Function to create a random generator matrix for a single regime's internal dynamics
function random_tridiagonal_generator(N::Int)
    G = zeros(Float64, N, N)

    # Populate the sub-diagonal and super-diagonal with random values
    for i in 2:N
        G[i, i-1] = rand()  # Sub-diagonal element
        G[i-1, i] = rand()  # Super-diagonal element
    end

    # Adjust diagonal elements to ensure each row sums to zero
    for i in 1:N
        G[i, i] = -sum(G[i, j] for j in 1:N if j != i)
    end

    return G
end

# Function to create a regime-switching generator matrix with tridiagonal G_i blocks
function regime_switching_generator(R::Int, N::Int)
    G = zeros(Float64, R * N, R * N)  # Full block matrix of size (R*N) x (R*N)

    # Random tridiagonal within-regime generator matrices for each regime
    for r in 1:R
        start_idx = (r - 1) * N + 1
        end_idx = r * N
        G[start_idx:end_idx, start_idx:end_idx] = random_tridiagonal_generator(N)
    end

    # Set the off-diagonal blocks for regime switching
    for r1 in 1:R
        for r2 in 1:R
            if r1 != r2
                λ = rand()  # Random switching rate between regimes
                start_idx1 = (r1 - 1) * N + 1
                end_idx1 = r1 * N
                start_idx2 = (r2 - 1) * N + 1
                end_idx2 = r2 * N

                G[start_idx1:end_idx1, start_idx2:end_idx2] = λ * I(N)
            end
        end
    end

    # Adjust diagonal elements globally to ensure rows sum to zero
    for r in 1:R
        start_idx = (r - 1) * N + 1
        end_idx = r * N
        for i in start_idx:end_idx
            G[i, i] = -sum(G[i, j] for j in 1:R*N if j != i)
        end
    end

    return G
end

# Example usage
R = 3  # Number of regimes
N = 4  # Number of states within each regime
G = regime_switching_generator(R, N)
data = eigen(G)
print(data.values)
# Generate 1000 random regime-switching generator matrices with within-regime dynamics
num_generators = 1000
num_regimes = 3     # Number of regimes
num_states = 4      # Number of states within each regime

generators = [random_regime_switching_generator(num_regimes, num_states) for _ in 1:num_generators]

function is_diagonalizable(Q::Matrix{Float64})
    eigen_decomposition = eigen(Q)
    # Check if the number of linearly independent eigenvectors is equal to the dimension of Q
    return rank(hcat(eigen_decomposition.vectors...)) == size(Q, 1)
end

is_diagonalizable(G)
# Check diagonalizability of each generator matrix and store the results
num_diagonalizable = 0
diagonalizability_results = []

for Q in generators
    is_diag = is_diagonalizable(Q)
    push!(diagonalizability_results, is_diag)
    if is_diag
        num_diagonalizable += 1
    end
    println("Is diagonalizable: ", is_diag)
end

num_diagonalizable
# Display an example generator matrix
println("Example regime-switching generator matrix with within-regime dynamics:")
println(generators[1])

pretty_table(generators[1])