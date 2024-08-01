using Test

# Test for calculateSufficientStats function
function test_calculateSufficientStats()
    v0, ϱ, ν, κ, T = 2.0, 1.0, 3.0, 0.5, 1.0 
    mean, std_dev = calculateSufficientStats(ν, ϱ, κ, v0, T)
    @test mean ≈ ν * (1 - exp(-ϱ * T))
    @test std_dev ≈ sqrt(ν * (1 - exp(-2 * ϱ * T)) / (2 * ϱ))
end

# Run the tests
@testset "Sanity Check" begin
    test_calculateSufficientStats()
end

# Dummy data for testing
V = rand(10, 5)  # Replace with appropriate dimensions
t = 2
M = 3

# Dummy version of the line
IntV = (reshape(sum(V[((t-1)*M+1):(t*M+1), :], dims=1), 5) + reshape(sum(V[((t-1)*M+2):(t*M), :], dims=1), 5) + 2 * (V[((t-1)*M+2), :] + V[t*M, :])) ./ (3 * M)
IntV = reshape(sum(V[((t-1)*M+1):(t*M+1), :], dims=1), 5)
reshape(sum(V[((t-1)*M+2):(t*M), :], dims=1), 5)
2 * (V[((t-1)*M+2), :] + V[t*M, :] + V[t*M, :])
(3 * M)
println("IntV: $IntV")