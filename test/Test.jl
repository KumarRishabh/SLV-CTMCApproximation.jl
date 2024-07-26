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