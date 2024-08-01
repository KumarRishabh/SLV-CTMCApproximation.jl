using Test, Parameters, Revise

@with_kw mutable struct HestonParams
    S0::Float64 = 100.0
    v0::Float64 = 0.04
    μ::Float64 = 0.0
    ρ::Float64 = -0.7
    κ::Float64 = 5.3
    ϱ::Float64 = 0.04
    ν::Float64 = 0.01
end


PS1 = HestonParams(S0=100, μ=0.02, ν=0.085, ϱ=6.21, κ=0.2, ρ=-0.7, v0=0.501)

# Heston model description
# S0: Initial stock price
# v0: Initial volatility
# rho: Correlation between the Wiener processes driving the stock price and the volatility
# kappa: Mean-reversion rate of the volatility
# theta: Long-term mean of the volatility


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


# Milstein discretization of the Heston model

function milstein_discretization(S0, V0, n, N, M, T, params; epsilon=1e-3)
    S = zeros(N, n)
    V = zeros(N, n)
    S[:, 1] .= S0
    V[:, 1] .= V0
    dt = T / n
    for i in 1:n
        dW1 = randn(N)
        dW2 = randn(N)
        S[:, i+1] .= S[:, i] .+ params.μ .* S[:, i] .* dt .+ sqrt(V[:, i]) .* S[:, i] .* dW1 .+ 0.5 .* V[:, i] .* S[:, i] .* (dW1.^2 .- dt)
        V[:, i+1] .= V[:, i] .+ params.κ .* (params.θ .- V[:, i]) .* dt .+ params.ν .* sqrt(V[:, i]) .* dW2 .+ 0.5 .* params.ν^2 .* (dW2.^2 .- dt)
    end
    return S, V
end
