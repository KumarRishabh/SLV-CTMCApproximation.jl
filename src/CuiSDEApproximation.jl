module CuiSDEApproximation
using Random
using Distributions
using Plots
using Parameters
using BenchmarkTools
using StatsBase
using Revise

# Use Zhenyu Cui's SDE approximation method to generate sample paths of the Heston model and 3/2 model
# use there sample paths to price European call options and American put and call options
# Also, Asian options
# Use the same parameters as in the previous notebook

@with_kw struct HestonParameters
    S0::Float64 = 100.0
    V0::Float64 = 0.04
    mu::Float64 = -0.5
    nu::Float64 = 0.01
    mean_reversion_coeff::Float64 = 5.3
    rho::Float64 = -0.7
    kappa::Float64 = -0.5
end 

@with_kw struct SLV32Parameters
    S0::Float64 = 100.0
    V0::Float64 = 0.04
    mu::Float64 = -0.5
    nu::Float64 = 0.01
    mean_reversion_coeff::Float64 = 5.3
    rho::Float64 = -0.7
    kappa::Float64 = -0.5
    gamma::Float64 = 0.5
end

@with_kw struct SimulationParameters
    epsilon::Float64 = 10e-06
    timelimit::Float64 = 1.0
    nsim::Int64 = 100000
end 

@with_kw struct PayoffParameters
    strike::Float64 = 100.0
    maturity::Float64 = 1.0
end 

# Do continuous time simulations of the Heston model and the 3/2 model
# The Heston model is given by the following SDEs:
# dS = mu * S * dt + sqrt(1 - ρ^2) * sqrt(V) * S * dW1 + ρ* sqrt(V) * S * dW2
# dV = (θ - ϱ*V )* dt + κ * sqrt(V) * dW2

# The 3/2 model is given by the following SDEs:
# dS = mu * S * dt + sqrt(V) * S * dW1
# dV = κ * (θ - V) * dt + γ * V^(3/2) * dW2

function calculateSufficientStats(ν, ϱ, κ, v0, T)

    mean = v0 * exp(-ϱ * T) + ν * (1 - exp(-ϱ * T)) / ϱ
    variance = κ^2 * v0 * (exp(-ϱ * T) - exp(-2 * ϱ * T)) / ϱ + ν * κ^2 * (1 - exp(-ϱ * T))^2 / (2 * ϱ^2)
    std_dev = sqrt(variance)
    return mean, std_dev
end


function calculateHestonVolatilityGenerator(paramsHeston, paramsSim, paramsPayoff; grids = 100, γ::Int32 = 5)
    # Calculate the CTMC generator approximating the volatility process in the Heston model.
    # This process (rather the index of the process) is denoted by α(t) in the following paper:
    # 1. Cui, Z., Lars Kirkby, J., & Nguyen, D. (2019). Continuous-time Markov chain and regime switching approximations with applications to options pricing. Springer.

    Q = zeros(grids, grids) 
    # Calculate the drift and diffusion coefficients
    mean, variance = calculateSufficientStats(paramsHeston.nu, paramsHeston.mean_reversion_coeff, paramsHeston.kappa, paramsHeston.V0, paramsPayoff.maturity)    
    
end

function calculatevolatilityGenerator(paramsModel, paramsSim, paramsPayoff; model = "Heston")
    if model == "Heston"
        return calculateHestonVolatilityGenerator(paramsModel, paramsSim, paramsPayoff)
    elseif model == "SLV32"
        return calculateSLV32VolatilityGenerator(paramsModel, paramsSim, paramsPayoff)
    else
        error("Model not supported")
    end
end 

function calculateSDEGenerator()

end 

function simulateQtransitions()

end 

function EuropeanCallOptionPrice()

end

end 

