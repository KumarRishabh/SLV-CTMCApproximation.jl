using Random
using Distributions
using Plots
using Parameters
using BenchmarkTools
using StatsBase
using Revise
include("../src/DiscreteTimeApproximation.jl")
include("../src/OptionPricing.jl")
using .DiscreteTimeApproximation
using .OptionPricing

@with_kw struct HestonParameters
    S0::Float64 = 100.0
    V0::Float64 = 0.04
    mu::Float64 = -0.5
    nu::Float64 = 0.01
    mean_reversion_coeff::Float64 = 5.3
    rho::Float64 = -0.7
    kappa::Float64 = -0.5
end

@with_kw struct ContinuousSimulationParameters
    epsilon::Float64 = 10e-06
    timelimit::Float64 = 1.0
    nsim::Int64 = 100000
end 

@with_kw struct PayoffParameters
    strike::Float64 = 100.0
    maturity::Float64 = 1.0
end

module ContinuousTimeBinomialOptionPricing


end