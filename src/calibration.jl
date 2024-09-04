module HestonCalibration

using Random
using Distributions
using Plots
using Parameters
using BenchmarkTools
using StatsBase
using Revise
using 
# What is calibration about? 
# The heston model has 5 parameters:
# mu: drift
# kappa: volatility of volatility
# rho: correlation between stock price and variance
# varrho: mean reversion speed
# nu: long term variance
# We want to find the parameters that best fit the data

@with_kw struct HestonParameters
    mu::Float64 = -0.5
    kappa::Float64 = 5.3
    rho::Float64 = -0.7
    varrho::Float64 = 0.5
    nu::Float64 = 0.04
end 

function fitHestonModel(data)
    # Fit the Heston model to the data
    # data is a matrix with 2 columns: time and stock price

    # Define the objective function
end 

# get NVIDIA data from yahoo finance
function getNVIDIAData()
    # Get the data from yahoo finance

end 
