using Random
using Distributions
using Plots
using Parameters
using BenchmarkTools
using StatsBase

# continuation value and payoff functions

call_payoff = (x, strike) -> max(x - strike, 0)
put_payoff = (x, strike) -> max(strike - x, 0)
