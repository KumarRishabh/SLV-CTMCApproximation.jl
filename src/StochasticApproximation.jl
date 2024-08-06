using LinearAlgebra
using Random
using Distributions
using Plots
using Parameters 
using BenchmarkTools
using StatsBase

function Laguerre_Polynomials(n, x)
    L = zeros(n+1)
    L[1] = 1
    L[2] = 1 - x
    for i in 3:n+1
        L[i] = ((2*(i-1) + 1 - x) * L[i-1] - (i-1) * L[i-2]) / i
    end
    return L
end

function Laguerre_Polynomials_3D(n, x::Tuple{<:Real, <:Real, <:Real}) # x is a tuple with 3 elements
    L = zeros(n+1, n+1, n+1)
    L[1, 1, 1] = 1
    L[2, 1, 1] = 1 - x[1]
    L[1, 2, 1] = 1 - x[2]
    L[1, 1, 2] = 1 - x[3]
    for i in 3:n+1
        L[i, 1, 1] = ((2*(i-1) + 1 - x[1]) * L[i-1, 1, 1] - (i-1) * L[i-2, 1, 1]) / i
        L[1, i, 1] = ((2*(i-1) + 1 - x[2]) * L[1, i-1, 1] - (i-1) * L[1, i-2, 1]) / i
        L[1, 1, i] = ((2*(i-1) + 1 - x[3]) * L[1, 1, i-1] - (i-1) * L[1, 1, i-2]) / i
    end
    for i in 2:n+1     
        for j in 2:n+1
            for k in 2:n+1
                L[i, j, k] = ((2*(i-1) + 1 - x[1]) * L[i-1, j, k] - (i-1) * L[i-2, j, k]) / i
                L[i, j, k] = ((2*(j-1) + 1 - x[2]) * L[i, j-1, k] - (j-1) * L[i, j-2, k]) / j
                L[i, j, k] = ((2*(k-1) + 1 - x[3]) * L[i, j, k-1] - (k-1) * L[i, j, k-2]) / k
            end
        end
    end
    return L
end
   

# Employ a symbolic approach to calculate the Laguerre_Polynomials



# Consider the asset price follows a Geometric Brownian Motion
# Employ a Monte Carlo Simulation to calculate the price of a European Call Option
# And then, change the strike price to get 

