# @doc"""
#     SLVCTMCApproximation.jl

# # SLVCTMCApproximation.jl

# This Julia module is designed to approximate the Heston Stochastic Volatility model using the Continuous Time Markov Chain (CTMC) method. The Heston model is a mathematical model that describes the evolution of the volatility of an asset. It is particularly useful in the field of financial mathematics for pricing options and other financial derivatives.

# ## Features

# - Approximation of the Heston Stochastic Volatility model using CTMC.
# - Support for different binning modes for volatility process discretization.

# ## Functions

# - `VolatilityBins(ν, ϱ, κ, v; binning_mode = "uniform")`: Generates bins for the volatility process based on the specified parameters. Supports "uniform" and "Lo-Skindilias" binning modes.

# ## Usage

# To use the `VolatilityBins` function, specify the long-term variance (`ν`), mean reversion rate (`ϱ`), volatility of the variance (`κ`), and the current variance (`v`). Optionally, specify the binning mode for the volatility process discretization.

# """

# module SLVCTMCApproximation
using LinearAlgebra
using Random
using Distributions
using Plots
using Parameters 
using BenchmarkTools
using StatsBase
# export VolatilityBins, HestonApproximation, SABRApproximation, ThreeTwoApproximation
# Write your package code here.
# Approximate the Heston Stochastic Volatility model using the CTMC method
# The Heston model is given by the following SDEs:
# dS(t) = μS(t)dt + sqrt((1 - ρ^2)v(t))S(t)dW1(t) + ρ* sqrt(v(t))S(t)dW2(t) 
# dv(t) = (ν - ϱv(t))dt + κ*sqrt(v(t))dW2(t)
# where W1(t) and W2(t) are independent Brownian motions
# The CTMC method is used to approximate the Heston model


    function calculateSufficientStats(ν, ϱ, κ, v0, T)
        # Calculate the mean and standard deviation of the CIR volatility process
        # ν: long-term variance
        # ϱ: mean reversion rate
        # κ: volatility of the variance
        # v0: initial variance
        # T: time horizon
        # Output: mean and standard deviation of the CIR volatility process
        # Calculate the mean and standard deviation of the CIR volatility process
        mean = v0*exp(-ϱ*T) + ν*(1 - exp(-ϱ*T))/ϱ
        variance = κ^2*v0*(exp(-ϱ*T) - exp(-2*ϱ*T))/ϱ + ν*κ^2*(1 - exp(-ϱ*T))^2/(2*ϱ^2)
        std_dev = sqrt(variance)
        return mean, std_dev
    end

    function VolatilityBins(v0, ν, ϱ, κ, T; binning_mode = "uniform", epsilon = 0.0001, γ = 2, num_bins = 100)
        # ν: long-term variance
        # ϱ: mean reversion rate
        # κ: volatility of the variance
        # v: variance
        # Output: the bins for the volatility process
        # Set the number of bins
        n = num_bins
        # calculate the mean and standard deviation of the CIR volatility process as defined in the comments 

        mean, std_dev = calculateSufficientStats(ν, ϱ, κ, v0, T)
        # Set the minimum and maximum values for the volatility process
        v_min = max(epsilon, mean - γ*std_dev)
        v_max = mean + γ*std_dev
        # Set the bin width
        if binning_mode == "uniform"
            dv = (v_max - v_min)/n
        elseif binning_mode == "Lo-Skindilias"
            # each dv is a function of v
            dv = 0
        # elseif binning_mode
        end
            # Initialize the bins
        bins = zeros(n)
        # Set the bin values
        for i in 1:n
            if binning_mode == "uniform"
                bins[i] = v_min + (i - 1)*dv
            elseif binning_mode == "Lo-Skindilias"
                # each bin is a function of v
                # TODO: Impelment the Lo-Skindilias binning mode
                continue
            elseif binning_mode == "other"
                # TODO: Implement other binning modes
                continue
            end
        end

        return bins
    end

    function SABRApproximation()
    
    end

    function ThreeTwoApproximation()
        
    end

    function calculatevolatilityGenerator(ν, ϱ, κ, v0, T; γ = 5, num_bins = 100, epsilon = 0.0001) # Calculate the Q matrix for the volatility process
        # calculate the generator matrix Q for the Volatility process
        # Set the number of bins
        n = num_bins
        mean, std_dev = calculateSufficientStats(ν, ϱ, κ, v0, T)
        # Set the minimum and maximum values for the volatility process
        v_min = max(epsilon, mean - γ*std_dev) # MEAN AND STD_DEV ARE THE MEAN AND STD_DEV OF THE VOLATILITY PROCESS
        v_max = mean + γ*std_dev # MEAN AND STD_DEV ARE THE MEAN AND STD_DEV OF THE VOLATILITY PROCESS
        # Set the bin width
        dv = (v_max - v_min)/n # Uniform bin width
        # TODO: Use VolatilityBins function to generate the bins for the volatility process
        # These bings can be uniform or non-uniform according to various processes 
        # Initialize the generator matrix Q
        volbins = VolatilityBins(v0, ν, ϱ, κ, T, γ = γ, num_bins = num_bins)

        
        Q = zeros(n, n)
        # Set the bin vaues
        for i in 1:n
            # Calculate the transition rates for the volatility process
            v_curr, v_next = v_min + (i - 1)*dv, v_min + i*dv
            dv_curr, dv_next = dv, dv # For now, assume uniform bin width
            if i == 1
                Q[i, i + 1] = max(0, ν - ϱ*v_curr)/dv_next + (κ^2*v_curr - dv_curr*min(0, ν - ϱ*v_curr) - dv_next*max(0, ν - ϱ*v_curr))/(dv_next*(dv_curr + dv_next))
                Q[i, i] = -Q[i, i + 1]
            elseif i == n
                Q[i, i - 1] = min(0, ν - ϱ*v_curr)/dv_curr + (κ^2*v_curr - dv_curr*min(0, ν - ϱ*v_curr) - dv_next*max(0, ν - ϱ*v_curr))/(dv_curr*(dv_curr + dv_next))
                Q[i, i] = -Q[i, i - 1]
            else
                Q[i, i+1] = max(0, ν - ϱ * v_curr) / dv_next + (κ^2 * v_curr - dv_curr * min(0, ν - ϱ * v_curr) - dv_next * max(0, ν - ϱ * v_curr)) / (dv_next * (dv_curr + dv_next))
                Q[i, i-1] = min(0, ν - ϱ * v_curr) / dv_curr + (κ^2 * v_curr - dv_curr * min(0, ν - ϱ * v_curr) - dv_next * max(0, ν - ϱ * v_curr)) / (dv_curr* (dv_curr + dv_next))
                Q[i, i] = -Q[i, i + 1] - Q[i, i - 1]
            end
        end

        return Q
    end

    @doc"""
    # Calculate the generator matrix Q for the Price and Volatility process

    After calculating the generator matrix Q for the Volatility process, we can calculate the generator matrix Q for the Price and Volatility process. 
    The Price and Volatility process is given by the following SDEs in the Heston model:
    
    dS(t) = μS(t)dt + sqrt((1 - ρ^2)v(t))S(t)dW1(t) + ρ* sqrt(v(t))S(t)dW2(t)
    dv(t) = (ν - ϱv(t))dt + κ*sqrt(v(t))dW2(t)
    
    where W1(t) and W2(t) are independent Brownian motions.

    Consider a volatility process of the following form:
    dv(t) = μ(t)dt + σ(t)dW(t)
    
    The volatility process can be approximated with a CTMC method with the following rates: 
    
    """
    function calculateRegimeSwitchingGenerator(μ, ν, κ, ρ, ϱ, S0, v0, T, N, M; mode = "Explicit-Kushner")
        # μ: drift of the stock price
        # ν: long-term variance
        # κ: mean reversion rate
        # ρ: correlation between the stock price and the variance
        # ϱ: volatility of the variance
        # S0: initial stock price
        # v0: initial variance
        # T: time horizon
        # N: number of time steps
        # M: number of Monte Carlo simulations
        # Output: stock price and variance paths
        # Modes of the CTMC method: "Kushner-Kushner", "Explicit-Kushner", "Explicit-MP"
        # Initialize the stock price and variance paths
        S = zeros(N+1, M)
        v = zeros(N+1, M)
        
        # Set the initial stock price and variance
        S[1, :] .= S0
        v[1, :] .= v0
        
        # Set the time step
        dt = T/N
        if mode == "Kushner-Kushner"
            # use Zhenyu Cui's method to calculate the generator matrix Q for the Volatility process
            bins = VolatilityBins(ν, ϱ, κ, v0)
            # Q = zeros(length(bins), length(bins))
            # Set the parameters for the Kushner-Kushner method
            # calculate the generator matrix Q for the Volatility process
            Q = calculatevolatilityGenerator(ν, ϱ, κ, v0)
        else
            # TODO: Write the Explicit-Kushner and Explicit-MP methods
        end 
    end 

   
    @doc"""
    Using the Explicit-Kushner method, we can calculate the generator matrix Q for the Price and Volatility process.
    The Generator matrix governs the CTMC process for the Volatility process. 
    """
    function simulateVolatilityTransitions(Q_Volatility, volatility_bins, T; v0 = 0.04)
        # Input: Q_Volatility: Generator matrix for the Volatility process
        # volatility_bins: Bins for the volatility process
        # Output: Transitions for the volatility process
        # Initialize the transitions
        current_time = 0.0
        # set the current state by assigning a volaility bin to the initial volatility 
        current_time, next_time = 0.0, 0.0
        # convert v0 to a volatility bin
        current_state = findfirst(volatility_bins .>= v0)
        while t < T
            # Generate the next transition
            # Simulate the time to the next transition
            next_time = current_time + rand(Exponential(1/Q[current_state, current_state]))
            
            # Calculate the transition probabilities
            # Sample the next state
            # Update the current state
            # Update the time
        
        end
    end
    function HestonApproximation(μ, ν, κ, ρ, ϱ, S0, v0, T, N, M; mode = "Explicit-Kushner")
        # μ: drift of the stock price
        # ν: long-term variance
        # κ: mean reversion rate
        # ρ: correlation between the stock price and the variance
        # ϱ: volatility of the variance
        # S0: initial stock price
        # v0: initial variance
        # T: time horizon
        # N: number of time steps
        # M: number of Monte Carlo simulations
        # Output: stock price and variance paths
        # Modes of the CTMC method: "Kushner-Kushner", "Explicit-Kushner", "Explicit-MP"
        # Initialize the stock price and variance paths
        S = zeros(N+1, M)
        v = zeros(N+1, M)
        
        # Set the initial stock price and variance
        S[1, :] .= S0
        v[1, :] .= v0
        
        # Set the time step
        # set the upper and lower bounds for the stock and volatility grids 

        dt = T/N
        if mode == "Kushner-Kushner"
            # use Zhenyu Cui's method to calculate the generator matrix Q for the Volatility process
            bins = VolatilityBins(ν, ϱ, κ, v0)
            Q = zeros(length(bins), length(bins))


            # Set the parameters for the Kushner-Kushner method
            # calculate the generator matrix Q for the Volatility process
        
        
        else
        if mode == "Explicit-Kushner"
            bins = VolatilityBins(ν, ϱ, κ, v0)
            Q_Volatility = calculatevolatilityGenerator(ν, ϱ, κ, v0)
        
        end
            
            # After calculating the volatility transitions, we can calculate the stock price transitions 


            # Set the parameters for the Kushner-Kushner method
            # calculate the generator matrix Q for the Volatility process
        # Generate the stock price and variance paths
        for i in 1:N
            # Generate the stock price and variance paths using the CTMC method
            for j in 1:M
                # Generate the stock price and variance paths using the CTMC method
                S[i+1, j] = S[i, j] + μ*S[i, j]*dt + sqrt((1 - ρ^2)*v[i, j])*S[i, j]*randn()*sqrt(dt) + ρ*sqrt(v[i, j])*S[i, j]*randn()*sqrt(dt)
                v[i+1, j] = v[i, j] + (ν - ϱ*v[i, j])*dt + κ*sqrt(v[i, j])*randn()*sqrt(dt)
            end
        end
        
        return S, v


    end

    # Approximate the Three-Two model using the CTMC method
    # The Three-Two model is given by the following SDEs:

end

function simulateQtransitions(Q, bins, T; v0 = 0.04)
    # Input: Q: Generator matrix for the Volatility process
    # bins: Bins for the volatility process
    # Output: Transitions for the volatility process (till time T)

    # Initialize the transitions
    current_time = 0.0
    # set the current state by assigning a volaility bin to the initial volatility
    current_time, next_time = 0.0, 0.0
    # convert v0 to a volatility bin
    current_state = findfirst(bins .> v0) - 1
    println("Current State: ", current_state)
    state_transitions = [current_state]
    transition_times = [current_time]
    while current_time < T
        # Generate the next transition
        # Simulate the time to the next transition
        # Calculate the transition probabilities
        # Sample the next state
        # Update the current state
        # Update the time
        # println("Current State rate: ", Q[current_state, current_state])
        next_time = current_time + rand(Exponential(-1/Q[current_state, current_state]))
        # Calculate the transition probabilities
        e_i = (i, num_states) -> (e = zeros(Float64, num_states); e[i] = 1.0; e)

        transition_probs = e_i(current_state, length(bins)) + Q[current_state, :] ./ (-1* Q[current_state, current_state])
        next_state = sample(1:length(bins), Weights(transition_probs))
        push!(state_transitions, next_state)
        push!(transition_times, next_time)
        current_state = next_state
        current_time = next_time
    end 
    return (state_transitions, transition_times)
end

function simulatePriceProcess(transition_times, volatilitychain, μ, ν, ρ, κ, S0, v0)
    # Input: Q: Generator matrix for the Price process
    # bins: Bins for the price process
# test the VolatilityGenerator function for some established parameters of the Heston Stochastic Volatility model
    # at every transition times, update the price process using the following equations 
    # S^{T_{i+1}} = S^{T_i} + μS^{T_i}dt + sqrt((1 - ρ^2)v^{T_i})S^{T_i}dW1 + ρ*sqrt(v^{T_i})S^{T_i}dW2
    log_prices = zeros(length(transition_times))
    dt = transition_times[1]
    log_prices[1] = log(S0) + sqrt(1 - ρ^2)*sqrt(v0 * dt)*randn() + (μ - ν * ϱ / κ) * dt + (ρ*ϱ/κ - 0.5)*v0*dt + ρ/κ*0
    for i in 2:length(transition_times)
        dt = transition_times[i] - transition_times[i - 1]
        log_prices[i] = log_prices[i - 1] + sqrt(1 - ρ^2)*sqrt(volatilitychain[i] * dt)*randn() + (μ - ν * ϱ / κ) * dt + (ρ*ϱ/κ - 0.5)*volatilitychain[i]*dt + ρ/κ*(volatilitychain[i] - volatilitychain[i - 1])
    end
    return log_prices
end 

# end 
# Set the parameters for the Heston model

PS1 = Dict(
    :S0 => 100,
    :μ => 0.02,
    :ν => 0.085,
    :ϱ => 6.21,
    :κ => 0.2,
    :ρ => -0.7,
    :V0 => 0.501
)

PS2 = Dict(
    :S0 => 100,
    :μ=> 0.02,
    :ν => 0.424,
    :ϱ => 6.00,
    :κ => 0.8,
    :ρ => -0.75,
    :V0 => 0.11
)

PS3 = Dict(:S0 => 100,
    :μ => 0.02,
    :ν => 0.225,
    :ϱ => 2.86,
    :κ => 0.6,
    :ρ => -0.96,
    :V0 => 0.07
)

# Extract the parameters
# S0, μ, ν, ϱ, κ, ρ, v0 = PS3[:S0], PS3[:μ], PS3[:ν], PS3[:ϱ], PS3[:κ], PS3[:ρ], PS3[:V0]
# S0, μ, ν, ϱ, κ, ρ, v0 = PS3[:S0], PS3[:μ], PS3[:ν], PS3[:ϱ], PS3[:κ], PS3[:ρ], PS3[:V0]
# Do this for Parameter Set 1
# S0, μ, ν, ϱ, κ, ρ, v0 = PS1[:S0], PS1[:μ], PS1[:ν], PS1[:ϱ], PS1[:κ], PS1[:ρ], PS1[:V0]
# Do this for Parameter Set 2
S0, μ, ν, ϱ, κ, ρ, v0 = PS2[:S0], PS2[:μ], PS2[:ν], PS2[:ϱ], PS2[:κ], PS2[:ρ], PS2[:V0]
T = 10
volbins= VolatilityBins(v0, ν, ϱ, κ, T, γ = 10, num_bins = 100)
# Sample volatility for volatility bins 

# Sample the volatility process

# The first bin that contains the initial volatility 
print("Initial volatility bin: ", findfirst(volbins .>= v0))

# volbins[findfirst(volbins .> v0) - 1]
# Calculate the bins for the volatility process
bins = VolatilityBins(ν, ϱ, κ, v0, T)   

mean, std_dev = calculateSufficientStats(ν, ϱ, κ, v0, T)
Q = calculatevolatilityGenerator(ν, ϱ, κ, v0, T, γ = 10, num_bins = 100)
for i in 1:100
    println(Q[i, i])
end
# Sample from the CTMC with Q as the generator matrix 

state_transitions, transition_times = simulateQtransitions(Q, volbins, T, v0 = v0)
volatilitychain = volbins[state_transitions]
logpriceprocess = simulatePriceProcess(transition_times, volatilitychain, μ, ν, ρ, κ, S0, v0)
priceprocess = exp.(logpriceprocess)
# length(state_transitions)
# length(transition_times)
volatilitychain

# Plot the price process and the volatility process
plot(transition_times, priceprocess, label = "Price Process", xlabel = "Time", ylabel = "Price", title = "Price Process vs Time")
plot(transition_times, volatilitychain, label = "Volatility Process", xlabel = "Time", ylabel = "Volatility", title = "Volatility Process vs Time")
function condition(bins, v0, ν, ϱ, κ)
    # First calculate the max difference between the bins 
    max_diff = maximum(diff(bins))
    # Find the min ratio of volatilityto the modulus of the drift
    min_ratio = minimum(κ^2 .* bins ./ abs.(ν .- ϱ .* bins))

    return max_diff <= min_ratio
end
# This condition is satisfied. 

# calculateSufficientStats(ν, ϱ, κ, v0, T)[1]                                                               
