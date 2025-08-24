using Random
using Distributions
using Plots
using Parameters
using BenchmarkTools
using StatsBase
using LinearAlgebra
using Parameters
# continuation value and payoff functions
using ProgressMeter
using Printf
@with_kw mutable struct HestonParams
    S0::Float64 = 100.0
    μ_h::Float64 = 0.02
    ν::Float64 = 0.085
    θ_h::Float64 = 0.04
    κ_h::Float64 = 6.21
    ρ::Float64 = -0.7
    v0::Float64 = 0.501
end

call_payoff = (x, strike) -> max(x - strike, 0)
put_payoff = (x, strike) -> max(strike - x, 0)

b(v; ν = 0.085, θ = 0.04, κ = 6.21) = ν - ϱ * v
b_plus(v) = max(b(v), 0)
b_minus(v) = max(-b(v), 0)


# Construct the 1-d generator for the volatility process using the MCAM method provided in the paper by Cui 
function construct_generator_reflecting(v::Vector{Float64},
                                        mu::Function,
                                        sigma::Function)
    m0 = length(v)
    Q = zeros(m0, m0)

    # Mesh widths h[i] = v[i+1] - v[i]
    h = [v[i+1] - v[i] for i in 1:(m0-1)]

    mu_plus(x) = max(mu(x), 0)
    mu_minus(x) = max(-mu(x), 0)

    # --- Interior points ---
    for i in 2:(m0-1)
        hL = h[i-1]  # = v[i] - v[i-1]
        hR = h[i]    # = v[i+1] - v[i]
        mp = mu_plus(v[i])
        mm = mu_minus(v[i])
        s2 = sigma(v[i])^2

        Q[i, i-1] = mm/hL + (s2 - (hL*mm + hR*mp)) / (hL*(hL + hR))
        Q[i, i+1] = mp/hR + (s2 - (hL*mm + hR*mp)) / (hR*(hL + hR))
        Q[i, i] = - (Q[i, i-1] + Q[i, i+1])
    end

    # --- Left boundary (reflecting at v[1]) ---
    # One-sided difference for i=1 (no i-1).
    hR = h[1]             # width from v[1] to v[2]
    mp = mu_plus(v[1])
    mm = mu_minus(v[1])
    s2 = sigma(v[1])^2

    # A simple reflection scheme might say:
    # outflow from i=1 goes only to i=2, using a one-sided formula:
    Q[1, 2] = mp/hR + (s2 - (hR*mp)) / (hR^2)

    # Negative drift is "reflected" rather than allowed to go below v[1].
    # So no Q[1,0].
    # Then enforce row sum = 0:
    Q[1, 1] = - Q[1, 2]

    # --- Right boundary (reflecting at v[m0]) ---
    # One-sided difference for i=m0 (no i+1).
    hL = h[m0-1]          # width from v[m0-1] to v[m0]
    mp = mu_plus(v[m0])
    mm = mu_minus(v[m0])
    s2 = sigma(v[m0])^2

    Q[m0, m0-1] = mm/hL + (s2 - (hL*mm)) / (hL^2)
    Q[m0, m0] = - Q[m0, m0-1]

    return Q
end

# Sample volatility process through CTMC simulation
function simulateQtransitions(Q, bins, T; v0 = 0.04)
    # Input: Q: Generator matrix for the Volatility process
    # bins: Bins for the volatility process
    # Output: Transitions for the volatility process (till time T)

    # Initialize the transitions
    current_time = 0.0
    # set the current state by assigning a volatility bin to the initial volatility
    current_time, next_time = 0.0, 0.0
    # convert v0 to a volatility bin
    current_state = findfirst(bins .> v0) - 1
    # println("Current State: ", current_state)
    state_transitions = [current_state]
    transition_times = [current_time]
    while current_time < T

        next_time = current_time + rand(Exponential(-1/Q[current_state, current_state])) # Calculate the next transition time
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

# Calculate the conditional mean and variance of the log price provided a sample path of the volatility process
# 
#\mu_{\mathrm{GP}}(\omega) = \left(\mu-\frac{\nu\rho}{\kappa}\right)t + \left(\frac{\rho\varrho}{\kappa}-\frac12\right)\!\int_{0}^{t}\omega(s)\,ds + \frac{\rho}{\kappa}\bigl(\omega(t)-\omega(0)\bigr), \\
# \sigma_{\mathrm{GP}}^{2}(\omega) = (1-\rho^{2})\!\int_{0}^{t}\omega(s)\,ds.
function conditional_mean_variance(log_price_path, volatility_path, transition_times, T)
    num_steps = length(log_price_path)
    conditional_mean = zeros(num_steps)
    conditional_variance = zeros(num_steps)
    # Integral term 
    integral_term = 0.0
    for i in 1:num_steps
        dt = transition_times[i] - (i > 1 ? transition_times[i-1] : 0.0)
        integral_term += volatility_path[i] * dt
    end
    conditional_mean = (μ - (ν * ρ) / κ) * T +
        (ρ * ϱ / κ - 0.5) * integral_term +
        (ρ / κ) * (volatility_path[end] - volatility_path[1])
    conditional_variance = (1 - ρ^2) * integral_term
    return conditional_mean, conditional_variance
end

function BS_european_option_price(S0, V0, K, μ_GP, σ_GP, ρ,  T; r = 0.0)
    d1 = (log(S0 / K) + (μ_GP + 0.5 * σ_GP^2) * T) / (σ_GP * (1 - ρ^2) * V0sqrt(T)) # Check the formula for d1
    d2 = d1 - σ_GP * sqrt((1 - ρ^2) * V0 * T) # Check the formula for d2
    call_price = S0 * cdf(Normal(0, 1), d1) - K * exp(-r * T) * cdf(Normal(0, 1), d2)
    return call_price
end


# -------------- Example usage ---------------
mu_fun(x) = 0.5 - x
sigma_fun(x) = 0.2 + 0.1*x
v_min = 1e-3 
v_max = 10.0
vgrid = range(1e-3, 10.0, length=101) |> collect
Q = construct_generator_reflecting(vgrid, mu_fun, sigma_fun)

@printf "Q is a %d x %d matrix with row sums ~ zero.\n" size(Q, 1) size(Q, 2)

# Once the generator of the volatility process is constructed, we can use the fokker plank equation to calculate the 
# V(t, x), which is the value function linked with the price of a European option at time t and stock price x. 
# The value function V(t, x) satisfies the following PDE:
# dV/dt = -μ_h * x * dV/dx - 0.5 * V * V_x^2 - 0.5 * V_x^2 * V_xx - 0.5 * V_x * V_x + θ_h * V_xx
# The boundary conditions are V(t, 0) = 0 and V(t, x) -> max(x - K, 0) as x -> ∞.
# The initial condition is V(0, x) = max(x - K, 0) for a call option.


"""
Use the parameters so that condition (C) in Michael Kouritzin's paper is satistified. 
The condition (C) says the following 
For the Heston model, defined as: 
dS_t = μ S_t dt + ρ sqrt(V_t) S_t dW_t^1 + sqrt(1 - ρ^2) sqrt(V_t) S_t dW_t^2
dV_t = ν - θ V_t dt + κ sqrt(V_t) dW_t^2
where W_t^1 and W_t^2 are uncorrelated Brownian motions, the following condition must be satisfied:

ν = n * κ^2 / 4, where n = 1,2,3 ...
"""
function explicit_heston_simulation(params::HestonParams, T, num_simulations; num_steps=1000)
    dt = T / num_steps
    times = range(0, T, length = num_steps+1)
    
    # Preallocate arrays: each row corresponds to a simulation path.
    S_paths = zeros(num_simulations, num_steps+1)
    V_paths = zeros(num_simulations, num_steps+1)
    
    # Set initial values.
    S_paths[:, 1] .= params.S0
    V_paths[:, 1] .= params.v0
    progress = Progress(num_steps, 1)
    for j in 2:(num_steps+1)
        next!(progress)
        for j in 2:(num_steps+1)
            
            # Generate independent Brownian increments for the two uncorrelated parts.
            dB = sqrt(dt) * randn(num_simulations)
            dβ = sqrt(dt) * randn(num_simulations)
            
            # --- Variance update (explicit solution) ---
            # Here we set Y_t = sqrt(V_t) so that:
            #   Y_{t+dt} = Y_t + (κ_h/2) dβ   and then V_{t+dt} = (Y_{t+dt})^2.
            sqrtV = sqrt.(V_paths[:, j-1])
            newY = sqrtV .+ (params.κ_h / 2) * dβ
            V_paths[:, j] = newY .^ 2
            
            # --- Asset price update ---
            # Using the increment for log(S) given by:
            #   d(log S) = sqrt(1-ρ²) * sqrt(V_t) * dB + (μ_h - (νρ)/κ_h)*dt - 0.5 * V_t*dt
            #              + (ρ/κ_h)*(V_{t+dt} - V_t)
            dExp = zeros(num_simulations)
            @. dExp = sqrt(1 - params.ρ^2) * sqrtV * dB +
                       (params.μ_h - (params.ν * params.ρ) / params.κ_h) * dt -
                       0.5 * V_paths[j-1] * dt +
                       (params.ρ / params.κ_h) * (V_paths[j] - V_paths[j-1])
            S_paths[:, j] = S_paths[:, j-1] .* exp.(dExp)
        end
        # Generate independent Brownian increments for the two uncorrelated parts.
        dB = sqrt(dt) * randn(num_simulations)
        dβ = sqrt(dt) * randn(num_simulations)
        
        # --- Variance update (explicit solution) ---
        # Here we set Y_t = sqrt(V_t) so that:
        #   Y_{t+dt} = Y_t + (κ_h/2) dβ   and then V_{t+dt} = (Y_{t+dt})^2.
        sqrtV = sqrt.(V_paths[:, j-1])
        newY = zeros(num_simulations)
        @. newY = sqrtV + (params.κ_h / 2) * dβ
        V_paths[:, j] = newY .^ 2
        
        # --- Asset price update ---
        # Using the increment for log(S) given by:
        #   d(log S) = sqrt(1-ρ²) * sqrt(V_t) * dB + (μ_h - (νρ)/κ_h)*dt - 0.5 * V_t*dt
        #              + (ρ/κ_h)*(V_{t+dt} - V_t)
        dExp = zeros(num_simulations)
        @. dExp = sqrt(1 - params.ρ^2) * sqrtV * dB + (params.μ_h - (params.ν * params.ρ) / params.κ_h) * dt -
              0.5 * V_paths[j-1] * dt +
              (params.ρ / params.κ_h) * (V_paths[j] - V_paths[j-1])
        S_paths[:, j] = S_paths[:, j-1] .* exp.(dExp)
    end
    return times, S_paths, V_paths
end

function explicit_european_option_price(params::HestonParams, T, K, num_simulations; num_steps=1000)
    times, S_paths, V_paths = explicit_heston_simulation(params, T, num_simulations; num_steps=num_steps)
    
    # Calculate the option payoff at maturity
    payoffs = max.(S_paths[:, end] .- K, 0.0)
    
    # Discount the expected payoff back to present value
    option_price = exp(-params.μ_h * T) * mean(payoffs)
    
    return option_price
end

function plot_simulation(times, S_paths, V_paths; title="Heston Model Simulation")
    p1 = plot(times, S_paths', legend=false, title="Stock Price Paths", xlabel="Time", ylabel="Stock Price")
    p2 = plot(times, V_paths', legend=false, title="Variance Paths", xlabel="Time", ylabel="Variance")
    plot(p1, p2, layout=(2,1), size=(800, 600))
end


# Check if condition (C) is satisfied 
# Not needed for CTMC approximation of the Volatity process 
function check_condition_C(params::HestonParams; max_iterations=1000)
    n = 1
    ν = params.ν
    κ = params.κ_h
    n = 4 * params.ν / params.κ_h^2
    println("n = $n")
    if isapprox(ν, round(Int, n) * κ^2 / 4)
        println("Condition (C) is satisfied for ν = $ν and κ = $κ")
        return true
    end
    return false
end
# Set the parameters

params = HestonParams(S0=100.0, μ_h=0.02, ν=0.085, θ_h=4.0, κ_h=0.15, ρ=-0.75, v0=0.04)
check_condition_C(params)


# Choose parameter set for the Heston model such that condition (C) is satisfied
# # params = HestonParams(S0=100.0, μ_h=0.02, ν=0.085, θ_h=4.0, κ_h=0.15, ρ=-0.75, v0=0.04)
# PS_explicit = HestonParams(S0=100.0, μ_h=0.0319, ν=0.093025, θ_h=6.21, κ_h=0.61, ρ=-0.7, v0=0.04)
# check_ondition_C(PS_explicit) 
# T = 1.0
# num_simulations = 5
# times, S_paths, V_paths = explicit_heston_simulation(PS_explicit, T, num_simulations)
# plot_simulation(times, S_paths, V_paths)c
# Approximate the volatility process using the MCAM method 

# Price of European call option using the MCAM volatility process with the explicit price equation of the Heston model 


