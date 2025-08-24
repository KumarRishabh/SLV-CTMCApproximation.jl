using Random, Distributions, Statistics

# Parameters for Black-Scholes option pricing
S0 = 100.0      # Initial stock price
K = 100.0       # Strike price
r = 0.01        # Risk-free rate
σ = 0.2         # Volatility
T = 1.0         # Time to maturity

# Heston model parameters
v0 = 0.04       # Initial variance (σ²)
κ = 1.0         # Mean reversion speed
θ = 0.04        # Long-term variance
σ_v = 0.2       # Volatility of variance
ρ = -0.7        # Correlation between stock and variance

# Analytical Black-Scholes formula
function black_scholes_call(S, K, r, σ, T)
    d1 = (log(S/K) + (r + 0.5*σ^2)*T) / (σ*sqrt(T))
    d2 = d1 - σ*sqrt(T)
    
    call_price = S * cdf(Normal(0,1), d1) - K * exp(-r*T) * cdf(Normal(0,1), d2)
    return call_price
end

# Monte Carlo simulation for Black-Scholes
function monte_carlo_option_price(S0, K, r, σ, T, n_sims=100000)
    Random.seed!(42)
    payoffs = Float64[]
    
    for i in 1:n_sims
        Z = randn()
        ST = S0 * exp((r - 0.5*σ^2)*T + σ*sqrt(T)*Z)
        payoff = max(ST - K, 0.0)
        push!(payoffs, payoff)
    end
    
    option_price = exp(-r*T) * mean(payoffs)
    return option_price, std(payoffs) / sqrt(n_sims)
end

# Heston model Monte Carlo simulation
function heston_monte_carlo(S0, K, r, T, v0, κ, θ, σ_v, ρ, n_sims=100000, n_steps=252)
    Random.seed!(42)
    dt = T / n_steps
    payoffs = Float64[]
    
    for i in 1:n_sims
        S = S0
        v = v0
        
        for j in 1:n_steps
            # Generate correlated random numbers
            Z1 = randn()
            Z2 = ρ * Z1 + sqrt(1 - ρ^2) * randn()
            
            # Milstein scheme for variance process
            sqrt_v = sqrt(max(v, 0))
            dv = κ * (θ - v) * dt + σ_v * sqrt_v * sqrt(dt) * Z2 + 
                 0.25 * σ_v^2 * dt * (Z2^2 - 1)
            v = max(v + dv, 0)  # Ensure non-negative variance
            
            # Milstein scheme for stock price
            sqrt_v_new = sqrt(max(v, 0))
            dS = r * S * dt + sqrt_v_new * S * sqrt(dt) * Z1 + 
                 0.5 * sqrt_v_new * S * v * dt * (Z1^2 - 1) / sqrt_v_new
            S = S + dS
        end
        
        payoff = max(S - K, 0.0)
        push!(payoffs, payoff)
    end
    
    option_price = exp(-r*T) * mean(payoffs)
    return option_price, std(payoffs) / sqrt(n_sims)
end

# Sanity checks
println("=== Option Pricing Sanity Check ===")
println()

# Black-Scholes comparison
bs_analytical = black_scholes_call(S0, K, r, σ, T)
bs_mc, bs_se = monte_carlo_option_price(S0, K, r, σ, T)

println("Black-Scholes Model:")
println("Analytical price: $(round(bs_analytical, digits=4))")
println("Monte Carlo price: $(round(bs_mc, digits=4)) ± $(round(bs_se, digits=4))")
println("Difference: $(round(abs(bs_analytical - bs_mc), digits=4))")
println()

# Heston model
heston_price, heston_se = heston_monte_carlo(S0, K, r, T, v0, κ, θ, σ_v, ρ)

println("Heston Model:")
println("Monte Carlo price: $(round(heston_price, digits=4)) ± $(round(heston_se, digits=4))")
println("Difference from BS: $(round(abs(bs_analytical - heston_price), digits=4))")