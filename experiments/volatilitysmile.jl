using Distributions
using Dates
using CSV
# Define the cumulative distribution function for the standard normal distribution
N(d) = cdf(Normal(0, 1), d)

# Black-Scholes formula for European call option price
function black_scholes_call_price(S, K, r, T, σ)
    d1 = (log(S / K) + (r + 0.5 * σ^2) * T) / (σ * sqrt(T))
    d2 = d1 - σ * sqrt(T)
    return S * N(d1) - K * exp(-r * T) * N(d2)
end

# Vega function: derivative of the option price with respect to volatility
function vega(S, K, r, T, σ)
    d1 = (log(S / K) + (r + 0.5 * σ^2) * T) / (σ * sqrt(T))
    return S * pdf(Normal(0, 1), d1) * sqrt(T)
end

# Function to calculate implied volatility using Newton-Raphson method
function implied_volatility(S, K, r, T, C_market; tol=1e-5, max_iter=1000)
    # Initial guess based on moneyness
    σ = abs(log(S/K)) / sqrt(T)
    σ = max(0.1, min(σ, 2.0))  # Bound initial guess
    
    for i in 1:max_iter
        C = black_scholes_call_price(S, K, r, T, σ)
        v = vega(S, K, r, T, σ)
        
        # Check for numerical stability
        if abs(v) < 1e-10
            return NaN
        end
        
        diff = C - C_market
        if abs(diff) < tol
            return σ
        end
        
        # Damped update step
        σ_new = σ - 0.5 * (diff / v)
        σ = max(0.001, min(σ_new, 5.0))  # Bound volatility
    end
    return NaN  # Return NaN instead of error
end

# According to 10-yearr US Treasury yield on 2025-01-20
r = 0.0454
# Option parameters
filtered_options_data = CSV.read("NVDA_filtered_options_expiry.csv", DataFrame)
strike_prices = [75.0, 100.0, 125.0, 150.0, 175.0, 200.0]
expiry_date = filtered_options_data.expirationDate[:]

options_data = CSV.read("NVDA_options_data.csv", DataFrame)
# get the lastprices for corresponding expiry date
market_prices = Dict{Date, Dict{Float64, Float64}}()

# Extract last prices for each expiry date and strike price
for date in expiry_date
    market_prices[date] = Dict{Float64, Float64}()
    
    # Filter options_data for current date
    date_options = options_data[options_data.expirationDate .== date, :]
    
    # Match strikes and get lastPrice
    for strike in strike_prices
        matching_row = date_options[date_options.strike .== strike, :]
        if !isempty(matching_row)
            market_prices[date][strike] = matching_row[1, :lastPrice]
        else
            market_prices[date][strike] = NaN
        end
    end
end
market_prices

# Calculate implied volatilities
implied_vols = Dict{Date, Dict{Float64, Float64}}()
for (date, prices) in market_prices
    implied_vols[date] = Dict{Float64, Float64}()
    for (strike, price) in prices
        if !isnan(price)
            implied_vols[date][strike] = implied_volatility(137.71, strike, r, Dates.value(date - Date(2025, 1, 20)), price)
        else
            implied_vols[date][strike] = NaN
        end
    end
end

implied_vols