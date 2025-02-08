using HTTP
using JSON
using Dates
using YFinance
using Revise
using DataFrames
function get_options(symbol::String, expiration_date::DateTime)
    options = get_Options(symbol)
    filtered_options = filter(x -> DateTime(x["expiration"]) == expiration_date, options)
    return filtered_options
end

function extract_call_data(options_data)
    calls = options_data["calls"]
    call_data = DataFrame(
        strike = calls["strike"],
        lastPrice = calls["lastPrice"],
        impliedVolatility = calls["impliedVolatility"],
        expiration = calls["expiration"]
    )
    return call_data
end
# Example usage: Fetch options data for NVIDIA (NVDA) with a specific expiration date
symbol = "NVDA"
today = DateTime("2025-01-20")
times_to_maturity = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5] # in years
expiration_dates = [today + Year(0) + Month(round(Int, t * 12)) for t in times_to_maturity]
options_data = get_options(symbol, expiration_dates[2])
options_data["calls"]
# Parsing the options data to get call prices
calls = extract_call_data(options_data)
market_prices = [(call["strike"], call["lastPrice"]) for call in calls]

# Print the strike prices and their corresponding market prices
println("Strike Prices and Market Prices:")
println(market_prices)

get_prices("AAPL",range="5y",interval="1mo",divsplits=true,exchange_local_time=false)
get_options("AAPL")

expirations = get_options_expirations("NVDA")
