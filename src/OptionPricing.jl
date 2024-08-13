module OptionPricing

# Write your package code here.

function EuropeanCall(StrikePrice, Premium)
    return maximum(0, Premium .- StrikePrice)

end

function EuropeanPut(StrikePrice, Premium)
    return maximum(0, StrikePrice .- Premium)

end

function AmericanCall(StrikePrice, Premium, time_to_maturity)
    return maximum(0, Premium .- StrikePrice) # This is the intrinsic value of the option
end

function AmericanPut(StrikePrice, Premium, time_to_maturity)
    return maximum(0, StrikePrice .- Premium) # This is the intrinsic value of the option

end

function AsianCall(StrikePrice, Premium)
    return maximum(0, mean(StrikePrice .- Premium))

end

function AsianPut(StrikePrice, Premium)
    return maximum(0, mean(Premium .- StrikePrice)

end

function BermudanCall()

end 

function BermudanPut()

end

function StochasticApproximation(basis_functions, N, T; χ = 2.0, γ = 2)
    # Define the basis functions
    basis_functions = basis_functions
    # Define the number of basis functions
end