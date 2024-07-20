module OptionPricing

# Write your package code here.

function EuropeanCall()

end

function EuropeanPut()

end

function AmericanCall()

end

function AmericanPut()

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

function SAApproximation()
end