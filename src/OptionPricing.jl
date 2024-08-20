module OptionPricing

# Write your package code here.

function call_payoff(strike, S)
    return max(S - strike, 0)
end

function put_payoff(strike, S)
    return max(strike - S, 0)
end

end 