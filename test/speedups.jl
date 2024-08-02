# Speed-ups through vectorization in Julia

using BenchmarkTools

# calculate speeds in vectorized Kahl Jackel's discretization of the Heston model
# calculate speeds in the naive version of Kahl Jackel's discretization of the Heston model

function KahlJackelHestonModel()
    # Parameters
    v0, ϱ, ν, κ, T = 2.0, 1.0, 3.0, 0.5, 1.0
    # Calculate the mean and standard deviation
    mean = ν * (1 - exp(-ϱ * T))
    std_dev = sqrt(ν * (1 - exp(-2 * ϱ * T)) / (2 * ϱ))
    return mean, std_dev
end

function KahlJackelHestonModelVectorized()
    # Parameters
    v0, ϱ, ν, κ, T = 2.0, 1.0, 3.0, 0.5, 1.0
    # Calculate the mean and standard deviation
    mean = ν .* (1 .- exp.(-ϱ .* T))
    std_dev = sqrt.(ν .* (1 .- exp.(-2 .* ϱ .* T)) ./ (2 .* ϱ))
    return mean, std_dev
end