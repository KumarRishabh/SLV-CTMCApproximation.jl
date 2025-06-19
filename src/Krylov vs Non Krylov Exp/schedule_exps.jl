inlcude "../src/CuiSDEApproximation.jl"
include "../src/DoubleDeathApproximation.jl"

using CSV, DataFrames, Plots

# define a small struct to hold model parameters
struct OptionParams
    S0::Float64
    K::Float64
    T::Float64
    r::Float64
    σ::Float64
end

# ground‐truth data: a Dict mapping each option‐type string to
# a Vector of (params => true_price) pairs
const GT = Dict(
    "EuropeanCall" => [
        OptionParams(100.0, 100.0, 1.0, 0.05, 0.2) => 10.4506,
        OptionParams(120.0, 100.0, 0.5, 0.03, 0.25) => 22.3145,
        # … add more cases
    ],
    "EuropeanPut" => [
        OptionParams(100.0, 100.0, 1.0, 0.05, 0.2) => 5.5735,
        # …
    ],
    "AmericanCall" => [
        OptionParams(100.0, 100.0, 1.0, 0.05, 0.2) => 10.4612,
        # …
    ],
    "AmericanPut" => [
        OptionParams(100.0, 100.0, 1.0, 0.05, 0.2) => 5.6821,
        # …
    ]
)

option_types = collect(keys(GT))
grid_sizes   = [10, 50, 100, 200, 500]

results = DataFrame(
    OptionType = String[],
    GridSize   = Int[],
    MAE        = Float64[]
)

for opt in option_types
    entries = GT[opt]
    truth   = [price for (_params => price) in entries]

    for gs in grid_sizes
        approx = [
            approximate_option_price(
                S0      = p.S0,
                K       = p.K,
                T       = p.T,
                r       = p.r,
                σ       = p.σ,
                american= startswith(opt, "American"),
                put     = endswith(opt, "Put"),
                krylov  = true,
                grid    = gs
            )
            for (p => _) in entries
        ]

        err = mean(abs.(approx .- truth))
        push!(results, (opt, gs, err))
    end
end

@df results plot(
    :GridSize, :MAE,
    group  = :OptionType,
    xlabel = "Grid Size",
    ylabel = "Mean Absolute Error",
    title  = "Krylov CTMC Pricing Accuracy by Option Type",
    legend = :outertopright
)