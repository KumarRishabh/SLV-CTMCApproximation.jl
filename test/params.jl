using Random, Statistics, LinearAlgebra, Plots, PyCall, PrettyTables
using Distributions
using Revise
# use the module DiscreteTimeApproximation
include("../src/DiscreteTimeApproximation.jl") 
using .DiscreteTimeApproximation
# python_functions = pyimport("../src/branching_particle_pricer.py")
T, delta_t, n = 20000, 1, 20
S_0, V_0, N, r = 100, 0.04, 200, 10

model_params = Dict(
    # "S_0" => 100,
    # "V_0" => 0.501,
    "mu" => -0.5,
    "nu" => 0.01,
    "mean_reversion_coeff" => 5.3,
    "rho" => -0.7,
    "kappa" => -0.5
)

PS_1 = Dict(
    # "S_0" => 100,
    # "V_0" => 0.501,
    "mu" => 0.02,
    "nu" => 0.085, # 8.5 * 0.2 * 0.2 / 4
    "mean_reversion_coeff" => 6.21, # ϱ
    "rho" => -0.7,
    "kappa" => 0.2
)

PS_2 = Dict(
    # "S_0" => 100,
    # "V_0" => 0.11,
    "mu" => 0.02,
    "nu" => 0.424,
    "mean_reversion_coeff" => 6.00, # ϱ
    "rho" => -0.75,
    "kappa" => 0.8
)

PS_3 = Dict(
    # "S_0" => 100,
    # "V_0" => 0.501,
    "mu" => 0.02,
    "nu" => 0.225,
    "mean_reversion_coeff" => 2.86, # ϱ
    "rho" => -0.96,
    "kappa" => 0.6
)

explicit_params = Dict(
    "mu" => 0.0319,
    "nu" => 0.093025,
    "mean_reversion_coeff" => 6.21, # ϱ
    "rho" => -0.7,
    "kappa" => 0.61
)
@time S, V, LogL = DiscreteTimeApproximation.branching_particle_filter(S_0, V_0, n, N, r, PS_3, T; delta_t = 0.001)
S
@time S, V, LogL = DiscreteTimeApproximation.weighted_heston(S_0, 0.010201, n, N, 6, T, explicit_params, delta_t = 0.002)
@time S, V, LogL = DiscreteTimeApproximation.weighted_heston_M2(S_0, 0.010201, n, N, T, explicit_params, delta_t = 0.002)
@time S, V, logL = DiscreteTimeApproximation.weighted_heston(S_0, 0.07, 3, 20, 10, 50, PS_3)
plot(V, title = "Stock Prices", label = "Stock Prices", xlabel = "Time", ylabel = "Stock Price", legend=false)
@time logS_history, V_history, logL_history = DiscreteTimeApproximation.branching_particle_filter(S_0, V_0, 20, 50, r, PS_3, n; delta_t)
for i in 1:50
    # println("At iteration $i: LogStockPrices: $(logS_history[i])")
    println("At iteration $i: Volatilities: $(V_history[i])")
end
output_file = "/home/rishabh/SLV-CTMCApproximation.jl/src/output.txt"
open(output_file, "w") do file
    for i in 1:50
        println(file, "At iteration $i: LogStockPrices: $(S[i, :])")
        println(file, "At iteration $i: Volatilities: $(V[i, :])")
        # println(file, "At iteration $i: LogLikelihoods: $(logL_history[i])")
    end
end
# plot the logS_history with respect to time for each particle in the simulation

function simulateCIRProcess(ν, ϱ, κ, V0, T; delta_t = 1e-3, epsilon = 1e-6)
    V = zeros(Int(T/delta_t))
    V[1] = V0
    for i in 2:Int(T/delta_t)
        dW = randn()
        V[i] = V[i-1] + (ν - ϱ*V[i-1])*delta_t + κ*sqrt(V[i-1]*delta_t)*dW + 0.25*κ^2*(delta_t*(dW^2 - 1))
        if V[i] < epsilon
            V[i] = epsilon - (V[i] - epsilon)
        end
        # println(V[i])
    end
    return V
end


function simulatemultipleCIRProcesses(ν, ϱ, κ, V0, T, N; delta_t = 1e-1, epsilon = 1e-6)
    V = zeros(Int(T/delta_t), N)
    for i in 1:N
        V[:,i] = simulateCIRProcess(ν, ϱ, κ, V0, T, delta_t = delta_t)
    end
    return V
end

V = simulatemultipleCIRProcesses(explicit_params["nu"], explicit_params["mean_reversion_coeff"], explicit_params["kappa"], V_0, T, 100)
V = simulatemultipleCIRProcesses(PS_2["nu"], PS_2["mean_reversion_coeff"], PS_2["kappa"], V_0, T, 100)
V = simulatemultipleCIRProcesses(PS_3["nu"], PS_3["mean_reversion_coeff"], PS_3["kappa"], V_0, T, 100)
plot(V, title = "Volatility Processes", label = "Volatility Processes", xlabel = "Time", ylabel = "Volatility", legend=false)

S, V = DiscreteTimeApproximation.KahlJackelVectorizedDixit(S_0, 0.07, 50000, 20, PS_3, Δt = 0.0001)
plot(V, title = "Volatility Processes", label = "Volatility Processes", xlabel = "Time", ylabel = "Volatility", legend=false)

# example ternary statement

# test this caluclation here
Y = sqrt(V_0/n) .* ones(N, n)
kappa, mrc = PS_3["kappa"], PS_3["mean_reversion_coeff"]
Y = exp(-mrc / 2) .* ((kappa / 2) .* rand(Normal(0, 1), N, n) .* sqrt(delta_t) .+ Y)
Vhat = sum(Y .^ 2, dims=2)

S_explicit, V_explicit = DiscreteTimeApproximation.explicit_heston(S_0, 0.010201, n, N, 6, T, explicit_params, vol_type = "Simpsons1/3")
S, V, L = DiscreteTimeApproximation.weighted_heston(S_0, 0.010201, n, N, 6, T, explicit_params)
S, V = DiscreteTimeApproximation.KahlJackelVectorizedDixit(S_0, 0.010201, T, N, explicit_params, Δt = 0.00001)

@time S, V, LogL = DiscreteTimeApproximation.weighted_heston_M2(S_0, 0.010201, n, N, T, explicit_params)
nu, kappa = explicit_params["nu"], explicit_params["kappa"]
max(floor(4 * nu / kappa^2 + 0.5), 1)