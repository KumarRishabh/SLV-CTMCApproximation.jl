using LinearAlgebra
using Random
using Distributions
using Plots
using Parameters 
using BenchmarkTools
using StatsBase
include("../src/DiscreteTimeApproximation.jl")
include("../src/OptionPricing.jl")
using .DiscreteTimeApproximation
using .OptionPricing
using Revise
using Profile 
using Printf
using ProfileView
using PrettyTables
@with_kw struct HestonParameters
    S0::Float64 = 100.0
    V0::Float64 = 0.04
    mu::Float64 = -0.5
    nu::Float64 = 0.01
    mean_reversion_coeff::Float64 = 5.3
    rho::Float64 = -0.7
    kappa::Float64 = -0.5
end

@with_kw struct SimulationParameters
    epsilon::Float64 = 10e-06
    nsim::Int64 = 100000
    nstep::Int64 = 50
    seed::Int64 = 91210
    step_size::Float64 = 1 / 50
    M::Int64 = 6
    n_normals::Int64 = 3
    n_calc::Bool = true
    rt::Int64 = 3
    c_branching_cst::Float64 = 1
    q_branching_cst::Float64 = 1.2
    epsilon_dynamic::Float64 = 10^-10
    c_eff_cst::Float64 = 1.2
    c_neff_cst::Float64 = 0.85
    gamma_SA::Int64 = 4
    exponent_SA::Float64 = 1 / 10
    number_of_polynomials::Int64 = 3
    multiple_size_of_sim_matrix::Float64 = 1.5
end

@with_kw struct PayoffParameters
    strike::Float64 = 100.0
    maturity::Float64 = 1.0
end
function Laguerre_Polynomials(n, x)
    L = zeros(n)
    L[1] = 1
    L[2] = 1 - x
    for i in 3:n
        L[i] = ((2*(i-2) + 1 - x) * L[i-1] - (i-2) * L[i-2]) / (i - 1) # Formula changed due to Julia Indexing
    end
    for i in 1:n
        L[i] = exp(-x / 2) * L[i]
    end
    return L
end

# poly1 = Laguerre_Polynomials(3, 1.0)
# poly2 = Laguerre_Polynomials(3, 2.0)
# poly3 = Laguerre_Polynomials(3, 3.0)

function Laguerre_Polynomials_3D(n, x, y, z)
    L = zeros(n^3)
    poly1 = Laguerre_Polynomials(n, x)
    poly2 = Laguerre_Polynomials(n, y)
    poly3 = Laguerre_Polynomials(n, z)
    for i in 1:n
        for j in 1:n
            for k in 1:n
                L[i + (j-1)*n + (k-1)*n*n] = poly1[i] * poly2[j] * poly3[k]
            end
        end
    end
    return L
end

# Laguerre_Polynomials_3D(3, 1.0, 2.0, 3.0)
function SAApproximation(param_Heston, param_simulation, param_payoff, S_matrix, V_matrix, L_matrix)
    # Define the basis functions
    
    N_simulations = size(S_matrix, 2)
    n_time_steps = size(S_matrix, 1)

    println("SA2")

    normalized_strike = param_payoff.strike / param_Heston.S0
    zeta = 0.0
    lambda = 0.0
    n_basis_functions = Int64(param_simulation.number_of_polynomials^3)

    alpha_J = zeros(n_basis_functions, n_time_steps)
    tau_J =  ones(Int64, N_simulations) * n_time_steps
    # tau_J .= n_time_steps - 1

    Z_matrix = zeros(n_time_steps, N_simulations)
    R_matrix = zeros(n_time_steps, N_simulations)

    R_matrix[1, :] .= 1.0

    for t in 2:n_time_steps
        for j in 1:N_simulations
            t_double = t * 1.0
            R_matrix[ t, j] = R_matrix[t-1, j] * (t_double / (t_double + 1.0)) + S_matrix[t, j] / param_Heston.S0 * (1.0 / (t_double + 1.0))
            # println("R_matrix[", t, ", ", j, "] = ", R_matrix[t, j])
        end
    end

    println("SA3")

    for t in 1:n_time_steps
        for j in 1:N_simulations
            Z_matrix[t, j] = exp(-param_Heston.mu * (t - 1) * param_simulation.step_size) * OptionPricing.call_payoff(normalized_strike, R_matrix[t, j])
        end
    end
    S_matrix_header = ["S_matrix $i" for i in 1:10]
    R_matrix_header = ["R_matrix $i" for i in 1:10]
    Z_matrix_header = ["Z_matrix $i" for i in 1:10]
    # println("S_matrix", S_matrix[1:10, 1:10])
    pretty_table(S_matrix[1:10, 1:10], header = S_matrix_header)
    # println("R_matrix", R_matrix[1:10, 1:10])
    pretty_table(R_matrix[1:10, 1:10], header = R_matrix_header)
    # println("Z_matrix", Z_matrix[1:10, 1:10])
    pretty_table(Z_matrix[1:10, 1:10], header = Z_matrix_header)

    println("SA4")

    for t in n_time_steps-1:-1:1
        println(t)
        nb_of_positive_payoffs = 0.0

        eJ_matrix = zeros(n_basis_functions, N_simulations)

        alpha_vec = zeros(n_basis_functions)
        alpha_average_vec = zeros(n_basis_functions)

        nb_of_positive_payoffs = 0.0

        for j in 1:N_simulations
            eJ_matrix[:, j] .= 0.0

            if Z_matrix[t, j] > 0.0
                nb_of_positive_payoffs += 1.0

                eJ_matrix[:, j] = Laguerre_Polynomials_3D(param_simulation.number_of_polynomials, R_matrix[t, j], S_matrix[t, j] / param_Heston.S0, V_matrix[t, j])
                eJ_vector = eJ_matrix[:, j]

                alpha_vec += (param_simulation.gamma_SA * L_matrix[t, j] / nb_of_positive_payoffs^param_simulation.exponent_SA) *
                             (Z_matrix[tau_J[j], j] - dot(eJ_vector, alpha_vec)) * eJ_vector

                alpha_average_vec = (alpha_average_vec * (nb_of_positive_payoffs - 1.0) + alpha_vec) / nb_of_positive_payoffs
            end
        end

        alpha_J[:, t] = alpha_average_vec

        sum_of_sq_error = 0.0

        for j in 1:N_simulations
            if Z_matrix[t, j] > 0.0
                sum_of_sq_error += (Z_matrix[tau_J[j], j] - dot(alpha_J[:, t], eJ_matrix[:, j]))^2.0
            end

            if Z_matrix[t, j] > 0.0 && Z_matrix[t, j] >= dot(alpha_J[:, t], eJ_matrix[:, j])
                tau_J[j] = t
            end
        end

        println(" t = ", t, " sum of sq errors = ", sum_of_sq_error)
    end

    weighted_payoff = 0.0
    weights = 0.0

    for j in 1:N_simulations
        weighted_payoff += L_matrix[tau_J[j], j] * Z_matrix[tau_J[j], j]
        weights += L_matrix[tau_J[j], j]
    end

    price = param_Heston.S0 * weighted_payoff / weights
    return price
end

# Employ a symbolic approach to calculate the Laguerre_Polynomials

param_Heston = HestonParameters(S0 = 100.0, V0 = 0.11, mu = 0.02, nu = 2.65*0.8*0.8/4, mean_reversion_coeff = 6, rho = -0.75, kappa = 0.8)
param_simulation = SimulationParameters(nsim = 1000000, number_of_polynomials = 5, gamma_SA = 1, exponent_SA = 1 / 10)
param_payoff = PayoffParameters()
# Consider the asset price follows a Geometric Brownian Motion
# Employ a Monte Carlo Simulation to calculate the price of a European Call Option
# And then, change the strike price to get 

PS_3 = Dict(
    # "S_0" => 100,
    # "V_0" => 0.501,
    "mu" => param_Heston.mu,
    "nu" => param_Heston.nu,
    "mean_reversion_coeff" => param_Heston.mean_reversion_coeff, # ϱ
    "rho" => param_Heston.rho,
    "kappa" => param_Heston.kappa
)
 
S_matrix, V_matrix, L_matrix, _ = DiscreteTimeApproximation.weighted_heston(param_Heston.S0, param_Heston.V0, param_simulation.n_normals, param_simulation.nsim, param_simulation.M, param_simulation.nstep, PS_3, delta_t = param_simulation.step_size)
L_matrix .= exp.(L_matrix)
@profile SAApproximation(param_Heston, param_simulation, param_payoff, S_matrix, V_matrix, L_matrix)
ProfileView.print()
Laguerre_Polynomials_3D(3, 1.5, 2.0, 3.0) 
size(S_matrix[1, :])