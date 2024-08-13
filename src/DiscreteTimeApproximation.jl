

module DiscreteTimeApproximation
using Random, Statistics, LinearAlgebra, Plots, PyCall, PrettyTables
using Distributions



function branching_particle_filter(S_0, V_0, N, T, r, params, n; delta_t = 1)
    mu, nu, mrc, rho, kappa = params["mu"], params["nu"], params["mean_reversion_coeff"], params["rho"], params["kappa"]

    nu_k = max(round(Int, 4 * nu / kappa^2 + 0.5), 1)
    a, b, c, d, e = sqrt(1 - rho^2), mu - nu * rho / kappa, rho * mrc / kappa - 0.5, rho / kappa, (nu - nu_k) / kappa^2
    f = e * (kappa^2 - nu - nu_k) / 2

    V_history = [[] for _ in 1:T+1]
    logS_history = [[] for _ in 1:T+1]
    logL_history = [[] for _ in 1:T+1]
    num_particles = N

    Vhat = fill(V_0, num_particles)
    logShat = log.(fill(S_0, num_particles))
    logLhat = zeros(num_particles)

    A = zeros(T + 1)

    V = V_history[1] = Vhat
    logS = logS_history[1] = logShat
    logL = logL_history[1] = logLhat

    Y = sqrt(V_0 / n) * ones(num_particles, n)
    for t in 2:T+1
        num_particles_ = 0
        # Y = exp(-mrc / 2) * ((kappa / 2) * randn(num_particles, n) * sqrt(delta_t) + Y)
        Y = -mrc/2 .* Y .* delta_t .+ kappa/2 .* randn(num_particles, n) .* sqrt(delta_t) .+ Y
        Vhat = sum(Y.^2, dims=2)

        logShat = [logS[i] + (a * sqrt(V[i] * delta_t) * randn() + b + c* (Vhat[i] - V[i]) * delta_t + d * (Vhat[i] - V[i])) for i in 1:num_particles]
        logLhat = [logL[i] + (e * (log(Vhat[i] / V[i]) + mrc) + f * (1 / Vhat[i] - 1 / V[i]) * delta_t) for i in 1:num_particles]
        logLhat = log.(exp.(logLhat) ./ sum(exp.(logLhat)))
        # convert Vhat to a vector 
        Vhat = reshape(Vhat, num_particles)
        A[t] = sum(exp.(logLhat)) / N

        logS, V, logL = [], [], []
        l = 0
        for j in 1:num_particles
            if A[t] / r < exp(logLhat[j]) < r * A[t]
                if (j - l) > length(logS)
                    push!(logS, logShat[j])
                    push!(V, Vhat[j])
                    push!(logL, logLhat[j])
                else
                    logS[j - l], V[j - l], logL[j - l] = logShat[j], Vhat[j], logLhat[j]
                end
            else
                l += 1
                if l > length(logS)
                    push!(logShat, logShat[j])
                    push!(Vhat, Vhat[j])
                    push!(logLhat, logLhat[j])
                else
                    logShat[l], Vhat[l], logLhat[l] = logShat[j], Vhat[j], logLhat[j]
                end
            end
        end
        num_particles_ = num_particles - l
        W = rand(l) / l
        W = [W[i] + i / l for i in 1:l]
        U = randperm(l)
        for j in 1:l
            N_j = floor(exp(logLhat[j]) / A[t]) + (U[j] <= exp(logLhat[j]) / A[t] - floor(exp(logLhat[j]) / A[t]) ? 1 : 0)
            for k in 1:N_j
                push!(logS, logShat[j])
                push!(V, Vhat[j])
                push!(logL, log(A[t]))
            end
            num_particles_ += N_j
        end
        num_particles = Int(num_particles_)
        println("Particles: ", num_particles)
        logS_history[t] = logS
        V_history[t] = V
        logL_history[t] = logL
        V = Vhat = convert(Vector{Float64}, V)
        logS = logShat = convert(Vector{Float64}, logS)
        logL = logLhat = convert(Vector{Float64}, logL)
        Y = sqrt.(Vhat / n) .* ones(num_particles, n)
    end
    return (logS_history, V_history, logL_history)
end

# mean_reversion_coefficient -> ϱ

function weighted_heston(S_0, V_0, n, N, M, T, params; epsilon=1e-3) # Does it work? 
    # 
    # Theorem 1 computations
    # 
    # Here M is the grid size for the time discretization
    # N is the number of particles
    # n is the number of factors
    mu, nu, mrc, rho, kappa = params["mu"], params["nu"], params["mean_reversion_coeff"], params["rho"], params["kappa"]
    nu_k = max(floor(4 * nu / kappa^2 + 0.5), 1)
    a, b, c, d, e = sqrt(1 - rho^2), mu - nu * rho / kappa, rho * mrc / kappa - 0.5, rho / kappa, (nu - nu_k)/kappa^2
    f = e * (kappa^2 - nu - nu_k) / 2
    sigma = kappa * sqrt((1 - exp(-mrc / M)) / (4 * mrc))
    alpha = exp(-mrc / (2 * M))
    # println("Sigma: $sigma, Alpha: $alpha")
    logS = zeros(Float64, T + 1, N) # logS_0
    logL = zeros(Float64, T + 1, N) # logL_0
    stoppingtimes = zeros(Int, N)
    Y = zeros(Float64, M * (T) + 1, N, n)
    V = zeros(Float64, M * (T) + 1, N)
    
    # '''Initialization'''
    logS[1, :] = log.(S_0 * ones(N))
    logL[1, :] .= 0
    V[1, :] .= V_0 * ones(N)
    stoppingtimes .= T +1
    Y[1, :, :] = (sqrt(V_0 / n)) * ones(N, n)
    
    for t in 2:T+1
        for k in M-1:-1:0
            Z = rand(Normal(0, 1), N, n)
            @. Y[(t - 1)* M - k + 1, :, :] = alpha * Y[(t - 1) * M - k, :, :] + sigma * Z
            V[(t - 1) * M - k + 1, :] = V[(t - 1) * M - k + 1, :] + sum(Y[(t - 1)* M - k + 1, :, :].^2, dims=2)
        end
        
        IntV = (4 * reshape(sum(V[((t-2)*M+2):2:((t - 1)* M), :], dims=1), N) + 2 * reshape(sum(V[((t - 2) * M + 3):2:((t - 1)* M), :], dims=1), N)  +  V[(t-2)*M + 1, :] + V[(t - 1)*M + 1, :]) ./ (3 * M)
    
        sqrt_IntV = sqrt.(IntV)
        V_diff = V[(t-1)*M+1, :] .- V[(t-2)*M+1, :]

        # Generate Nsample
        Nsample = a .* sqrt_IntV .* rand(Normal(0, 1), N)

        # Update logS
        @. logS[t, :] = logS[t-1, :] + Nsample + b + c * IntV + d * V_diff
        
        # '''Iterating over particles'''
        for j in 1:N
            if t <= stoppingtimes[j]
                if minimum(V[(t-2)*M+2:(t-1)*M+1, j]) > epsilon
                    IntVinv = (4 * sum(1 ./ V[(t-2)*M+2:2:(t-1)*M, j]) + 2 * sum(1 ./ V[(t-2)*M+3:2:(t-1)*M, j]) + 1 ./ V[(t-2)*M+1, j] + 1 ./ V[(t-1)*M+1, j]) / (3 * M)   
                    logL[t, j] = logL[t-1, j] + e * (log(V[(t - 1)*M + 1, j] / V[(t-2)*M + 1, j]) + mrc) + f* IntVinv
                else
                    stoppingtimes[j] = t - 1
                end
            end
        end
    end
    V_history = zeros(T + 1, N) 
    for i in 1:N
        V_history[:, i] .= V[1:M:((T)*M + 1), i]
    end
    # V_history = [V[1:M:((T)*M + 1), i] for i in 1:N].transpose()
    return (exp.(logS), V_history, logL)
end  

function integerCondition(ν, κ; epsilon=1e-6)
    # First calculate the max difference between the bins 
    return 4*ν/κ^2 ≈ round(4*ν/κ^2)
end
function explicit_heston(S_0, V_0, n, num_simulations, M, T, params; epsilon=1e-3, vol_type = "Trapezoidal")
    # check if the condition is satisfied
    mu, nu, mrc, rho, kappa = params["mu"], params["nu"], params["mean_reversion_coeff"], params["rho"], params["kappa"]'
    a, b, c, d, e, f= sqrt(1 - rho^2), mu - nu * rho / kappa, rho * mrc / kappa - 0.5, rho / kappa, 0, 0
    sigma = kappa * sqrt((1 - exp(-mrc / M)) / (4 * mrc))
    alpha = exp(-mrc / (2 * M))
    if integerCondition(params["nu"], params["kappa"]) == false
        error("The condition (C) is not satisfied")
    else
        println("The condition (C) is satisfied")
    end
    # T = time_limit/Δt # Number of time steps
    logS = zeros(Float64, T + 1, num_simulations)
    V = zeros(Float64, M*T + 1, num_simulations)

    logS[1, :] .= log(S_0) .* ones(num_simulations)
    # Simulate an OU process as Y
    Y = zeros(Float64, M*T + 1, num_simulations, n)
    Y[1, :, :] .= sqrt(V_0 / n) * ones(num_simulations, n)
    V[1, :] .= V_0 * ones(num_simulations)

    for i in 2:T+1
        for j in M-1:-1:0
            Z = rand(Normal(0, 1), num_simulations, n)
            @. Y[(i-1)*M-j+1, :, :] = alpha * Y[(i-1)*M-j, :, :] + sigma * Z
            V[(i-1)*M - j + 1, :] = V[(i-1)*M - j + 1, :] + sum(Y[(i-1)*M - j + 1, :, :].^2, dims=2)
        end
        if vol_type == "Trapezoidal"
            IntV = (2*reshape(sum(V[(i-2)*M+2:(i-1)*M, :], dims=1), num_simulations) .+ V[(i-1)*M+1, :] .+ V[(i-2)*M+1, :]) / (2*M)
        elseif vol_type == "Simpsons1/3"
            IntV = (4*reshape(sum(V[(i-2)*M+2:2:(i-1)*M, :], dims=1), num_simulations) .+ 2*reshape(sum(V[(i-2)*M+3:2:(i-1)*M, :], dims=1), num_simulations) .+ V[(i-1)*M+1, :] .+ V[(i-2)*M+1, :]) / (3*M)
        elseif vol_type == "Simpsons3/8"
            IntV = (3*reshape(sum(V[(i-2)*M+2:3:(i-1)*M, :], dims=1), num_simulations) .+ 3*reshape(sum(V[(i-2)*M+3:3:(i-1)*M, :], dims=1), num_simulations) .+ 2*reshape(sum(V[(i-2)*M+4:3:(i-1)*M, :], dims=1), num_simulations) .+ V[(i-1)*M+1, :] .+ V[(i-2)*M+1, :]) / (8*M/3)
        else
            error("Invalid vol_type")
        end
        sqrt_IntV = sqrt.(IntV)
        V_diff = V[(i-1)*M+1, :] .- V[(i-2)*M+1, :]
        Nsample = sqrt.(a .* sqrt_IntV) .* rand(Normal(0, 1), num_simulations)
        @. logS[i, :] = logS[i-1, :] + Nsample + b + c * IntV + d * V_diff
    end
    V_history = zeros(T + 1, num_simulations)
    V_history .= V[1:M:((T)*M + 1), :]
    return exp.(logS), V_history
end

# S_0, V_0, n, N, T = 100, 0.501, 20, 10, 10
# S_history, V_history, logL_history = weighted_heston_new_indexing(S_0, V_0, n, N, 6, T, PS_1)

function StochasticApproximation(basis_functions, N, T; χ = 2.0, γ = 2)
    λ, ζ = 0.0, 0  
    # Implement the stochastic approximation algorithm for pricing American Call options
end


# Kahl, C., & Jäckel, P. (2006). Fast strong approximation Monte Carlo schemes for stochastic volatility models. Quantitative Finance, 6(6), 513–536. https://doi.org/10.1080/14697680600841108
function HestonDiscretizationKahlJackel(S0, V0, T, N, params, Δt = 1e-3)
    # Implement the Heston Discretization using the Kahl-Jackel method
    V = zeros(Float64, N, T+1)
    logS = zeros(Float64, N, T+1)
    V[:, 1] .= V0
    logS[:, 1] .= log(S0)
    # Implement the Heston Discretization using the Kahl-Jackel method
    for i in 1:N
        for j in 2:T+1
            Δβ = sqrt(Δt) * randn()
            ΔB = sqrt(Δt) * randn() # independent brownian motions
            V[i, j] = V[i, j-1] + (params["nu"] - params["mean_reversion_coeff"] * V[i, j-1]) * Δt + params["kappa"] * sqrt(V[i, j-1]) * Δβ + 0.25 * params["kappa"]^2 * (Δβ^2 - Δt)
            logS[i, j] = logS[i, j-1] + params["mu"] * Δt - 0.25 * (V[i, j-1] + V[i, j]) * Δt + params["rho"] * sqrt(V[i, j-1]) * Δβ + 0.5 * (sqrt(V[i, j-1]) + sqrt(V[i, j]))*(ΔB - params["rho"] * Δβ) + 0.25 * params["rho"] *(Δβ - Δt)
        end
    end
    return exp.(logS), V
end

function KahlJackelVectorized(S0, V0, T, N, params, Δt = 1e-6)
    V = zeros(Float64, N, T+1)
    logS = zeros(Float64, N, T+1)
    V[:, 1] .= V0
    logS[:, 1] .= log(S0)
    Δβ = sqrt(Δt) * randn(N, T)
    ΔB = sqrt(Δt) * randn(N, T)
    for j in 2:T+1
        V[:, j] = V[:, j-1] + (params["nu"] .- params["mean_reversion_coeff"] .* V[:, j-1]) * Δt + params["kappa"] .* sqrt.(V[:, j-1]) .* Δβ[:, j-1] + 0.25 * params["kappa"]^2 .* (Δβ[:, j-1].^2 .- Δt)
        logS[:, j] = logS[:, j-1] .+ params["mu"] * Δt .- 0.25 .* (V[:, j-1] + V[:, j]) .* Δt + params["rho"] .* sqrt.(V[:, j-1]) .* Δβ[:, j - 1] .+ 0.5 .* (sqrt.(V[:, j-1]) + sqrt.(V[:, j])) .* (ΔB[:, j-1] - params["rho"] .* Δβ[:, j - 1]) .+ 0.25 .* params["rho"] .* (Δβ[:, j - 1] .- Δt)
    end
    return exp.(logS), V
end

function KahlJackelVectorizedDixit(S0, V0, T, N, params; Δt = 1e-6)
    V = zeros(Float64, N, T+1)
    logS = zeros(Float64, N, T+1)
    V[:, 1] .= V0
    logS[:, 1] .= log(S0)
    Δβ = sqrt(Δt) * randn(N, T)
    ΔB = sqrt(Δt) * randn(N, T)
    for j in 2:T+1
        prevV = V[:, j-1]
        prevβ = Δβ[:, j-1]
        @. @view(V[:, j]) = prevV + (params["nu"] - params["mean_reversion_coeff"] * prevV) * Δt + params["kappa"] * sqrt(prevV) * prevβ + 0.25 * params["kappa"]^2 * (prevβ^2 - Δt)
        @. @view(logS[:, j]) = logS[:, j-1] + params["mu"] * Δt - 0.25 * (prevV + V[:, j]) * Δt + params["rho"] * sqrt(prevV) *  + 0.5 * (sqrt(prevV) + sqrt(V[:, j])) * (ΔB[:, j-1] - params["rho"] * prevβ) + 0.25 * params["rho"] * (prevβ - Δt)
    end
    return exp.(logS), V
end

function BroadieKayaHestonSimulation(S0, V0, T, N, params, Δt=1e-6) # TODO: Correct this to the Actual Broadie Kaya simulation where the volatility is simulated through a Non-Central Chi-Squared distribution
    V = zeros(Float64, N, T + 1)
    logS = zeros(Float64, N, T + 1)
    V[:, 1] .= V0
    logS[:, 1] .= log(S0)
    Δβ = sqrt(Δt) * randn(N, T)
    ΔB = sqrt(Δt) * randn(N, T)
    for j in 2:T+1
        prevV = V[:, j-1]
        prevβ = Δβ[:, j-1]
        @. @view(V[:, j]) = prevV + (params["nu"] - params["mean_reversion_coeff"] * prevV) * Δt + params["kappa"] * sqrt(prevV) * prevβ + 0.25 * params["kappa"]^2 * (prevβ^2 - Δt)
        @. @view(logS[:, j]) = logS[:, j-1] + params["mu"] * Δt - 0.25 * (prevV + V[:, j]) * Δt + params["rho"] * sqrt(prevV) * +0.5 * (sqrt(prevV) + sqrt(V[:, j])) * (ΔB[:, j-1] - params["rho"] * prevβ) + 0.25 * params["rho"] * (prevβ - Δt)
    end
    return exp.(logS), V
end
end
# S0, V0, T, N = 100, 0.11, 1000, 100
  

# @time S_1, V_1 = HestonDiscretizationKahlJackel(S0, V0, T, N, params)
# @time S_2, V_2 = KahlJackelVectorized(S0, V0, T, N, params)
# @time S_3, V_3 = KahlJackelVectorizedDixit(S0, V0, T, N, params)
# plot(1:T+1, S_1', xlabel = "Time", ylabel = "Stock Prices", title = "Stock Prices vs Time")
# plot(1:T+1, V_1', xlabel = "Time", ylabel = "Volatilities", title = "Volatilities vs Time")


# 2.000000004 ≈ 2.0