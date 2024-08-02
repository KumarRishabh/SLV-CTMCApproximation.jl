using Random, Statistics, LinearAlgebra, Plots, PyCall, PrettyTables
np = pyimport("numpy")
function branching_particle_filter(S_0, V_0, N, T, r, params, n, delta_t; seed = 42)
    np.random.seed(seed) # set the seed for reproducibility with PyCall & numpy
    mu, nu, mrc, rho, kappa = params["mu"], params["nu"], params["mean_reversion_coeff"], params["rho"], params["kappa"]

    nu_k = max(round(4 * nu / kappa^2 + 0.5), 1)
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
    logL = logL_history[1] = zeros(num_particles)

    Y = sqrt(V_0 / n) * ones(num_particles, n)
    for t in 2:T+1
        num_particles_ = 0
        Y = exp(-mrc / 2) .* ((kappa / 2) .* np.random.randn(num_particles, n) .* sqrt(delta_t) .+ Y)
        Vhat = sum(Y .^ 2, dims=2)

        logShat = logS .+ (a .* sqrt.(V .* delta_t) .* np.random.randn(num_particles) .+ b * (Vhat .- V) .* delta_t .+ d .* (Vhat .- V))
        logLhat = logL .+ (e .* (log.(Vhat ./ V) .+ mrc) .+ f * (1 ./ Vhat .- 1 ./ V) .* delta_t)
        # println("At iteration $t: $logShat")
        logLhat = log.(exp.(logLhat) ./ sum(exp.(logLhat)))
        
        # println("At iteration $t: LogLHat is $logLhat")
        # println("At iteration $t: $logLhat")
        # print the shape of logLhat
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
                if l <= length(logS)
                    logShat[l], Vhat[l], logLhat[l] = logShat[j], Vhat[j], logLhat[j]
                else
                    logShat = vcat(logShat, logShat[j]) # since logShat is a matrix, we need to append a row to it rather than treat it as a vector
                    # since logShat is a matrix, we need to append a row to it rather than treat it as a vector
                    Vhat = vcat(Vhat, Vhat[j])
                    logLhat = vcat(logLhat, logLhat[j])
                end
            end
        end

        num_particles_ = num_particles - l
        W = np.random.rand(l) / (l)
        W = np.array([W[i] + i / (l) for i in 1:l]) # Stratified Uniform samples
        U = np.random.permutation(W)
        for j in 1:l
            N_j = round(exp(logLhat[j]) / A[t]) + (
                if U[j] <= exp(logLhat[j]) / A[t] - round(exp(logLhat[j]) / A[t])
                    1
                else
                    0
                end
            )
            for k in 1:N_j
                push!(logS, logShat[j])
                push!(V, Vhat[j])
                push!(logL, log(A[t]))
                # println("Particle $k in iteration $t: $N_j")
                num_particles_ += 1
            end
            # println("Particles in iteration $t: $num_particles_")
        end

        num_particles = copy(num_particles_)
        println("Particles in iteration $t: $num_particles")
        logS_history[t] = copy(logS)
        V_history[t] = copy(V)
        logL_history[t] = copy(logL)

        V = Vhat = convert(Array, V)
        logS = logShat = convert(Array, logS)
        logL = logLhat = convert(Array, logL)
        Y = sqrt.(Vhat / n) .* ones(num_particles, n)
    end
    return logS_history, V_history, logL_history
end

T, delta_t, n = 200, 1, 20
S_0, V_0, N, r = 100, 0.04, 200, 1.4

params = Dict(
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
    "nu" => 0.085, 
    "mean_reversion_coeff" => 6.21, 
    "rho" => -0.7, 
    "kappa" => 0.2
)
@time logS_history, V_history, logL_history = branching_particle_filter(S_0, V_0, N, T, r, params, n, delta_t)
output_file = "/Users/rishabhkumar/SLV-CTMCApproximation.jl/src/output.txt"
open(output_file, "w") do file
    for i in 1:T+1
        println(file, "At iteration $i: LogStockPrices: $(logS_history[i])")
        println(file, "At iteration $i: Volatilities: $(V_history[i])")
        println(file, "At iteration $i: LogLikelihoods: $(logL_history[i])")
    end
end
# plot the logS_history with respect to time for each particle in the simulation
function weighted_heston(S_0, V_0, n, N, M, T, params; epsilon=1e-3)
    # 
    # Theorem 1 computations
    # 
    # Here M is the grid size for the time discretization
    # N is the number of particles
    # n is the number of factors
    mu, nu, mrc, rho, kappa = params["mu"], params["nu"], params["mean_reversion_coeff"], params["rho"], params["kappa"]
    nu_k = max(Int(floor(4 * nu / kappa^2 + 0.5)), 1)
    a, b, c, d, e = sqrt(1 - rho^2), mu - nu * rho / kappa, rho * mrc / kappa - 0.5, rho / kappa, (nu - nu_k) / kappa^2
    f = e * (kappa^2 - nu - nu_k) / 2
    sigma = kappa * sqrt((1 - exp(-mrc / M)) / (4 * mrc))
    alpha = exp(-mrc / (2 * M))
    logS = zeros(Float64, T + 1, N)
    logL = zeros(Float64, T + 1, N)
    stoppingtimes = zeros(Int, N)
    Y = zeros(Float64, M * (T) + 1, N, n)
    V = zeros(Float64, M * (T) + 1, N)
    
    # '''Initialization'''
    logS[1, :] = log.(S_0 * ones(N))
    logL[1, :] .= 0
    V[1, :] .= V_0 
    stoppingtimes .= T +1
    Y[1, :, :] = (sqrt(V_0 / n)) * ones((N, n))
    
    for t in 2:T+1
        for k in M-1:-1:0
            Z = randn(N, n)
            Y[(t - 1)* M - k + 1, :, :] = alpha * Y[(t - 1) * M - k, :, :] + sigma * Z
            V[(t - 1) * M - k + 1, :] = V[(t - 1) * M - k + 1, :] + sum(Y[(t - 1)* M - k + 1, :, :].^2, dims=2)
        end
        
        IntV = (reshape(sum(V[((t-2)*M+1):((t - 1)* M + 1), :], dims=1), N) + reshape(sum(V[((t - 2) * M + 2):( (t - 1)* M), :], dims=1), N) + 2 * (V[((t - 2)* M + 2 ), :] + V[(t - 1) * M, :])) ./ (3 * M)
        # IntV = (sum(V[(t-1)*M+1:t*M+1, :], dims=1) + sum(V[(t-1)*M+2:t*M, :], dims=1) + 2 * (V[(t-1)*M+2, :] + V[t * M, :])) ./ (3 * M)
        # println("IntV: $IntV")
        Nsample = a .* sqrt.(IntV) .* randn(N)
        logS[t, :] = logS[t-1, :] .+ Nsample .+ b .+ c * IntV .+ d .* (V[(t - 1) * M + 1, :] .- V[(t-2) * M + 1, :])
        
        # '''Iterating over particles'''
        for j in 1:N
            if t <= stoppingtimes[j]
                if minimum(V[(t-2)*M+1:(t-1)*M, j]) > epsilon
                    IntVinv = (sum(1 ./ V[(t-2)*M+1:(t-1)*M+1, j]) + sum(1 ./ V[(t-2)*M+2:(t - 1)*M, j]) + 2 * (1 / V[(t-2)*M+2, j] + 1 / V[(t - 1)*M, j])) / (3 * M)
                    logL[t, j] = logL[t-1, j] + e * (log(V[(t - 1)*M + 1, j] / V[(t-2)*M + 1, j]) + mrc) + f* IntVinv
                else
                    stoppingtimes[j] = t - 1
                end
            end
        end
    end
    V_history = zeros(T + 1, N) 
    V_history .= V[1:M:((T)*M + 1), :]
    return (exp.(logS), V_history, logL)
end


function weighted_heston_new_indexing(S_0, V_0, n, N, M, T, params; epsilon=1e-3)
    # 
    # Theorem 1 computations
    # 
    # Here M is the grid size for the time discretization
    # N is the number of particles
    # n is the number of factors
    mu, nu, mrc, rho, kappa = params["mu"], params["nu"], params["mean_reversion_coeff"], params["rho"], params["kappa"]
    nu_k = max(Int(floor(4 * nu / kappa^2 + 0.5)), 1)
    a, b, c, d, e = sqrt(1 - rho^2), mu - nu * rho / kappa, rho * mrc / kappa - 0.5, rho / kappa, (nu - nu_k) / kappa^2
    f = e * (kappa^2 - nu - nu_k) / 2
    sigma = kappa * sqrt((1 - exp(-mrc / M)) / (4 * mrc))
    alpha = exp(-mrc / (2 * M))
    logS = zeros(Float64, N, T + 1)
    logL = zeros(Float64, N, T + 1)
    stoppingtimes = zeros(Int, N)
    Y = zeros(Float64, N, n, M * (T) + 1)
    V = zeros(Float64, N, M * (T) + 1)

    # '''Initialization'''
    logS[:, 1] = log.(S_0 * ones(N))
    logL[:, 1] .= 0
    V[:, 1] .= V_0
    stoppingtimes .= T + 1
    Y[:, :, 1] = (sqrt(V_0 / n)) * ones((N, n))

    for t in 2:T+1
        for k in M-1:-1:0
            Z = randn(N, n)
            Y[:, :, (t-1)*M-k+1] = alpha * Y[:, :, (t-1)*M-k] + sigma * Z
            V[:, (t-1)*M-k+1] = V[:, (t-1)*M-k+1] + sum(Y[:, :, (t-1)*M-k+1] .^ 2, dims=2)
        end

        IntV = (reshape(sum(V[:, ((t-2)*M+1):((t-1)*M+1)], dims=2), N) + reshape(sum(V[:, ((t-2)*M+2):((t-1)*M)], dims=2), N) + 2 * (V[:, ((t-2)*M+2)] + V[:, (t-1)*M])) ./ (3 * M)
        # IntV = (sum(V[:, (t-1)*M+1:t*M+1], dims=2) + sum(V[:, (t-1)*M+2:t*M], dims=2) + 2 * (V[:, (t-1)*M+2] + V[:, t*M])) ./ (3 * M)
        # println("IntV: $IntV")
        Nsample = a .* sqrt.(IntV) .* randn(N)
        logS[:, t] = logS[:, t-1] .+ Nsample .+ b .+ c * IntV .+ d .* (V[:, (t-1)*M+1] .- V[:, (t-2)*M+1])

        # '''Iterating over particles'''
        for j in 1:N
            if t <= stoppingtimes[j]
                if minimum(V[j, (t-2)*M+1:(t-1)*M]) > epsilon
                    IntVinv = (sum(1 ./ V[j, (t-2)*M+1:(t-1)*M+1]) + sum(1 ./ V[j, (t-2)*M+2:(t-1)*M]) + 2 * (1 / V[j, (t-2)*M+2] + 1 / V[j, (t-1)*M])) / (3 * M)
                    logL[j, t] = logL[j, t-1] + e * (log(V[j, (t-1)*M+1] / V[j, (t-2)*M+1]) + mrc) + f * IntVinv
                else
                    stoppingtimes[j] = t - 1
                end
            end
        end
    end
    V_history = zeros(N, T + 1)
    V_history .= V[:, 1:M:((T)*M+1)]
    return (exp.(logS), V_history, logL)
end

S_0, V_0, n, N, T = 100, 0.501, 20, 1000, 10
S_history, V_history, logL_history = weighted_heston_new_indexing(S_0, V_0, n, N, 6, T, PS_1)
# for i in 1:T+1
#     println(logS_history[i, :])
# end

# for i in 1:T+1
#     println(V_history[i, :])
# end

# for i in 1:T+1
#     println(logL_history[i, :])
# end

print(S_history[:, 1]')
p =plot(1:T+1, S_history', xlabel = "Time", ylabel = "Log Stock Prices", title = "Log Stock Prices vs Time")
plot(1:T+1, V_history', xlabel = "Time", ylabel = "Volatilities", title = "Volatilities vs Time")
display(p)
# logS_history[1]


function StochasticApproximation(basis_functions, N, T; χ = 2.0, γ = 2)
    λ = 0.0, ζ = 0  
    # Implement the stochastic approximation algorithm for pricing American Call options
end


# Example values
T = 10
M = 2
V = rand(22, 5)  # Random matrix with 22 rows and 5 columns

# Create V_history with appropriate dimensions
V_history = zeros(Float64, div((T+1)*M + 1, M), size(V, 2))
# Perform the operation
V_history .= V[1:M:((T)*M + 1), :]

# Print the result
# println(V_history)
# pretty print V_history
pretty_table(V_history)

# Kahl, C., & Jäckel, P. (2006). Fast strong approximation Monte Carlo schemes for stochastic volatility models. Quantitative Finance, 6(6), 513–536. https://doi.org/10.1080/14697680600841108
function HestonDiscretizationKahlJackel(S0, V0, T, N, params, Δt = 1e-6)
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
            Δβ = 0.04
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

function KahlJackelVectorizedDixit(S0, V0, T, N, params, Δt = 1e-6)
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
S0, V0, T, N = 100, 0.501, 100000, 100
  
@time S_1, V_1 = HestonDiscretizationKahlJackel(S0, V0, T, N, params)
@time S_2, V_2 = KahlJackelVectorized(S0, V0, T, N, params)
@time S_3, V_3 = KahlJackelVectorizedDixit(S0, V0, T, N, params)
plot(1:T+1, S_2', xlabel = "Time", ylabel = "Stock Prices", title = "Stock Prices vs Time")
