using Random, Statistics, LinearAlgebra, Plots

function branching_particle_filter(S_0, V_0, N, T, r, params, n, delta_t)
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
        Y = exp(-mrc / 2) .* ((kappa / 2) .* randn(num_particles, n) .* sqrt(delta_t) .+ Y)
        Vhat = sum(Y .^ 2, dims=2)

        logShat = logS .+ (a .* sqrt.(V .* delta_t) .* randn(num_particles) .+ b * (Vhat .- V) .* delta_t .+ d .* (Vhat .- V))
        println("At iteration $t: $Vhat")
        logLhat = logL .+ (e .* (log.(Vhat ./ V) .+ mrc) .+ f * (1 ./ Vhat .- 1 ./ V) .* delta_t)
        # println("At iteration $t: $logShat")
        
        logLhat = log.(exp.(logLhat) ./ sum(exp.(logLhat)))
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
        W = rand(l) ./ l .+ (0:l - 1) ./ l
        U = shuffle(W)
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

T, delta_t, n = 100, 1, 200
S_0, V_0, N, r = 100, 0.04, 10, 1.5

params = Dict(
    "mu" => -0.04, 
    "nu" => 0.01,
    "mean_reversion_coeff" => 5.3,
    "rho" => -0.7,
    "kappa" => -0.5
)
@time logS_history, V_history, logL_history = branching_particle_filter(S_0, V_0, N, T, r, params, n, delta_t)
print(logL_history)
# plot the logS_history with respect to time for each particle in the simulation
function weighted_heston(S_0, V_0, n, N, M, T, params; epsilon=1e-3)
    # 
    # Theorem 1 computations
    # 
    mu, nu, mrc, rho, kappa = params["mu"], params["nu"], params["mean_reversion_coeff"], params["rho"], params["kappa"]
    nu_k = max(Int(floor(4 * nu / kappa^2 + 0.5)), 1)
    a, b, c, d, e = sqrt(1 - rho^2), mu - nu * rho / kappa, rho * mrc / kappa - 0.5, rho / kappa, (nu - nu_k) / kappa^2
    sigma = kappa * sqrt((1 - exp(-mrc / M)) / (4 * mrc))
    alpha = exp(-mrc / (2 * M))
    logS = zeros(Float64, T + 1, N)
    logL = zeros(Float64, T + 1, N)
    stoppingtimes = zeros(Int, N)
    Y = zeros(Float64, M * T + 1, N, n)
    V = zeros(Float64, M * T + 1, N)
    
    # '''Initialization'''
    logS[1, :] = log(S_0 * ones(N))
    logL[1, :] = 0 
    stoppingtimes .= T
    Y[1, :, :] = (sqrt(V_0 / n)) * ones((N, n))
    
    for t in 2:T+1
        for k in M:-1:0
            Z = randn(N, n)
            Y[t * M - k, :, :] = alpha * Y[t * M - (k+1), :, :] + sigma * Z
            V[t * M - k, :] = V[t * M - k, :] + sum(Y[t * M - k, :, :].^2, dims=2)
        end
        
        IntV = (sum(V[(t-1)*M+1:t*M+1, :], dims=1) + sum(V[(t-1)*M+2:t*M, :], dims=1) + 2 * (V[(t-1)*M+2, :] + V[t*M, :])) / (3 * M)
        Nsample = a * sqrt(IntV) .* randn(N)
        logS[t, :] = logS[t-1, :] + Nsample + b + c * IntV + d * (V[t * M, :] - V[(t-1) * M, :])
        
        # '''Iterating over particles'''
        for j in 1:N
            if t <= stoppingtimes[j]
                if minimum(V[(t-1)*M+1:t*M, j]) > epsilon
                    IntVinv = (sum(1 ./ V[(t-1)*M+1:t*M+1, j]) + sum(1 ./ V[(t-1)*M+2:t*M, j]) + 2 * (1 / V[(t-1)*M+2, j] + 1 / V[t*M, j])) / (3 * M)
                    logL[t, j] = logL[t-1, j] + e * (log(V[t*M, j] / V[(t-1)*M, j]) + mrc) + f * IntVinv
                else
                    stoppingtimes[j] = t - 1
                end
            end
        end
    end
    
    return (logS, V, logL)
end

weighted_heston(S_0, V_0, n, N, 200, T, params)
# logS_history[1]


function StochasticApproximation()
    # Implement the stochastic approximation algorithm for pricing American Call options
end

