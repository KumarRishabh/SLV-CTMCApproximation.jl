using Random, Statistics, LinearAlgebra

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
        Y .= exp(-mrc / 2) .* ((kappa / 2) * randn(num_particles, n) * sqrt(delta_t) .+ Y)
        Vhat .= sum(Y .^ 2, dims=2)

        logShat .= logS .+ (a * sqrt.(V * delta_t) .* randn(num_particles) .+ b * (Vhat .- V) * delta_t .+ d * (Vhat .- V))
        logLhat .= logL .+ (e * (log.(Vhat ./ V) .+ mrc) .+ f * (1 ./ Vhat .- 1 ./ V) * delta_t)
        logLhat .= log.(exp.(logLhat) ./ sum(exp.(logLhat)))
        A[t] = sum(exp.(logLhat)) / N

        logS, V, logL = [], [], []
        l = 0
        for j in 1:num_particles
            if A[t] / r < exp(logLhat[j]) < r * A[t]
                if (j - l) >= length(logS)
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
                    push!(logShat, logShat[j])
                    push!(Vhat, Vhat[j])
                    push!(logLhat, logLhat[j])
                end
            end
        end

        num_particles_ = num_particles - l
        W = rand(l) ./ l .+ (1:l) ./ l
        U = shuffle(W)
        for j in 1:l
            N_j = Int(exp(logLhat[j]) / A[t]) + (
                if U[j] <= exp(logLhat[j]) / A[t] - Int(exp(logLhat[j]) / A[t])
                    1
                else
                    0
                end
            )
            for k in 1:N_j
                push!(logS, logShat[j])
                push!(V, Vhat[j])
                push!(logL, log(A[t]))
                num_particles_ += 1
            end
        end

        num_particles = num_particles_
        println("Particles in iteration $t: $num_particles")
        logS_history[t] = logS
        V_history[t] = V
        logL_history[t] = logL

        V = Vhat = convert(Array, V)
        logS = logShat = convert(Array, logS)
        logL = logLhat = convert(Array, logL)
        Y = sqrt.(Vhat / n) .* ones(num_particles, n)
    end
    return logS_history, V_history, logL_history
end

T, delta_t, n = 100, 1, 200
S_0, V_0, N, r = 100, 0.04, 1000, 1.5

params = Dict(
    "mu" => -0.04, 
    "nu" => 0.01,
    "mean_reversion_coeff" => 5.3,
    "rho" => -0.7,
    "kappa" => -0.5
)
logS_history, V_history, logL_history = branching_particle_filter(S_0, V_0, N, T, r, params, n, delta_t)

p = plot(1:T+1, logS_history, label="S")
display(p)
function WeightedHestonSimulation()

end



function StochasticApproximation()

end
