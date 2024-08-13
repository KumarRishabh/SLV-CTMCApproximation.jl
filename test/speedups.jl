using Random
using LinearAlgebra

struct ParametersSimulation
    nsim::Int
    c_neff_cst::Float64
    c_eff_cst::Float64
    rt::Float64
end

function effective_branching!(St::Vector{Float64}, Vt::Vector{Float64}, Lt::Vector{Float64},
    Y_sim::Matrix{Float64}, eta::Vector{Float64}, It::Matrix{Int},
    Nt::Int, t::Int, param_simulation::ParametersSimulation, max_nb_sim::Int)

    Nt_old = length(St)
    Nt_new = 0

    # Store simulated values in temporary vectors
    St_hat = copy(St)
    Vt_hat = copy(Vt)
    Lt_hat = copy(Lt)
    It_hat = copy(It)
    eta_hat = copy(eta)
    Yt_hat = copy(Y_sim)

    # Reset the simulation vectors
    St .= zeros(max_nb_sim)
    Vt .= zeros(max_nb_sim)
    Lt .= zeros(max_nb_sim)
    It .= zeros(Int, max_nb_sim, size(It, 2))
    eta .= zeros(max_nb_sim)
    Y_sim .= zeros(size(Yt_hat, 1), max_nb_sim)

    # Calculate average weight At
    At = sum(Lt_hat) / param_simulation.nsim

    sum_Lt_squared = sum(Lt_hat .^ 2)

    # Initialize non-resample count
    non_resample_count = 0

    # Calculate r_t and check each particle for branching
    Nt_eff = (param_simulation.nsim * param_simulation.nsim * At * At) / sum_Lt_squared
    Nt_neff = Nt_old - Nt_eff

    rt_effective = param_simulation.c_neff_cst +
                   (param_simulation.c_eff_cst - param_simulation.c_neff_cst) * Nt_eff / Nt_old

    println("At = $At sum_Lt_squared = $sum_Lt_squared")
    println("Nt_eff = $Nt_eff Nt_neff = $Nt_neff rt = $rt_effective")

    for k in 1:Nt_old
        if (Lt_hat[k] > (param_simulation.rt * At)) || (Lt_hat[k] < (At / param_simulation.rt))
            Lt_hat[k-non_resample_count] = Lt_hat[k]
            St_hat[k-non_resample_count] = St_hat[k]
            Vt_hat[k-non_resample_count] = Vt_hat[k]
            eta_hat[k-non_resample_count] = eta_hat[k]
            Yt_hat[:, k-non_resample_count] .= Yt_hat[:, k]
            It_hat[k-non_resample_count, :] .= It_hat[k, :]
        else
            non_resample_count += 1
            Lt[non_resample_count] = Lt_hat[k]
            St[non_resample_count] = St_hat[k]
            Vt[non_resample_count] = Vt_hat[k]
            eta[non_resample_count] = eta_hat[k]
            Y_sim[:, non_resample_count] .= Yt_hat[:, k]
            It[non_resample_count, :] .= It_hat[k, :]
            It[non_resample_count, t] = non_resample_count
        end
    end

    println("Branching % = ", (Nt_old - non_resample_count) / Nt_old)

    Nt_new = non_resample_count

    # Simulate Nt_old - non_resample_count uniforms 
    simulated_uniforms = rand(Nt_old - non_resample_count)

    for i in 1:(Nt_old-non_resample_count)
        simulated_uniforms[i] = (i - 1 + simulated_uniforms[i]) / (Nt_old - non_resample_count)
    end

    shuffle!(simulated_uniforms)

    for k in non_resample_count+1:Nt_old
        Nkt = floor(Int, Lt_hat[k-non_resample_count] / At) +
              (simulated_uniforms[k-non_resample_count] < (Lt_hat[k-non_resample_count] / At - floor(Lt_hat[k-non_resample_count] / At)))

        for j in 1:Nkt
            Lt[Nt_new+j] = At
            St[Nt_new+j] = St_hat[k-non_resample_count]
            Vt[Nt_new+j] = Vt_hat[k-non_resample_count]
            eta[Nt_new+j] = eta_hat[k-non_resample_count]
            Y_sim[:, Nt_new+j] .= Yt_hat[:, k-non_resample_count]
            It[Nt_new+j, :] .= It_hat[k-non_resample_count, :]
            It[Nt_new+j, t] = Nt_new + j
        end

        Nt_new += Nkt
    end

    Nt = Nt_new
end

# Use effective_branching! function to simulate the 