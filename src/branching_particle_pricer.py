import numpy as np
import matplotlib.pyplot as plt
import numpy.random as npr
import seaborn as sns

def branching_particle_filter(S_0, V_0, N, T, r):
    '''
    Theorem-1 Computations
    '''
    mu, nu, mrc, rho, kappa = params['mu'], params['nu'], params['mean_reversion_coeff'], params['rho'], params['kappa']
    '''
    num_particles => N_{t-1}
    num_particles_ => N_{t}
    '''
    ''' Parameters '''
    # nu_k = max(int(4*nu/kappa**2 + 0.5), 1)
    # int_candidate = max(floor(4*nu/kappa**2 + 0.5), 1)
    int_candidate = max(int(4*nu/kappa**2 + 0.5), 1)
    nu_k = int_candidate * kappa**2/4.0
    a, b, c, d, e = np.sqrt(1 - rho**2), mu - nu*rho/kappa, rho*mrc/kappa - 0.5, rho/kappa, (nu - nu_k)/kappa**2 
    f = e*(kappa**2 - nu - nu_k)/2
    '''Declaration of Variables'''
#     V = np.zeros((T+1, N)) 
#     logS = np.zeros((T+1, N)) 
#     logL = np.zeros((T+1, N)) 
#     Vhat = np.zeros((T+1, N)) 
#     logShat = np.zeros((T+1, N)) 
#     logLhat = np.zeros((T+1, N)) 
    
    V_history = [[] for i in range(T+1)]
    logS_history = [[] for i in range(T+1)]
    logL_history = [[] for i in range(T+1)]
    num_particles = N
    '''Initializing Variables'''
    Vhat = V_0 * np.ones(num_particles) # V[0] = V_0
    logShat = np.log(S_0 * np.ones(num_particles)) 
    logLhat = np.zeros(num_particles) #L[0] = 1
    
    A = np.zeros(T+1) # A[0] is dummy
 
    V = V_history[0] = Vhat
    logS = logS_history[0] = logShat
    logL = logL_history[0] = np.zeros(num_particles)
    
    '''Calculate Vhat terms'''
    Y = np.sqrt(V_0/n) * np.ones((num_particles, n))
    for t in range(1, T+1):
        num_particles_ = 0 # Initializing new particle count 
        Y = np.exp(-mrc/2)*((kappa/2) *npr.randn(num_particles, n)*np.sqrt(delta_t) + Y) # Propagate Y_i^j[t] -> Y_i^j[t+1]
        Vhat = np.sum(Y**2, axis=1)
           
#         print("length of V: ", Vhat.shape)
        # Propagation
        logShat = logS + (a * np.sqrt(V*delta_t)*npr.randn(num_particles) + b + c * (Vhat - V)*delta_t + d*(Vhat - V)) # Euler Discretization
        logLhat = logL + (e * (np.log(Vhat/V) + mrc)) + (f * (1/Vhat - 1/V) * delta_t) # Simple Euler Discretization
        logLhat = np.log(np.exp(logLhat)/np.sum(np.exp(logLhat)))
        A[t] = np.sum(np.exp(logLhat))/N # average weight averaged w.r.t. the "initial # of particles", not num_particles

        logS, V, logL = [], [], [] # Cleaning up the slate to write onto the particle history
        # Select which to branch
        l = 0 
        for j in range(0, num_particles): # Non Branched Particles
            if A[t]/r < np.exp(logLhat[j]) < r*A[t]:
                if (j - l) >= len(logS):
                    logS.append(logShat[j])
                    V.append(Vhat[j])
                    logL.append(logLhat[j]) 
                else:
                    logS[j - l], V[j - l], logL[j - l] = logShat[j], Vhat[j], logLhat[j]
            else: # Branched Particles
                '''
                This part of the algorithm is erroneous in the paper, at least in my opinion. 
                '''
                l = l + 1
                if (l - 1) >= len(logS): # Don't branch this
                    logShat = np.append(logShat, logShat[j])
                    Vhat = np.append(Vhat, Vhat[j])
                    logLhat = np.append(logLhat, logLhat[j])
                else: # Send this for branching
                    logShat[l-1], Vhat[l-1], logLhat[l-1] = logShat[j], Vhat[j], logLhat[j]
        # Branching part of the algorithm
        num_particles_ = num_particles - l
        W = npr.rand(l)/(l)
        W = np.array([W[i] + i/(l) for i in range(l)]) # Stratified Uniform samples
        U = npr.permutation(W)
        for j in range(0, l):
            N_j = int(np.exp(logLhat[j])/A[t]) + (1 if (U[j] <= np.exp(logLhat[j])/A[t] - int(np.exp(logLhat[j])/A[t])) else 0)
#             print(N_j)
            for k in range(0, N_j):
                logS.append(logShat[j]) 
                V.append(Vhat[j]) 
                logL.append(np.log(A[t]))
            num_particles_ += N_j
#         print("Length of V: ", len(V))
        num_particles = num_particles_
        print("Particles:", num_particles)
        logS_history[t] = logS
        V_history[t] = V
        logL_history[t] = logL
        # Re-setting variables
        V = Vhat = np.array(V)
#         print(V.shape)
        logS = logShat = np.array(logS)
        logL = logLhat = np.array(logL)
        Y = np.sqrt(Vhat/n).reshape(num_particles, 1)*np.ones((num_particles, n))
    return (logS_history, V_history, logL_history)