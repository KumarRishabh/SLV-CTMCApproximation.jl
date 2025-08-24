docs"""
    Consider the following SDE:
    dS_t = b_11(S_t, V_t) dt + σ_11(S_t, V_t) dW_1t + σ_12(S_t, V_t) dW_2t
    dV_t = b_2(V_t) dt + σ_2(V_t) dW_2t

    Consider the following change of variables: 
    X_t = f(S_t, V_t)
    Y_t = g(V_t)

    Through the above change of variables, we can transform the SDE into an alternative SDE:

    where X_t and Y_t are the CTMC state variables.
"""


