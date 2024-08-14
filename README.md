# SLVCTMCApproximation

[![Build Status](https://github.com/KumarRishabh98/SLVCTMCApproximation.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/KumarRishabh98/SLVCTMCApproximation.jl/actions/workflows/CI.yml?query=branch%3Amain)

<!-- TODO: Update Readme with the goals of the project -->

This package focuses on pricing a variety of path-dependent options through various Stochastic Volatility Models (hence the SLV in the name), which are approximated using a Continuous time Markov Process with at most countable state space. This approach is non-standard, where the standard approach is using a discrete time approximation of these stochastic local volatility models. The Stochastic Local Volatilities of interest are: 
- [ ] Heston Model 
- [ ] 3/2 Model
- [ ] Jump Diffusions - ABCJ/SLVJ or Bates 
- [ ] SABR Model
<!-- Make a list in markdown -->

Calculate the generator matrix Q for the Price and Volatility process

    After calculating the generator matrix Q for the Volatility process, we can calculate the generator matrix Q for the Price and Volatility process. 
    The Price and Volatility process is given by the following SDEs in the Heston model:
    
    $$dS(t) = μS(t)dt + sqrt((1 - ρ^2)v(t))S(t)dW1(t) + ρ* sqrt(v(t))S(t)dW2(t)$$
    $$dv(t) = (ν - ϱv(t))dt + κ*sqrt(v(t))dW2(t)$$
    
    where W1(t) and W2(t) are independent Brownian motions.

    Consider a volatility process of the following form:
    dv(t) = μ(t)dt + σ(t)dW(t) 

    The volatility process can be approximated using the following birth-death markov chain, with the following bounded finite states 
    
## TODO List

- [x] Calculate the generator matrix Q for the Volatility process
- [ ] Calculate the generator matrix Q for the Price and Volatility process
- [ ] Write unit tests for the models
- [ ] Update documentation