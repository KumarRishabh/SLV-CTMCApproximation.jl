using LinearAlgebra
using FastExpm
# consider a Tridiagonal matrix, compute the exponential of the matrix
names(FastExpm, all = true)

A = Tridiagonal(rand(99), rand(100), rand(99))
dense_A = Matrix(A)
expA = fastExpm(dense_A)
println(expA)

# Derive the option price model using: 
# 1. Explicit heston scheme
# 2. Double Death and Double Birth scheme
option_price = exp(-r * T) * (π0'*P*G)[1]

function expm(A::Tridiagonal)
    n = length(A.dv)
    expA = Tridiagonal(zeros(n), zeros(n-1))
    expA.dv .= exp.(A.dv)
    for i in 1:n-1
        expA.ev[i] = A.ev[i] * (exp(A.dv[i]) - 1)
    end
    return expA
end