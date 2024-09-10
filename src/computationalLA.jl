using LinearAlgebra

# consider a Tridiagonal matrix, compute the exponential of the matrix


function expm(A::Tridiagonal)
    n = length(A.dv)
    expA = Tridiagonal(zeros(n), zeros(n-1))
    expA.dv .= exp.(A.dv)
    for i in 1:n-1
        expA.ev[i] = A.ev[i] * (exp(A.dv[i]) - 1)
    end
    return expA
end