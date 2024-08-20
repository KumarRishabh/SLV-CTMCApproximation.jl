#include <gtest/gtest.h>
#include <eigen3/Eigen/Dense>
using namespace std;

// Mock function for M_first_laguerre_weighted_eigen
Eigen::VectorXd M_first_laguerre_weighted_eigen(int M, double x)
{

    Eigen::VectorXd polynomials(M);

    polynomials(0) = 1.0;
    polynomials(1) = 1.0 - x;

    for (int kk = 2; kk < M; kk++)
    {
        // cout << "kk: " << kk << endl;
        polynomials(kk) = ((2.0 * ((double)(kk)-1.0) + 1.0 - x) * polynomials(kk - 1) - ((double)(kk)-1.0) * polynomials(kk - 2)) / (double)(kk);
    }

    for (int kk = 0; kk < M; kk++)
    {
        polynomials(kk) = exp(-x / 2.0) * polynomials(kk);
    }

    return (polynomials);
}

// Function to be tested
Eigen::VectorXd laguerre_basis_3d_eigen(int M, double x0, double x1, double x2) {
    Eigen::VectorXd poly_basis(M * M * M);

    Eigen::VectorXd poly0 = M_first_laguerre_weighted_eigen(M, x0);
    Eigen::VectorXd poly1 = M_first_laguerre_weighted_eigen(M, x1);
    Eigen::VectorXd poly2 = M_first_laguerre_weighted_eigen(M, x2);

    for (int k0 = 0; k0 < M; k0++) {
        for (int k1 = 0; k1 < M; k1++) {
            for (int k2 = 0; k2 < M; k2++) {
                poly_basis(k0 + k1 * M + k2 * M * M) = poly0(k0) * poly1(k1) * poly2(k2);
            }
        }
    }

    return poly_basis;
}

// Test case
int main() {
    Eigen::VectorXd poly_basis;
    poly_basis = laguerre_basis_3d_eigen(3, 1.0, 2.0, 3.0);
    // Iterate over the basis
    cout << "Laguerre basis: " << endl;
    cout << poly_basis << endl;
    Eigen::VectorXd poly1 = M_first_laguerre_weighted_eigen(3, 1.0);
    Eigen::VectorXd poly2 = M_first_laguerre_weighted_eigen(3, 2.0);
    Eigen::VectorXd poly3 = M_first_laguerre_weighted_eigen(3, 3.0);
    cout << "Laguerre basis 1: " << endl;
    cout << poly1 << endl;
    cout << "Laguerre basis 2: " << endl;
    cout << poly2 << endl;
    cout << "Laguerre basis 3: " << endl;
    cout << poly3 << endl;
    return 0;
}