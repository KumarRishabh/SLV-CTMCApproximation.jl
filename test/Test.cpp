#include <iostream>
#include <Eigen/Dense>
#include <cmath>
#include <random>
#include <chrono>
#include <map>

using namespace Eigen;
using namespace std;
using namespace std::chrono;

MatrixXd HestonDiscretizationKahlJackel(double S0, double V0, int T, int N, 
                                        const map<string, double>& params, double Δt = 1e-7) {
    MatrixXd V = MatrixXd::Zero(N, T + 1);
    MatrixXd logS = MatrixXd::Zero(N, T + 1);
    V.col(0).setConstant(V0);
    logS.col(0).setConstant(log(S0));

    random_device rd;
    mt19937 gen(rd());
    normal_distribution<> d(0, 1);

    for (int i = 0; i < N; ++i) {
        for (int j = 1; j <= T; ++j) {
            double Δβ = sqrt(Δt) * d(gen);
            double ΔB = sqrt(Δt) * d(gen);  // independent brownian motions
            V(i, j) = V(i, j - 1) + (params.at("nu") - params.at("mean_reversion_coeff") * V(i, j - 1)) * Δt 
                      + params.at("kappa") * sqrt(V(i, j - 1)) * Δβ 
                      + 0.25 * pow(params.at("kappa"), 2) * (pow(Δβ, 2) - Δt);
            logS(i, j) = logS(i, j - 1) + params.at("mu") * Δt 
                         - 0.25 * (V(i, j - 1) + V(i, j)) * Δt 
                         + params.at("rho") * sqrt(V(i, j - 1)) * Δβ 
                         + 0.5 * (sqrt(V(i, j - 1)) + sqrt(V(i, j))) * (ΔB - params.at("rho") * Δβ) 
                         + 0.25 * params.at("rho") * (Δβ - Δt);
        }
        cout << "Simulation " << i + 1 << " completed" << endl;
    }

    MatrixXd S = logS.array().exp();
    return S;
}

int main(){
    double S0 = 100, V0 = 0.04;
    int T = 100000, N = 100;
    map<string, double> PS_1 = {{"nu", 0.04}, {"mean_reversion_coeff", 0.1}, {"kappa", 0.5}, {"mu", 0.05}, {"rho", -0.75}};

    auto start_time = high_resolution_clock::now();
    MatrixXd S = HestonDiscretizationKahlJackel(S0, V0, T, N, PS_1);
    auto end_time = high_resolution_clock::now();

    duration<double> execution_time = end_time - start_time;
    cout << "Execution time: " << execution_time.count() << " seconds" << endl;

    return 0;
}