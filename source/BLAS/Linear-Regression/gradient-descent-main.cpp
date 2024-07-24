#include <openblas/cblas.h>
#include <limits>
#include <cstring>
#include <cmath>
#include <assert.h>
#include <iostream>

int main(int argc, char *argv[], char *envp[])
{
    // Linear Regression by Gradient Descent

    // y = Xβ + ϵ
    // y: target
    // x: n features
    // β: n coeffiects // This is what we would like to estimate
    // ϵ: errors // This should ideally be 0

    // we have m training examples
    // y: m × 1
    // X: m × n

    // Gradient Descent
    // β: n × 1
    // ∇J(β) = (1/m) (X^T X β - X^T y) = (1/m) (X^T (X β - y)) = (1/m) (X^T (hat{y} - y)) = (1/m) (X^T ϵ)
    // β_(t+1) = β_t - α ∇J(β_t)

    // Learning Rate
    constexpr double const alpha = 1e-3;

    constexpr int const iteration_count = 15000;

    // Cost Change Threshold
    constexpr double const gamma = 1e-4;

    constexpr int const m = 4;
    constexpr int const raw_n = 4;

    double Raw_X[m * raw_n] = {
        2104.0, 5.0, 1.0, 45.0,
        1416.0, 3.0, 2.0, 40.0,
        1534.0, 3.0, 2.0, 30.0,
        852.0, 2.0, 1.0, 36.0};

    double y[m] = {
        400.0,
        232.0,
        315.0,
        178.0};

    constexpr int const n = raw_n + 1;

    double X[m * n];

    // Feature Scaling
    double Mu_X[raw_n];
    for (int i = 0; i < raw_n; ++i)
    {
        Mu_X[i] = (1.0 / double(m)) * cblas_dsum(m, &Raw_X[i], raw_n);
    }

    double Sigma_X[raw_n];
    for (int i = 0; i < raw_n; ++i)
    {
        double Column_X[m];
        cblas_dcopy(m, &Raw_X[i], raw_n, Column_X, 1);

        cblas_daxpy(m, -1.0, &Mu_X[i], 0, &Column_X[0], 1);

        Sigma_X[i] = std::sqrt((1.0 / double(m)) * cblas_ddot(m, Column_X, 1, Column_X, 1));
    }

    for (int i = 0; i < raw_n; ++i)
    {
        cblas_dcopy(m, &Raw_X[i], raw_n, &X[i + 1], n);

        cblas_daxpy(m, -1.0, &Mu_X[i], 0, &X[i + 1], n);

        cblas_dscal(m, 1.0 / Sigma_X[i], &X[i + 1], n);
    }

    // Intercept Term
    {
        double const one = 1.0;
        cblas_dcopy(m, &one, 0, &X[0], n);
    }

    double theta[n] = {
        0.0,
        0.0,
        0.0,
        0.0,
        0.0};

    double previous_cost = std::numeric_limits<double>::max();

    for (int t = 0; t < iteration_count; ++t)
    {
        double prediction[m];
        {
            // Matrix Multiplication
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, 1, n, 1.0, &X[0], n, &theta[0], 1, 0.0, &prediction[0], 1);
        }

        double error[m];
        {
            // Matrix subtraction
            std::memcpy(error, prediction, sizeof(prediction));
            cblas_daxpy(m * 1, -1.0, y, 1, error, 1);
        }

        // Check Convergence
        double current_cost = (0.5 / double(m)) * cblas_ddot(m, &error[0], 1, &error[0], 1);
        if (previous_cost < current_cost)
        {
            std::cout << "Learning rate may be too high." << std::endl;
            break;
        }
        else if ((previous_cost - current_cost) < gamma)
        {
            std::cout << "Convergence achieved after " << t + 1 << " iterations." << std::endl;
            break;
        }
        previous_cost = current_cost;

        double gradient[n];
        {
            // Matrix Transpose and Multiplication
            cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, n, 1, m, 1.0, &X[0], n, &error[0], 1, 0.0, &gradient[0], 1);

            // Matrix Scale
            cblas_dscal(n * 1, 1.0 / double(m), &gradient[0], 1);
        }

        double theta_change[n];
        {
            // Matrix Scale
            std::memcpy(theta_change, gradient, sizeof(gradient));
            cblas_dscal(n * 1, alpha, &theta_change[0], 1);
        }

        // Matrix subtraction
        cblas_daxpy(n * 1, -1.0, &theta_change[0], 1, theta, 1);
    }

    std::cout << "Linear Regression Coefficients: ";
    for (int i = 0; i < n; ++i)
    {
        std::cout << theta[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}
