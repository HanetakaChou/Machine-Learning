#include <openblas/cblas.h>
#include <openblas/lapacke_config.h>
#include <openblas/lapacke.h>
#include <assert.h>
#include <cstring>
#include <iostream>

int main(int argc, char *argv[], char *envp[])
{
    // Linear Regression by Normal Equation

    // y = Xβ + ϵ
    // y: target
    // x: n features
    // β: n coeffiects // This is what we would like to estimate
    // ϵ: errors // This should ideally be 0

    // we have m training examples
    // y: m × 1
    // X: m × n

    // Normal Equation
    // β: n × 1
    // β = (X^T X)^(-1) X^T y

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

    // Normal Equation

    double XT_mul_X[n * n];
    {
        // Matrix Transpose and Multiplication
        cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, n, n, m, 1.0, &X[0], n, &X[0], n, 0.0, &XT_mul_X[0], n);
    }

    double XT_mul_X_inv[n * n];
    {
        // Matrix Inverse
        lapack_int ipiv[2 * n];
        std::memcpy(XT_mul_X_inv, XT_mul_X, sizeof(XT_mul_X));
        int result_lu_decomposition = LAPACKE_dgetrf(LAPACK_ROW_MAJOR, n, n, &XT_mul_X_inv[0], n, ipiv);
        assert(0 == result_lu_decomposition);
        int result_inverse = LAPACKE_dgetri(LAPACK_ROW_MAJOR, n, &XT_mul_X_inv[0], n, ipiv);
        assert(0 == result_inverse);
    }

    double XT_mul_y[n];
    {
        // Matrix Transpose and Multiplication
        cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, n, 1, m, 1.0, &X[0], n, &y[0], 1, 0.0, &XT_mul_y[0], 1);
    }

    double theta[n];
    {
        // Matrix Multiplication
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, 1, n, 1.0, &XT_mul_X_inv[0], n, &XT_mul_y[0], 1, 0.0, &theta[0], 1);
    }

    std::cout << "Linear Regression Coefficients: ";
    for (int i = 0; i < n; ++i)
    {
        std::cout << theta[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}
