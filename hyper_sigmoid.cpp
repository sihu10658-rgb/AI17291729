#include <vector>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

// --- 0. 13차 체비쇼브 급수 기반 극한 정밀 exp (Target Error: ~1e-14) ---
inline double cheb_exp(double x) {
    // 구간 [-1, 1] 클램핑
    if (x < -1.0) x = -1.0;
    if (x > 1.0) x = 1.0;

    // 체비쇼브 다항식 T_0부터 T_13까지 점화식(T_{n} = 2x*T_{n-1} - T_{n-2})을 통한 고속 생성
    double T[14];
    T[0] = 1.0;
    T[1] = x;
    for (int i = 2; i <= 13; ++i) {
        T[i] = 2.0 * x * T[i - 1] - T[i - 2];
    }

    // e^x의 제1종 수정 베셀 함수 계수들 (c_0 = I_0(1), c_n = 2 * I_n(1), n >= 1)
    // 13차까지의 고정밀 계수 값들
    const double c[14] = {
        1.2660658777520084,  // c0
        1.1303182079849694,  // c1
        0.2714953395340944,  // c2
        0.0443368498486638,  // c3
        0.0054742404420942,  // c4
        0.0005429268312080,  // c5
        0.0000449773229547,  // c6
        0.0000031984366137,  // c7
        0.0000001992551072,  // c8
        0.0000000110354414,  // c9
        0.0000000005515219,  // c10
        0.0000000000254035,  // c11
        0.0000000000010901,  // c12
        0.0000000000000438   // c13
    };

    // 체비쇼브 급수 총합 계산: sum(c_i * T_i)
    double result = 0.0;
    for (int i = 0; i <= 13; ++i) {
        result += c[i] * T[i];
    }
    return result;
}

// --- 1. 기본 행렬 연산 ---
std::vector<std::vector<double>> matrix_add(
    const std::vector<std::vector<double>>& A, 
    const std::vector<std::vector<double>>& B) {
    int rows = A.size();
    int cols = A[0].size();
    std::vector<std::vector<double>> C(rows, std::vector<double>(cols, 0.0));

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            C[i][j] = A[i][j] + B[i][j];
        }
    }
    return C;
}

std::vector<std::vector<double>> matrix_multiply(
    const std::vector<std::vector<double>>& A, 
    const std::vector<std::vector<double>>& B) {
    int M = A.size();
    int K = A[0].size();
    int N = B[0].size();

    std::vector<std::vector<double>> C(M, std::vector<double>(N, 0.0));

    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            double sum = 0.0;
            for (int k = 0; k < K; ++k) {
                sum += A[i][k] * B[k][j];
            }
            C[i][j] = sum;
        }
    }
    return C;
}

// --- 2. 활성화 함수 (13차 체비쇼브 기반) ---
std::vector<std::vector<double>> sigmoid(const std::vector<std::vector<double>>& Z) {
    int rows = Z.size();
    int cols = Z[0].size();
    std::vector<std::vector<double>> A(rows, std::vector<double>(cols, 0.0));

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            double ex = cheb_exp(-Z[i][j]);
            A[i][j] = 1.0 / (1.0 + ex);
        }
    }
    return A;
}

std::vector<std::vector<double>> sigmoid_backward(
    const std::vector<std::vector<double>>& dA, 
    const std::vector<std::vector<double>>& A) {
    int rows = A.size();
    int cols = A[0].size();
    std::vector<std::vector<double>> dZ(rows, std::vector<double>(cols, 0.0));

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            dZ[i][j] = dA[i][j] * A[i][j] * (1.0 - A[i][j]);
        }
    }
    return dZ;
}

std::vector<std::vector<double>> relu(const std::vector<std::vector<double>>& Z) {
    int rows = Z.size();
    int cols = Z[0].size();
    std::vector<std::vector<double>> A(rows, std::vector<double>(cols, 0.0));

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            A[i][j] = std::max(0.0, Z[i][j]);
        }
    }
    return A;
}

std::vector<std::vector<double>> relu_backward(
    const std::vector<std::vector<double>>& dA, 
    const std::vector<std::vector<double>>& Z) {
    int rows = Z.size();
    int cols = Z[0].size();
    std::vector<std::vector<double>> dZ(rows, std::vector<double>(cols, 0.0));

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            dZ[i][j] = Z[i][j] > 0 ? dA[i][j] : 0.0;
        }
    }
    return dZ;
}

// --- 3. 손실 함수 & 최적화 ---
double mse_loss(const std::vector<double>& y_pred, const std::vector<double>& y_true) {
    double sum = 0.0;
    int n = y_pred.size();
    for (int i = 0; i < n; ++i) {
        double diff = y_pred[i] - y_true[i];
        sum += diff * diff;
    }
    return sum / n;
}

std::vector<std::vector<double>> update_weights(
    const std::vector<std::vector<double>>& weights,
    const std::vector<std::vector<double>>& gradients,
    double learning_rate) {
    
    int rows = weights.size();
    int cols = weights[0].size();
    std::vector<std::vector<double>> updated(rows, std::vector<double>(cols, 0.0));

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            updated[i][j] = weights[i][j] - (learning_rate * gradients[i][j]);
        }
    }
    return updated;
}

// Pybind11 바인딩 등록
PYBIND11_MODULE(Matrix, m) {
    m.doc() = "Custom high-performance C++ AI Engine Core with 13th-Order Chebyshev Approximation";
    
    m.def("matrix_add", &matrix_add, "Matrix Addition");
    m.def("matrix_multiply", &matrix_multiply, "Matrix Multiplication");
    
    m.def("sigmoid", &sigmoid, "13th-Order Chebyshev Sigmoid Activation");
    m.def("sigmoid_backward", &sigmoid_backward, "Sigmoid Gradient Backward");
    m.def("relu", &relu, "ReLU Activation");
    m.def("relu_backward", &relu_backward, "ReLU Gradient Backward");
    
    m.def("mse_loss", &mse_loss, "Mean Squared Error Loss");
    m.def("update_weights", &update_weights, "Gradient Descent Weight Update");
}