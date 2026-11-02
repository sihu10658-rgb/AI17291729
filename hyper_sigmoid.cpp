#include <vector>
#include <cmath>
#include <stdexcept>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

// 1. 행렬 덧셈
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

// 2. 행렬 곱셈
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

// 3. 손실 함수 (MSE)
double mse_loss(const std::vector<double>& y_pred, const std::vector<double>& y_true) {
    double sum = 0.0;
    int n = y_pred.size();
    for (int i = 0; i < n; ++i) {
        double diff = y_pred[i] - y_true[i];
        sum += diff * diff;
    }
    return sum / n;
}

// 4. 최적화 (가중치 갱신)
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
    m.doc() = "Custom high-performance C++ AI Engine Core: Matrix & Learning Ops";
    m.def("matrix_add", &matrix_add, "Matrix Addition");
    m.def("matrix_multiply", &matrix_multiply, "Matrix Multiplication");
    m.def("mse_loss", &mse_loss, "Mean Squared Error Loss");
    m.def("update_weights", &update_weights, "Gradient Descent Weight Update");
}
