import os
import sys
import subprocess

# --- 0단계: C++ 원샷 자동 빌드 & 로드 ---
def ensure_cpp_compiled():
    cpp_file = "hyper_sigmoid.cpp"
    if os.name == 'nt':
        out_file = "Matrix.pyd"
        cmd = f'cl /O2 /LD {cpp_file} /Fe:{out_file}'
    else:
        try:
            ext_suffix = subprocess.check_output("python3-config --extension-suffix", shell=True).decode().strip()
        except Exception:
            ext_suffix = ".so"
        out_file = f"Matrix{ext_suffix}"
        
        includes = subprocess.check_output("python3 -m pybind11 --includes", shell=True).decode().strip()
        cmd = f'c++ -O3 -march=native -flto -ffast-math -Wall -shared -std=c++23 -fPIC {includes} {cpp_file} -o {out_file} -undefined dynamic_lookup'

    if not os.path.exists(out_file) or os.path.getmtime(cpp_file) > os.path.getmtime(out_file):
        print("🛠️ [브릿지 빌드]: C++ 코어 최적화 컴파일 중...")
        result = subprocess.run(cmd, shell=True)
        if result.returncode != 0:
            print("❌ [빌드 실패]")
            sys.exit(1)
        print("✅ [브릿지 빌드 성공]: C++ 코어가 파이썬 브릿지와 연결되었습니다.\n")

ensure_cpp_compiled()
import Matrix


# --- 1단계: 브릿지 아키텍처 (Neural Network Layers) ---

class Linear:
    """C++ 매트릭스 연산을 활용하는 선형(Dense) 레이어 브릿지"""
    def __init__(self, weight_init, bias_init=None):
        self.W = weight_init
        self.b = bias_init
        self.X_cache = None
        self.dW = None

    def forward(self, X):
        self.X_cache = X
        # Z = X * W
        return Matrix.matrix_multiply(X, self.W)

    def backward(self, dZ, learning_rate):
        # dW = X^T * dZ (간이 1x2 형태 대응)
        self.dW = [[self.X_cache[0][0] * dZ[0][0], self.X_cache[0][1] * dZ[0][0]]] # 필요시 확장 가능
        # 가중치 갱신
        self.W = Matrix.update_weights(self.W, self.dW, learning_rate)
        return self.W


class SigmoidActivation:
    """13차 체비쇼브 기반 시그모이드 활성화 함수 브릿지"""
    def __init__(self):
        self.A_cache = None

    def forward(self, Z):
        self.A_cache = Matrix.sigmoid(Z)
        return self.A_cache

    def backward(self, dA):
        return Matrix.sigmoid_backward(dA, self.A_cache)


class MSELoss:
    """MSE 손실 함수 브릿지"""
    def compute(self, y_pred, y_true):
        return Matrix.mse_loss(y_pred, y_true)
