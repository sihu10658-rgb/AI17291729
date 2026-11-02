import os
import sys
import subprocess

# ==========================================
# 0단계: C++ 원샷 자동 컴파일 (pybind11 링크)
# ==========================================
def ensure_cpp_compiled():
    cpp_file = "hyper_sigmoid.cpp"
    
    if os.name == 'nt':  # Windows
        out_file = "Matrix.pyd"
        cmd = f'cl /O2 /LD {cpp_file} /Fe:{out_file}'
    else:  # Mac / Linux
        try:
            ext_suffix = subprocess.check_output("python3-config --extension-suffix", shell=True).decode().strip()
        except Exception:
            ext_suffix = ".so"
        out_file = f"Matrix{ext_suffix}"
        
        includes = subprocess.check_output("python3 -m pybind11 --includes", shell=True).decode().strip()
        # -O3 최적화와 함께 C++20 표준으로 13차 체비쇼브 코어 컴파일
        cmd = f'c++ -O3 -Wall -shared -std=c++20 -fPIC {includes} {cpp_file} -o {out_file} -undefined dynamic_lookup'

    if not os.path.exists(out_file) or os.path.getmtime(cpp_file) > os.path.getmtime(out_file):
        print("🛠️ [원샷 빌드]: 13차 체비쇼브 기반 C++ AI 엔진 컴파일 중...")
        result = subprocess.run(cmd, shell=True)
        if result.returncode != 0:
            print("❌ [빌드 실패]: C++ 코어 소스를 확인해주세요.")
            sys.exit(1)
        print("✅ [빌드 성공]: 파이썬-C++ 바인딩 완료!\n")

# 빌드 실행 후 모듈 임포트
ensure_cpp_compiled()
import Matrix


# ==========================================
# 1단계: 극한 정밀도 AI 신경망 학습 루프 실행
# ==========================================
def run_high_precision_training():
    print("==================================================")
    print(" 🚀 13차 체비쇼브 기반 AI 엔진 학습 테스트 (Epoch 1000)")
    print("==================================================")

    # 1. 입력 및 가중치 초기화
    X = [[0.3, 0.7]]
    W = [[0.2], [0.5]]
    y_true = [[0.85]]
    learning_rate = 0.5

    print(f"🔹 초기 입력값 X: {X}")
    print(f"🔹 초기 가중치 W: {W}")
    print(f"🔹 목표 정답 y: {y_true}\n")

    initial_loss = 0.0

    # 2. 1,000 에폭 학습 루프
    for epoch in range(1, 1001):
        # 순전파: Z = X * W, A = Sigmoid(Z) [13차 체비쇼브 exp 적용]
        Z = Matrix.matrix_multiply(X, W)
        A = Matrix.sigmoid(Z)
        
        # 손실 계산 (MSE)
        loss = Matrix.mse_loss([A[0][0]], [y_true[0][0]])
        if epoch == 1:
            initial_loss = loss

        # 역전파: 기울기 계산 및 가중치 갱신
        dA = [[2.0 * (A[0][0] - y_true[0][0])]]
        dZ = Matrix.sigmoid_backward(dA, A)
        dW = [[X[0][0] * dZ[0][0]], [X[0][1] * dZ[0][0]]]
        W = Matrix.update_weights(W, dW, learning_rate)

        if epoch == 1 or epoch % 200 == 0:
            print(f"🔄 [Epoch {epoch:4d}] 예측값: {A[0][0]:.8f} | Loss: {loss:.12f}")

    print("==================================================")
    print(f"🎉 [학습 완료]")
    print(f"  └─ 초기 Loss : {initial_loss:.12f}")
    print(f"  └─ 최종 Loss : {loss:.12f}")
    print("==================================================\n")

if __name__ == "__main__":
    run_high_precision_training()