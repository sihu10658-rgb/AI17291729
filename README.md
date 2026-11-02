# AI17291729
# 🚀 C++/Python Hybrid High-Performance AI Engine

> **고정밀 C++20 코어**와 **Python**을 `pybind11`으로 바인딩하여, 오차 없는 수학적 연산과 극한의 성능을 동시에 추구하는 하이브리드 AI 엔진 프로젝트입니다.

---

## 🛠️ Key Features (핵심 기능)

1. **C++20 Native Core (`hyper_sigmoid.cpp`)**
   * **Matrix Operations**: 고속 행렬 덧셈(`matrix_add`) 및 곱셈(`matrix_multiply`)
   * **Loss Function**: 정밀한 평균 제곱 오차(MSE Loss) 계산
   * **Gradient Descent**: 역전파 및 가중치 갱신(`update_weights`) 연산[cite: 1]
2. **One-Shot Automated Build Pipeline**
   * 파이썬 스크립트 실행 시 C++ 소스코드의 변경 사항을 스스로 감지하여(`os.path.getmtime`), 터미널 빌드 과정 없이 자동으로 컴파일 후 모듈을 동적 로드합니다.
3. **Cross-Platform Compatibility**
   * macOS, Linux, Windows 환경 대응 및 링커 옵션 자동 최적화 (`-undefined dynamic_lookup`)

---

## 📂 Project Structure

```text
├── hyper_sigmoid.cpp   # C++ 고성능 AI 연산 코어 (pybind11 바인딩)
├── main.py             # 원샷 자동 컴파일 및 엔진 테스트 오케스트레이터
└── README.md           # 프로젝트 문서
