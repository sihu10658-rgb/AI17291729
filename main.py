import math
import re
import Matrix

class Embedder:
    """텍스트 정보(단어)를 C++ 엔진이 연산할 수 있는 숫자 행렬(벡터)로 변환하는 임베더"""
    def __init__(self):
        self.vocab = {}
        
    def fit(self, documents):
        """문서 집합을 읽어 단어장(Vocab)을 구축"""
        tokens_set = set()
        for doc in documents:
            tokens = self._tokenize(doc)
            tokens_set.update(tokens)
        self.vocab = {word: idx for idx, word in enumerate(sorted(tokens_set))}
        
    def _tokenize(self, text):
        """특수문자 제거 및 단어 단위 분리"""
        clean_text = re.sub(r'[^\w\s]', '', text)
        return clean_text.split()

    def transform(self, text):
        """문장을 C++ 행렬 연산용 1xD 벡터(리스트)로 변환"""
        vector = [0.0] * len(self.vocab)
        tokens = self._tokenize(text)
        for token in tokens:
            if token in self.vocab:
                vector[self.vocab[token]] += 1.0
        
        # 벡터 정규화 (크기를 1로 맞춰 정밀도 향상)
        norm = math.sqrt(sum(v ** 2 for v in vector))
        if norm > 0:
            vector = [v / norm for v in vector]
        return vector


class KnowledgeEngine:
    def __init__(self, knowledge_base):
        self.knowledge_base = knowledge_base
        self.embedder = Embedder()
        
        # 1. 텍스트 정보 수집 및 임베더 학습
        docs = list(knowledge_base.keys())
        self.embedder.fit(docs)
        
        # 2. 모든 지식 텍스트를 임베딩 행렬(Knowledge Matrix)로 변환
        self.doc_vectors = [self.embedder.transform(doc) for doc in docs]
        print(f" [시스템]: 총 {len(docs)}개의 지식 문서를 {len(self.embedder.vocab)}차원 벡터로 임베딩 완료.")

    def query(self, user_question):
        # 1. 질문 텍스트를 임베딩 벡터로 전환
        q_vec = self.embedder.transform(user_question)
        
        # 2. C++ 행렬 곱셈 연산을 이용해 질문과 지식간 유사도(Dot Product) 계산
        # Doc_Matrix (N x D) * Q_Vector_Transposed (D x 1)
        best_score = -1.0
        best_doc_idx = -1
        
        for idx, doc_vec in enumerate(self.doc_vectors):
            # C++ matrix_multiply 호출을 위해 2차원 리스트 형태 맞춤
            # [1 x D] * [D x 1] -> [1 x 1] 연산
            mat_a = [q_vec]
            mat_b = [[v] for v in doc_vec]
            
            # C++ 네이티브 곱셈 수행!
            res = Matrix.matrix_multiply(mat_a, mat_b)
            score = res[0][0]
            
            if score > best_score:
                best_score = score
                best_doc_idx = idx
                
        docs = list(self.knowledge_base.keys())
        if best_score > 0.1: # 일정 유사도 이상일 때만 지식 반환
            matched_doc = docs[best_doc_idx]
            answer = self.knowledge_base[matched_doc]
            return answer, best_score
        else:
            return "죄송합니다, 관련된 지식을 데이터베이스에서 찾을 수 없습니다.", 0.0


def main():
    # 💡 인공지능이 참조할 지식 데이터베이스 (텍스트 정보)
    knowledge_db = {
        "대한민국 수도": "대한민국의 수도는 서울특별시입니다.",
        "파이썬 C++ 연동": "pybind11을 사용하면 파이썬과 C++을 손쉽게 연결할 수 있습니다.",
        "임베딩 벡터": "임베딩은 사람이 쓰는 문자와 정보 데이터를 C++ 엔진이 다룰 수 있는 숫자 벡터로 변환하는 기술입니다.",
        "C++ AI 엔진": "이 시스템은 C++20과 파이썬이 결합된 하이브리드 지식 연산 AI 구조입니다."
    }

    engine = KnowledgeEngine(knowledge_db)

    print("\n==================================================")
    print(" 🧠 임베딩 기반 C++ 지식 검색 AI 콘솔 ")
    print("==================================================")
    print("질문을 입력하면 임베더가 문자를 숫자로 변환하고 C++ 엔진이 검색합니다.")
    print("종료: 'exit'\n")

    while True:
        try:
            user_input = input("[질문 입력]: ").strip()
            if user_input.lower() in ["exit", "종료"]:
                break
            if not user_input:
                continue

            # 지식 검색 및 C++ 연산 실행
            answer, confidence = engine.query(user_input)
            
            print(f"  └─ 💡 [AI 답변]: {answer}")
            print(f"  └─ 🔍 [C++ 연산 유사도 신뢰도]: {confidence * 100:.1f}%\n")

        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    main()