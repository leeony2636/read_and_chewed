# 🎙️ 현대 음성 인식(STT)의 진화: Self-Supervised Learning (wav2vec 2.0)에서 Large Scaling Laws (OWLS)까지

음성 인식(Speech-to-Text, STT) 기술은 라벨링된 정답 데이터가 극히 드문 환경을 극복하기 위한 **자기지도학습(Self-Supervised Learning)**의 도입부터, 거대한 모델과 대규모 데이터를 바탕으로 한 **스케일링 법칙(Scaling Laws)** 연구로 비약적인 발전을 이루었습니다.

음성 인식 발전사의 두 축을 담당하는 **wav2vec 2.0 (2020)**과 **OWLS (2025)** 논문의 핵심 내용과 유기적 흐름을 정리합니다.

<img width="839" height="410" alt="image (4)" src="https://github.com/user-attachments/assets/d8ec3b70-278e-4fe6-b01a-a2155b636dac" />

---

## 1. wav2vec 2.0 (Facebook AI Research, 2020)
> **논문명:** *wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations*

### 📌 한 줄 요약
> 사람이 작성한 정답 텍스트(Label)가 없어도, **라벨 없는 원본 음성 파형(Raw Waveform)만으로 음성의 구조적 표현을 사전 학습(Pre-training)**하여 적은 정답 데이터로도 높은 음성 인식 성능을 달성한 모델입니다.

---

### 💡 왜 이 논문이 필요했는가? (음성 데이터의 한계)
전 세계에는 7,000개 이상의 언어가 존재하지만, 고성능 음성 인식 모델을 학습시킬 만큼 **사람이 직접 글자로 적어둔 라벨 데이터(Transcribed Audio)가 충분한 언어는 극소수**입니다.

* **핵심 질문:** *"정답 텍스트가 적더라도, 정답 없는 음성 데이터를 먼저 스스로 많이 듣고 공부하면 음성 인식을 잘할 수 있지 않을까?"*
* **해결책:** NLP의 BERT가 문장에서 단어를 가리고 맞히듯, 원본 음성 파형의 일부를 가리고(Masking) 맞히는 **자기지도학습(Self-Supervised Learning)** 구조인 wav2vec 2.0을 제안했습니다.

---

### 🔑 핵심 키워드 정리

| 키워드 | 개념 설명 |
| :--- | :--- |
| **wav2vec 2.0** | 라벨 없는 음성 데이터로 잠재 표현(Latent Representation)을 먼저 학습하는 자기지도학습 모델 |
| **Raw Waveform** | 사람이 수동으로 가공한 특징값(Spectrogram 등)이 아닌 원본 음성 파형 신호 그대로를 입력으로 사용 |
| **CNN Feature Encoder** | 원본 음성 파형을 입력받아 짧은 구간의 프레임 단위 음성 잠재 표현으로 변환하는 모듈 |
| **Quantization Module** | 연속적인(Continuous) 음성 잠재 표현을 이산적인(Discrete) 음성 단위(Codebook Vector)로 변환 |
| **Contrastive Learning** | 진짜 정답 음성 단위 1개와 여러 개의 거짓(Distractor) 후보군을 놓고 진짜를 구분하도록 학습 |
| **CTC Loss** | 파인튜닝 단계에서 입력 음성 프레임과 정답 텍스트 간의 길이 불일치 및 순서를 맞추기 위한 손실 함수 |

---

wav2vec 2.0은 크게 3가지 핵심 모듈로 구성됩니다.

```text
[Raw Waveform] ➔ [1. CNN Feature Encoder] ➔ [2. Transformer Context Network] ➔ [Contextualized Repr.]
                                   │
                                   └─────────➔ [3. Quantization Module]     ➔ [Quantized Target (Codebook)]

### 🏗️ 모델 구조 및 학습 메커니즘

1. **CNN Feature Encoder**: 원본 음성 파형(Raw Waveform)을 입력받아 잠재 표현 $z_t$를 추출합니다.
2. **Transformer Context Network**: 마스킹 처리된 잠재 표현을 입력받아 음성 전체의 전역적 문맥 표현 $c_t$를 생성합니다.
3. **Quantization Module (Gumbel-Softmax)**: 잠재 표현 $z_t$를 이산적 코드북 벡터 $q_t$로 양자화(Quantization)하여 Contrastive Learning의 정답 타깃으로 사용합니다.

> ❓ **왜 음성을 Quantization(양자화) 해야 할까?**  
> 음성 신호를 연속적인(Continuous) 숫자값 그대로 타깃으로 삼으면, 모델이 노이즈나 세부 주파수 변화 같은 무의미한 정보까지 과도하게 학습하려 합니다. Quantization을 적용하면 음성을 일정한 의미 단위(Discrete Unit)로 묶어주어 학습 안정성과 일반화 성능이 극대화됩니다.

---

### 📊 주요 실험 및 성과

* **10분 라벨 데이터**: 단 10분의 정답 텍스트 데이터만으로 파인튜닝했을 때, LibriSpeech `test-clean/test-other`에서 **WER 4.8 / 8.2**라는 놀라운 성과를 달성했습니다.
* **1시간 라벨 데이터**: 1시간의 라벨 데이터만 사용해도 기존의 100시간 라벨 데이터를 사용한 지도학습 모델보다 뛰어난 성능을 보였습니다.
* **결론**: wav2vec 2.0은 데이터 자원이 부족한 **저자원 언어(Low-resource Languages)** 환경에서 압도적인 효율성을 증명했습니다.

---

## 2. OWLS (Open Whisper-style Language-Speech Models, 2025)

> **논문명**: *OWLS: Scaling Laws and Capabilities of Open Speech Recognition and Translation Models (Chen et al., 2025)*

### 📌 한 줄 요약
> 150개 언어, 최대 36만 시간의 대규모 데이터와 0.25B~18B 파라미터 모델을 통해 **음성 인식 및 음성 번역에서의 스케일링 법칙(Scaling Laws)과 한계점을 체계적으로 규명**한 논문입니다.

---

### 💡 왜 이 논문이 필요했는가?
LLM 분야에서는 모델 크기, 데이터 양, 연산량(Compute)의 증가에 따른 성능 향상이 법칙(Scaling Law)으로 잘 정리되어 있었으나, 다국어 음성 인식 및 번역 영역에서는 체계적인 스케일링 연구가 부족했습니다.

OWLS 연구진은 Whisper 아키텍처 기반으로 **"모델 크기, 데이터의 양과 다양성, 계산량이 음성 성능에 미치는 정확한 한계선은 어디인가?"**라는 질문을 검증하고자 시작했습니다.

---

### 📊 OWLS 핵심 실험 결과 요약

| 실험 구분 | 핵심 내용 및 발견점 |
| :--- | :--- |
| **모델 스케일링 (0.25B ~ 18B)** | 모델 크기가 확장될수록 대부분의 언어에서 WER(단어 오류율) 및 CER이 일관되게 감소함. |
| **저자원 언어 성능** | 학습 데이터가 35시간 미만인 50개 언어 기준, 모델을 1B에서 9B로 키웠을 때 평균 WER이 **59%에서 45%로 대폭 개선**됨. |
| **데이터 분포의 한계 (포화 현상)** | 데이터 양을 무작정 늘려도 **동일한 분포/환경의 데이터만 추가할 경우 성능 향상이 정체(Saturation)**됨. |
| **데이터 다양성의 중요성** | 기존 180K 데이터에 새로운 분포인 YODAS(180K 시간)를 추가하자 정체되었던 한국어·폴란드어 등의 성능이 다시 급격히 개선됨. |
| **발현적 능력 (Emergent Abilities)** | 9B, 18B 등 초대형 모델로 갈수록 **표기법 변환 이해, 코드 스위칭(혼용 언어), In-Context Learning을 통한 미학습 언어 처리** 능력이 나타남. |

---

### 🏠 OWLS 논문의 일상생활 비유

* **모델 크기** = 학생의 사고 능력(지능)
* **데이터 양 및 다양성** = 공부하는 교재의 양과 문제 유형
* **연산량 (Compute)** = 학생이 공부에 투자한 시간

> 아무리 사고 능력이 뛰어난 학생(대형 모델)이라도 교재(데이터)가 아예 없으면 배울 수 없고, 똑같은 문제집(동일 분포 데이터)만 계속 반복하면 성적이 더 이상 오르지 않습니다. **더 똑똑한 학생에게 다양하고 새로운 유형의 문제집을 줄 때 비로소 최고의 성능**이 나타납니다.

---

## 🎯 최종 종합 결론

1. **wav2vec 2.0 (2020)**: 정답 라벨이 없어도 음성 자체의 구조를 학습하는 **Self-Supervised Learning**을 통해 저자원 언어 및 적은 데이터 환경에서의 STT 가능성을 열었습니다.
2. **OWLS (2025)**: 대규모 다국어 환경에서 **모델 크기·데이터 다양성·연산량 간의 스케일링 법칙**을 정립하여, 음성 AI 모델을 효율적으로 확장하기 위한 가이드라인을 완성했습니다.

> 💡 **한 줄 요약**: 음성 STT 기술은 **'라벨 없는 데이터로 소리의 구조를 배우는 기법(wav2vec 2.0)'**에서 **'대규모 데이터와 모델 확장을 통한 다국어 스케일링 법칙의 정립(OWLS)'**으로 발전해 왔습니다.
