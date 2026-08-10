# STT 데이터 정제 및 평가 로직

AI Hub 자유대화 음성 데이터에서 요리 관련 발화를 추출하고, 단계별 필터링을 통해 Whisper STT 평가용 데이터를 구축했다. 전체 과정은 **키워드 기반 1차 검색 → 의미 기반 2차 필터 → 요리 도메인 중심 3차 필터 → WAV 분할 → Whisper 평가 → WER/CER 계산** 순서로 진행했다.

## 1. 데이터 정제 과정

| 단계 | 적용 기준 | 데이터 수 |
|---|---|---:|
| 원본 | AI Hub 라벨 JSON 90개 | 전체 발화 |
| 1차 검색 | 레시피 기반 요리 키워드 약 83,213개와 대화문 비교 | 14,495개 |
| 2차 필터 | 음식/재료, 조리행동, 계량표현 중 2개 이상 포함 | 약 222개 |
| 3차 필터 | 음식·재료 + 조리행동 또는 음식·재료 + 계량표현 | 185개 |

### 1차 검색

레시피 CSV에서 생성한 `cooking_keywords.txt`를 이용해 AI Hub 대화 전체에서 요리 관련 가능성이 있는 문장을 넓게 수집했다. 이 단계에서는 누락을 최소화하는 것이 목적이었기 때문에 일반적인 일상 대화도 함께 포함되었다.

```python
matched_keywords = [
    keyword
    for keyword in cooking_keywords
    if keyword in text
]
```

### 2차 필터

1차로 추출한 14,495개 발화에서 **음식/재료, 조리행동, 계량표현**을 기준으로 관련성을 다시 판단했다. 세 가지 조건 중 두 가지 이상을 만족하는 발화를 선택하고, STT에 사용하기 어려운 너무 짧거나 긴 음성을 제외하기 위해 발화 길이를 1~15초로 제한했다.

```python
has_action = any(word in text for word in cooking_actions)
has_unit = any(word in text for word in cooking_units)
has_food = any(word in text for word in food_keywords)

score = int(has_action) + int(has_unit) + int(has_food)

return score >= 2
```

### 3차 필터

2차 결과에도 일반 대화가 일부 남아 있어 조건을 더 강화했다. 단순히 음식 관련 단어 하나가 등장하는 문장은 제외하고 **음식·재료와 실제 조리행동이 함께 등장하거나, 음식·재료와 계량표현이 함께 등장하는 경우만 최종 데이터로 선택**했다. 비요리 표현과 중복 문장도 제거했다.

```python
if has_food and has_action:
    return True

if has_food and has_unit:
    return True

return False
```

```python
final_df = final_df.drop_duplicates(subset=["text"])
```

최종적으로 요리 도메인 평가에 사용할 발화 185개를 확보했다.

## 2. WAV 자동 분할

AI Hub 라벨 JSON에는 각 발화의 `StartTime`과 `EndTime`이 포함되어 있다. 해당 시간을 이용해 긴 원본 WAV에서 필요한 구간만 자동으로 잘라 개별 음성 파일을 생성했다.

```python
start_frame = int(start_time * frame_rate)
end_frame = int(end_time * frame_rate)

src.setpos(start_frame)
frames = src.readframes(end_frame - start_frame)
```

분리된 WAV와 정답 문장은 `metadata.csv`로 연결하여 Whisper 평가에 사용했다.

## 3. Whisper STT 평가

최종 음성을 `openai/whisper-base`에 입력하고 한국어 음성 인식을 수행했다. Whisper가 생성한 문장을 원본 정답 문장과 비교해 STT 성능을 측정했다.

```python
result = stt_pipeline(
    audio_path,
    generate_kwargs={
        "language": "korean",
        "task": "transcribe"
    }
)
```

평가 전에는 특수문자와 불필요한 공백 차이가 오류율에 영향을 주지 않도록 정답과 예측 문장을 동일하게 정규화했다.

```python
text = str(text).lower().strip()
text = re.sub(r"[^가-힣a-z0-9\s]", "", text)
text = re.sub(r"\s+", " ", text).strip()
```

## 4. WER / CER 평가

STT 결과는 **WER(Word Error Rate)**과 **CER(Character Error Rate)**을 사용해 평가했다. WER은 단어 단위 오류율, CER은 문자 단위 오류율이며 두 값 모두 낮을수록 인식 성능이 좋다.

```python
wer_score = wer(references, predictions)
cer_score = cer(references, predictions)
```

| 평가 | 데이터 | WER | CER |
|---|---:|---:|---:|
| 1차 Baseline | 221개 | 0.7848 | 0.6570 |
| 최종 정제 후 | 185개 | 0.7658 | 0.5677 |

동일한 `Whisper-base`를 사용하고 평가 데이터만 정제했을 때 **WER은 0.7848 → 0.7658, CER은 0.6570 → 0.5677로 감소**했다. 특히 CER이 크게 감소하면서 일반 대화를 제거하고 요리 관련성이 높은 데이터로 정제한 효과를 확인할 수 있었다.

## 5. 전체 처리 흐름

```text
AI Hub 자유대화
        ↓
요리 키워드 기반 1차 검색
14,495개
        ↓
음식 · 조리행동 · 계량표현 2차 필터
약 222개
        ↓
요리 도메인 중심 3차 필터
185개
        ↓
StartTime / EndTime 기준 WAV 분할
        ↓
Whisper-base STT
        ↓
정답과 예측 문장 정규화
        ↓
WER / CER 평가
```

현재 단계는 **모델 자체를 변경한 실험이 아니라 평가 데이터를 요리 도메인에 맞게 정제한 실험**이다. 이후 동일한 최종 평가 데이터를 기준으로 `Whisper-small` 및 요리 도메인 파인튜닝 모델을 적용해 성능 변화를 비교할 예정이다.
