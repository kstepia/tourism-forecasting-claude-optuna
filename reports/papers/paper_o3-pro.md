
## “Nature Sustainability” 투고용 상세 연구계획서

_(o1‑pro ― 재현성 · 템플릿, o3 ― 리뷰 전략 · 정책 함의, gpt‑4.5 ― 계량 엄밀성 장점을 통합)_

---

### 1. 저널 규격 및 투고 포맷 확정

|규정 항목|Nature Sustainability **Article** 요건|적용 방안|
|---|---|---|
|본문 길이|**3,500 단어 이하**(Methods·참고문헌·Figure legends 제외) ([nature.com](https://www.nature.com/natsustain/content?utm_source=chatgpt.com "Content Types \| Nature Sustainability"))|3,300 ± 100 단어로 편집|
|요약(Abstract)|150 단어, 인용 불가 ([nature.com](https://www.nature.com/natsustain/content?utm_source=chatgpt.com "Content Types \| Nature Sustainability"))|148 단어 목표|
|디스플레이|Figures + Tables ≤ 6개 ([nature.com](https://www.nature.com/natsustain/content?utm_source=chatgpt.com "Content Types \| Nature Sustainability"))|그림 4, 표 2|
|자료·코드 공개|제출 시점부터 GitHub URL + Zenodo DOI 명시 의무 ([nature.com](https://www.nature.com/natsustain/editorial-policies/reporting-standards?utm_source=chatgpt.com "Reporting standards and availability of data, materials, code and ..."))|“Data & Code availability” 문단 확보|

> **결정** : Article 형식을 선택하되, o1‑pro식 _스캐폴드_로 분량을 준수하고, Methods는 Supplementary로 상세화해 본문을 압축한다.

---

### 2. 연구 배경 & 목표

- **배경** : 팬데믹·항공 네트워크 충격 등으로 제주 관광수요는 구조적 변동(Break)과 높은 예측 불확실성을 보임.
    
- **연구 공백** :
    
    1. 기존 구조단절 연구는 Break 탐지 이후 예측모형으로 직접 연결하지 않음.
        
    2. Tourism Forecasting에서 LSTM·Transformer 사용례는 있으나 진단 가능성·정책 시뮬레이션이 부족 ([nature.com](https://www.nature.com/articles/s41598-025-94268-8?utm_source=chatgpt.com "The use of artificial neural network algorithms to enhance tourism ..."), [mdpi.com](https://www.mdpi.com/2071-1050/17/5/2210?utm_source=chatgpt.com "Tourism Demand Forecasting Based on a Hybrid Temporal Neural ...")).
        
- **학술·사회적 가치** :  
    _Break 탐지 ↔ AI 예측 ↔ 경제적 손실_을 하나의 파이프라인으로 제시하면 지속가능 관광 거버넌스 의사결정을 지원.
    

---

### 3. 가설 (간결형)

|가설|서술|검증 방법|
|---|---|---|
|**H1** (Structural Break)|2005–2025 제주 관광수요 시계열에는 유의미한 **다중 구조단절**(≤ 5개)이 존재한다.|Wild Binary Segmentation + supF test (α = 0.05) ([researchgate.net](https://www.researchgate.net/publication/292047580_Multiple_change-point_detection_for_non-stationary_time_series_using_Wild_Binary_Segmentation?utm_source=chatgpt.com "Multiple change-point detection for non-stationary time series using ..."))|
|**H2** (Predictive Superiority)|**LSTM·Transformer**가 SARIMA·BSTS 대비 **MAE·RMSE**를 유의하게 감소시킨다.|Nested CV, Newey‑West 보정 DM‑test|
|**H3** (Policy Impact)|AI 예측 정확도 향상(10 %p) → 관광쇼크 시 **연간 GRDP 손실 ≥ 0.2 % 감소**.|BSTS counter‑factual 시뮬레이션, Input‑Output 비율 적용|

_H3_는 o3 버전의 정책 함의 로직을 반영하되, 수치 목표(0.2 %)를 사전에 명시하여 리뷰어의 “실질적 임팩트?” 의문을 차단한다.

---

### 4. 데이터 설계

|카테고리|변수|기간/해상도|출처|
|---|---|---|---|
|**종속**|제주 월별 입도객 수|2005.01 – 2025.12|제주관광공사 API|
|**설명**|국제선 좌석수, 항공료, 실효환율, 기상(강수·기온), COVID‑19 확진|동일 기간, 월|ICAO, KOSIS, KMA, WHO|
|**파생**|Google Trends 지수, SNS 감성점수|일 → 월 리샘플링|Google API, Twitter Academic|

- **데이터 공개** : `/data/raw`(원본)·`/data/processed`(파생) 구조, 버전관리 = DVC(+LFS).
    

---

### 5. 방법론 세부 (o1‑pro 스캐폴드 + gpt‑4.5 엄밀성)

#### 5.1 구조단절 탐지

1. **모델** : Piecewise‑linear SARIMA.
    
2. **알고리즘** : Wild Binary Segmentation(WBS), penalty γ = log T.
    
3. **검정** : supF test, BIC 최적 break ≤ 5.
    

#### 5.2 예측 모델

|계열|구조·하이퍼|과적합 방지|
|---|---|---|
|**SARIMA**|자동 p,d,q 선택 (AICc)|Ljung‑Box 잔차 검정|
|**BSTS**|Local Linear Trend + Holiday dummy|Spike‑&‑Slab Prior|
|**LSTM**|2 layers×32, seq = 24, dropout = 0.2|Early‑Stopping(Δval ≤ 0.001, 5 epoch)|
|**Transformer**|2 encoder layers, heads = 2, d_model = 64|Weight decay 1e‑4|

> LSTM/Transformer 설정은 최근 Tourism 수요 연구에서 성능·컴퓨팅 균형이 확인된 값 ([mdpi.com](https://www.mdpi.com/2071-1050/17/5/2210?utm_source=chatgpt.com "Tourism Demand Forecasting Based on a Hybrid Temporal Neural ...")).

#### 5.3 평가 & 통계 검정

- **지표** : RMSE, MAE, MAPE, QLIKE(변동성).
    
- **모델 우위** : Newey‑West DM‑test(h = 1, 3, 12).
    
- **다중 비교 보정** : Holm‑Bonferroni.
    

#### 5.4 설명 가능성

- **SHAP** 전체 중요도 + **Attention Map**(Transformer) 시각화 → Fig 3 (메인).
    
- **모델 해석 텍스트** : “항공좌석수·환율” 상위 기여 확인 기대.
    

#### 5.5 정책 시나리오

1. 관광객 급감 Shock = Actual – Counterfactual(BSTS)
    
2. 손실 GRDP = Shock × 인당 소비 × 비용승수(Leontief IO 표).
    
3. LSTM/Transformer 정확도 개선 → 손실 보정률 계산 (Table 2).
    

---

### 6. 재현성 인프라 (o1‑pro 템플릿)

|폴더|내용|
|---|---|
|`/repro/env.yml`|Conda 3.12, Python 3.11, TensorFlow 2.16, R 4.3 (bsts)|
|`/scripts/run_all.sh`|**단일 커맨드** 전체 파이프라인|
|`/notebooks/EDA.ipynb`|시계열 특성 · 결측 처리|
|`/models/`|Checkpoints(.h5), SARIMA `.pkl`|
|`/docs/Reporting_Summary.docx`|Nature Portfolio 양식 초안|
|Continuous CI|GitHub Actions: lint + pytest + unit test|

Zenodo DOI 생성 시 GitHub Release v1.0 고정 → 투고 PDF에 DOI 삽입.

---

### 7. 일정 & 인력 배분 (총 10 주)

|주차|작업|담당·산출물|
|---|---|---|
|1‑2|데이터 수집·정리|Data Engineer → `/data/raw`|
|3|EDA·Break 탐지|Quant Lead → Fig 1, Supplement S1|
|4‑6|모델 구현·훈련|ML Engineer (딥러닝), Statistician(SARIMA/BSTS)|
|7|SHAP·시나리오 분석|Policy Analyst → Fig 3, Table 2|
|8|내부 크로스리뷰|모든 저자 → 코멘트수정|
|9|저널 포맷팅·Reporting Summary|Corresponding Author|
|10|프리서브미션 Enquiry & 정식 Submission|Corresponding Author (메일, ScholarOne)|

---

### 8. reject 리스크 사전 점검 (o3 컨설턴트 방식)

|가능 지적|예방 조치|
|---|---|
|**범위 과다**|딥러닝은 “검증·강건성” 섹션으로 제한, 본문 900 단어 이하|
|**재현성 부족**|코드 100 % 공개 + seed·GPU 명시|
|**블랙박스 우려**|SHAP Figure + 요약표로 변수기여 명확화|
|**데이터 누수**|Sliding Window Split 그림(Extended Data 1)|

---

### 9. 예상 결과 형태 (목차·분량 예시)

|섹션|주요 콘텐츠|예상 단어수|
|---|---|---|
|Abstract|목적·방법·핵심 결과·시사점|150|
|Introduction|문제 인식·연구 공백·가설|650|
|Results|Break 탐지·예측 성능·정책 시뮬|1,300|
|Discussion|학술·정책 함의·한계·미래연구|750|
|Methods*|데이터·모형·검정·윤리|1,200 (섹션 분리)|
|Figures|Fig 1 Break, Fig 2 Workflow, Fig 3 SHAP, Fig 4 Prediction plot|4|
|Tables|Table 1 성능 비교, Table 2 GRDP 시나리오|2|

*Methods는 단어 제한 밖(온라인).

---

### 10. 검증 가능한 참고 근거(발췌)

1. Nature Sustainability Article 분량·표/그림 한도 ([nature.com](https://www.nature.com/natsustain/content?utm_source=chatgpt.com "Content Types | Nature Sustainability"))
    
2. Data·Code 공개·재현성 정책 ([nature.com](https://www.nature.com/natsustain/editorial-policies/reporting-standards?utm_source=chatgpt.com "Reporting standards and availability of data, materials, code and ..."))
    
3. LSTM 기반 관광수요 예측 SCI 논문 사례 ([nature.com](https://www.nature.com/articles/s41598-025-94268-8?utm_source=chatgpt.com "The use of artificial neural network algorithms to enhance tourism ..."), [mdpi.com](https://www.mdpi.com/2071-1050/17/5/2210?utm_source=chatgpt.com "Tourism Demand Forecasting Based on a Hybrid Temporal Neural ..."))
    
4. Wild Binary Segmentation 구조단절 검정 이론 ([researchgate.net](https://www.researchgate.net/publication/292047580_Multiple_change-point_detection_for_non-stationary_time_series_using_Wild_Binary_Segmentation?utm_source=chatgpt.com "Multiple change-point detection for non-stationary time series using ..."))
    

---

## 최종 메시지

이 계획서는 Nature Sustainability **편집 규정**과 최근 **AI‑기반 관광 연구 선례**를 바탕으로,

- **가설 3개**,
    
- **표/그림 6개**,
    
- **3,300 단어 본문**으로 간소화하면서도  
    딥러닝(LSTM·Transformer)·구조단절 계량 분석·정책 시뮬레이션을 모두 포함합니다.
    

o1‑pro의 **재현성 워크플로**, o3의 **리뷰어 친화 전략**, gpt‑4.5의 **엄밀한 통계 검정**을 결합했으므로 데스크‑리젝트 위험을 최소화할 수 있습니다. 필요 시 본 템플릿을 기반으로 바로 **프리서브미션 enquiry** 메일 초안을 작성해 드릴 수 있으니 알려주세요!