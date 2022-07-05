# Drinking Prediction Model with Health Data

### 문제 인식
* 의료 계열에서 종사하면서 다양한 오류들을 확인
* 특히, 임플란트 수술의 성공여부에는 많은 변수가 존재하는데 이런 변수를 줄이기 위해 수술 전 환자의 건강 상태를 확인하는 것이 우선
* 음주 여부는 잇몸의 상태 또는 수술 시 출혈량에도 영향을 끼칠 수 있음

### 문제 정의
* 음주 여부(=Target)에 대해 어떤 특성이 영향을 끼치는지 데이터 통계 및 가시화를 통해 분석하고자 한다.

### 가설 설정
* 알코올을 분해하는 간과 연관된 특성(GTP, AST, ALT)이 음주여부(=Target)에 가장 큰 영향을 미칠 것이다.

### 목표
* 음주 여부를 판단하는데 중요한 지표가 될 특성을 확인하고 이를 예측할 수 있는 모델을 구현한다.

### Benefit
* 평소 식습관, 정신 상태, 수면 습관 등 다양한 방면으로 데이터를 수집하고 음주 여부 판단뿐 아니라 특정 타겟을 예측 모델 학습에 적용시킨다면 국민의 보건의료 수준과 삶의 질을 항샹시키는데 큰 기여를 할 수 있을 것


## Requirements

```python
%%capture
import sys

if 'google.colab' in sys.modules:
    # Install packages in Colab
    !pip install eli5
    !pip install pdpbox
    !pip install shap
```

## Tools
* Python
* numpy
* matplotlib, seaborn
* sklearn
* eli5
* pdpbox
* shap
* XGboost

## Process
- [Data](#data)
- [Metrics and Score](#metrics-and-score)
  * [Predictive Model Explain](#5-predictive-model-explain)
- [Results](#results)
- [Reviews](#reviews)

## Data
건강검진 데이터(국민보험공단)
약 2만명의 데이터를 기반으로 분석진행
[(features explain)](https://github.com/kimmy-git/Health_data_project/blob/main/features.txt)
### alcohol == Target

### 1. Data EDA
* Data Preprocessing
    - 사용한 데이터
        - 10만개의 데이터 중 2만개의 데이터만 사용(= 빠른 구현을 위함)

    - 결측치 처리 및 필요 없는 Feature 삭제
        - '치아우식증유무', '결손치유무', '치아마모증유무', '제3대구치(사랑니)이상',
          '치석', '데이터공개일자', '기준년도'
        - 치아의 상태이 음주 여부에 중요한 특성이 될 수 있겠지만 너무 많은 결측치이기 때문에 제외
        - '요단백', 'LDL콜레스테롤' 최빈값, 중간값 적용(= 평균치는 이상치에 영향이 있기 때문에 중간값으로 지정)
        - 이외의 결측치는 각 feature마다 많이 존재하지 않기 때문에 제외

    - 전처리
        - Features 이름 변경
        - BMI 지수 Feature 생성(= 체중 / ( (신장 / 100) ** 2 ))
        - 시력 클래스 중 9.9 = 실명자 ==> 제외
        - 이상치 처리(= 상위 0.1%) 
        - => 이상치 제거 후 데이터의 구조 및 이상치 다시 확인 필요
        - => 이상치 제거를 통해 어떤 부분을 학습 데이터로 사용할 건지에 대한 설명이 필요함
 
* Data Visualization
    * heatmap - feature correlation(!주의! 상관관계는 인과관계를 뜻하는 것이 아님)
        - 간수치에 대한 특성이 양의 선형관계를 보여줄 줄 알았는데 거의 0에 가까움(= 거의 의미가 없다는 뜻)
        - 간수치 특성중 GTP와 흡연 여부가 양의 션형관계를 보임
        - 추가적으로 0에 근접한 Feature 제외(= '총콜레스테롤', '시도코드', '요단백')
        - '가입자번호'도 제거할 필요가 있음(= 범주형이기 때문에 큰 의미가 없음)
    * seaborn - 데이터 분포 확인
        - 데이터는 남/여 거의 비슷한 분포를 가지고 Target의 분포 또한 일정함
        - 남성이 많이 음주를 한다고 답한 반면 여성은 반대 양상을 보여줌(= 현재 데이터한에서)
        - 체중과 신장의 경우 음주를 하는 경우 보통 큰 값을 가지는데 이는 남성이 주로 체격이 크기 때문에 이러한 양상을 띔
        - 흡연(= smoke)을 하는 사람이 주로 음주도 하는 것으로 보임
        - GTP의 경우 값이 많이 편향되어 있기 때문에 log변환을 통한 시각화(= 음주를 하는 사람이 대체로 값이 큰 값을 가진다고 볼 수 있음)
        - 시력하고는 특별한 관계를 설명할 수 없음(= 크게 의미가 없는 것으로 보임)
        - 헤모글로빈(= hemo)의 경우는 안마시는 사람보다 마시는 사람이 값이 높고 마시는 사람과 안마시는 사람의 값이 반대 양상을 보임

### 2. Data split
- split, stratify=target 적용
- => 각각의 class 비율을 train / validation에 유지
- => 한 쪽에 쏠려서 분배되는 것을 방지


### 3. Modeling(Randomforestclassifier, XGBClassifier)
- Bagging, RandoForest
    - 질문을 여러가지로 나눠서 좋은 결과의 총합을 평균내어 예측

- Boosting, XGBoost
    - 순차적으로 학습하고 오차를 본완해가며 정확도를 향상시키는 모델
    
### Baseline Model - 50%
```python
# 전체 데이터 기준모델 설정
major= df[target].mode()[0]
pred= [major] * len(df[target])
baseline= accuracy_score(df[target], pred)
print('baseline_accuarcy_Score=', baseline)
```
분류기준모델의 성능은 특성의 최빈값 비율로 설정

**why?** 예측모델이 사람보다 분류를 잘한다면 의미있는 모델이 구축 됐다고 볼 수 있다.

## Metrics and Score
* accuracy - 현재 분포 비율이 비슷하기때문에 Target값을 정확히 예측한 평가지표
* f1_score - 실제 술을 마신다는 사람을 마신다고 예측한 것에(=재현율) 비중을 둘 수 있는 평가지표
* auc_roc curve - Target을 잘 구분하는지를 판단하는 평가지표(= 주된 지표로 진행)

### 4. Hyperparameter Tunning
* make_pipeline 임의값 설정 후 model fit, score 확인
* 모델 성능 향상을 위한 Hyperparameter Tunning 진행
```python
RandomForest
dists = {
    'randomforestclassifier__criterion': ['entrophy', 'gini'], # 분할 품질을 측정하는 기능
    'randomforestclassifier__n_estimators': randint(50, 120), # 모델에 사용할 트리 개수
    'randomforestclassifier__max_depth': np.arange(1, 10), # 트리 깊이
    'randomforestclassifier__max_features': uniform(0, 1), # 분할에 사용할 특성 수
    'randomforestclassifier__min_samples_leaf': np.arange(1, 15, 5) # 리프 노드에 있어야 할 최소 샘플 수
}

clf_rf = RandomizedSearchCV(
    pipe_rf, 
    param_distributions=dists, 
    n_iter=50, 
    cv=3, 
    scoring= 'accuracy',  
    verbose=1,
    n_jobs=-1
)


XGboost
dists_xg = {
    'xgbclassifier__n_estimators': randint(50, 120), # 모델에 사용할 트리 개수
    'xgbclassifier__max_depth': np.arange(1, 10), # 트리 깊이
    'xgbclassifier__max_features': uniform(0, 1), # 분할에 사용할 특성 수
    'xgbclassifier__min_samples_leaf': np.arange(1, 15, 5) # 리프 노드에 있어야 할 최소 샘플 수
}
clf_xg = RandomizedSearchCV(
    pipe_xg, 
    param_distributions=dists_xg, 
    n_iter=50, 
    cv=3, 
    scoring='accuracy',  
    verbose=1,
    n_jobs=-1
)

clf_xg.fit(X_train, y_train);
```
### Confusion Matrix 확인
<p align="center"><img src="https://user-images.githubusercontent.com/83389640/144187159-6e685fed-43b1-491a-a7b4-efdee5ac0635.png"></p>

### Model 성능
|model|RandomForest(Before)|RandomForest(After)|XGboost(Before)|XGboost(After)|
|:-----|:---------|:--------|:--------|:----------|
|f1_score|69.827|72.7781|72.9101|73.1569|
|roc_auc_score|77.2028|80.8852|81.2516|81.3268|
|accuracy_score|69.9617|72.0663|72.5128|72.8316|

### 성능이 제일 높은 XGboost 사용
* Test data - PermutationImportance 순열중요도 확인
* SHAP, PDP를 활용하여 예측모델 설명(PDP => ppt, ipynb 확인)
* SHAP, PDP = Target에 대한 각 특성들의 영향

### 5. Predictive Model Explain
- ![ppt참고](https://github.com/kimmy-git/Health_data_project/blob/main/Health_data_project(ppt).pptx)

#### 1) PermutationImportance
<p align="center"><img src="https://user-images.githubusercontent.com/83389640/144193904-183dc256-f477-43e0-b6d6-59aa9d985156.png"></p>

#### 2) SHAP Visualization
* 예측값과 실제값 비교 DataFrame 형성
* 예측 확률 임계치 0.5 으로 설정( Right, Wrong 구분 )

#### 예측성공
* True Positive - 술을 마시며 마신다고 예측 = 91%
* True Negative - 술을 마시지 않으며 마시지 않는다고 예측 = 87%
![initial](https://user-images.githubusercontent.com/83389640/144185757-731d1ac0-967c-4cf1-92f9-aef6a2b4c8a6.png)
#### 예측실패
* False Negative - 술을 마시지만 마시지 않는다고 예측 = 67%
* False Positve - 술을 마시지 않지만 마신다고 예측 = 68%
![initial](https://user-images.githubusercontent.com/83389640/144185800-f7540fea-9873-44f0-946b-d02b5d5c96dc.png)


## Results
* **XGboost 82% 성능 모델 구현**
* **가설 : 음주는 간수치 특성이 가장 영향이 있을 것, 가설은 틀렸나???**
![Process](https://user-images.githubusercontent.com/83389640/177256142-b29037f8-407c-454c-9663-b37b32c1d3cb.jpg)
    * = 음주여부를 판단하는데 가장 크게 영향을 주는 것은 성별과 흡연여부, gtp
    * => 분석관점보다는 모델 성능에 중점으로 진행했기 때문에 데이터 구조 파악 및 분석 내용이 많이 부실함
    * => 그렇기 때문에 특정 데이터에서 어떤 형태의 분포를 보이는지에 대한 설명력 부족, 이는 가설을 뒷받침하기 어려움

* 결과적으로 모델 성능면에서는 좋다고 판단이 되나 분석면에서 설명력이 부족 => 결과에 대한 신뢰도가 떨어진다.
    * = 더 면밀한 데이터 분석이 필요함
    * = 데이터에서도 어느 부분의 데이터를 학습을 시킬건지에 대한 기준을 잡을 필요가 있음
    * => [Comento 직무부트캠프 수강](kaggle)

## Reviews
* 어려웠던 점
    - 주제가 따로 정해진 것이 없어서 주제 선정 부분부터 많은 어려움이 있었음
        - 난 무엇을 하고자 하는가???
        - 이전 일을 하면서 느꼈던 경험을 통해 선정하게됨
        - 건강데이터는 분석 목적으로 많이 공개한다는 것을 알게됨
    - 데이터를 어디까지 봐야하는지에 대한 기준도 없었고 기술적인 역량이 많이 부족했음
        - 현재 데이터는 어떤 양상을 보이는지 면밀히 보지 못한 아쉬움이 큼 
* 느낀점
    - 각 Feature가 무엇을 뜻 하는지 어떤 영향이 있을지 직접 찾아본 것은 좋은 공부가 되었음.
    - 프로젝트를 진행하기에 앞서 먼저 내가 무엇을 하고자하는가에 대한 관점이 필요함
        - 난 무엇을 하고자 하는가, 주제를 찾는 부분에서 많은 시간을 할애함
        - 분석을 일관성 있게 적용하기 위해서 왜 해당 프로젝트를 진행할려고 했는지 이를 통해서 해결하고자하는 것은 무엇인지에 대해 명확하게 명시
    - 하지만 데이터 분석관점보다는 예측 모델의 성능을 중점으로 앞서 정했던 가설과 목적에 대한 설명력 부족으로 모호한 결과를 가져옴
        - 모델을 위한 분석이 아닌 분석을 위한 모델링과 방법론이 되어야한다는 것을 느낌

* 개선점
    - 데이터 구조를 먼저 파악해야함(범주형, 수치형)
        - Target은 범주형인데 현재 수치형 상태로 그대로 모델에 적용
        - 즉, 숫자의 크기에 따라 학습에 영향이 갈 수 있음

    - 적용할 특성(=독립변수)을 정함으로써 설명력을 높일 필요가 있음
        - 각 Feature의 구조적 특징을 확인하고 독립변수로 사용한 이유에 대한 설명이 필요함
        - 현재 Feature의 설명력이 많이 부족하고 왜 이상치는 제거를 했는지 또한 설명이 없음으로 결과에 대한 신뢰도가 떨어짐
    
