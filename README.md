# Drinking Prediction Model with Health Data
* 의료 계열에서 종사하면서 다양한 오류들을 확인
* 특히, 임플란트 수술의 성공여부에는 많은 변수가 존재하고 이런 변수를 줄이고자 수술 전 환자의 건강 상태를 확인하는 것은 필수

**이 과정에서 음주 여부를 측정할 수 있는 서비스를 구축하여 성공적인 수술에 도움이 되기 위함**


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


## Data
건강검진 데이터(국민보험공단)
약 2만명의 데이터를 기반으로 분석진행
[(features explain)](https://github.com/kimmy-git/Health_data_project/blob/main/features.txt)

## Model
이진분류문제 - 분류모델 사용(Randomforestclassifier, XGBClassifier)

## Metrics and Score
* accuracy
* f1_score
* auc_roc curve

## Train Process
### Data Preprocessing(EDA) -> Data Visualization -> (Train, val, Test) split -> 
### Modeling(Hyperparameter Tunning) -> Metrics and Score -> Model Explain

### 1. Data Preprocessing
* Data 일부만 추출(2만명)
* 결측치가 많은 Features 제거 + 결측치 최빈값, 중간값으로 설정
* BMI 지수 Feature 추가(EDA)
* describe로 이상치 확인 => 상위 1% 제거

### 2. Data Visualization
* seaborn - 데이터 분포 확인
* heatmap - feature correlation 확인

### 3. (Train, val, Test) split
* Target = 'alcohol', 데이터 split

### 4. Modeling(Randomforestclassifier, XGBClassifier)
* make_pipeline 임의값 설정 후 model fit
```python
RandomForest
pipe_rf = make_pipeline(
    RandomForestClassifier(n_jobs=-1, 
                           max_depth= 2,
                           min_samples_split=2,
                           min_samples_leaf= 10,
                           random_state=2)
)

XGboost
pipe_xg = make_pipeline(
    XGBClassifier(max_depth=2,
                  random_state=10,
                  min_samples_split=2,
                  min_samples_leaf= 10,
                  n_jobs=-1) 

)
```

### 5. 모델 성능 향상을 위한 Hyperparameter Tunning
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

### 6. XGboost Model 
**Baseline Model** - 50%
```python
# 전체 데이터 기준모델 설정
major= df[target].mode()[0]
pred= [major] * len(df[target])
baseline= accuracy_score(df[target], pred)
print('baseline_accuarcy_Score=', baseline)
```
분류기준모델의 성능은 특성의 최빈값 비율로 설정

**why?** 예측모델이 사람보다 분류를 잘한다면 의미있는 모델이 구축 됐다고 볼 수 있다.

### 성능이 제일 높은 XGboost 사용
* Test data - PermutationImportance 순열중요도 확인
* SHAP, PDP를 활용하여 예측모델 설명(PDP => ppt, ipynb 확인)
* SHAP, PDP = Target에 대한 각 특성들의 영향

### 7. Predictive Model Explain

#### 1) PermutationImportance
<p align="center"><img src="https://user-images.githubusercontent.com/83389640/144193904-183dc256-f477-43e0-b6d6-59aa9d985156.png"></p>

#### 2) SHAP Visualization
* 예측값과 실제값 비교 DataFrame 형성
* 예측 확률 임계치 0.5 으로 설정( Right, Wrong 구분 )

**index로 접근했기 때문에 number + 1 = 환자번호**
#### 예측성공
* True Positive - 술을 마시며 마신다고 예측 = 91%
* True Negative - 술을 마시지 않으며 마시지 않는다고 예측 = 87%
![initial](https://user-images.githubusercontent.com/83389640/144185757-731d1ac0-967c-4cf1-92f9-aef6a2b4c8a6.png)
#### 예측실패
* False Negative - 술을 마시지만 마시지 않는다고 예측 = 67%
* False Positve - 술을 마시지 않지만 마신다고 예측 = 68%
![initial](https://user-images.githubusercontent.com/83389640/144185800-f7540fea-9873-44f0-946b-d02b5d5c96dc.png)

## Tools
* Python
* numpy
* matplotlib, seaborn
* sklearn
* eli5
* pdpbox
* shap
* XGboost

## Results
* XGboost 82% 성능 모델 구현
* 음주여부에 가장 크게 영향을 주는 것은 성별과 흡연여부, gtp 순으로 높은 것을 확인
* ==> 해당 특성을 중점으로 피드백을 통해 생활 습관 

## Reviews
1. 더 많은 데이터를 사용했다면 더 성능이 좋은 모델을 구축할 수 있을까?
2. 상위 1%이상치 제거, 전처리는 충분한가? 
3. 82%의 성능은 충분한가?
