<ul align="center">
  
![middle](https://capsule-render.vercel.app/api?type=cylinder&color=0147FF&height=150&section=header&text=Wassup&fontColor=FFFFFF&fontSize=70&animation=fadeIn&fontAlignY=55)
  
**WR(Water_Rocket)** <br>
김도연 : Team Leader (97_회로전공,ENTJ)<br> 
김선들 : Git Branch management (96_컴퓨터전공,INTJ)<br>
김정현 : 데이터 분석,  모델 성능 개선(01_소프트웨어전공,ISFJ)<br>
박건수 : 데이터 분석,  모델 성능 개선(96_정보통신공학전공,ISTP)<br></ul>

# 실험보고서
|실험내용|실험보고서|발표자료|
|------|---|---|
|[대구교통사고실험.xlsx](https://github.com/electronicguy97/est_wassup_01/files/13645556/default.xlsx)|[인공지능 보고서.docx](https://github.com/electronicguy97/est_wassup_01/files/13645944/default.docx)|[발표차료_첫째주.pdf](https://github.com/electronicguy97/est_wassup_01/files/13645586/_.pdf)|
|||[중간발표.pdf](https://github.com/electronicguy97/est_wassup_01/files/13645589/default.pdf)|
|||[최종발표.pdf](https://github.com/electronicguy97/est_wassup_01/files/13645591/default.pdf)|
  
# prepocess_config .py  
  
DROP_X_COLS - data에서 사용하지 않을 컬럼을 지정(실험용으로 제작 사용하지 않으셔도 됩니다.-> 빈리스트)
USE_X_COLS - data에서 사용할 컬럼을 지정  
TARGET_COLS - data의 target 컬럼을 지정  
EMBEDDING_COLS - 임베딩(벡터화)시킬 컬럼 설정

+ config  
  + input_data  
      + train_csv : train data path  
      + test_csv : test data path  

  + output_data
      + train_feas_csv : 가공되어 저장 train data path  
      + test_feas_csv : 가공되어 저장 test data path  
      + train_target_csv : train data의 TARGET컬럼만 추출 후 저장 path  
      + y_scaler_save : train_target에 사용한 scaler를 저장할 path  
  + add_data
    + merge할 추가 csv
  
  + options
      + embedding_cols : 임베딩(벡터화)할 컬럼
      + index_col : train과 test에서 index로 사용할 컬럼 이름  
      + target_cols : 타겟 칼럼  
      + ignore_drop_cols : True - 컬럼들을 drop 하지 않음, False - DROP_X_COLS의 컬럼들을 drop 함  
      + drop_cols : 제외 할 피쳐  
      + use_cols : 사용 할 피쳐  
      + fill_num_strategy : null 값이 있는 부분을 채울 방법을 min, mean, max 중에서 선택  
      + x_scaler : SCALER['minmax']() put None for no X scaling process  
      + y_scaler : SCALER['minmax']() put None for no y scaling process  

# config.py  

+ SCHEDULER
    + CAWarmRestarts: optim.lr_scheduler.CosineAnnealingWarmRestarts  
        + [참조링크](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingWarmRestarts.html#torch.optim.lr_scheduler.CosineAnnealingWarmRestarts)  
    + CALR: optim.lr_scheduler.CosineAnnealingLR  
        +  [참조링크](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html)  
    + lambdaLR: optim.lr_scheduler.LambdaLR  
        + [참조링크](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.LambdaLR.html)  
+ EMBEDDING_COLS = [임베딩 시킬 피쳐]
+ EMBEDDING_NODE = 늘릴 차원 수
+ CUSTOM_WEIGHT = 가중치 원하는 크기 ex)[10,5,3,1] 사망자~부상자 가중치
+ NUM+OF_TASKS = 가중치 총길이

+ EXPERIMENT_NAME = output_files의 경로 중간의 삽입되는 이름  

+ CUSTOM_LOSS
  + multi_task_mse : 가중치를 적용한 mse
  + rmsle : rmsle 

+ EXPERIMENT_NAME = 최종 데이터(loss,eval등) 저장할 경로
+ config  
    + input_files  
        + X_csv : 가공한 train data path, preprocess의 output_data의 train_feas_csv 경로  
        + y_csv : 가공한 target data path, preprocess의 output_data의 train_target_csv 경로  
        + tst_csv : 가공한 test data path, preprocess의 output_data의 test_feas_csv 경로  
  
    + output_files  
        + output_train : 훈련 후 저장할 모델의 저장 할 경로  
        + output_eval : 훈련 후 저장할 평가지표 저장 할 경로  
        + 'output_pred : 훈련 후 test데이터로 예측한 결과 저장 할 경로  
  
    + model: ANN - 사용할 모델 class  
    + model_params  
        + input_dim: 'auto' - 입력한 데이터의 차원수를 자동으로 맞춰 줌  
        + hidden_dim : 훈련 시 hidden_dim의 차원과 레이어 수  
        + use_drop : drop out의 사용 유무 True - 사용, False - 사용 안함  
        + drop_ratio : drop out 사용 시 수치  
        + activation : 사용할 활성 함수 선택  
  
    + train_params  - 스케줄러 관한 설정  
        + use_scheduler: 스케줄러 사용 유/무 설정, True - 사용, False - 사용 안 함  
        + scheduler_cls: SCHEDULER['CALR']  
        + scheduler_params  
            + 'T_max': cos 반복 주기  
            + 'eta_min': 최저 lr
  
        + loss : 사용할 손실 함수 선택  
        + optim : 사용할 옵티마이저 선택 ,[sigmoid, relu, tanh, prelu 중 선택
        + main_metric : pbar에 보일 성능지표
        + metrics  
            'rmse': torchmetrics.MeanSquaredError(squared=False)  
            'mse': torchmetrics.MeanSquaredError()  
        + device : 연산할 device를 선택(cpu, cuda 등)  
        + epochs : train과 eval 실행 시 사용할 epoch 수  
        + data_loader_params  
            + batch_size : 배치사이즈 선택  
        + optim_params  
            + lr : learning rate 선택  
    + cv_params  
        + n_split : evaluation실행 시 사용할 fold 수 선택

# EDA

![image](https://github.com/electronicguy97/est_wassup_01/assets/103613730/51435928-e6f8-4579-8ac3-d39744dededf)<br>
ECLO = 사망자수 * 10 + 중상자수 * 5 + 경상자수 * 3 + 부상자수<br>
ECLO를 뜻하는 사상자수 외에는 큰 상관관계를 찾을 수 없다. 비선형관계 또는 Multicollinearity(다중공산성)등으로 예측 된다.<br>


![image](https://github.com/electronicguy97/est_wassup_01/assets/103613730/cee6b82b-6d8f-4065-b2e1-f997b3e0e5b8)<br>
ECLO가 18인 경우 9월 외에는 다른 데이터의 분포는 비슷한 것을 보아 상관관계가 높지 않다는 것을 알 수 있다.

![image](https://github.com/electronicguy97/est_wassup_01/assets/103613730/a96cb048-5a44-4bee-ab85-042ab8ecba61)<br>
해당 feature는 데이터의 불균형이 심해 오히려 상관관계가 높지 않은 것으로 예측된다.

---
<img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=PyTorch&logoColor=white"> <img src = "https://img.shields.io/badge/python-3776AB?style=for-the-badge&logo=python&logoColor=white">
