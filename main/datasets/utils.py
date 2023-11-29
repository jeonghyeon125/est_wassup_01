import pandas as pd
from typing import Type


def custom_X_preprocess_cat(X_df: Type[pd.DataFrame]) -> Type[pd.DataFrame]:
  X_df = X_df.copy()
  
  # ['사고일시', '요일', '기상상태', '시군구', '도로형태', '노면상태', '사고유형']
  # 사고일시 yyyy-mm-dd hh datetime - 시간 컬럼 생성
  if '사고일시' in X_df.columns:
    X_df['시간'] = X_df['사고일시'].str.split(' ').str[-1].astype(int)
    X_df.drop(columns=['사고일시'], inplace=True)
  
  # 도로형태 e.g. 교차로 - 교차로안 -> 도로형태와 세부도로형태로 분리
  if '도로형태' in X_df.columns:
    X_df[['도로형태-1', '도로형태-2']] = X_df['도로형태'].str.split(' - ', expand=True)
    X_df.drop(columns=['도로형태'], inplace=True)
  
  # 시군구 대전데이터 한정으로, 군, 동 정보만 남긴다.
  if '시군구' in X_df.columns:
    # 군, 구(동)
    X_df[['군', '구(동)']] = X_df['시군구'].str.split(' ', expand=True)[[1,2]]
    X_df.drop(columns=['시군구'], inplace=True)
    
    # df_test 의 unique values 와 df_train 의 unique values 가 다르므로 우선 제외시킴
    X_df.drop(columns=['구(동)'], inplace=True)
  
  if '노면상태' in X_df.columns:
    # 건조 젖음/습기 기타 서리/결빙 침수 적설
    X_df['노면상태'] = X_df['노면상태'].map(lambda x: 
      0 if x == '건조' else
      1 if x == '기타' else
      2 if x == '젖음/습기' else
      3 if x == '서리/결빙' else
      4
    )
  
  if '기상상태' in X_df.columns:
    # 맑음 비 흐림 기타 눈
    X_df['기상상태'] = X_df['기상상태'].map(lambda x: 
      0 if x == '맑음' else
      1 if x == '기타' else
      2 if x == '흐림' else
      3 if x == '비' else
      4
    )
  
  if '요일' in X_df.columns:
    # 월 화 수 목 금 토 일
    X_df['요일'] = X_df['요일'].map(lambda x: 
      0 if x == '월' else
      1 if x == '화' else
      2 if x == '수' else
      3 if x == '목' else
      4 if x == '금' else
      5 if x == '토' else
      6
    )
  
  return X_df