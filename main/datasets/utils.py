import pandas as pd
from typing import Type


def custom_X_preprocess_cat(X_df: pd.DataFrame, add_df_list: list[pd.DataFrame]) -> pd.DataFrame:
    X_df = X_df.copy()

    # ['사고일시', '요일', '기상상태', '시군구', '도로형태', '노면상태', '사고유형']
    # 사고일시 yyyy-mm-dd hh datetime - 시간 컬럼 생성
    if '사고일시' in X_df.columns:
        X_df['사고일시'] = pd.to_datetime(X_df['사고일시'])
        X_df['시간'] = X_df['사고일시'].dt.hour
        X_df['월'] = X_df['사고일시'].dt.month
        X_df.drop(columns=['사고일시'], inplace=True)

    # 도로형태 e.g. 교차로 - 교차로안 -> 도로형태와 세부도로형태로 분리
    # 전국 도로형태별 2019 ~ 2021 치사율
    if '도로형태' in X_df.columns:
        X_df[['도로형태-1', '도로형태-2']] = X_df['도로형태'].str.split(' - ', expand=True)
        X_df.drop(columns=['도로형태'], inplace=True)

    # 시군구 대전데이터 한정으로, 군, 동 정보만 남긴다.
    if '시군구' in X_df.columns:
        # 구, 동
        X_df[['구', '동']] = X_df['시군구'].str.split(' ', expand=True)[[1, 2]]
        X_df.drop(columns=['시군구'], inplace=True)

        # Add new features from external datasets
        X_df = merge_features_from_externals(X_df, add_df_list)
        # X_df = X_df.drop(columns=['동'])

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
        3 if x == '비' or x == '안개' else
        4
      )

    if '요일' in X_df.columns:
      # 월 화 수 목 금 토 일
      X_df['요일'] = X_df['요일'].map(lambda x:
        0 if x == '월요일' else
        1 if x == '화요일' else
        2 if x == '수요일' else
        3 if x == '목요일' else
        4 if x == '금요일' else
        5 if x == '토요일' else
        6
      )

    return X_df


def merge_features_from_externals(X_df: pd.DataFrame, list_add_df: list[pd.DataFrame]) -> pd.DataFrame:
    X_df = X_df.copy()
    for i in range(len(list_add_df)):
        # if i == 4: continue
        method = getattr(MergeData, f'_merge_data_{i}')
        if method:
            num_data = len(X_df)
            X_df = method(X_df, list_add_df[i])
            if len(X_df) != num_data:
                raise ValueError(f"num data changed: {num_data} -> {len(X_df)}: idx:{i}")
            if X_df.isnull().values.any():
                raise ValueError(f"nan exists in {i}th data")
    return X_df


# MergeData
class MergeData:
    @staticmethod
    def _merge_data_0(X_df: pd.DataFrame, add_df: pd.DataFrame) -> pd.DataFrame:
        """
        bad_cars.csv
        """
        merged_df = pd.merge(X_df, add_df, left_on='구', right_on='index', how='left')
        merged_df = merged_df.drop(columns=['index'])
        if merged_df.isnull().values.any():
            raise ValueError(f"has nan")
        return merged_df

    @staticmethod
    def _merge_data_1(X_df: pd.DataFrame, add_df: pd.DataFrame) -> pd.DataFrame:
        """
        bad_pedestrain.csv
        """
        merged_df = pd.merge(X_df, add_df, left_on='동', right_on='동', how='left')
        merged_df = merged_df.rename(columns={
            "사고건수": "사고건수_bad_pedestrian",
            "사상자수": "사상자수_bad_pedestrian",
            "사망자수": "사망자수_bad_pedestrian",
            "중상자수": "중상자수_bad_pedestrian",
            "경상자수": "경상자수_bad_pedestrian",
        })
        if merged_df.isnull().values.any():
            merged_df = merged_df.fillna(0)
        return merged_df

    @staticmethod
    def _merge_data_2(X_df: pd.DataFrame, add_df: pd.DataFrame) -> pd.DataFrame:
        """
        car_per_area_size_dong.csv
        """
        merged_df = pd.merge(X_df, add_df, left_on=['구', '동'], right_on=['구군', '읍면동'], how='left')
        merged_df = merged_df.drop(columns=['구군', '읍면동'])
        if merged_df.isnull().values.any():
            raise ValueError(f"has nan")
        return merged_df

    @staticmethod
    def _merge_data_3(X_df: pd.DataFrame, add_df: pd.DataFrame) -> pd.DataFrame:
        """
        car_speed.csv
        """
        merged_df = pd.merge(X_df, add_df, left_on=['시간', '구', '동'], right_on=['시간', '시군구명', '동'], how='left')
        merged_df = merged_df.drop(columns=['시군구명'])

        if merged_df.isnull().values.any():
            new_cols = ['평균속도']
            for new_col in new_cols:
                merged_df[new_col] = merged_df[new_col].fillna(merged_df[new_col].mean())
        return merged_df

    @staticmethod
    def _merge_data_4(X_df: pd.DataFrame, add_df: pd.DataFrame) -> pd.DataFrame:
        """
        car_traffic.csv
        """
        merged_df = pd.merge(X_df, add_df, left_on=['시간', '구', '동'], right_on=['시간', '시군구명', '동'], how='left')
        merged_df = merged_df.drop(columns=['시군구명'])
        merged_df = merged_df.rename(columns={
            '전체': '전체_car_traffic',
            '승용차': '승용차_car_traffic',
            '버스': '버스_car_traffic',
            '트럭': '트럭_car_traffic',
        })
        if merged_df.isnull().values.any():
            new_cols = [
                '전체_car_traffic',
                '승용차_car_traffic',
                '버스_car_traffic',
                '트럭_car_traffic',
            ]
            for new_col in new_cols:
                merged_df[new_col] = merged_df[new_col].fillna(merged_df[new_col].mean())
        return merged_df

    @staticmethod
    def _merge_data_5(X_df: pd.DataFrame, add_df: pd.DataFrame) -> pd.DataFrame:
        """
        frozen_road.csv
        """
        merged_df = pd.merge(X_df, add_df, left_on='동', right_on='dong', how='left')
        merged_df = merged_df.rename(columns={
            "사고건수": "사고건수_frozen_road",
            "사상자수": "사상자수_frozen_road",
            "사망자수": "사망자수_frozen_road",
            "중상자수": "중상자수_frozen_road",
            "경상자수": "경상자수_frozen_road",
        })
        if merged_df.isnull().values.any():
            merged_df = merged_df.fillna(0)
        merged_df = merged_df.drop(columns=['dong'])
        return merged_df

    @staticmethod
    def _merge_data_6(X_df: pd.DataFrame, add_df: pd.DataFrame) -> pd.DataFrame:
        """
        moving_pop_by_age.csv
        """
        merged_df = pd.merge(X_df, add_df, left_on=['월', '구', '동'], right_on=['월', '구', '동'], how='left')
        merged_df = merged_df.rename(columns={
            '10대': '10대_moving_pop_by_age',
            '20대': '20대_moving_pop_by_age',
            '30대': '30대_moving_pop_by_age',
            '40대': '40대_moving_pop_by_age',
            '50대': '50대_moving_pop_by_age',
            '60대': '60대_moving_pop_by_age',
        })

        if merged_df.isnull().values.any():
            new_cols = [
                '10대_moving_pop_by_age',
                '20대_moving_pop_by_age',
                '30대_moving_pop_by_age',
                '40대_moving_pop_by_age',
                '50대_moving_pop_by_age',
                '60대_moving_pop_by_age',
            ]
            for new_col in new_cols:
                merged_df[new_col] = merged_df[new_col].fillna(merged_df[new_col].mean())
        return merged_df

    @staticmethod
    def _merge_data_7(X_df: pd.DataFrame, add_df: pd.DataFrame) -> pd.DataFrame:
        """
        moving_pop_by_days.csv
        """
        merged_df = pd.merge(X_df, add_df, left_on=['월', '구', '동'], right_on=['월', '구', '동'], how='left')
        if merged_df.isnull().values.any():
            new_cols = ['월요일', '화요일', '수요일', '목요일', '금요일', '토요일', '일요일']
            for new_col in new_cols:
                merged_df[new_col] = merged_df[new_col].fillna(merged_df[new_col].mean())
        return merged_df

    @staticmethod
    def _merge_data_8(X_df: pd.DataFrame, add_df: pd.DataFrame) -> pd.DataFrame:
        """
        moving_pop_by_time.csv
        """
        merged_df = pd.merge(X_df, add_df, left_on=['월', '구', '동'], right_on=['월', '구', '동'], how='left')
        new_cols = [
            '0',
            '1',
            '2',
            '3',
            '4',
            '5',
            '6',
            '7',
            '8',
            '9',
            '10',
            '11',
            '12',
            '13',
            '14',
            '15',
            '16',
            '17',
            '18',
            '19',
            '20',
            '21',
            '22',
            '23',
        ]
        for new_col in new_cols:
            if merged_df[new_col].isnull().values.any():
                merged_df[new_col] = merged_df[new_col].fillna(merged_df[new_col].mean())

        return merged_df

    @staticmethod
    def _merge_data_9(X_df: pd.DataFrame, add_df: pd.DataFrame) -> pd.DataFrame:
        """
        old_pedestrian.csv
        """
        merged_df = pd.merge(X_df, add_df, left_on='동', right_on='dong', how='left')
        merged_df = merged_df.rename(columns={
            "사고건수": "사고건수_old_pedestrian",
            "사상자수": "사상자수_old_pedestrian",
            "사망자수": "사망자수_old_pedestrian",
            "중상자수": "중상자수_old_pedestrian",
            "경상자수": "경상자수_old_pedestrian",
        })
        if merged_df.isnull().values.any():
            merged_df = merged_df.fillna(0)
        merged_df = merged_df.drop(columns=['dong'])
        return merged_df

    @staticmethod
    def _merge_data_10(X_df: pd.DataFrame, add_df: pd.DataFrame) -> pd.DataFrame:
        """
        pedestrian.csv
        """
        merged_df = pd.merge(X_df, add_df, left_on='동', right_on='dong', how='left')
        merged_df = merged_df.rename(columns={
            "사고건수": "사고건수_pedestrian",
            "사상자수": "사상자수_pedestrian",
            "사망자수": "사망자수_pedestrian",
            "중상자수": "중상자수_pedestrian",
            "경상자수": "경상자수_pedestrian",
        })
        if merged_df.isnull().values.any():
            merged_df = merged_df.fillna(0)
        merged_df = merged_df.drop(columns=['dong'])
        return merged_df

    @staticmethod
    def _merge_data_11(X_df: pd.DataFrame, add_df: pd.DataFrame) -> pd.DataFrame:
        """
        population_mildo_gu_basis.csv
        """
        merged_df = pd.merge(X_df, add_df, left_on=['구'], right_on=['구'], how='left')
        if merged_df.isnull().values.any():
            raise ValueError(f"has nan")
        return merged_df

    @staticmethod
    def _merge_data_12(X_df: pd.DataFrame, add_df: pd.DataFrame) -> pd.DataFrame:
        """
        rain_mm_monthly.csv
        """
        add_df = add_df.copy()
        add_df['월'] = add_df['월'].map(lambda x: int(x[:-1]))
        merged_df = pd.merge(X_df, add_df, left_on='월', right_on='월', how='left')
        if merged_df.isnull().values.any():
            raise ValueError(f"has nan")
        return merged_df

    @staticmethod
    def _merge_data_13(X_df: pd.DataFrame, add_df: pd.DataFrame) -> pd.DataFrame:
      """
        road_long_by_dong.csv
      """
      add_df = add_df.copy()
      merged_df = pd.merge(X_df, add_df, on=['구','동'], how='left').fillna(0)
      if merged_df.isnull().values.any():
            raise ValueError(f"has nan")
      return merged_df
    
    @staticmethod
    def _merge_data_14(X_df: pd.DataFrame, add_df: pd.DataFrame) -> pd.DataFrame:
      """
        snow_mean_days_monthly.csv
      """
      add_df = add_df.copy()
      add_df.columns = ['월', '눈_평균일수']
      add_df['월'] = add_df['월'].map(lambda x: int(x[:-1]))
      merged_df = pd.merge(X_df, add_df, on='월', how='left')
      if merged_df.isnull().values.any():
            raise ValueError(f"has nan")
      return merged_df
    
    @staticmethod
    def _merge_data_15(X_df: pd.DataFrame, add_df: pd.DataFrame) -> pd.DataFrame:
      """
        sun_records_monthly_mean.csv
      """
      add_df = add_df.copy()
      add_df['month'] = add_df['month'].astype(int)
      merged_df = pd.merge(X_df, add_df, left_on='월', right_on='month', how='left')
      merged_df = merged_df.drop(columns=['month'])
      if merged_df.isnull().values.any():
            raise ValueError(f"has nan")
      return merged_df
    
    @staticmethod
    def _merge_data_16(X_df: pd.DataFrame, add_df: pd.DataFrame) -> pd.DataFrame:
      """
        truck.csv
      """
      add_df = add_df.copy()
      new_col = [ col if col == 'dong' else f'truck_{col}' for col in add_df.columns]
      add_df.columns = new_col
      merged_df = pd.merge(X_df, add_df, left_on='동', right_on='dong', how='left')
      merged_df = merged_df.drop(columns=['dong'])
      merged_df = merged_df.fillna(0)
      if merged_df.isnull().values.any():
            raise ValueError(f"has nan")
      return merged_df
    
    @staticmethod
    def _merge_data_17(X_df: pd.DataFrame, add_df: pd.DataFrame) -> pd.DataFrame:
      """
        weather_overall_monthly.csv
      """
      add_df = add_df.copy()
      add_df = add_df.reset_index()
      add_df['index'] = add_df['index'].map(lambda x: int(x[:-1]))
      merged_df = pd.merge(X_df, add_df, left_on='월', right_on='index', how='left')
      merged_df = merged_df.drop(columns=['index'])
      if merged_df.isnull().values.any():
            raise ValueError(f"has nan")
      return merged_df