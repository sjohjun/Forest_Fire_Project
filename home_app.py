# -*- coding:utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
import pandas_gbq

import requests
from bs4 import BeautifulSoup
import json
import lxml

from datetime import datetime, timedelta
import time

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import folium
from streamlit_folium import st_folium
from folium.plugins import MarkerCluster
import branca.colormap as cm

from shapely.geometry import Point, Polygon, MultiPolygon, shape
from shapely import wkt

import statsmodels.api as sm

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit, train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix, classification_report,  roc_curve, auc, RocCurveDisplay
from imblearn.over_sampling import SMOTE

from xgboost import XGBClassifier, plot_importance
from lightgbm import LGBMClassifier, plot_importance

import googlemaps
from google.cloud import bigquery
from google.oauth2 import service_account

import utils
import eda_app
import stat_app
import model_app
import service_app

from utils import credentials, servicekey

import os
import warnings
warnings.filterwarnings("ignore")

def get_weather_days_data(serviceKey, weather_stations, start_date_str=None, end_date_str=None):
    """
    지정한 기상 관측소의 일별 날씨 데이터를 조회하여 데이터프레임으로 반환합니다.

    Args:
        serviceKey (str): 공공데이터포털에서 발급받은 인증키.
        weather_stations (pandas.DataFrame): 기상 관측소 정보가 포함된 데이터프레임.
        start_date_str (str, optional): 조회 시작 날짜를 나타내는 문자열 (예: "20220101").
            기본값은 None이며, 기본값일 경우 2013년 1월 1일로 설정됩니다.
        end_date_str (str, optional): 조회 끝 날짜를 나타내는 문자열 (예: "20220331").
            기본값은 None이며, 기본값일 경우 어제 날짜로 설정됩니다.

    Returns:
        pandas.DataFrame: 조회된 일별 날씨 데이터를 담은 데이터프레임 객체.
    """

    url = 'http://apis.data.go.kr/1360000/AsosDalyInfoService/getWthrDataList'

    # 시작 날짜와 끝 날짜를 생성합니다
    if start_date_str is None:
        start_date = datetime.now() - timedelta(days=9)  # 시작 날짜를 2013년 1월 1일로 설정합니다
    else:
        start_date = datetime.strptime(start_date_str, "%Y%m%d")

    if end_date_str is None:
        end_date = datetime.now() - timedelta(days=2)  # 어제 날짜를 구하기 위해 현재 날짜에서 1일을 뺍니다
    else:
        end_date = datetime.strptime(end_date_str, "%Y%m%d")

    end_date_str = end_date.strftime("%Y%m%d")

    all_data = []  # 전체 데이터를 저장할 리스트를 생성합니다

    for stnNm in weather_stations["stnId"]:
        params = {
            'serviceKey': serviceKey,
            'pageNo': '1',  # 초기 페이지 번호를 1로 설정합니다
            'numOfRows': '999',  # 한 페이지에 최대로 가져올 데이터 수를 설정합니다
            'dataType': 'json',
            'dataCd': 'ASOS',
            'dateCd': 'DAY',
            'startDt': start_date.strftime("%Y%m%d"),  # 시작 날짜를 문자열로 변환하여 설정합니다
            'endDt': end_date_str,  # 끝 날짜를 어제 날짜로 설정합니다
            'stnIds': stnNm
        }

        while True:
            try:
                response = requests.get(url, params=params)
                response.raise_for_status()  # 오류가 발생하면 예외를 발생시킴
                data = response.json()
                all_data.extend(data['response']['body']['items']['item'])

                # 다음 페이지로 이동
                params['pageNo'] = str(int(params['pageNo']) + 1)
                if int(params['pageNo']) > int(
                        int(data['response']['body']['totalCount']) / int(params['numOfRows'])) + 1:
                    break
            except requests.exceptions.HTTPError as e:
                print("API 요청 오류:", e.response.text)  # API 요청 오류 메시지 출력
                break
            except Exception as e:
                print(params)
                print(response.content)
                print("예외 발생:", e)  # 기타 예외 발생 시 메시지 출력
                break

    # 리스트에서 데이터프레임을 생성합니다
    weather_days = pd.DataFrame(all_data)

    return weather_days

def get_dataframe_from_bigquery(dataset_id, table_id):
    """
    주어진 BigQuery 테이블에서 데이터를 조회하여 DataFrame으로 반환합니다.

    Args:
        dataset_id (str): 대상 데이터셋의 ID.
        table_id (str): 대상 테이블의 ID.
        key_path (str): 서비스 계정 키 파일의 경로.

    Returns:
        pandas.DataFrame: 조회된 데이터를 담은 DataFrame 객체.
    """

    # BigQuery 클라이언트 생성
    client = bigquery.Client(credentials=credentials, project=credentials.project_id)

    # 테이블 레퍼런스 생성
    table_ref = client.dataset(dataset_id).table(table_id)

    # 테이블 데이터를 DataFrame으로 변환
    df = client.list_rows(table_ref).to_dataframe()

    return df


def get_geodataframe_from_bigquery(dataset_id, table_id):
    """
    주어진 BigQuery 테이블에서 데이터를 조회하여 Geopandas GeoDataFrame으로 반환합니다.

    Args:
        dataset_id (str): 대상 데이터셋의 ID.
        table_id (str): 대상 테이블의 ID.
        key_path (str): 서비스 계정 키 파일의 경로.

    Returns:
        geopandas.GeoDataFrame: 조회된 데이터를 담은 Geopandas GeoDataFrame 객체.
    """

    # 빅쿼리 클라이언트 객체 생성
    client = bigquery.Client(credentials=credentials)

    # 쿼리 작성
    query = f"SELECT * FROM `{dataset_id}.{table_id}`"

    # 쿼리 실행
    df = client.query(query).to_dataframe()

    # 'geometry' 열의 문자열을 다각형 객체로 변환
    df['geometry'] = df['geometry'].apply(wkt.loads)

    # GeoDataFrame으로 변환
    gdf = gpd.GeoDataFrame(df, geometry='geometry')
    gdf.crs = "EPSG:4326"

    return gdf

def today_weather(weather_stations):
    """
    날씨 데이터를 전처리하는 함수입니다.

    Args:
        - weather_stations (pandas.DataFrame): 기상 관측소 정보가 포함된 데이터프레임

    Returns:
        pandas.DataFrame: 전처리된 날씨 데이터가 포함된 데이터프레임
    """
    # 날씨 데이터 가져오기
    weather_days = get_weather_days_data(servicekey, weather_stations)

    # 필요없는 열 제거
    weather_days = weather_days.drop(['stnNm', 'minTaHrmt', 'maxTaHrmt', 'mi10MaxRn', 'mi10MaxRnHrmt', 'hr1MaxRn', 'hr1MaxRnHrmt',
                       'sumRnDur', 'hr24SumRws', 'maxWd', 'avgTd', 'avgPv', 'avgPa', 'maxPs', 'maxPsHrmt',
                       'minPs', 'minPsHrmt', 'avgPs', 'ssDur', 'sumSsHr', 'hr1MaxIcsrHrmt', 'hr1MaxIcsr',
                       'sumGsr', 'ddMefs', 'ddMefsHrmt', 'ddMes', 'ddMesHrmt', 'sumDpthFhsc', 'avgTs', 'minTg',
                       'avgCm5Te', 'avgCm10Te', 'avgCm20Te', 'avgCm30Te', 'avgM05Te', 'avgM10Te', 'avgM15Te',
                       'avgM30Te', 'avgM50Te', 'sumLrgEv', 'sumSmlEv', 'n99Rn', 'iscs', 'sumFogDur',
                       'maxInsWsWd', 'maxInsWsHrmt', 'maxWsWd', 'maxWsHrmt', 'minRhmHrmt', 'avgTca', 'avgLmac'], axis=1)

    # 날짜 데이터 타입으로 변환
    weather_days['tm'] = pd.to_datetime(weather_days['tm'], errors='coerce')

    # 숫자로 변환할 열 선택
    numeric_columns = weather_days.columns.drop("tm")

    # 숫자로 변환
    weather_days[numeric_columns] = weather_days[numeric_columns].apply(pd.to_numeric, errors='coerce')

    # 결측값 0으로 채우기
    weather_days['sumRn'].fillna(0, inplace=True)

    # stnId 별로 데이터프레임 분할
    dfs = []
    for stn_id, group in weather_days.groupby("stnId"):
        # Shift된 열에 처음 값을 추가
        group["h1"] = group["avgRhm"].shift(1)
        group.loc[group.index[0], "h1"] = group["avgRhm"].iloc[0]

        group["h2"] = group["h1"].shift(1)
        group.loc[group.index[0], "h2"] = group["avgRhm"].iloc[0]

        group["h3"] = group["h2"].shift(1)
        group.loc[group.index[0], "h3"] = group["avgRhm"].iloc[0]

        group["h4"] = group["h3"].shift(1)
        group.loc[group.index[0], "h4"] = group["avgRhm"].iloc[0]

        # 실효습도 계산
        r = 0.7
        group["effRhm"] = ((group["avgRhm"]) + (r**1)*(group["h1"]) + (r**2)*(group["h2"]) + (r**3)*(group["h3"]) + (r**4)*(group["h4"])) * (1-r)

        # 6일전부터 기준일까지 7일간 강수량(mm)
        window_size = 7
        group['sumRn7'] = group['sumRn'].rolling(window_size, min_periods=1).sum()

        # 강수 여부, 비 옴 1 / 비 안옴 0
        group['Rntf'] = group['sumRn'].apply(lambda x: 1 if x > 0 else 0)

        # 6일전부터 기준일까지 7일간 최대풍속
        group['maxwind7'] = group['maxWs'].rolling(window_size, min_periods=1).max()

        # 비가 오지 않은 날의 일수를 저장할 새로운 칼럼 추가
        group['noRn'] = 0

        # 일강수량이 0인 날의 연속된 일수를 계산하여 noRn 칼럼에 저장
        count = 0
        for i, value in enumerate(group['sumRn']):
            if value == 0:
                count += 1
            else:
                group.loc[group.index[i], 'noRn'] = count
                count = 0

        dfs.append(group)

    # 데이터프레임 합치기
    weather_days = pd.concat(dfs)

    # 기상 관측소 정보와 병합
    weather_days = weather_days.merge(weather_stations, on='stnId')

    # 필요없는 열 제거
    weather_days = weather_days.drop(['stnId', 'stnAddress', 'stnLatitude', 'stnLongitude', 'h1', 'h2', 'h3', 'h4'], axis=1)

    # 날짜 설정
    target_date = (datetime.now() - timedelta(days=2)).strftime("%Y-%m-%d")

    # 필요한 열을 기준으로 그룹화하고 평균 계산
    weather_days = weather_days[weather_days["tm"] == target_date].reset_index(drop=True)
    weather_days = weather_days.groupby(["w_regions", "tm"]).agg({
        "avgTa": "mean",
        "minTa": "mean",
        "maxTa": "mean",
        "sumRn": "mean",
        "maxInsWs": "mean",
        "maxWs": "mean",
        "avgWs": "mean",
        "minRhm": "mean",
        "avgRhm": "mean",
        "effRhm": "mean",
        "sumRn7": "mean",
        "Rntf": lambda x: int(np.any(x == 1)),
        "maxwind7": "mean",
        "noRn": "mean",
    }).reset_index()

    # 소수점 자리수 설정
    weather_days = weather_days.round({"avgTa": 2, "minTa": 2, "maxTa": 2, "sumRn": 2, "maxInsWs": 2, "maxWs": 2, "avgWs": 2, "minRhm": 2, "avgRhm": 2, "effRhm": 2, "sumRn7": 2})

    # 필요없는 열 제거
    weather_days = weather_days.drop(['tm'], axis=1)

    return weather_days


def split_train_test(data):
    """
    입력된 데이터를 학습 및 테스트 데이터로 분할하고 클래스 불균형을 해결하기 위해 SMOTE를 적용합니다.

    Args:
        data (DataFrame): 피처와 레이블을 포함하는 입력 데이터.

    Returns:
        tuple: 4개의 요소를 포함하는 튜플:
            - X_train_over (DataFrame): SMOTE를 적용한 학습용 피처 데이터.
            - X_test (DataFrame): 테스트용 피처 데이터.
            - y_train_over (Series): SMOTE를 적용한 학습용 레이블 데이터.
            - y_test (Series): 테스트용 레이블 데이터.
    """

    # 기간을 고려하여 train, test 데이터 나누기
    train_start = '2013-01-01'
    train_end = '2020-12-31'
    test_start = '2021-01-01'
    test_end = '2022-12-31'

    train_mask = (data['tm'] >= train_start) & (data['tm'] <= train_end)
    test_mask = (data['tm'] >= test_start) & (data['tm'] <= test_end)

    train_data = data[train_mask]
    test_data = data[test_mask]

    X_train = train_data.drop(['w_regions', 'tm', 'fire_occur'], axis=1)
    y_train = train_data['fire_occur']

    X_train = X_train.astype(float)
    y_train = y_train.astype(int)

    # SMOTE 적용
    smote = SMOTE(random_state=42)

    X_train_over, y_train_over = smote.fit_resample(X_train, y_train)

    X_test = test_data.drop(['w_regions', 'tm', 'fire_occur'], axis=1)
    y_test = test_data['fire_occur']

    X_test = X_test.astype(float)
    y_test = y_test.astype(int)

    return X_train_over, X_test, y_train_over, y_test

def train_logistic_regression(X_train, y_train):
    # Train logistic regression model
    lr_model = LogisticRegression(solver='liblinear', random_state=0)
    lr_model.fit(X_train, y_train)
    return lr_model

def train_xgboost(X_train, y_train):
    # Train XGBoost model
    params = {
        'objective': 'binary:logistic',
        'max_depth': 4,
        'alpha': 10,
        'learning_rate': 1.0,
        'n_estimators': 100
    }
    xgb_model = XGBClassifier(booster='gbtree', importance_type='gain', **params)
    xgb_model.fit(X_train, y_train)
    return xgb_model

def train_lightgbm(X_train, y_train):
    # Train LightGBM model
    params = {
        'class_weight': 'balanced',
        'drop_rate': 0.9,
        'min_data_in_leaf': 100,
        'max_bin': 255,
        'n_estimators': 500,
        'min_sum_hessian_in_leaf': 1,
        'learning_rate': 0.1,
        'bagging_fraction': 0.85,
        'colsample_bytree': 1.0,
        'feature_fraction': 0.1,
        'lambda_l1': 5.0,
        'lambda_l2': 3.0,
        'max_depth': 9,
        'min_child_samples': 55,
        'min_child_weight': 5.0,
        'min_split_gain': 0.1,
        'num_leaves': 45,
        'subsample': 0.75
    }
    lgb_model = LGBMClassifier(boosting_type='dart', importance_type='gain', **params)
    lgb_model.fit(X_train, y_train)
    return lgb_model

def get_dwi_by_pred(pred_proba, num_intervals=10):
    """
    예측값을 기반으로 DWI(Drought Warning Index) 값을 계산하여 출력합니다.

    Args:
        pred_proba (array-like): 예측값.
        num_intervals (int, optional): DWI 등급의 개수. 기본값은 10입니다.

    Returns:
        int: DWI 값
    """

    interval_idx = int(pred_proba * num_intervals)
    if interval_idx == num_intervals:
        interval_idx -= 1
    dwi = interval_idx + 1
    return dwi

def create_dwi_choropleth_map(dataframe, geometry_column, dwi_columns):
    """
    GeoDataFrame을 기반으로 DWI 등급 Choropleth 맵을 생성합니다.

    Args:
        dataframe (geopandas.GeoDataFrame): Choropleth 맵을 생성할 GeoDataFrame.
        geometry_column (str): 지오메트리 정보를 포함하는 열의 이름.
        dwi_columns (str): DWI 등급 값을 포함하는 열의 이름.

    Returns:
        folium.Map: 생성된 Choropleth 맵 객체.
    """
    # 표현할 좌표계 설정
    dataframe.crs = "EPSG:4326"

    # 지도 생성
    map = folium.Map(location=[37.7, 128.3], zoom_start=8)

    # DWI 등급 수와 범위 설정
    num_intervals = 10
    min_value = 1
    max_value = 10
    interval_size = (max_value - min_value) / num_intervals

    # 등급 색상 맵 설정
    colormap = cm.linear.YlOrRd_09.scale(min_value, max_value)

    # 테두리 선 스타일 함수
    def style_function(feature):
        dwi_value = feature['properties'][dwi_columns]
        color = colormap(dwi_value)
        return {
            'fillColor': color,
            'fillOpacity': 0.7,
            'color': 'black',
            'weight': 1,
            'dashArray': '5, 5'
        }

    # Choropleth 맵 생성
    folium.GeoJson(
        dataframe,
        name='choropleth',
        style_function=style_function,
        tooltip=folium.GeoJsonTooltip(fields=[dwi_columns], labels=True, sticky=False),
        highlight_function=lambda x: {'weight': 3},
    ).add_to(map)

    # 범례 추가
    colormap.add_to(map)
    map.add_child(colormap)

    st_folium(map)

def home_app():
    """
        Renders the introduction section of the app, including tabs for overview, objectives, and analysis phases.
    """
    weather_stations = get_dataframe_from_bigquery("PREPROCESSING_DATA", "weather_stations").sort_values(["stnId"])
    gangwon_regions = get_geodataframe_from_bigquery("PREPROCESSING_DATA", "gangwon_regions")

    weather_days = today_weather(weather_stations)

    model_data = {
        "강원북부내륙": ("GangwonNorthInland", train_lightgbm),
        "강원북부산지": ("GangwonNorthMount", train_logistic_regression),
        "강원북부해안": ("GangwonNorthCoast", train_logistic_regression),
        "강원중부내륙": ("GangwonCentralInland", train_logistic_regression),
        "강원중부산지": ("GangwonCentralMount", train_lightgbm),
        "강원중부해안": ("GangwonCentralCoast", train_logistic_regression),
        "강원남부내륙": ("GangwonSouthInland", train_logistic_regression),
        "강원남부산지": ("GangwonSouthMount", train_logistic_regression),
        "강원남부해안": ("GangwonSouthInland", train_logistic_regression)
    }

    dwi_data = []
    for region, (data_table, model_func) in model_data.items():
        data = get_dataframe_from_bigquery("ANALSIS_DATA", data_table).sort_values(["tm"]).reset_index(drop=True)
        data = data[data['tm'] < '2023-01-01']
        X_train, X_test, y_train, y_test = split_train_test(data)
        model = model_func(X_train, y_train)
        pred_proba = model.predict_proba(weather_days[weather_days["w_regions"] == region].drop(['w_regions'], axis=1))[:, 1]
        dwi = get_dwi_by_pred(pred_proba)
        dwi_data.append((region, dwi))

    dwi_df = pd.DataFrame(dwi_data, columns=['w_regions', 'DWI'])
    merged_df = gangwon_regions.merge(dwi_df, on='w_regions', how='left')
    target_date = (datetime.now() - timedelta(days=2)).strftime("%Y-%m-%d")


    empyt1, con1, empty2 = st.columns([0.2, 1.0, 0.2])
    empyt1, con2, empty2 = st.columns([0.4, 1.0, 0.4])
    empyt1, con3, empty2 = st.columns([0.5, 0.5, 0.5])
    empyt1, con4, empty2 = st.columns([0.4, 1.0, 0.4])

    with con1:
        st.markdown(
            "<h2 style='text-align: center; color: black;'>강원도 산불 예측 및 피해 최소화 프로젝트</span>", unsafe_allow_html=True)
        st.write('<hr>', unsafe_allow_html=True)
    with con2:
        st.markdown("<h4 style='font-size: 24px; text-align: center; color: black;'>🔥🌳 실시간 산불위험지수(DWI) 🌳🔥</h4>",
                    unsafe_allow_html=True)
        st.markdown("")
        st.markdown(f"<h6 style='font-size: 16px; text-align: center; color: black;'> ({target_date}) 실시간 DWI 지수 시각화 </h4>",
                    unsafe_allow_html=True)
    with con3:
        create_dwi_choropleth_map(merged_df, "geometry", "DWI")
    with con4:
        st.markdown(
            "<h6 style='font-size: 16px; text-align: center; color: black;'> 강원도 9개 지역별 설정한 ML MODEL 에 입력하여, </h4>",
            unsafe_allow_html=True)
        st.markdown("")
        st.markdown(
            "<h6 style='font-size: 16px; text-align: center; color: black;'> 얻어진 확률들의 예측치를 이용하여 산불위험지수(DWI) 지도시각화 </h4>",
            unsafe_allow_html=True)

    st.write('<hr>', unsafe_allow_html=True)

    # Link
    c1, c2, c3 = st.columns(3)
    with c1:
        st.info('**Project: [@KingBeeM/ForestFire](https://github.com/KingBeeM/ForestFire)**', icon="💡")
    with c2:
        st.info('**GitHub: [@KingBeeM](https://github.com/KingBeeM)**', icon="💻")
    with c3:
        st.info('**Data: [Public API](https://www.data.go.kr/data/15059093/openapi.do)**', icon="📕")