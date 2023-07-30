# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import geopandas as gpd
import pandas_gbq
import json
from shapely.geometry import Point, Polygon, MultiPolygon, shape
from shapely import wkt
import googlemaps
from google.cloud import bigquery
from google.oauth2 import service_account
from PIL import Image

import warnings
warnings.filterwarnings("ignore")

credentials = service_account.Credentials.from_service_account_info(st.secrets["gcp_service_account"])
servicekey = st.secrets.api_key.data_serviceKey

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



@st.cache_data
def load_data(type):
    """
        데이터를 로드하는 함수입니다.

        Args:
            type (str): 데이터 유형 ("RAW_DATA", "PREPROCESSING_DATA", "ANALSIS_DATA")

        Returns:
            dataframe : 데이터 프레임 또는 지리정보 데이터 프레임
    """
    if type == "RAW_DATA":
        # BigQuery 에 RAW_DATA Load
        weather_stations = get_dataframe_from_bigquery("RAW_DATA", "weather_stations").sort_values(["stnId"])
        weather_days = get_dataframe_from_bigquery("RAW_DATA", "weather_days").sort_values(["stnId", "tm"])
        forestfire_occurs = get_dataframe_from_bigquery("RAW_DATA", "forestfire_occurs").sort_values(["objt_id", "occu_date"])
        forestfire_occurs_add = get_dataframe_from_bigquery("RAW_DATA", "forestfire_occurs_add").sort_values(["objt_id", "occu_date"])
        gangwon_code = get_dataframe_from_bigquery("RAW_DATA", "gangwon_code").sort_values(["code"])
        gangwon_SGG = get_geodataframe_from_bigquery("RAW_DATA", "gangwon_SGG").sort_values(["ADM_SECT_C", "SGG_NM"])
        gangwon_UMD = get_geodataframe_from_bigquery("RAW_DATA", "gangwon_UMD").sort_values(["EMD_CD"])

        return weather_stations, weather_days, forestfire_occurs, forestfire_occurs_add, gangwon_code, gangwon_SGG, gangwon_UMD

    elif type == "PREPROCESSING_DATA":
        # BigQuery 에 PREPROCESSING_DATA Load
        weather_stations = get_dataframe_from_bigquery("PREPROCESSING_DATA", "weather_stations").sort_values(["stnId"])
        weather_days = get_dataframe_from_bigquery("PREPROCESSING_DATA", "weather_days").sort_values(["stnId", "tm"])
        forestfire_occurs = get_dataframe_from_bigquery("PREPROCESSING_DATA", "forestfire_occurs").sort_values(["objt_id", "occu_date"])
        gangwon_regions = get_geodataframe_from_bigquery("PREPROCESSING_DATA", "gangwon_regions")

        return weather_stations, weather_days, forestfire_occurs, gangwon_regions

    elif type == "ANALSIS_DATA":
        data_1 = get_dataframe_from_bigquery("ANALSIS_DATA", "GangwonNorthInland").sort_values(["tm"]).reset_index(drop=True)
        data_2 = get_dataframe_from_bigquery("ANALSIS_DATA", "GangwonNorthMount").sort_values(["tm"]).reset_index(drop=True)
        data_3 = get_dataframe_from_bigquery("ANALSIS_DATA", "GangwonNorthCoast").sort_values(["tm"]).reset_index(drop=True)
        data_4 = get_dataframe_from_bigquery("ANALSIS_DATA", "GangwonCentralInland").sort_values(["tm"]).reset_index(drop=True)
        data_5 = get_dataframe_from_bigquery("ANALSIS_DATA", "GangwonCentralMount").sort_values(["tm"]).reset_index(drop=True)
        data_6 = get_dataframe_from_bigquery("ANALSIS_DATA", "GangwonCentralCoast").sort_values(["tm"]).reset_index(drop=True)
        data_7 = get_dataframe_from_bigquery("ANALSIS_DATA", "GangwonSouthInland").sort_values(["tm"]).reset_index(drop=True)
        data_8 = get_dataframe_from_bigquery("ANALSIS_DATA", "GangwonSouthMount").sort_values(["tm"]).reset_index(drop=True)
        data_9 = get_dataframe_from_bigquery("ANALSIS_DATA", "GangwonSouthInland").sort_values(["tm"]).reset_index(drop=True)

        data_1 = data_1[data_1['tm'] < '2023-01-01']
        data_2 = data_2[data_2['tm'] < '2023-01-01']
        data_3 = data_3[data_3['tm'] < '2023-01-01']
        data_4 = data_4[data_4['tm'] < '2023-01-01']
        data_5 = data_5[data_5['tm'] < '2023-01-01']
        data_6 = data_6[data_6['tm'] < '2023-01-01']
        data_7 = data_7[data_7['tm'] < '2023-01-01']
        data_8 = data_8[data_8['tm'] < '2023-01-01']
        data_9 = data_9[data_9['tm'] < '2023-01-01']

        return data_1, data_2, data_3, data_4, data_5, data_6, data_7, data_8, data_9