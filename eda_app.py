# -*- coding:utf-8 -*-

import streamlit as st
import pandas as pd
import geopandas as gpd
import pandas_gbq
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.font_manager as fm
import seaborn as sns
import plotly.express as px
import folium
from folium.plugins import MarkerCluster, HeatMap
from streamlit_folium import st_folium
from streamlit_folium import folium_static
from shapely.geometry import Point, Polygon, MultiPolygon, shape
from shapely import wkt
from PIL import Image

import utils
import home_app
import stat_app
import model_app
import service_app

import os
import warnings
warnings.filterwarnings("ignore")

@st.cache_data()
def font_set():
    # # matplotlib 한글 폰트 설정
    # font_dirs = [os.getcwd() + '/nanum']
    # font_files = fm.findSystemFonts(fontpaths=font_dirs)
    # for font_file in font_files:
    #     fm.fontManager.addfont(font_file)
    # plt.rcParams['font.family'] = 'NanumGothic'

    # 폰트 경로 설정
    font_dirs = [os.getcwd() + '/nanum']

    # 폰트 파일 탐색
    font_files = fm.findSystemFonts(fontpaths=font_dirs)

    # 폰트 매니저에 폰트 추가
    for font_file in font_files:
        fm.fontManager.addfont(font_file)

    # Matplotlib의 폰트 설정
    plt.rcParams['font.family'] = fm.FontProperties(fname=font_files[0]).get_name()  # 첫 번째 폰트를 사용하도록 설정

def visualize_forestfire_by_region(dataframe, region_column, value_column, cmap_name, title):
    """
    지역별 특정 값에 따른 시각화를 수행합니다.

    Args:
        dataframe (pandas.DataFrame): 지역별 값 데이터가 포함된 DataFrame.
        region_column (str): 지역 정보가 포함된 열의 이름.
        value_column (str): 시각화할 값이 포함된 열의 이름.
        cmap_name (str): 색상 맵 이름.
        title (str): 시각화 제목.

    Returns:
        None
    """
    dataframe.crs = "EPSG:4326"

    # 색상 설정
    cmap = LinearSegmentedColormap.from_list(cmap_name, ['green', 'orange', 'darkred'])

    # 지도 그리기
    fig, ax = plt.subplots(figsize=(10, 12))
    dataframe.plot(column=value_column, cmap=cmap, linewidth=0.8, ax=ax, edgecolor='0.8')

    # 범례 설정
    sm = plt.cm.ScalarMappable(cmap=cmap)
    sm.set_array([])
    cbar = plt.colorbar(sm, **{'orientation': 'vertical', 'shrink': 0.6})  # cbar_kwargs를 사용하여 높이 조절

    # 축 제거
    ax.axis('off')

    # 제목 설정
    plt.title(title)

    # 시각화 출력
    st.pyplot(fig)

def plot_boxplot(data_frames, column_name, labels, colors, title, ylabel):
    """
    여러 데이터프레임의 특정 컬럼에 대한 boxplot을 그리는 함수입니다.

    Args:
        data_frames (list): 데이터프레임들의 리스트
        column_name (str): boxplot에 사용할 컬럼의 이름
        labels (list): 각 boxplot에 대한 레이블 리스트
        colors (list): 각 boxplot의 색상 리스트
        title (str): 그래프의 제목
        ylabel (str): y축 레이블

    Returns:
        None
    """
    font_set()
    # font_Names = [f.name for f in fm.fontManager.ttflist]
    # plt.rc('font', family=font_Names)
    plt.style.use('ggplot')
    plt.rcParams['figure.figsize'] = (10, 5)
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.unicode_minus'] = False

    fig, ax = plt.subplots()

    box_width = 1.5
    median_color = 'red'

    for i, df in enumerate(data_frames):
        data = [df[column_name]]
        box = ax.boxplot(data, patch_artist=True, positions=[i*4+2], widths=box_width,
                         boxprops={'linewidth': 1.7, 'edgecolor': 'black'},
                         whiskerprops={'linewidth': 1.7})

        for patch, color in zip(box["boxes"], colors):
            patch.set_facecolor(color)

        for median in box["medians"]:
            median.set(color=median_color)

    ax.set_xticks([i*4+2 for i in range(len(data_frames))])

    yticks = np.arange(-20, 50, 10)
    ax.set_yticks(yticks)

    legend_patches = [mpatches.Patch(color=color, label=label) for color, label in zip(colors, labels)]
    ax.legend(handles=legend_patches, loc='lower center', bbox_to_anchor=(0.5, -0.2),
              fontsize=8.5, ncol=len(data_frames))

    ax.set_xticklabels(["지역 " + str(i+1) for i in range(len(data_frames))])
    ax.set_title(title, fontweight='bold')
    ax.set_ylabel(ylabel)
    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9)

    st.pyplot(fig)

def plot_boxplot_rhm(data_frames, column_name, labels, colors, title, ylabel):
    """
    여러 데이터프레임의 특정 컬럼에 대한 boxplot을 그리는 함수입니다.

    Args:
        data_frames (list): 데이터프레임들의 리스트
        column_name (str): boxplot에 사용할 컬럼의 이름
        labels (list): 각 boxplot에 대한 레이블 리스트
        colors (list): 각 boxplot의 색상 리스트
        title (str): 그래프의 제목
        ylabel (str): y축 레이블

    Returns:
        None
    """

    font_set()
    # font_Names = [f.name for f in fm.fontManager.ttflist]
    # plt.rc('font', family=font_Names)
    plt.style.use('ggplot')
    plt.rcParams['figure.figsize'] = (10, 5)
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.unicode_minus'] = False

    fig, ax = plt.subplots()

    box_width = 1.5
    median_color = 'red'

    for i, df in enumerate(data_frames):
        data = [df[column_name]]
        box = ax.boxplot(data, patch_artist=True, positions=[i*4+2], widths=box_width,
                         boxprops={'linewidth': 1.7, 'edgecolor': 'black'},
                         whiskerprops={'linewidth': 1.7})

        for patch, color in zip(box["boxes"], colors):
            patch.set_facecolor(color)

        for median in box["medians"]:
            median.set(color=median_color)

    ax.set_xticks([i*4+2 for i in range(len(data_frames))])

    yticks = np.arange(10, 120, 10)
    ax.set_yticks(yticks)

    legend_patches = [mpatches.Patch(color=color, label=label) for color, label in zip(colors, labels)]
    ax.legend(handles=legend_patches, loc='lower center', bbox_to_anchor=(0.5, -0.2),
              fontsize=8.5, ncol=len(data_frames))

    ax.set_xticklabels(["지역 " + str(i+1) for i in range(len(data_frames))])
    ax.set_title(title, fontweight='bold')
    ax.set_ylabel(ylabel)
    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9)

    st.pyplot(fig)

def plot_boxplot_wd(data_frames, column_name, labels, colors, title, ylabel):
    """
    여러 데이터프레임의 특정 컬럼에 대한 boxplot을 그리는 함수입니다.

    Args:
        data_frames (list): 데이터프레임들의 리스트
        column_name (str): boxplot에 사용할 컬럼의 이름
        labels (list): 각 boxplot에 대한 레이블 리스트
        colors (list): 각 boxplot의 색상 리스트
        title (str): 그래프의 제목
        ylabel (str): y축 레이블

    Returns:
        None
    """
    font_set()
    # font_Names = [f.name for f in fm.fontManager.ttflist]
    # plt.rc('font', family=font_Names)
    plt.style.use('ggplot')
    plt.rcParams['figure.figsize'] = (10, 5)
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.unicode_minus'] = False

    fig, ax = plt.subplots()

    box_width = 1.5
    median_color = 'red'

    for i, df in enumerate(data_frames):
        data = [df[column_name]]
        box = ax.boxplot(data, patch_artist=True, positions=[i*4+2], widths=box_width,
                         boxprops={'linewidth': 1.7, 'edgecolor': 'black'},
                         whiskerprops={'linewidth': 1.7})

        for patch, color in zip(box["boxes"], colors):
            patch.set_facecolor(color)

        for median in box["medians"]:
            median.set(color=median_color)

    ax.set_xticks([i*4+2 for i in range(len(data_frames))])

    yticks = np.arange(-5, 40, 10)
    ax.set_yticks(yticks)

    legend_patches = [mpatches.Patch(color=color, label=label) for color, label in zip(colors, labels)]
    ax.legend(handles=legend_patches, loc='lower center', bbox_to_anchor=(0.5, -0.2),
              fontsize=8.5, ncol=len(data_frames))

    ax.set_xticklabels(["지역 " + str(i+1) for i in range(len(data_frames))])
    ax.set_title(title, fontweight='bold')
    ax.set_ylabel(ylabel)
    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9)

    st.pyplot(fig)

def eda_app():
    weather_stations, weather_days, forestfire_occurs, gangwon_regions = utils.load_data("PREPROCESSING_DATA")

    visual_feature = forestfire_occurs[["w_regions", "ar", "amount", "latitude", "longitude"]].groupby(
        ["w_regions"]).agg({
        "ar": lambda x: round(x.astype(float).sum()),
        "amount": lambda x: round(x.astype(float).sum()),
        "latitude": "mean",
        "longitude": "mean"
    }).reset_index()

    region_img = Image.open("streamlit_img/region.png")

    selected_chart = st.sidebar.selectbox("Select Chart", ["EDA 과정", "지도 시각화", "기상 데이터"])

    if selected_chart == "EDA 과정":

        st.markdown("<h4> 지도 시각화를 위한 지역 분할 과정 </h4>", unsafe_allow_html=True)
        st.markdown("---")

        st.image(region_img)

        st.markdown("")
        st.markdown("✔ **강원도 기상청 예특보 관할구역에 대한 내용을 참고하여 지역 분할 진행**")
        st.markdown("")
        st.markdown("✔ **최초 읍면동 기준으로 12분할을 진행하였지만 기상 관측소 위치에서 벗어난 지역들이 발생하여 수정이 필요하다고 판단**")
        st.markdown("")
        st.markdown("✔ **지역적 특성과 기상 관측소 위치를 고려하여 총 9분할로 진행**")

    if selected_chart == '지도 시각화':

        st.markdown("<h4> 강원도 지역 지도 시각화 </h4>", unsafe_allow_html=True)

        tab1, tab2, tab3 = st.tabs(["**발생 건수**","**피해 범위**", "**피해 금액**"])

        with tab1:
            col1, col2, col3 = st.columns([1, 8, 1])
            with col1:
                pass

            with col2:
                # w_regions별 value_counts 계산
                region_counts = forestfire_occurs['w_regions'].value_counts().reset_index()
                region_counts.columns = ['w_regions', 'Counts']

                # 데이터프레임 생성
                region_counts_df = pd.DataFrame({'w_regions': region_counts['w_regions'],
                                                 'Fire_Counts': region_counts['Counts']})

                merged_Count = gangwon_regions.merge(region_counts_df, on='w_regions', how='left')

                visualize_forestfire_by_region(merged_Count, "w_regions", "Fire_Counts", "fire_counts_cmap", "Forest Fire Counts by Region")

                st.markdown("✔ **2013 ~ 2022년의 산불 발생 건수 지도**")
                st.markdown("✔ **상대적으로 유동 인구가 많은 내륙 지역에서 산불 발생이 많이 발생한 것을 확인**")

            with col3:
                pass

        with tab2:

            col1, col2, col3 = st.columns([1, 8, 1])

            with col1:
                pass

            with col2:
                Damage_Area = pd.DataFrame({'w_regions': visual_feature['w_regions'],
                                            'DamageArea': visual_feature['ar']})

                merged_Area = gangwon_regions.merge(Damage_Area, on='w_regions', how='left')

                visualize_forestfire_by_region(merged_Area, "w_regions", "DamageArea", "DamageArea_cmap", "DamageArea by Region")

                st.markdown("✔ **2013 ~ 2022년의 산불 피해 범위 지도**")
                st.markdown("✔ **2022년에 발생했던 강릉,동해 산불로 인해 강릉 지역이 가장 큰 피해를 입은 것으로 나타남**")

            with col3:
                pass

        with tab3:

            col1, col2, col3 = st.columns([1, 8, 1])

            with col1:
                pass

            with col2:
                Damage_Amount = pd.DataFrame({'w_regions': visual_feature['w_regions'],
                                              'Amount': visual_feature['amount']})

                merged_Amount = gangwon_regions.merge(Damage_Amount, on='w_regions', how='left')

                visualize_forestfire_by_region(merged_Amount, "w_regions", "Amount", "Amount_cmap", "Amount by Region")

                st.markdown("✔ **2013 ~ 2022년의 산불 피해 금액 지도**")
                st.markdown("✔ **고성에서 강릉으로 연결되는 해안 지역이 대형 산불에 대한 피해가 큰 것으로 나타남**")
                st.markdown("✔ **그로 인해 피해 금액도 높게 나타남**")

            with col3:
                pass

    if selected_chart == "기상 데이터":

        st.markdown("<h4> 강원도 지역 기상 데이터 </h4>", unsafe_allow_html=True)

        data_frames = utils.load_data("ANALSIS_DATA")
        tab1, tab2, tab3, tab4 = st.tabs(["**기온**","**습도**","**풍속**","**강수**"])

        with tab1:

            # 9개 지역별 최고 기온 데이터
            column_name = "maxTa"
            labels = ['최고 기온']
            colors = ["lightgreen"]
            title = '9개 지역별 최고 기온 데이터'
            ylabel = '온도 (℃)'

            plot_boxplot(data_frames, column_name, labels, colors, title, ylabel)

            # 9개 지역별 최저 기온 데이터
            column_name = "minTa"
            labels = ['최저 기온']
            colors = ["lightblue"]
            title = '9개 지역별 최저 기온 데이터'
            ylabel = '온도 (℃)'

            plot_boxplot(data_frames, column_name, labels, colors, title, ylabel)

            # 9개 지역별 평균 기온 데이터
            column_name = "avgTa"
            labels = ['평균 기온']
            colors = ["pink"]
            title = '9개 지역별 평균 기온 데이터'
            ylabel = '온도 (℃)'

            plot_boxplot(data_frames, column_name, labels, colors, title, ylabel)

        with tab2:

            # 9개 지역별 최소 상대습도 데이터
            column_name = "minRhm"
            labels = ['최소 상대습도']
            colors = ["lightblue"]
            title = '9개 지역별 최소 상대습도 데이터'
            ylabel = '단위 (%)'

            plot_boxplot_rhm(data_frames, column_name, labels, colors, title, ylabel)

            # 9개 지역별 평균 상대습도 데이터
            column_name = "avgRhm"
            labels = ['평균 상대습도']
            colors = ["pink"]
            title = '9개 지역별 평균 상대습도 데이터'
            ylabel = '단위 (%)'

            plot_boxplot_rhm(data_frames, column_name, labels, colors, title, ylabel)

            # 9개 지역별 실효 습도 데이터
            column_name = "effRhm"
            labels = ['실효습도']
            colors = ["orange"]
            title = '9개 지역별 실효습도 데이터'
            ylabel = '단위 (%)'

            plot_boxplot_rhm(data_frames, column_name, labels, colors, title, ylabel)

        with tab3:

            # 9개 지역별 평균 풍속 데이터
            column_name = "avgWs"
            labels = ['평균 풍속']
            colors = ["pink"]
            title = '9개 지역별 평균 풍속 데이터'
            ylabel = '단위 (m/s)'

            plot_boxplot_wd(data_frames, column_name, labels, colors, title, ylabel)

            # 9개 지역별 최대 순간 풍속 데이터
            column_name = "maxInsWs"
            labels = ['최대 순간 풍속']
            colors = ["lightgreen"]
            title = '9개 지역별 최대 순간 풍속 데이터'
            ylabel = '단위 (m/s)'

            plot_boxplot_wd(data_frames, column_name, labels, colors, title, ylabel)

            # 9개 지역별 최대 풍속 데이터
            column_name = "maxWs"
            labels = ['최대 풍속']
            colors = ["lightblue"]
            title = '9개 지역별 최대 풍속 데이터'
            ylabel = '단위 (m/s)'

            plot_boxplot_wd(data_frames, column_name, labels, colors, title, ylabel)

            # 9개 지역별 7일간 최대 풍속 데이터
            column_name = "maxwind7"
            labels = ['7일간 최대 풍속']
            colors = ["violet"]
            title = '9개 지역별 7일간 최대 풍속 데이터'
            ylabel = '단위 (m/s)'

            plot_boxplot_wd(data_frames, column_name, labels, colors, title, ylabel)

        with tab4:

            regions = ["지역 {}".format(i + 1) for i in range(len(weather_days))]
            # 지역별 강수 여부
            labels = ['지역 1', '지역 2', '지역 3', '지역 4', '지역 5', '지역 6', '지역 7', '지역 8', '지역 9']

            font_set()
            font_Names = [f.name for f in fm.fontManager.ttflist]
            plt.rc('font', family=font_Names)
            plt.style.use('ggplot')
            plt.rcParams['figure.figsize'] = (12, 8)
            plt.rcParams['font.size'] = 12
            plt.rcParams['axes.unicode_minus'] = False

            fig, ax = plt.subplots()

            width = 0.35
            x = range(len(labels))
            colors = ['steelblue', 'darkorange']

            legend_labels = ['0', '1']

            for i, df in enumerate(data_frames):
                region = labels[i]
                counts = df['Rntf'].value_counts()

                ax.bar(x[i] - width / 2, counts[0], width, label=legend_labels[0], color=colors[0])
                ax.bar(x[i] + width / 2, counts[1], width, label=legend_labels[1], color=colors[1])

                for j, count in enumerate(counts):
                    ax.text(x[i] + width * (j - 0.5), count + 2, str(count), ha='center', va='bottom')

            ax.set_xticks(x)
            ax.set_xticklabels(labels)

            ax.set_ylabel('단위 (건)')
            ax.set_title('지역별 강수 여부')

            legend = ax.legend(handles=[ax.patches[0], ax.patches[len(labels)]], labels=legend_labels, title='강수 여부',
                               loc='lower center', ncol=2)

            legend.set_bbox_to_anchor((0.5, -0.15))
            legend._legend_box.align = "center"

            st.pyplot(fig)
            st.markdown("✔ **강수 데이터를 명목형 변수 (0, 1)로 변환**")
            st.markdown("✔ **0 = 비가 오지 않았음, 1 = 비가 왔음**")