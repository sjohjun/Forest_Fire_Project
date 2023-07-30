# -*- coding:utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
import scipy.stats as stats
from pingouin import ttest

import utils
import home_app
import stat_app
import model_app
import service_app
from utils import credentials, servicekey

from PIL import Image

import os
import warnings
warnings.filterwarnings("ignore")

def twoMeans(data):
    """
    두 개의 평균을 비교하여 통계 분석을 수행하고 Plotly를 사용하여 막대 그래프를 생성합니다.

    Args:
        data (pandas.DataFrame): 분석에 사용될 데이터 프레임. 컬럼은 'avgTa', 'sumRn', 'avgWs', 'avgRhm', 'effRhm', 'Rntf', 'noRn', 'fire_occur'로 구성되어야 합니다.

    Returns:
        None
    """
    selected_data = data[['avgTa', 'sumRn', 'avgWs', 'avgRhm', 'effRhm', 'Rntf', 'noRn', 'fire_occur']]
    # 컬럼 선택
    selected_col = st.selectbox("SELECT COLUMN", ['avgTa', 'sumRn', 'avgWs', 'avgRhm', 'effRhm', 'noRn'])

    # 'fire_occur' 그룹별로 선택된 컬럼(col)의 평균 계산
    summary_df = np.round(selected_data.groupby('fire_occur').agg({'fire_occur': 'size', selected_col: 'mean'}), 1)
    st.dataframe(summary_df)

    st.markdown("Independent Test")

    # 'fire_occur' 값에 따라 데이터를 나누기
    data1 = selected_data[selected_data['fire_occur'] == 1][selected_col]
    data0 = selected_data[selected_data['fire_occur'] == 0][selected_col]

    # NaN 값 처리
    data1 = data1.dropna()
    data0 = data0.dropna()

    result = ttest(data1, data0, correction=False, confidence=0.99)
    st.dataframe(result)

    if result['p-val'].values > 0.05:
        st.markdown(":green[$H_0$] : **The means for the two sales are equal.**")
        st.markdown("<hr>", unsafe_allow_html=True)
    else:
        st.markdown(":green[$H_1$] : **The means for the two populations are not equal.**")
        st.markdown("<hr>", unsafe_allow_html=True)

    grouped = selected_data.groupby('fire_occur')
    means = grouped[selected_col].agg('mean')

    # Define the data for the plot
    data = pd.DataFrame({'fire_occur': means.index, selected_col: means.values})

    # Create the Plotly bar plot
    fig = px.bar(data, x=selected_col, y='fire_occur', orientation='h', title=f'Average {selected_col} by Fire_occur',
                 labels={'col': f'{selected_col}', 'species': 'Fire_occur'})

    # Customize the layout
    fig.update_layout(
        plot_bgcolor='white',
        width=800,
        height=400
    )

    st.plotly_chart(fig)

def stat_app():
    st.sidebar.markdown("## SubMenu")
    submenu = st.sidebar.radio("Submenu", ['Two Means', 'Logistic Regression'], label_visibility='collapsed')

    if submenu == 'Two Means':
        data_1, data_2, data_3, data_4, data_5, data_6, data_7, data_8, data_9 = utils.load_data("ANALSIS_DATA")

        selected_data = st.selectbox("SELECT DATA", ["Site_1", "Site_2", "Site_3",
                                                     "Site_4", "Site_5", "Site_6",
                                                     "Site_7", "Site_8", "Site_9"])

        if selected_data == "Site_1":
            st.markdown("---")
            st.markdown(f"### {selected_data} Two Means")
            st.markdown("---")
            twoMeans(data_1)
        elif selected_data == "Site_2":
            st.markdown("---")
            st.markdown(f"### {selected_data} Two Means")
            st.markdown("---")
            twoMeans(data_2)
        elif selected_data == "Site_3":
            st.markdown("---")
            st.markdown(f"### {selected_data} Two Means")
            st.markdown("---")
            twoMeans(data_3)
        elif selected_data == "Site_4":
            st.markdown("---")
            st.markdown(f"### {selected_data} Two Means")
            st.markdown("---")
            twoMeans(data_4)
        elif selected_data == "Site_5":
            st.markdown("---")
            st.markdown(f"### {selected_data} Two Means")
            st.markdown("---")
            twoMeans(data_5)
        elif selected_data == "Site_6":
            st.markdown("---")
            st.markdown(f"### {selected_data} Two Means")
            st.markdown("---")
            twoMeans(data_6)
        elif selected_data == "Site_7":
            st.markdown("---")
            st.markdown(f"### {selected_data} Two Means")
            st.markdown("---")
            twoMeans(data_7)
        elif selected_data == "Site_8":
            st.markdown("---")
            st.markdown(f"### {selected_data} Two Means")
            st.markdown("---")
            twoMeans(data_8)
        elif selected_data == "Site_9":
            st.markdown("---")
            st.markdown(f"### {selected_data} Two Means")
            st.markdown("---")
            twoMeans(data_9)

    elif submenu == 'Logistic Regression':
        st.markdown("---")
        st.markdown(f"### Logistic Regression STATS")
        st.markdown("---")

        tab1, tab2, tab3 = st.tabs(["**Statistic**", "**Correnlation Matrix**", "**Regression Model Formula**"])

        with tab1:
            selected_data = st.selectbox("SELECT DATA", ["Site_1", "Site_2", "Site_3",
                                                         "Site_4", "Site_5", "Site_6",
                                                         "Site_7", "Site_8", "Site_9"])

            stats1_img = Image.open("streamlit_img/region_stats1.png")
            stats2_img = Image.open("streamlit_img/region_stats2.png")
            stats3_img = Image.open("streamlit_img/region_stats3.png")
            stats4_img = Image.open("streamlit_img/region_stats4.png")
            stats5_img = Image.open("streamlit_img/region_stats5.png")
            stats6_img = Image.open("streamlit_img/region_stats6.png")
            stats7_img = Image.open("streamlit_img/region_stats7.png")
            stats8_img = Image.open("streamlit_img/region_stats8.png")
            stats9_img = Image.open("streamlit_img/region_stats9.png")

            if selected_data == "Site_1":
                st.image(stats1_img)
            elif selected_data == "Site_2":
                st.image(stats2_img)
            elif selected_data == "Site_3":
                st.image(stats3_img)
            elif selected_data == "Site_4":
                st.image(stats4_img)
            elif selected_data == "Site_5":
                st.image(stats5_img)
            elif selected_data == "Site_6":
                st.image(stats6_img)
            elif selected_data == "Site_7":
                st.image(stats7_img)
            elif selected_data == "Site_8":
                st.image(stats8_img)
            elif selected_data == "Site_9":
                st.image(stats9_img)

        with tab2:
            corr1_img = Image.open("streamlit_img/Correlation matrix1.png")
            corr2_img = Image.open("streamlit_img/Correlation matrix2.png")

            selected_corr = st.radio("Correlation matrix", ["significant variables", "insignificant variables"], label_visibility='collapsed')

            if selected_corr == "significant variables":
                st.image(corr1_img)
            elif selected_corr == "insignificant variables":
                st.image(corr2_img)

        with tab3:
            lr_img = Image.open("streamlit_img/logistic regression.png")
            st.image(lr_img)