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
from shapely.geometry import Point, Polygon, MultiPolygon, shape
from shapely import wkt

import statsmodels.api as sm

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit, train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix, classification_report,  roc_curve, auc, RocCurveDisplay
from imblearn.over_sampling import SMOTE

from xgboost import XGBClassifier, plot_importance
from lightgbm import LGBMClassifier, plot_importance

from PIL import Image

import utils
import home_app
import eda_app
import stat_app
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


def classification_report_to_dataframe(report):
    """
    분류 보고서를 데이터프레임으로 변환합니다.

    Args:
        report (str): 클래스 분류 보고서 문자열. 보통 sklearn.metrics.classification_report 함수의 결과를 사용합니다.

    Returns:
        pandas.DataFrame: 분류 보고서를 기반으로 생성된 데이터프레임. 각 클래스에 대한 Precision, Recall, F1-Score, Support가 열로 표시됩니다.
    """
    report_data = []
    lines = report.split('\n')
    for line in lines[2:-3]:
        row = line.split()
        if len(row) >= 5:
            class_name = row[0]
            precision = float(row[1])
            recall = float(row[2])
            f1_score = float(row[3])
            support = int(row[4])
            report_data.append([class_name, precision, recall, f1_score, support])

    df = pd.DataFrame(report_data, columns=['Class', 'Precision', 'Recall', 'F1-Score', 'Support'])
    df.set_index('Class', inplace=True)

    return df

def display_results (y_real, pred, pred_proba):
    """
    분류 결과를 출력합니다.

    Args:
        y_real (array-like): 실제 레이블 값.
        pred (array-like): 예측값.
        pred_proba (array-like): 예측 확률 값.

    Returns:
        None
    """

    st.write("confusion matrix is:")
    st.write(confusion_matrix(y_real, pred))
    st.markdown("---")
    st.write("Accuracy score is:", accuracy_score(y_real, pred))
    st.write("Precision is : ", precision_score(y_real, pred))
    st.write("Recall is : ", recall_score(y_real, pred))
    st.write("F1 score is : ", f1_score(y_real, pred))
    st.write("ROC AUC score is : ", roc_auc_score(y_real, pred_proba))
    st.markdown("---")
    st.write("Classification report is:")
    st.write(classification_report_to_dataframe(classification_report(y_real, pred)))

def plot_feature_importance_lr(model, X_train):
    """
       Logistic Regression 모델의 피처 중요도를 시각화합니다.

       Args:
           model (sklearn.linear_model.LogisticRegression): Logistic Regression 모델 객체.
           X_train (pandas.DataFrame): 학습 데이터의 피처 데이터프레임.

       Returns:
           None
    """
    font_set()
    # font_Names = [f.name for f in fm.fontManager.ttflist]
    # plt.rc('font', family=font_Names)
    plt.style.use('default')
    plt.rcParams['figure.figsize'] = (10, 5)
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.unicode_minus'] = False

    coefficients = model.coef_[0]
    feature_importance = pd.DataFrame({'Feature': X_train.columns, 'Importance': np.abs(coefficients)})
    feature_importance = feature_importance.sort_values('Importance', ascending=True)
    feature_importance.plot.barh(x='Feature', y='Importance', figsize=(10, 6), legend=False, title='Feature Importance')
    st.pyplot(plt)

def plot_feature_importance(model, feature_names):
    """
        모델의 피처 중요도를 시각화합니다.

        Args:
            model: 피처 중요도를 계산할 모델 객체. feature_importances_ 속성을 가져야 합니다.
            feature_names (list): 피처 이름을 담고 있는 리스트.

        Returns:
            None
    """
    font_set()
    # font_Names = [f.name for f in fm.fontManager.ttflist]
    # plt.rc('font', family=font_Names)
    plt.style.use('default')
    plt.rcParams['figure.figsize'] = (10, 5)
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.unicode_minus'] = False

    importances = model.feature_importances_
    indices = np.argsort(importances)
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(indices)), importances[indices], align='center')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Importance')
    plt.title('Feature Importance')
    st.pyplot(plt)


def calculate_tpr_fpr(y_real, y_pred):
    """
    실제 값과 예측 값으로부터 True Positive Rate(TPR)와 False Positive Rate(FPR)를 계산합니다.

    Args:
        y_real (array-like): 실제 레이블 값.
        y_pred (array-like): 예측값.

    Returns:
        tpr (float): True Positive Rate(TPR).
        fpr (float): False Positive Rate(FPR).
    """

    # Calculates the confusion matrix and recover each element
    cm = confusion_matrix(y_real, y_pred)
    TN = cm[0, 0]
    FP = cm[0, 1]
    FN = cm[1, 0]
    TP = cm[1, 1]

    # Calculates tpr and fpr
    tpr = TP / (TP + FN)  # sensitivity - true positive rate
    fpr = 1 - TN / (TN + FP)  # 1-specificity - false positive rate

    return tpr, fpr

def get_n_roc_coordinates(y_real, y_proba, n=50):
    """
    실제 값과 예측 확률 값으로부터 주어진 개수(n)에 따른 TPR과 FPR 좌표를 얻습니다.

    Args:
        y_real (array-like): 실제 레이블 값.
        y_proba (array-like): 예측 확률 값.
        n (int, optional): TPR과 FPR 좌표 개수. 기본값은 50입니다.

    Returns:
        tpr_list (list): TPR 좌표 리스트.
        fpr_list (list): FPR 좌표 리스트.
    """

    tpr_list = [0]
    fpr_list = [0]

    for i in range(n):
        threshold = i / n
        y_pred = y_proba[:, 1] > threshold
        tpr, fpr = calculate_tpr_fpr(y_real, y_pred)
        tpr_list.append(tpr)
        fpr_list.append(fpr)

    return tpr_list, fpr_list

def plot_roc_curve(tpr, fpr, scatter = True):
    """
    TPR과 FPR 좌표를 사용하여 ROC 곡선을 그립니다.

    Args:
        tpr (array-like): TPR 좌표 값.
        fpr (array-like): FPR 좌표 값.
        scatter (bool, optional): 점 표시 여부. 기본값은 True입니다.

    Returns:
        None
    """
    font_set()
    # font_Names = [f.name for f in fm.fontManager.ttflist]
    # plt.rc('font', family=font_Names)
    plt.style.use('default')
    plt.rcParams['figure.figsize'] = (10, 5)
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.unicode_minus'] = False

    plt.figure(figsize = (5, 5))
    sns.lineplot(x = fpr, y = tpr)
    sns.lineplot(x = [0, 1], y = [0, 1], color = 'green', linestyle="dashed")
    plt.xlabel("False Positive Rate (Positive label: 1)")
    plt.ylabel("True Positive Rate (Positive label: 1)")
    st.pyplot(plt)

def plot_dwi_intervals(y_real, pred, pred_proba, num_intervals=10):
    """
    DWI 등급 구간에 따라 Forestfire_occur와 Non_Forestfire_occur 값을 계산하고, 그래프를 그립니다.

    Args:
        y_real (Series): 테스트 레이블 데이터.
        pred (array-like): 예측값.
        pred_proba (array-like): 예측 확률 값.
        num_intervals (int, optional): DWI 등급의 개수. 기본값은 10입니다.

    Returns:
        None
    """

    # DWI 등급 구간 설정
    interval_labels = [f"{int((i + 1) * 10)}%" for i in range(num_intervals)]

    # DWI 등급
    dwis = list(range(1, num_intervals + 1))

    # Forestfire_occur와 Non_Forestfire_occur 값
    forestfire_occur = np.zeros(num_intervals)
    non_forestfire_occur = np.zeros(num_intervals)

    # 예측값의 숫자에 따라 count
    for pred in pred_proba:
        idx = int((1 - pred) * 10)
        if idx == 10:
            idx -= 1
        for i in range(idx, num_intervals):
            forestfire_occur[i] += 1

    # Non_Forestfire_occur 값
    total_samples = len(pred_proba)
    non_forestfire_occur = total_samples - forestfire_occur

    # 구간 범위 추정
    interval_ranges = []
    for i in range(num_intervals):
        if i == 0:
            lower_percentile = np.percentile(pred_proba, i * 10)
            upper_percentile = np.percentile(pred_proba, (i + 1) * 10)
            interval_range = f"[{0.0000:.4f}∼{upper_percentile:.4f}]"

        elif i == num_intervals - 1:
            lower_percentile = np.percentile(pred_proba, i * 10)
            upper_percentile = np.percentile(pred_proba, (i + 1) * 10)
            interval_range = f"[{lower_percentile:.4f}∼{1.0000:.4f}]"

        else:
            lower_percentile = np.percentile(pred_proba, i * 10)
            upper_percentile = np.percentile(pred_proba, (i + 1) * 10)
            interval_range = f"[{lower_percentile:.4f}∼{upper_percentile:.4f}]"
        interval_ranges.append(interval_range)

    # 데이터 프레임 생성
    data = {
        'Interval ratio': interval_labels,
        'DWI': dwis,
        'Interval range': interval_ranges,
        'Forestfire_occur': forestfire_occur,
        'Non_Forestfire_occur': non_forestfire_occur
    }

    df = pd.DataFrame(data)

    font_set()
    # font_Names = [f.name for f in fm.fontManager.ttflist]
    # plt.rc('font', family=font_Names)
    plt.style.use('default')
    plt.rcParams['figure.figsize'] = (10, 5)
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.unicode_minus'] = False

    plt.figure()
    # 그래프 그리기
    plt.plot(df['DWI'], df['Forestfire_occur'], marker='o', label='Forestfire_occur', color="red")
    plt.plot(df['DWI'], df['Non_Forestfire_occur'], marker='o', label='Non_Forestfire_occur', color="blue")

    # 수치 표시
    for x, y1, y2 in zip(df['DWI'], df['Forestfire_occur'], df['Non_Forestfire_occur']):
        plt.text(x, y1 + 10, str(int(y1)), ha='center', va='bottom')
        plt.text(x, y2 + 10, str(int(y2)), ha='center', va='bottom')

    # 축 및 레이블 설정
    plt.xlabel('DWI')
    plt.ylabel('Count')
    plt.xticks(df['DWI'])
    plt.title('Forestfire_occur and Non_Forestfire_occur by DWI')

    # 범례 표시
    plt.legend(loc='upper right', ncol=2)

    max_value = max(df['Forestfire_occur'].max(), df['Non_Forestfire_occur'].max())
    plt.ylim(bottom=-50, top=max_value + 150)

    st.pyplot(plt)

def processing(data, model):
    """
       데이터를 전처리하고 주어진 모델을 학습하고 평가하는 함수입니다.

       Args:
           data (DataFrame): 전처리할 데이터 프레임.
           model (str): 사용할 모델의 이름. 'LogisticRegression', 'XGBoost', 'LightGBM' 중 하나를 선택해야 합니다.

       Returns:
           None
    """
    X_train, X_test, y_train, y_test = split_train_test(data)
    feature_names = X_train.columns

    # LR
    lr_model = train_logistic_regression(X_train, y_train)
    # XGB
    xgb_model = train_xgboost(X_train, y_train)
    # LGBM
    lgb_model = train_lightgbm(X_train, y_train)

    empty1, con1, empty2, con2, empty3 = st.columns([0.1, 0.5, 0.1, 0.5, 0.1])
    empyt1, con3, empty2, con4, empty3 = st.columns([0.1, 0.5, 0.1, 0.5, 0.1])
    empyt1, con5, empty2 = st.columns([0.1, 1.0, 0.1])

    if model == "LogisticRegression":
        with con3:
            st.markdown("---")
            st.markdown(f"<h2 style='font-size: 24px; text-align: center; color: black;'>Train feature_importance</span>", unsafe_allow_html=True)
            st.markdown("---")
            pred = lr_model.predict(X_train)
            pred_proba = lr_model.predict_proba(X_train)[:, 1]

            display_results(y_train, pred, pred_proba)
            plot_feature_importance_lr(lr_model, X_train)

        with con4:
            st.markdown("---")
            st.markdown(f"<h2 style='font-size: 24px; text-align: center; color: black;'>Test feature_importance</span>", unsafe_allow_html=True)
            st.markdown("---")
            pred = lr_model.predict(X_test)
            pred_proba = lr_model.predict_proba(X_test)[:, 1]

            display_results(y_test, pred, pred_proba)
            plot_feature_importance_lr(lr_model, X_test)

        with con2:
            st.markdown("---")
            st.markdown(f"<h2 style='font-size: 24px; text-align: center; color: black;'>DWI(산불위험지수)</span>", unsafe_allow_html=True)
            st.markdown("---")
            plot_dwi_intervals(y_test, pred, pred_proba)

    elif model == "XGBoost":
        with con3:
            st.markdown("---")
            st.markdown(f"<h2 style='font-size: 24px; text-align: center; color: black;'>Train feature_importance</span>", unsafe_allow_html=True)
            st.markdown("---")
            pred = xgb_model.predict(X_train)
            pred_proba = xgb_model.predict_proba(X_train)[:, 1]

            display_results(y_train, pred, pred_proba)
            plot_feature_importance(xgb_model, feature_names)

        with con4:
            st.markdown("---")
            st.markdown(f"<h2 style='font-size: 24px; text-align: center; color: black;'>Test feature_importance</span>", unsafe_allow_html=True)
            st.markdown("---")
            pred = xgb_model.predict(X_test)
            pred_proba = xgb_model.predict_proba(X_test)[:, 1]

            display_results(y_test, pred, pred_proba)
            plot_feature_importance(xgb_model, feature_names)

        with con2:
            st.markdown("---")
            st.markdown(f"<h2 style='font-size: 24px; text-align: center; color: black;'>DWI(산불위험지수)</span>", unsafe_allow_html=True)
            st.markdown("---")
            plot_dwi_intervals(y_test, pred, pred_proba)

    elif model == "LightGBM":
        with con3:
            st.markdown("---")
            st.markdown(f"<h2 style='font-size: 24px; text-align: center; color: black;'>Train feature_importance</span>", unsafe_allow_html=True)
            st.markdown("---")
            pred = lgb_model.predict(X_train)
            pred_proba = lgb_model.predict_proba(X_train)[:, 1]

            display_results(y_train, pred, pred_proba)
            plot_feature_importance(lgb_model, feature_names)

        with con4:
            st.markdown("---")
            st.markdown(f"<h2 style='font-size: 24px; text-align: center; color: black;'>Test feature_importance</span>", unsafe_allow_html=True)
            st.markdown("---")
            pred = lgb_model.predict(X_test)
            pred_proba = lgb_model.predict_proba(X_test)[:, 1]

            display_results(y_test, pred, pred_proba)
            plot_feature_importance(lgb_model, feature_names)

        with con2:
            st.markdown("---")
            st.markdown(f"<h2 style='font-size: 24px; text-align: center; color: black;'>DWI (산불위험지수)</span>", unsafe_allow_html=True)
            st.markdown("---")
            plot_dwi_intervals(y_test, pred, pred_proba)


    with con1:
        # LR
        lr_model.fit(X_train, y_train)
        y_pred_lr = lr_model.predict(X_test)
        y_proba_lr = lr_model.predict_proba(X_test)

        # XGB
        xgb_model.fit(X_train, y_train)
        y_pred_xgb = xgb_model.predict(X_test)
        y_proba_xgb = xgb_model.predict_proba(X_test)

        # LGBM
        lgb_model.fit(X_train, y_train)
        y_pred_lgb = lgb_model.predict(X_test)
        y_proba_lgb = lgb_model.predict_proba(X_test)

        # Plots the ROC curve using the sklearn methods (sklearn + matplolib.pyplot)
        fpr_lr, tpr_lr, _ = roc_curve(y_test, y_proba_lr[:, 1])
        fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_proba_xgb[:, 1])
        fpr_lgb, tpr_lgb, _ = roc_curve(y_test, y_proba_lgb[:, 1])

        roc_auc_lr = auc(fpr_lr, tpr_lr)
        roc_auc_xgb = auc(fpr_xgb, tpr_xgb)
        roc_auc_lgb = auc(fpr_lgb, tpr_lgb)

        plt.figure(figsize=(6, 4))
        plt.grid(False)

        plt.plot(fpr_lr, tpr_lr, label=f'LogisticRegression (AUC = {roc_auc_lr:.3f})', color='cornflowerblue')
        plt.plot(fpr_xgb, tpr_xgb, label=f'XGBClassifier (AUC = {roc_auc_xgb:.3f})', color='darkorange')
        plt.plot(fpr_lgb, tpr_lgb, label=f'LGBMClassifier (AUC = {roc_auc_lgb:.3f})', color='red')
        plt.plot([0, 1], [0, 1], label=f'RandomClassfier (AUC = 0.5)', color='black', linestyle="dashed")
        plt.xlabel("False Positive Rate (Positive label: 1)")
        plt.ylabel("True Positive Rate (Positive label: 1)")
        plt.title(f"ROC-AUC Curve")
        plt.legend()

        st.markdown("---")
        st.markdown(f"<h2 style='font-size: 24px; text-align: center; color: black;'>모델 성능 비교</span>", unsafe_allow_html = True)
        st.markdown("---")
        st.pyplot(plt)

def model_app():
    data_1, data_2, data_3, data_4, data_5, data_6, data_7, data_8, data_9 = utils.load_data("ANALSIS_DATA")

    with st.sidebar:
        selected_data = st.selectbox("SELECT DATA", ["Site_1", "Site_2", "Site_3",
                                                     "Site_4", "Site_5", "Site_6",
                                                     "Site_7", "Site_8", "Site_9"])
        selected_model = st.selectbox("SELECT MODEL", ["LogisticRegression", "XGBoost", "LightGBM"])

    if selected_data == "Site_1" and selected_model == "LogisticRegression":
        st.markdown(f"<h2 style='text-align: center; color: black;'>{selected_data} - {selected_model} Model</span>", unsafe_allow_html = True)
        st.markdown("---")
        processing(data_1, "LogisticRegression")
    elif selected_data == "Site_1" and selected_model == "XGBoost":
        st.markdown(f"<h2 style='text-align: center; color: black;'>{selected_data} - {selected_model} Model</span>", unsafe_allow_html = True)
        st.markdown("---")
        processing(data_1, "XGBoost")
    elif selected_data == "Site_1" and selected_model == "LightGBM":
        st.markdown(f"<h2 style='text-align: center; color: black;'>{selected_data} - {selected_model} Model</span>", unsafe_allow_html = True)
        st.markdown("---")
        processing(data_1, "LightGBM")

    if selected_data == "Site_2" and selected_model == "LogisticRegression":
        st.markdown(f"<h2 style='text-align: center; color: black;'>{selected_data} - {selected_model} Model</span>", unsafe_allow_html = True)
        st.markdown("---")
        processing(data_2, "LogisticRegression")
    elif selected_data == "Site_2" and selected_model == "XGBoost":
        st.markdown(f"<h2 style='text-align: center; color: black;'>{selected_data} - {selected_model} Model</span>", unsafe_allow_html = True)
        st.markdown("---")
        processing(data_2, "XGBoost")
    elif selected_data == "Site_2" and selected_model == "LightGBM":
        st.markdown(f"<h2 style='text-align: center; color: black;'>{selected_data} - {selected_model} Model</span>", unsafe_allow_html = True)
        st.markdown("---")
        processing(data_2, "LightGBM")

    if selected_data == "Site_3" and selected_model == "LogisticRegression":
        st.markdown(f"<h2 style='text-align: center; color: black;'>{selected_data} - {selected_model} Model</span>", unsafe_allow_html = True)
        st.markdown("---")
        processing(data_3, "LogisticRegression")
    elif selected_data == "Site_3" and selected_model == "XGBoost":
        st.markdown(f"<h2 style='text-align: center; color: black;'>{selected_data} - {selected_model} Model</span>", unsafe_allow_html = True)
        st.markdown("---")
        processing(data_3, "XGBoost")
    elif selected_data == "Site_3" and selected_model == "LightGBM":
        st.markdown(f"<h2 style='text-align: center; color: black;'>{selected_data} - {selected_model} Model</span>", unsafe_allow_html = True)
        st.markdown("---")
        processing(data_3, "LightGBM")

    if selected_data == "Site_4" and selected_model == "LogisticRegression":
        st.markdown(f"<h2 style='text-align: center; color: black;'>{selected_data} - {selected_model} Model</span>", unsafe_allow_html = True)
        st.markdown("---")
        processing(data_4, "LogisticRegression")
    elif selected_data == "Site_4" and selected_model == "XGBoost":
        st.markdown(f"<h2 style='text-align: center; color: black;'>{selected_data} - {selected_model} Model</span>", unsafe_allow_html = True)
        st.markdown("---")
        processing(data_4, "XGBoost")
    elif selected_data == "Site_4" and selected_model == "LightGBM":
        st.markdown(f"<h2 style='text-align: center; color: black;'>{selected_data} - {selected_model} Model</span>", unsafe_allow_html = True)
        st.markdown("---")
        processing(data_4, "LightGBM")

    if selected_data == "Site_5" and selected_model == "LogisticRegression":
        st.markdown(f"<h2 style='text-align: center; color: black;'>{selected_data} - {selected_model} Model</span>", unsafe_allow_html = True)
        st.markdown("---")
        processing(data_5, "LogisticRegression")
    elif selected_data == "Site_5" and selected_model == "XGBoost":
        st.markdown(f"<h2 style='text-align: center; color: black;'>{selected_data} - {selected_model} Model</span>", unsafe_allow_html = True)
        st.markdown("---")
        processing(data_5, "XGBoost")
    elif selected_data == "Site_5" and selected_model == "LightGBM":
        st.markdown(f"<h2 style='text-align: center; color: black;'>{selected_data} - {selected_model} Model</span>", unsafe_allow_html = True)
        st.markdown("---")
        processing(data_5, "LightGBM")

    if selected_data == "Site_6" and selected_model == "LogisticRegression":
        st.markdown(f"<h2 style='text-align: center; color: black;'>{selected_data} - {selected_model} Model</span>", unsafe_allow_html = True)
        st.markdown("---")
        processing(data_6, "LogisticRegression")
    elif selected_data == "Site_6" and selected_model == "XGBoost":
        st.markdown(f"<h2 style='text-align: center; color: black;'>{selected_data} - {selected_model} Model</span>", unsafe_allow_html = True)
        st.markdown("---")
        processing(data_6, "XGBoost")
    elif selected_data == "Site_6" and selected_model == "LightGBM":
        st.markdown(f"<h2 style='text-align: center; color: black;'>{selected_data} - {selected_model} Model</span>", unsafe_allow_html = True)
        st.markdown("---")
        processing(data_6, "LightGBM")

    if selected_data == "Site_7" and selected_model == "LogisticRegression":
        st.markdown(f"<h2 style='text-align: center; color: black;'>{selected_data} - {selected_model} Model</span>", unsafe_allow_html = True)
        st.markdown("---")
        processing(data_7, "LogisticRegression")
    elif selected_data == "Site_7" and selected_model == "XGBoost":
        st.markdown(f"<h2 style='text-align: center; color: black;'>{selected_data} - {selected_model} Model</span>", unsafe_allow_html = True)
        st.markdown("---")
        processing(data_7, "XGBoost")
    elif selected_data == "Site_7" and selected_model == "LightGBM":
        st.markdown(f"<h2 style='text-align: center; color: black;'>{selected_data} - {selected_model} Model</span>", unsafe_allow_html = True)
        st.markdown("---")
        processing(data_7, "LightGBM")

    if selected_data == "Site_8" and selected_model == "LogisticRegression":
        st.markdown(f"<h2 style='text-align: center; color: black;'>{selected_data} - {selected_model} Model</span>", unsafe_allow_html = True)
        st.markdown("---")
        processing(data_8, "LogisticRegression")
    elif selected_data == "Site_8" and selected_model == "XGBoost":
        st.markdown(f"<h2 style='text-align: center; color: black;'>{selected_data} - {selected_model} Model</span>", unsafe_allow_html = True)
        st.markdown("---")
        processing(data_8, "XGBoost")
    elif selected_data == "Site_8" and selected_model == "LightGBM":
        st.markdown(f"<h2 style='text-align: center; color: black;'>{selected_data} - {selected_model} Model</span>", unsafe_allow_html = True)
        st.markdown("---")
        processing(data_8, "LightGBM")

    if selected_data == "Site_9" and selected_model == "LogisticRegression":
        st.markdown(f"<h2 style='text-align: center; color: black;'>{selected_data} - {selected_model} Model</span>", unsafe_allow_html = True)
        st.markdown("---")
        processing(data_9, "LogisticRegression")
    elif selected_data == "Site_9" and selected_model == "XGBoost":
        st.markdown(f"<h2 style='text-align: center; color: black;'>{selected_data} - {selected_model} Model</span>", unsafe_allow_html = True)
        st.markdown("---")
        processing(data_9, "XGBoost")
    elif selected_data == "Site_9" and selected_model == "LightGBM":
        st.markdown(f"<h2 style='text-align: center; color: black;'>{selected_data} - {selected_model} Model</span>", unsafe_allow_html = True)
        st.markdown("---")
        processing(data_9, "LightGBM")