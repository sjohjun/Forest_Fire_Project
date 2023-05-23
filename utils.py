# -*- coding:utf-8 -*-


import lxml
import requests
from bs4 import BeautifulSoup
import pandas as pd
from pathlib import Path
import streamlit as st
import time
import math

@st.cache_data
def load_data():
    comp_dir = Path('data/')

    service_key = ''  # 인증키
    url = f'{service_key}'  # 인증키 포함 주소
    req = requests.get(url)
    req.content
    soup = BeautifulSoup(req.content, "lxml")  # XML 생성
    content = req.json()    # JSON 생성

    # json을 데이터프레임으로 전환
    row = content['주소']['row']
    df = pd.DataFrame(row)

    return soup, content, df


# Date Selection
def date_select():
    return