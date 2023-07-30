# -*- coding: utf-8 -*-
import pandas as pd
import requests
from bs4 import BeautifulSoup
import re
import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
import streamlit as st
from streamlit_player import st_player, _SUPPORTED_EVENTS

def fireWarring():
    # Define the URL of the webpage
    url = "http://forestfire.nifos.go.kr/main.action"
    st.markdown(f"[Forest Fire Information System]({url})")
    st.image("http://forestfire.nifos.go.kr/images/map/img_forest_today.gif")

def crawling():
    """
    네이버 뉴스를 크롤링하여 제목과 URL을 수집하는 함수입니다.

    Returns:
        DataFrame: 크롤링 결과를 담은 데이터프레임
    """
    search = '강원도 산불'
    page = 5
    #start수를 1, 11, 21, 31 ...만들어 주는 함수
    page_num = 0

    if page == 1:
        page_num =1
    elif page == 0:
        page_num =1
    else:
        page_num = page+9*(page-1)

    news_title = []
    news_url = []
    news_content = []

    for i in range(1, page_num + 1, 10):
      #url 생성
      url = "https://search.naver.com/search.naver?where=news&sm=tab_pge&query=" + search + "&start=" + str(page_num)

      # ConnectionError방지
      headers = { "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/100.0.48496.75" }

      #html불러오기
      original_html = requests.get(url, headers=headers)
      html = BeautifulSoup(original_html.text, "html.parser")

      # 검색결과
      articles = html.select("div.group_news > ul.list_news > li div.news_area > a")
      articles2 = html.select("div.group_news > ul.list_news > li div.news_area > div.dsc_wrap")


      for i in articles:
          news_title.append(i.attrs['title'])
          news_url.append(i.attrs['href'])
      for i in articles2:
          news_content.append(i.text)


    news_df = pd.DataFrame({'Title': news_title, 'URL': news_url})

    return news_df

def news():
    """
    네이버 뉴스를 크롤링하여 표 형태로 출력하는 함수입니다.

    Returns:
        None
    """
    st.subheader("News")
    news_df = crawling()
    # Add link to phone number column
    news_df['URL'] = news_df['URL'].apply(lambda x: f'<a href="{x}">{x}</a>')
    # Convert DataFrame to HTML table with center-aligned content and column names
    news_df = news_df.to_html(index=False, classes=["center-aligned"], justify="center", escape=False)
    # Apply CSS styling to center-align the table
    news_df = f"<style>.center-aligned {{ text-align: center; }}</style>{news_df}"
    # Display the HTML table in Streamlit
    st.markdown(news_df, unsafe_allow_html=True)

def youtubeNews():
    """
        YouTube 동영상 재생기를 포함한 뉴스 섹션을 생성하는 함수입니다.

        Returns:
            None
    """
    c1, c2, c3 = st.columns([3, 3, 2])

    with c3:
        st.subheader("Parameters")

        options = {
            "events": st.multiselect("Events to listen", _SUPPORTED_EVENTS, ["onProgress"]),
            "progress_interval": 1000,
            "volume": st.slider("Volume", 0.0, 1.0, 1.0, .01),
            "playing": st.checkbox("Playing", False),
            "loop": st.checkbox("Loop", False),
            "controls": st.checkbox("Controls", True),
            "muted": st.checkbox("Muted", False),
        }

    with c1:
        url1 = st.text_input("First URL", "https://www.youtube.com/watch?v=ZP4s4CEGMJw")
        st_player(url1, **options, key="youtube_player1")

    with c2:
        url2 = st.text_input("Second URL", "https://www.youtube.com/watch?v=ya8MurTg4x0")
        st_player(url2, **options, key="youtube_player2")

    c4, c5, c6 = st.columns([3, 3, 2])
    with c4:
        url3 = st.text_input("Third URL", "https://www.youtube.com/watch?v=ECZBcCVNogI")
        st_player(url3, **options, key="youtube_player3")

    with c5:
        url4 = st.text_input("Fourth URL", "https://www.youtube.com/watch?v=cRIGefdVj-g")
        st_player(url4, **options, key="youtube_player4")

    with c6:
        with st.expander("SUPPORTED PLAYERS", expanded=False):
            st.write("""
               - Dailymotion
               - Facebook
               - Mixcloud
               - SoundCloud
               - Streamable
               - Twitch
               - Vimeo
               - Wistia
               - YouTube
               <br/><br/>
               """, unsafe_allow_html=True)

def callNumber():
    """
        전화번호 정보를 보여주는 탭을 포함한 함수입니다.

        Returns:
            None
    """
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["강원도", "산림청", "시/군", "소방서", "국립공원", "한국전력공사"])
    with tab1 :
        st.subheader("강원도")
        gangwon = pd.read_csv("data/gangwon.csv", encoding='cp949')
        # Add link to phone number column
        gangwon['대표전화번호'] = gangwon['대표전화번호'].apply(lambda x: f'<a href="tel:{x}">{x}</a>')
        # Convert DataFrame to HTML table with center-aligned content and column names
        gangwon_table = gangwon.to_html(index=False, classes=["center-aligned"], justify="center", escape=False)
        # Apply CSS styling to center-align the table
        gangwon_table = f"<style>.center-aligned {{ text-align: center; }}</style>{gangwon_table}"
        # Display the HTML table in Streamlit
        st.markdown(gangwon_table, unsafe_allow_html=True)
    with tab2:
        st.subheader("산림청")
        mountain = pd.read_csv("data/mountain.csv", encoding='cp949')
        # Add link to phone number column
        mountain['대표전화번호'] = mountain['대표전화번호'].apply(lambda x: f'<a href="tel:{x}">{x}</a>')
        # Convert DataFrame to HTML table with center-aligned content and column names
        mountain_table = mountain.to_html(index=False, classes=["center-aligned"], justify="center", escape=False)
        # Apply CSS styling to center-align the table
        mountain_table = f"<style>.center-aligned {{ text-align: center; }}</style>{mountain_table}"
        # Display the HTML table in Streamlit
        st.markdown(mountain_table, unsafe_allow_html=True)
    with tab3:
        st.subheader("시/군")
        mountain = pd.read_csv("data/mountain.csv", encoding='cp949')
        # Add link to phone number column
        mountain['대표전화번호'] = mountain['대표전화번호'].apply(lambda x: f'<a href="tel:{x}">{x}</a>')
        # Convert DataFrame to HTML table with center-aligned content and column names
        mountain_table = mountain.to_html(index=False, classes=["center-aligned"], justify="center", escape=False)
        # Apply CSS styling to center-align the table
        mountain_table = f"<style>.center-aligned {{ text-align: center; }}</style>{mountain_table}"
        # Display the HTML table in Streamlit
        st.markdown(mountain_table, unsafe_allow_html=True)
    with tab4:
        st.subheader("소방서")
        firestation = pd.read_csv("data/firestation.csv", encoding='cp949')
        # Add link to phone number column
        firestation['대표전화번호'] = firestation['대표전화번호'].apply(lambda x: f'<a href="tel:{x}">{x}</a>')
        # Convert DataFrame to HTML table with center-aligned content and column names
        firestation_table = firestation.to_html(index=False, classes=["center-aligned"], justify="center", escape=False)
        # Apply CSS styling to center-align the table
        firestation_table = f"<style>.center-aligned {{ text-align: center; }}</style>{firestation_table}"
        # Display the HTML table in Streamlit
        st.markdown(firestation_table, unsafe_allow_html=True)
    with tab5:
        st.subheader("국립 공원")
        park = pd.read_csv("data/mountain.csv", encoding='cp949')
        # Add link to phone number column
        park['대표전화번호'] = park['대표전화번호'].apply(lambda x: f'<a href="tel:{x}">{x}</a>')
        # Convert DataFrame to HTML table with center-aligned content and column names
        park_table = park.to_html(index=False, classes=["center-aligned"], justify="center", escape=False)
        # Apply CSS styling to center-align the table
        park_table = f"<style>.center-aligned {{ text-align: center; }}</style>{park_table}"
        # Display the HTML table in Streamlit
        st.markdown(park_table, unsafe_allow_html=True)
    with tab6:
        st.subheader("한국 전력 공사")
        elect = pd.read_csv("data/elect.csv", encoding='cp949')
        # Add link to phone number column
        elect['대표전화번호'] = elect['대표전화번호'].apply(lambda x: f'<a href="tel:{x}">{x}</a>')
        # Convert DataFrame to HTML table with center-aligned content and column names
        elect_table = elect.to_html(index=False, classes=["center-aligned"], justify="center", escape=False)
        # Apply CSS styling to center-align the table
        elect_table = f"<style>.center-aligned {{ text-align: center; }}</style>{elect_table}"
        # Display the HTML table in Streamlit
        st.markdown(elect_table, unsafe_allow_html=True)

def declaration():
    """
        신고 및 문의를 위한 정보를 입력받고 전송하는 함수입니다.

        Returns:
            None
    """
    buff, col, buff2 = st.columns([1, 3, 1])
    with col :
        input_user_name=st.text_input("이름", key="name",max_chars=5)
        input_phone_number=st.text_input("전화번호",key="phonenumber",max_chars=13)
        input_text=st.text_area("신고 및 문의 내용",key="declaration",height=30)
    buff3, col1, buff4 = st.columns([2, 1, 1.5])
    with col1:
        checkbox = st.checkbox('개인 정보 이용 동의')
        btn_clicked = st.button('전송',disabled=(checkbox is False))
        if btn_clicked:
            con = st.container()
            con.caption("Result")
            if not str(input_user_name):
                con.error("이름을 확인해 주세요")
            elif not str(input_phone_number):
                con.error("전화 번호를 확인해 주세요")
            elif not str(input_text):
                con.error("접수 내용을 확인해 주세요")
            else:
                con.warning('접수가 완료 되었습니다. 순차적으로 처리하여 연락드리겠습니다.')
        else:
            st.write('전송을 누르면 접수가 완료됩니다.')

def fireStats():
    """
       대형 화재 통계 정보를 표시하는 함수입니다.

       Returns:
           None
    """
    bigfire = pd.read_csv("data/bigfire.csv", encoding='cp949')
    # Convert DataFrame to HTML table with center-aligned content and column names
    bigfire_table = bigfire.to_html(index=False, classes=["center-aligned"], justify="center", escape=False, na_rep="")
    # Apply CSS styling to center-align the table
    bigfire_table = f"<style>.center-aligned {{ text-align: center; }}</style>{bigfire_table}"
    # Display the HTML table in Streamlit
    st.markdown(bigfire_table, unsafe_allow_html=True)

def service_app():
    st.sidebar.markdown("## SubMenu")
    Service_List = st.sidebar.radio(" ", ['강원 비상 연락망', '네이버 뉴스', '강원 산불 영상', '신고 서비스', '대형 산불 통계'], label_visibility='collapsed')
    if Service_List == '네이버 뉴스':
        st.header("강원 산불 뉴스")
        news()
    elif Service_List == '강원 산불 영상':
        st.header("강원 산불 뉴스 영상")
        youtubeNews()
    elif Service_List == '강원 비상 연락망':
        st.header("비상 연락망")
        callNumber()
    elif Service_List == '신고 서비스':
        st.header("신고 서비스")
        declaration()
    elif Service_List == '대형 산불 통계':
        st.header("대형 산불 통계")
        fireStats()