# ForestFire 
## Measures to Minimize Forest Fire Damage

[Streamlit 링크]() <br/><br/>
[발표 영상]() <br/><br/>
[데모 시연]() <br/><br/>
[포트 폴리오]() <br/><br/>
![screensh]()

## 1.프로젝트의 시작 (2023.05.22 ~ 2023.06.23)
- 산불 피해 최소화 대책 
 
## 2. 대회 목표 : 산불 예측 대응 체계를 위한 위험 예보 시스템 구축
- 공공 데이터 API 를 통한 실시간 데이터 수집
- 이미지 크롤링 및 딥러닝을 통한 예측 시스템 구축
- 산불에 관련된 여러 변수들을 파악하여 회귀분석 진행
- 
## 3. 세부 수행 내용
- 행정안전부 빅데이터 공통기반 시스템, 기상청(동네예보, KLAPS, 초단기 실황), 산악기상망 등의 빅데이터와 산림청, 국가정보자원관리원의 공간 API로 Data 수집
- 수집된 DB 로딩 및 전처리
- 로지스틱 회귀분석, 공간 정보 분석을 통한 데이터 분석 및 시각화
- 산불위험 제보, 산불 지역 알림 분석 및 시각화
- 이용자의 GPS를 이용하여 인접 위치에 대한 산림정보 및 산불위험지수 등급화, 산불위험 제보 서비스
- 행정구역별 산불 위험등급, 대형 산불 위험정보, 산불위험 통계, 과거 자료 검색 등의 서비스

## 4. 데이터
- 공공 데이터 API
  + 기상청
  + 국가정보자원관리원 
  + 공공 데이터
  + 지도
  + 
## 5. ERD (개체 관계 모델)
![screensh]()

## 6. 팀 구성
- 사용언어 : Python : 3.9.13v
- 작업툴 : VS code
- 인원 : 4명
- 주요 업무 :
  + 공공데이터 API를 통한 데이터 수집
  + 회귀분석, 공간 정보 분석을 통한 데이터 분석 및 시각화
  + 이미지 크롤링을 통한 딥러닝
  + Streamlit 대시보드 개발을 통한 웹 서비스
- 기간 : 2023.05.22 ~ 2023.06.23
***

## 7. 주요 기능
- Home
  + 
- Description
  + 
- Data
  +  
- EDA
  +   
  + 
  + 
  + 
- STAT
  + 
  + 
- DL
  + 
  + 
***

## 8. 설치 방법
### Windows
+ 버전 확인
    - vscode : 1.74.1
    - python : 3.9.13
    - 라이브러리 : pandas (1.5.3), numpy (1.23.5), plotly (5.14.1), matplotlib (3.7.1), streamlit (1.21.0), seaborn (0.12.2), pingouin (0.5.3), statsmodels (0.13.2), scikit-learn (1.2.2), xgboost (1.7.5), pandas-profiling (3.6.3), streamlit-option-menu (0.3.2), streamlit_pandas_profiling (0.1.3), scipy(1.9.1), 


- 프로젝트 파일을 다운로드 받습니다. 

```bash
git clone https://github.com/ChoiJMS2/forestfire.git
```

- 프로젝트 경로에서 가상환경 설치 후 접속합니다. (Windows 10 기준)
```bash
virtualenv venv
source venv/Scripts/activate
```

- 라이브러리를 설치합니다. 
```bash
pip install -r requirements.txt
```

- streamlit 명령어를 실행합니다. 
```bash
streamlit run app.py
```

## 9. 주요 업데이트 내용
