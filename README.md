# ForestFire (~ 2023.06.23)
<br/>

## [1. Streamlit Service](https://kingbeem-forestfire-app-zxbk0n.streamlit.app/ "Streamlit Link")<br/>

## [2. Personal Code](https://github.com/KingBeeM/ForestFire/tree/main/file/code/ "Code Link")<br/>

## [3. Deep Learning](https://github.com/KingBeeM/ForestFire/blob/main/file/code/DL_EfficientNet.ipynb/ "DL Link")<br/>

## [4. PDF File](https://github.com/KingBeeM/ForestFire/blob/main/file/ppt/ForestFire.pdf/ "PDF Link")<br/>

---

## ✔ 목적
강원도 산불 예측 및 피해 최소화 프로젝트 : 머신러닝과 딥러닝을 활용한 모델 개발
<br/>

## ✔ 데이터
| 제공 사이트          | 제공 기관   | 한글 이름  | 사용 테이블 이름 |
|---------------------|------------|--------|-------------|
| 공공데이터포털      | 기상청     | 기상청_지상(종관, ASOS) 일자료 조회서비스 | weather_days|
| 공공데이터포털      | 기상청     | 기상청_관측지점정보 | weather_stations|
| 공공데이터포털      | 산림청    | 산림청_산불발생통계(대국민포털) | forestfire_occurs_add |
| 공공데이터포털      | 행정안전부 | 산불발생이력 | forestfire_occurs|
| 국가공간정보포털    | 국토교통부 | 행정구역_읍면동(법정동) | gangwon_UMD |
| 국가공간정보포털    | 국토교통부 | 행정구역시군구_경계 | gangwon_SSG |
| 행정표준코드관리시스템 | 국토교통부 | 행정구역_코드(법정동) | gangwon_code|
<br/>

## ✔ ERD
![image](https://github.com/KingBeeM/ForestFire/blob/main/file/img/ERD.png)
<br/>

## ✔ Flow Chart
![image](https://github.com/KingBeeM/ForestFire/blob/main/file/img/flowchart.png)
<br/>

## ✔ 팀 구성
- 사용언어 : Python
- 작업툴 : VS Code / PyCharm / Google Colab / Google BigQuery / QGIS / IBM SPSS Statistics
- 인원 : 4명
- 주요 업무 : 강원도 산불 예측 및 피해 최소화 프로젝트 : 머신러닝과 딥러닝을 활용한 모델 개발
- 기간 : 2023.05.22 ~ 2023.06.23

## ✔ 주요 기능
- **HOME**
  - 강원도 산불위험지수(DWI) 지도시각화
    - 기상요인을 고려해 강원도 지역을 9개로 나누어서 각각 지역에 대해 ML 모델 생성
    - 실시간 API 요청을 통한 각 지역별 실시간 산불위험지수(DWI) 지도시각화

![image1](https://github.com/KingBeeM/ForestFire/blob/main/file/img/home_img.png)
- **EDA**
  - 강원도 기상 정보를 바탕으로 강원도 지역 9 분할 과정
    - 강원지방기상청 관할 예·특보구역에 따라 12 분할
    - 강원도 기상관측소 위치정보를 고려해 9 분할로 수정

![image2](https://github.com/KingBeeM/ForestFire/blob/main/file/img/EDA_img.png)
- **STAT**
  - 기상요인을 고려해 강원도 지역을 9개로 분할한 각 지역에 대해 통계분석
    - Python 환경에서 통계분석에 한계가 있어 SPSS 에서 진행
    - 각 지역별 종속변수에 영향을 미치는 독립변수 요소에 대해 파악

![image3](https://github.com/KingBeeM/ForestFire/blob/main/file/img/stat_img1.png)
![image4](https://github.com/KingBeeM/ForestFire/blob/main/file/img/stat_img2.png)
- **ML**
  - 기상요인을 고려해 강원도 지역을 9개로 분할한 각 지역에 대해 ML 모델 생성
  - 각 지역별로 LogisticRegression / XGBoost / LightGBM 모델 생성
  - 각 모델별 ROC-AUC 비교를 통한 각 지역별 적합한 모델 선정

![image5](https://github.com/KingBeeM/ForestFire/blob/main/file/img/model_img.png)
- **DL**
  - EfficentNet-B7 모델을 사용한 산불 이미지 분류 모델
  - 6개의 Class 에 대해 모델 훈련
  - 신뢰성을 고려하여 성능지표로 Top-2 Accuracy 사용 (상위 2개)
  - Warning message 와 GPS 기반 주소 및 이미지 전송

<img src="/file/img/DL_img.png" width="500" height="600">

## ✔ 설치 방법

### Windows
- 버전 확인
  - VS Code / PyCharm : Python 3.10.9
  - Google Colab
  - 라이브러리 : 
```
beautifulsoup4==4.11.1
bs4==0.0.1
db-dtypes==1.1.1
Flask==2.2.2
folium==0.14.0
geopandas==0.13.0
google-cloud-bigquery==3.11.0
googlemaps==4.10.0
keras==2.12.0
lxml==4.9.1
matplotlib==3.7.0
missingno==0.5.2
numpy==1.23.5
opencv-python==4.7.0.72
pandas==1.5.3
pandas-gbq==0.19.2
pingouin==0.5.3
plotly==5.9.0
scikit-learn==1.2.1
seaborn==0.12.2
selenium==4.8.3
shapely==2.0.1
statsmodels==0.13.5
streamlit==1.20.0
streamlit-option-menu==0.3.5
streamlit-pandas-profiling==0.1.3
tensorflow==2.12.0
torch==2.0.0
torchvision==0.15.1
tqdm==4.64.1
xgboost==1.7.5
```
- 프로젝트 파일을 다운로드 받습니다.
```
git clone https://github.com/KingBeeM/ForestFire.git
```
- 프로젝트 경로에서 가상환경 설치 후 접속합니다. (Windows 11 기준)
```
virtualenv venv
source venv/Scripts/activate
```
- 라이브러리를 설치합니다.
```
pip install -r requirements.txt
```
- streamlit 명령어를 실행합니다.
```
streamlit run app.py
```