# rain
2022년 1학기 기계학습 전통 18 
기상청 기상자료개방포털에서 제공하는 데이터를 사용하여 다음날 비가 올지를 예측합니다.

일 별 데이터에서 각 지역 최소 기온 , 최대 기온, 강수량, 최대 풍속, 일조량, 증발량 , 풍향의 데이터를 합친 10년치 데이터를 사용하였습니다.
시간 별 데이터에서 오전 9시와 오후 3시의 기온, 풍속 , 풍향 , 습도 , 전운량 , 현지기압을 사용하였습니다.
해당 날의 최소기온 ,최대 기온등과 오전 9시와 오전 3시 데이터의 값들이 한 횡을 이룹니다.

데이터를 구성하는 column 들 입니다.
Index(['location', 'MinTemp', 'MaxTemp', 'Rainfall', 'WindGustSpeed',
       'WindGustDir', 'Sunshine', 'Evaporation', 'TodayRain', 'Pressure9am',
       'WindDir9am', 'Cloud9am', 'Humidity9am', 'WindSpeed9am', 'Temp9am',
       'Pressure3pm', 'WindDir3pm', 'Cloud3pm', 'Humidity3pm', 'WindSpeed3pm',
       'Temp3pm', 'Year', 'Month', 'Day'],
      dtype='object')




weather.csv
기상청 데이터에서 가져온 일 별 , 시간 별 데이터를 합친 데이터





전처리 

pre_labeled_weather.csv 의 경우 :
laction(지역),WindGustDir,WindDir9am,WindDir3pm 의 경우는 labeling 하였습니다.
결측치의 경우 카테고리 값(laction(지역),WindGustDir,WindDir9am,WindDir3pm) 은 최빈값을 
나머지 숫자 결측치는 평균값을 사용 하였습니다.





MiceImputed_weather.csv 의 경우:
IterativeImputer을 사용하여 전처리한 데이터 




drop_Micelmputed_weather.csv 의 경우:
MiceImputed_weather.csv에서 이상치를 제거한 데이터




scaled_Micelmputed_weather.csv 의 경우:
MiceImputed_weather.csv에서 로그 스케일링을 한 데이터
