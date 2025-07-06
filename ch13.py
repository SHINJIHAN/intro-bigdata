# 예제 9: 4장의 예제 5에 주어진 자료를 사용하여
# 키(height)를 기반으로 몸무게(weight)를 예측하는 단순 선형회귀 분석을 수행하라.

import numpy as np

# 키와 몸무게 데이터 (각 52개)
height = np.array([181, 161, 170, 160, 158, 168, 162, 179, 183, 178, 
                   171, 177, 163, 158, 160, 160, 158, 173, 160, 163, 
                   167, 165, 163, 173, 178, 170, 167, 177, 175, 169, 
                   152, 158, 160, 160, 159, 180, 169, 162, 178, 173, 
                   173, 171, 171, 170, 160, 167, 168, 166, 164, 173, 
                   180]) 

weight = np.array([78, 49, 52, 53, 50, 57, 53, 54, 71, 73, 
                   55, 73, 51, 53, 65, 48, 59, 64, 48, 53, 
                   78, 45, 56, 70, 68, 59, 55, 64, 59, 55, 
                   38, 45, 50, 46, 50, 63, 71, 52, 74, 52, 
                   61, 65, 68, 57, 47, 48, 58, 59, 55, 74, 
                   74])

#----------------------------------------#

import pandas as pd
import statsmodels.formula.api as smf

# 데이터 딕셔너리 생성 후 DataFrame으로 변환
d = {'height': height, 'weight': weight}
dat = pd.DataFrame(data=d)

# 단순 선형 회귀모형 적합 (종속변수: weight, 독립변수: height)
fit = smf.ols('weight ~ height', data=dat).fit()
print(fit.summary())  # 회귀 분석 결과 출력

#----------------------------------------#

# 회귀 해석 요약:
# 추정된 회귀식: weight = -100.782 + 0.9479 * height
# ⇒ 키가 1cm 증가할 때 몸무게는 평균적으로 약 0.9479kg 증가
# R-squared = 0.542 ⇒ 약 54.2%의 몸무게 변동이 키로 설명됨
# height에 대한 p값 < 0.01 ⇒ 키는 몸무게 예측에 통계적으로 유의한 변수

# 회귀계수의 95% 신뢰구간 출력
conf_int = fit.conf_int(0.05)

#----------------------------------------#

# 시각화를 위한 추가 모듈 import
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
import statsmodels.api as sm
from scipy import stats

# 한글 폰트 설정 (윈도우의 경우: 맑은 고딕 사용)
font_name = font_manager.FontProperties(fname='C:/Windows/Fonts/malgun.ttf').get_name()
rc('font', family=font_name, size=10)
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 깨짐 방지

# 전체 그래프 크기 설정 (2x2 서브플롯)
plt.figure(figsize=(10, 8))

# [1] 산점도 + 회귀직선
plt.subplot(2, 2, 1)
slope, intercept = np.polyfit(height, weight, 1)
abline_values = [slope * i + intercept for i in height]

plt.plot(height, weight, 'o', label='Data')      # 실제 관측값 (산점도)
plt.plot(height, abline_values, 'b', label='Fit')  # 회귀직선
plt.title('키(height)와 몸무게의 관계')
plt.xlabel('키 (height)')
plt.ylabel('몸무게 (weight)')
plt.legend()

# [2] 잔차도: 잔차 vs 키
plt.subplot(2, 2, 2)
residuals = fit.resid
plt.plot(height, residuals, 'o')
plt.title('잔차도 (Residual Plot)')
plt.xlabel('키 (height)')
plt.ylabel('잔차 (Residuals)')

# [3] 잔차의 정규확률도: 정규분포 가정 확인용
plt.subplot(2, 2, 3)
stats.probplot(residuals, dist='norm', plot=plt)
plt.title('잔차의 정규성 검정')
plt.xlabel('이론 분위수 (Theoretical Quantiles)')
plt.ylabel('표본 분위수 (Sample Quantiles)')

# 여백 자동 조정
plt.tight_layout()
plt.show()

#----------------------------------------#

# 종합 해석:
# 1. 회귀 계수 p값이 매우 작기 때문에, 키는 몸무게를 예측하는 데 유의미한 변수다.
# 2. 잔차도에서 특정한 패턴이 없으므로 등분산성 가정이 적절하다.
# 3. 잔차의 정규 확률도에서 점들이 직선에 가깝게 분포하므로, 잔차는 정규분포를 따르는 것으로 보인다.
# ⇒ 따라서 이 회귀모형은 통계적으로 유의하며, 예측에도 활용할 수 있다.