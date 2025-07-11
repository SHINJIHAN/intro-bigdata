# 예제13: 음료수 80병의 용량 측정 데이터 (단위: ml)

import numpy as np
import pandas as pd

# drink 변수에 음료수 용량 데이터를 NumPy 배열로 저장
drink = np.array([98, 99, 100, 99, 99.4, 101.7, 98.8, 101.8, 101.5, 
                  101.8, 102.6, 101, 98.8, 101.4, 99.7, 99.7, 99.7, 
                  100.9, 98.6, 101.4, 102.1, 102.9, 100.8, 101.8, 
                  100, 101.2, 100.5, 101.2, 100.1, 101.6, 101.3, 99.9, 
                  99.4, 99.3, 99.4,101.6, 96.1, 100, 99.7, 99.1, 100.7, 
                  100.8, 100.8, 95.5,100.1, 100.5, 98.9, 99.9, 96.8, 
                  102.4, 100, 103.7, 101.4,99.7, 97.4, 99.5, 97.5, 
                  99.9, 100.3, 100.2, 101.5, 99.4, 99.7, 98.2, 100.3, 
                  100.2, 100.5, 100.4, 101.5, 98.4, 101.4, 98.8, 100.9, 
                  101.1, 100.9, 98.1, 98.7, 99.2, 98.1, 97.2])

# ------------------------------------------------------------
# 각종 통계량 계산
# ------------------------------------------------------------

# 평균 (데이터들의 산술평균)
print(np.mean(drink))

# 중앙값 (데이터를 크기순으로 정렬했을 때 중앙에 위치한 값)
print(np.median(drink))

# 분산 (데이터가 평균에서 얼마나 떨어져 있는지, 표본분산: 자유도 1로 계산)
print(np.var(drink, ddof=1))

# 분산 (모집단 분산: 자유도 0으로 계산, 표본이 모집단 전체인 경우)
print(np.var(drink))

# 표준편차 (분산의 제곱근, 표본 표준편차)
print(np.std(drink, ddof=1))

# 표준편차 (모집단 표준편차)
print(np.std(drink))

# 범위 (최대값과 최소값의 차이)
print(np.max(drink) - np.min(drink))

# 사분위수 범위 (IQR, 중간 50% 데이터의 범위: 75% 분위수 - 25% 분위수)
q1, q3 = np.percentile(drink, [25, 75])
print(q3 - q1)

# ------------------------------------------------------------
# pandas DataFrame을 활용한 요약 통계량 출력
# ------------------------------------------------------------

# NumPy 배열을 pandas DataFrame으로 변환
drink_df = pd.DataFrame(drink)

# 기술 통계량 요약 출력 (개수, 평균, 표준편차, 최소/최대값, 사분위수 등)
summary = drink_df.describe()
print(summary)
