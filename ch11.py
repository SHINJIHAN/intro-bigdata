# 예제 9: 예제 4의 자료를 기반으로 평균 세균수에 대한 90% 신뢰구간을 구하고,
# 예제 4의 가설검정을 다시 파이썬으로 실행한 결과를 비교한다. (p.346)

# 예제 4: 도시 상수원 호수의 수질 검사 결과,
# 단위부피당 평균 세균수가 200 이상이면 부적합하다.
# 열 곳의 샘플 데이터를 통해 평균 세균수가 200보다 작은지 검정한다. (p.330)

import numpy as np
bacteria = np.array([175, 190, 215, 198, 184, 207, 210, 193, 196, 180])

# 데이터 기초 통계량 계산
xbar_b = np.mean(bacteria); print(xbar_b)       # 평균
var_b = np.var(bacteria, ddof=1); print(var_b)  # 분산 (표본 분산)
sd_b = np.std(bacteria, ddof=1); print(sd_b)    # 표준편차
median_b = np.median(bacteria); print(median_b) # 중앙값
min_b = np.min(bacteria); print(min_b)          # 최솟값
max_b = np.max(bacteria); print(max_b)          # 최댓값
sum_b = np.sum(bacteria); print(sum_b)          # 합계
n = bacteria.size; print(n)                     # 표본 크기

from scipy import stats

# 표본표준오차(Standard Error)
se_b = stats.sem(bacteria); print(se_b)

# 신뢰구간 계산 (90% 신뢰수준 → 양쪽 5%씩 → t값)
t_alpha = stats.t.ppf(1 - 0.1 / 2, n - 1); print(t_alpha)
interval = t_alpha * se_b; print(interval)  # 신뢰구간의 반길이

# 평균에 신뢰구간 반영
CI = [xbar_b - interval, xbar_b + interval]; print(CI)

# 가설검정: 귀무가설 H0: μ = 200, 대립가설 H1: μ < 200 (단측 검정)
tval = (xbar_b - 200) / se_b; print(tval)  # t 통계량
pval = stats.t.cdf(tval, n - 1); print(pval)  # 누적 확률(p값)

# 해석: p값 0.1211은 유의수준 0.05보다 크므로
# 귀무가설을 기각할 수 없다 → 평균 세균수가 200보다 작다고 단정할 수 없다.


# 예제 10: 새로운 자료 x에 대해 정규성 검토 및 평균이 38보다 큰지 검정 (p.348)

x = np.array([31, 35, 37, 38, 38, 38, 39, 40, 40, 41, 42, 43, 44, 44, 46, 48])

# 기초 통계량 계산
xbar_x = np.mean(x); print(xbar_x)         # 평균
var_x = np.var(x, ddof=1); print(var_x)    # 분산
sd_x = np.std(x, ddof=1); print(sd_x)      # 표준편차
median_x = np.median(x); print(median_x)   # 중앙값
min_x = np.min(x); print(min_x)            # 최솟값
max_x = np.max(x); print(max_x)            # 최댓값
sum_x = np.sum(x); print(sum_x)            # 합계
n = x.size; print(n)                       # 데이터 개수

# 정규성 검토: Q-Q plot
import matplotlib.pyplot as plt
import statsmodels.api as sm

sm.qqplot(x, line='s')  # 표준 정규분포 기준선과 비교
plt.title("Normal Q-Q Plot")
plt.show()

# 평균에 대한 95% 신뢰구간 계산
se = stats.sem(x); print(se)
t_alpha = stats.t.ppf(1 - 0.05 / 2, n - 1); print(t_alpha)
interval = t_alpha * se; print(interval)
CI = [xbar_x - interval, xbar_x + interval]; print(CI)

# 가설검정: 귀무가설 H0: μ = 38, 대립가설 H1: μ > 38 (단측 검정)
tval = (xbar_x - 38) / se; print(tval)
pval = 1 - stats.t.cdf(tval, n - 1); print(pval)

# 해석: p값이 0.026으로 유의수준 5%보다 작으므로
# 귀무가설을 기각하고, 평균이 38보다 크다고 할 수 있다.
