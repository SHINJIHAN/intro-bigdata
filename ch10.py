# 예제 13: 중1 남학생 평균 키의 95% 신뢰구간

import numpy as np
from scipy import stats

# 중학교 1학년 남학생 30명의 키 데이터
height = np.array([163, 161, 168 , 161, 157, 162, 153, 159, 164, 170,
                   152, 160, 157, 168, 150, 165, 156, 151, 162, 150,
                   156, 152, 161, 165, 168, 167, 165, 168, 159, 156])

# 표본통계량 계산
xbar_h = np.mean(height)            # 평균
var_h = np.var(height, ddof=1)     # 분산 (자유도 1)
sd_h = np.std(height, ddof=1)      # 표준편차
se_h = stats.sem(height)           # 표본표준오차 (SEM)

print(f"평균: {xbar_h:.2f}, 표준편차: {sd_h:.2f}, 표본오차: {se_h:.2f}")

# 신뢰구간 계산 (95% → 양쪽 각각 2.5%)
z_alpha = stats.norm.ppf(1 - 0.05 / 2)
interval = z_alpha * se_h
CI = [xbar_h - interval, xbar_h + interval]
print(f"95% 신뢰구간: {CI}")


# 예제 14: 평균키가 159cm와 통계적으로 다른가?

# 귀무가설: 평균키 = 159cm
# 대립가설: 평균키 ≠ 159cm

zval = (xbar_h - 159) / se_h        # z통계량 계산
pval = 2 * (1 - stats.norm.cdf(abs(zval)))  # 양측검정 p값

print(f"z값: {zval:.3f}, p값: {pval:.3f}")

# 해석:
# p값이 크므로 (보통 0.05보다 크면) → 귀무가설 기각 불가
# → 평균키가 159cm와 통계적으로 유의한 차이가 없다고 본다


# 예제 15: 평균 교통소음에 대한 98% 신뢰구간과 가설검정

# 교통소음 측정 데이터 (49개)
noise = np.array([55.9, 63.8, 57.2, 59.8, 65.7, 62.7, 60.8, 51.3, 61.8, 56.0,
                  66.9, 56.8, 66.2, 64.6, 59.5, 63.1, 60.6, 62.0, 59.4, 67.2,
                  63.6, 60.5, 66.8, 61.8, 64.8, 55.8, 55.7, 77.1, 62.1, 61.0,
                  58.9, 60.0, 66.9, 61.7, 60.3, 51.5, 67.0, 60.2, 56.2, 59.4,
                  67.9, 64.9, 55.7, 61.4, 62.6, 56.4, 56.4, 69.4, 57.6, 63.8])

# 표본통계량
xbar_n = np.mean(noise)
sd_n = np.std(noise, ddof=1)
se_n = stats.sem(noise)

print(f"평균: {xbar_n:.2f}, 표준편차: {sd_n:.2f}, 표본오차: {se_n:.2f}")


# 98% 신뢰구간
z_alpha = stats.norm.ppf(1 - 0.02 / 2)  # 98% CI → 양쪽 1%씩
interval = z_alpha * se_n
CI = [xbar_n - interval, xbar_n + interval]
print(f"98% 신뢰구간: {CI}")


# 유의수준 5%에서 평균이 60을 초과하는지 검정
# 귀무가설: μ ≤ 60
# 대립가설: μ > 60  (단측검정)

zval = (xbar_n - 60) / se_n
pval = stats.norm.sf(zval)  # 상위 단측 검정

print(f"z값: {zval:.3f}, p값: {pval:.3f}")

# 해석:
# p값이 0.021로 유의수준 0.05보다 작음 → 귀무가설 기각
# → 평균 교통소음이 60을 초과한다고 할 수 있음
