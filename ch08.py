# 예제 7: 어느 대학교의 일반수학 중간고사 성적은 평균 63, 분산 100인 정규분포를 따른다고 가정한다. (p.232)

from scipy.stats import norm

# (1) 성적이 50점 이하인 학생의 비율 계산
# 평균(loc)=63, 표준편차(scale)=10
print(norm.cdf(x=50, loc=63, scale=10))

# 해석: 50점 이하인 학생 비율은 약 9.68%이다.

# (2) 상위 10%의 학생에게 A를 준다면, 해당되는 점수 계산 (상위 10% → 누적확률 90%)
print(norm.ppf(q=0.9, loc=63, scale=10))

# 해석: 상위 10% 커트라인은 약 75.82점이므로, A는 75.8점 이상부터 부여한다.



# 예제 12: 어떤 데이터가 정규분포를 따르는지 판단하기 위해 정규확률그림을 작성한다. (p.247)

import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

# 샘플 데이터: 1인당 연간 소비 금액 등과 같은 예제일 수 있음
data1 = np.array([
    4001, 3927, 3048, 4298, 4000, 3445, 4949, 3530, 3075, 4012,
    3797, 3550, 4027, 3571, 3738, 5157, 3598, 4749, 4263, 3894,
    4262, 4232, 3852, 4256, 3271, 4315, 3078, 3607, 3889, 3147,
    3421, 3531, 3987, 4120, 4349, 4071, 3683, 3332, 3285, 3739,
    3544, 4103, 3401, 3601, 3717, 4846, 5005, 3991, 2866, 3561,
    4003, 4387, 3510, 2884, 3819, 3173, 3470, 3340, 3214, 3670, 3694
])

# 정규확률그림(Q-Q plot)을 통해 데이터가 정규분포를 따르는지 확인
sm.qqplot(data1, line="s")
plt.title("Normal Q-Q Plot")

# 해석: 데이터 점들이 대각선 근처에 밀집하면 정규성을 가정해도 무방하다.



# 예제 13: 숲속 나무의 체적 데이터를 통해 히스토그램과 정규성 검토 (p.249)

import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

# 나무의 체적 데이터 (m³ 단위로 추정)
data2 = np.array([
    39.3, 14.8, 6.3, 0.9, 6.5, 3.5, 8.3, 10.0, 1.3, 7.1, 6.0, 17.1, 16.8, 0.7,
    7.9, 2.7, 26.2, 24.3, 17.7, 3.2, 7.4, 6.6, 5.2, 8.3, 5.9, 3.5, 8.3, 44.8,
    8.3, 13.4, 19.4, 19.0, 14.1, 1.9, 12.0, 19.7, 10.3, 3.4, 16.7, 4.3, 1.0,
    7.6, 28.33, 26.2, 31.7, 8.7, 18.9, 3.4, 10.0
])

# 히스토그램 작성: 데이터 분포 형태 확인
plt.hist(data2, bins=5, range=(0, 50), edgecolor="black")
plt.xlabel('Tree Volume (m³)')
plt.ylabel('Frequency')
plt.title('Histogram of Tree Volume')
plt.show()

# 해석: 분포가 왼쪽으로 치우쳐 있어(좌편향) 정규분포를 따르지 않을 수 있다.

# Q-Q 플롯: 정규성 확인용
sm.qqplot(data2, line="s")
plt.title("Normal Q-Q Plot (Original Data)")
plt.show()

# 해석: 데이터가 직선을 크게 벗어나 정규분포를 따르지 않음

# ----------------------------
# 데이터 변환 후 정규성 재검토
# ----------------------------

# (1) 제곱근 변환 (sqrt): 좌편향 완화 시도
data3 = np.sqrt(data2)
sm.qqplot(data3, line="s")
plt.title("Q-Q Plot after sqrt(data)")
plt.show()

# (2) 4제곱근 변환 (4th root): 좀 더 강한 좌편향 보정
data4 = np.power(data2, 0.25)
sm.qqplot(data4, line="s")
plt.title("Q-Q Plot after 4th Root Transform")
plt.show()

# 해석: 제곱근 및 4제곱근 변환을 통해 데이터 분포가 정규분포에 근접해졌음을 시각적으로 확인할 수 있음
