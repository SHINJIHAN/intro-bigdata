# 예제 4: 이산 균등분포에서 표본평균의 분포 확인 (p.268)

# 0부터 9까지의 정수(총 10개)를 같은 확률로 갖는 모집단에서,
# 크기 5인 표본을 100번 추출하여 표본평균을 구한다.
# 이 표본평균들의 분포를 히스토그램과 Q-Q 플롯으로 시각화한다.


import numpy as np

# 무작위 정수 5개를 0~99 범위에서 추출 (시드 설정 없음)
a = np.random.randint(0, 100, size=5)
b = np.random.randint(0, 100, size=5)

# 시드(seed)를 1로 고정하면 이후 추출되는 난수가 항상 동일함
np.random.seed(1)
c = np.random.randint(0, 100, size=5)

np.random.seed(1)
d = np.random.randint(0, 100, size=5)

# 확인용 출력
print("a :", a)
print("b :", b)
print("c :", c)
print("d :", d)

# 해석:
# c와 d는 시드가 같으므로 동일한 난수가 생성됨 → 재현 가능한 결과 확보 가능


m = []  # 표본평균을 저장할 리스트

np.random.seed(1234)  # 결과 재현을 위해 시드 고정

# 표본 크기 5, 반복 100회
for i in range(100):
    sample = np.random.randint(0, 10, size=5)  # 0~9 사이 정수 5개 추출
    m.append(np.mean(sample))  # 표본평균 계산하여 저장

m = np.array(m)  # 리스트를 넘파이 배열로 변환
print(m)  # 표본평균 100개 출력


import matplotlib.pyplot as plt

plt.hist(m, bins=7, edgecolor="black")  # 히스토그램 작성 (구간 수 7)
plt.xlabel('Sample Mean')  # x축 레이블
plt.ylabel('Frequency')    # y축 레이블
plt.title('Histogram of Sample Means')  # 제목
plt.show()

# 해석:
# 이산 균등분포(0~9)에서 추출한 표본평균들의 분포가 
# 종 모양(Bell-shaped)으로 나타남 → 중심극한정리에 따라
# 표본평균이 정규분포를 따를 수 있음을 시사


import statsmodels.api as sm

sm.qqplot(m, line='s')  # Q-Q 플롯 작성 ("s"는 표준 정규분포 기준선 의미)
plt.title("Normal Q-Q Plot of Sample Means")
plt.show()

# 해석:
# 대부분의 점이 직선 근처에 위치 → 데이터가 정규분포에 가까운 형태
# 중심극한정리의 시각적 검증 도구로 활용 가능
