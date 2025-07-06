# 예제 1: 유전자 형태 분포가 이론적 비율(1:2:1)과 일치하는지를 검정

import numpy as np
from scipy import stats

# 실제 관측값: A형 18명, B형 55명, C형 27명
O = np.array([18, 55, 27])  # Observed (관측 빈도)

# 이론적 비율 (A:B:C = 1:2:1 → 비율로는 0.25 : 0.5 : 0.25)
Pr = np.array([0.25, 0.5, 0.25])  # 이론적 비율

# 총 표본 수
n = O.sum()

# 이론적 기대값 = 전체 × 각 비율
E = n * Pr  # Expected (기대 빈도)

# 자유도 = 범주 수 - 1
df = len(O) - 1

# 카이제곱 적합도 검정
chi2, p = stats.chisquare(O, E)

# 결과 출력
print("Chi-squared test for given probabilities\n")
print("Chi-Squared :", round(chi2, 4))
print("df          :", df)
print("P-Value     :", round(p, 4))

# 해석:
# 귀무가설(H₀): 관측된 유전자 분포는 이론적 비율(1:2:1)과 같다.
# 대립가설(H₁): 분포는 이론적 비율과 다르다.
# P-값이 0.2698로 유의수준 0.05보다 크므로 → 귀무가설을 기각하지 않음.
# 따라서, 실험 결과는 이론과 잘 일치한다고 판단된다.



# 예제 2: 식이요법 A와 B에 따라 건강 상태 분포가 다른지를 검정

import numpy as np
import pandas as pd
from scipy import stats

# 행: 식이요법 그룹 (A/B)
# 열: 건강 상태 (Good/Normal/Bad)
diet = np.array([
    [37, 24, 19],  # A 그룹
    [17, 33, 20]   # B 그룹
])

# 판다스 DataFrame으로 보기 좋게 구성
column_names = ['Good', 'Normal', 'Bad']
row_names = ['diet_A', 'diet_B']
table = pd.DataFrame(diet, columns=column_names, index=row_names)
print(table)

# 독립성 검정을 위한 카이제곱 검정
chi2, p, dof, expected = stats.chi2_contingency(diet)

# 결과 출력
print("\nPearson's Chi-squared test\n")
print("Chi-Squared :", round(chi2, 4))
print("df          :", dof)
print("P-Value     :", round(p, 4))

# 해석:
# 귀무가설(H₀): 식이요법과 건강상태는 서로 독립이다 (즉, 관련 없음)
# 대립가설(H₁): 식이요법에 따라 건강상태가 달라진다 (즉, 관련 있음)
# P-값이 0.0164로 유의수준 0.05보다 작음 → 귀무가설 기각
# 즉, 식이요법 A와 B는 건강 상태에 유의한 차이를 만든다.