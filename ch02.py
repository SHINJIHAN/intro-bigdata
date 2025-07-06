# 1. 도수분포표 생성

# 예제1: 사망자 목록 중 130명을 임의로 추출, 이들의 사망원인을 10가지로 분류하였다.(p.35)
import numpy as np
import pandas as pd

# 변수 death에 NumPy 배열을 할당
death = np.array([2, 1, 2, 4, 2, 5, 3, 3, 5, 6, 3, 8, 3, 
                  3, 6, 3, 6, 5, 3, 5, 2, 6, 2, 3, 4, 3, 
                  2, 9, 2, 2, 3, 2, 7, 3, 2, 10, 6, 2, 3, 
                  1, 2, 3, 3, 4, 3, 2, 6, 2, 2, 3, 2, 3, 
                  4, 3, 2, 3, 5, 2, 5, 5, 3, 4, 3, 6, 2, 
                  1, 2, 3, 2, 6, 3, 3, 6, 3, 2, 3, 6, 4, 
                  6, 5, 3, 5, 6, 2, 6, 3, 2, 3, 2, 6, 2, 
                  6, 3, 3, 2, 6, 9, 6, 3, 6, 6, 2, 3, 2, 
                  3, 5, 3, 5, 2, 3, 2, 3, 3, 1, 3, 3, 2, 
                  3, 3, 4, 3, 6, 6, 3, 3, 3, 2, 3, 3, 6])


# death 배열의 각 숫자(1~10)는 사망원인에 해당하며,
# 이를 crosstab으로 빈도표로 정리

table = pd.crosstab(index=death, colnames=["질병"], columns="도수") 

# 질병 코드 → 질병명 매핑 (index에 적용)
# 1: 감염성, 2: 암, 3: 순환기, 4: 호흡기, 5: 소화기
# 6: 사고사, 7: 비뇨기, 8: 정신병, 9: 노환, 10: 신경계

table.index = [
    "감염" , "각종암", "순환기", "호흡기", "소화기", 
    "사고사", "비뇨기", "정신병", "노환", "신경계"
]
print(table)



# 2. 막대 그래프 시각화

import matplotlib.pyplot as plt
from matplotlib import font_manager

num_bars = len(table.index)        # 막대의 개수
base_color = [0, 1, 1]             # 기본 색상 (cyan 계열)
bar_width = 0.8                    # 막대 폭 (미사용, 0.5로 고정됨)

# 한글 폰트 설정 (NanumGothic)
font_path = r'C:\Users\jkl12\Downloads\NanumGothic.otf'
font_prop = font_manager.FontProperties(fname=font_path)

# 각 질병별 도수에 따라 막대 그리기 (색상은 점점 연하게)
for i, (index, row) in enumerate(table.iterrows()):
    color = np.array(base_color) * (num_bars - i) / num_bars  # 색상 밝기 조절
    plt.bar(index, row[0], color=color, width=0.5)            # 막대 그리기

# 축 레이블 설정
plt.xlabel('질병', fontproperties=font_prop, fontsize=14)
plt.ylabel('빈도수', fontproperties=font_prop, fontsize=14)

# 그래프 제목 설정
plt.title('질병에 따른 막대그래프', fontproperties=font_prop, fontsize=20)

# x축 눈금 라벨 회전 없이 표시
plt.xticks(rotation=0, fontproperties=font_prop)

plt.show()




# 3. 원형 그래프 시각화

# 한글 폰트 및 기본 폰트 크기 설정
plt.rcParams['font.family'] = 'NanumGothic'
plt.rcParams['font.size'] = 12

# 원형 그래프 크기 지정 (가로x세로)
plt.figure(figsize=(10, 10))

# 블루 계열 색상 리스트
colors = [
    'dodgerblue', 'royalblue', 'deepskyblue', 
    'lightskyblue', 'cornflowerblue', 'skyblue', 'deepskyblue'
]

# 사망 원인별 비율을 나타내는 원형 그래프 생성
patches, texts, autotexts = plt.pie(
    table['도수'], labels=table.index, autopct='%1.1f%%', 
    startangle=0, colors=colors
)

# 그래프 제목 설정
plt.title("사망원인에 대한 원형 그래프", fontsize=20)

plt.show()



# 4. 파레토 차트 시각화

import matplotlib.patches as mpatches

# 사망 원인별 도수 내림차순 정렬
sorted_table = table.sort_values(by='도수', ascending=False)

# 누적 상대도수 계산 (%)
cumulative_percentage = sorted_table['도수'].cumsum() / sorted_table['도수'].sum() * 100

# 두 축을 가진 그래프 생성 (크기 지정)
fig, ax = plt.subplots(figsize=(10, 8))

# 빈도수 막대그래프 표시
sorted_table['도수'].plot(
    kind='bar', color='royalblue', ax=ax,
    width=0.8, position=0.5
)

# x축, y축 레이블 설정
ax.set_xlabel('질병 종류', fontproperties=font_prop, fontsize=14)
ax.set_ylabel('빈도수', fontproperties=font_prop, fontsize=14)

# 누적 상대도수를 선그래프로 표시 (보조 y축)
ax2 = ax.twinx()
cumulative_percentage.plot(
    color='black', ax=ax2,
    style='-o', use_index=False
)
ax2.set_ylabel('누적 상대도수 (%)', fontproperties=font_prop, fontsize=14)

# y축 값에 % 포맷 적용
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}%'))

# 범례 표시
p_legend1 = mpatches.Patch(color='black', label='누적 상대도수')
p_legend2 = mpatches.Patch(color='royalblue', label='빈도수')
plt.legend(handles=[p_legend1, p_legend2], loc='center right', prop=font_prop)

# x축 눈금 라벨 각도 및 폰트 설정
ax.set_xticklabels(sorted_table.index, rotation=0, fontproperties=font_prop)

# 그래프 제목 설정
plt.title("사망 원인", fontproperties=font_prop, fontsize=20)

plt.show()



# 5. 도수다각형
# 예제2: 정량 100인 음료수 80병을 임의 추출하여 내용물의 양을 측정한 자료 (p.42)

# 데이터 배열 선언
drink = np.array([101.8, 101.5, 101.8, 102.6, 101, 96.8, 102.4, 100, 98.8, 98.1, 
                  98.8, 98, 99.4, 95.5, 100.1, 100.5, 97.4, 100.2, 101.4, 98.7, 
                  101.4, 99.4, 101.7, 99, 99.7, 98.9, 99.5, 100, 99.7, 100.9, 
                  99.7, 99, 98.8, 99.7, 100.9, 99.9, 97.5, 101.5, 98.2, 99.2, 
                  98.6, 101.4, 102.1, 102.9, 100.8, 99.4, 103.7, 100.3, 100.2, 
                  101.1, 101.8, 100, 101.2, 100.5, 101.2, 101.6, 99.9, 100.5, 
                  100.4, 98.1, 100.1, 101.6, 99.3, 96.1, 100, 99.7, 99.7, 99.4, 
                  101.5, 100.9, 101.3, 99.9, 99.1, 100.7, 100.8, 100.8, 101.4, 
                  100.3, 98.4, 97.2])

# 히스토그램 생성: 빈(bin) 수 10, 막대 색상과 투명도 지정
plt.figure()
n, bins, patches = plt.hist(drink, bins=10, facecolor="blue", alpha=0.3)

# 히스토그램 빈 중심 계산: 인접한 bin 경계의 평균값으로 구함
x = [(bins[i] + bins[i + 1]) / 2 for i in range(len(bins) - 1)]
w_bin = bins[1] - bins[0]  # bin 너비
x.insert(0, x[0] - w_bin)  # 시작점에 빈 중심 추가 (히스토그램 범위 확장)
x.append(x[-1] + w_bin)    # 끝점에 빈 중심 추가

# 빈도수 배열 n의 앞뒤에 0 추가하여 도수다각형 닫기
n = np.insert(n, 0, 0.0)
n = np.append(n, 0.0)

# 도수다각형 데이터 포인트 플로팅 (파란색 실선, 원형 마커)
plt.plot(x, n, 'b-', marker='o')

# 축 라벨 및 그래프 제목 설정
plt.xlabel('음료', fontsize=14, fontproperties=font_prop)
plt.ylabel('빈도수', fontsize=14, fontproperties=font_prop)
plt.title('음료의 히스토그램 및 도수다각형', fontsize=16, fontproperties=font_prop)

plt.show()