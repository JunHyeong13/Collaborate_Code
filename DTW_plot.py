import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# 데이터 파일이 있는 디렉토리 경로
directory_path = 'C:\\Users\\user\\Downloads\\TEPA\\Collaborate_Code\\DTWS'
output_dir = os.path.join(directory_path, 'DTW_plots')

# 출력 폴더가 없으면 생성
os.makedirs(output_dir, exist_ok=True)

# 그래프에 사용할 열 목록
columns_to_plot = ['Head', 'Core', 'Right_Leg', 'Left_Leg', 'Right_Arm', 'Left_Arm',
                   'Acceleration', 'Velocity', 'Position', 'Joint Angle', 'rotation', 'Overall']

# 1️⃣ 각 컬럼별 y축 범위를 계산하는 함수
def get_y_ranges(directory_path, columns):
    y_ranges = {}
    for column in columns:
        y_min, y_max = float('inf'), float('-inf')
        for file_name in os.listdir(directory_path):
            if not file_name.lower().endswith('.csv'):
                continue
            file_path = os.path.join(directory_path, file_name)
            try:
                df = pd.read_csv(file_path, header=0, index_col=0)
            except pd.errors.ParserError:
                df = pd.read_csv(file_path, header=0)
                df.index = df.iloc[:, 0].astype(str)
                df = df.iloc[:, 1:]
            df.columns = df.columns.str.strip()
            if column in df.columns:
                y_min = min(y_min, df[column].min())
                y_max = max(y_max, df[column].max())
        # 안전하게 0/0 체크
        if y_min == float('inf') or y_max == float('-inf'):
            y_min, y_max = 0, 1
        y_ranges[column] = (y_min, y_max)
    return y_ranges

# 모든 컬럼별 y축 범위 계산
y_ranges = get_y_ranges(directory_path, columns_to_plot)

# 2️⃣ 파일별 처리
def process_file(file_path):
    try:
        df = pd.read_csv(file_path, header=0, index_col=0)
    except pd.errors.ParserError:
        df = pd.read_csv(file_path, header=0)
        df.index = df.iloc[:, 0].astype(str)
        df = df.iloc[:, 1:]
    df.columns = df.columns.str.strip()

    # 인덱스를 문자열로 변환
    if isinstance(df.index, pd.MultiIndex):
        df.index = df.index.to_flat_index()
    df.index = df.index.map(lambda x: '' if pd.isna(x) else str(x))

    # pre/main/post 조건별 데이터 필터링
    pre_data = df[df.index.str.contains('pre', case=False, na=False)]
    main_data = df[df.index.str.contains('main', case=False, na=False)]
    post_data = df[df.index.str.contains('post', case=False, na=False)]

    current_time = datetime.now().strftime("%Y_%m_%d")
    colors = ['gray', 'blue', 'purple']  # Pre, Main, Post 색상

    for column in columns_to_plot:
        if column not in df.columns:
            print(f"Column '{column}' not found in the data. 건너뜀")
            continue

        pre_avg = pre_data[column].mean() if (column in pre_data.columns and not pre_data.empty) else 0
        main_avg = main_data[column].mean() if (column in main_data.columns and not main_data.empty) else 0
        post_avg = post_data[column].mean() if (column in post_data.columns and not post_data.empty) else 0

        print(f"{column} - Pre: {pre_avg:.2f}, Main: {main_avg:.2f}, Post: {post_avg:.2f}")

        means = [pre_avg, main_avg, post_avg]
        labels = ['Pre', 'Main', 'Post']

        # 그래프 그리기
        plt.figure(figsize=(6, 4))
        plt.bar(labels, means, color=colors, width=0.5)
        plt.title(f'{column} Average per Condition')
        plt.ylabel('Average Value')

        # 3️⃣ 컬럼별 y축 범위 적용
        y_min, y_max = y_ranges[column]
        plt.ylim(y_min, y_max)

        # 저장 경로
        file_name = os.path.basename(file_path).split('.')[0]
        save_path = os.path.join(output_dir, f'{file_name}_{current_time}_{column}_Averages.png')

        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        print(f"그래프가 저장되었습니다: {save_path}")

# 폴더 내 모든 CSV 처리
for file_name in os.listdir(directory_path):
    if file_name.lower().endswith('.csv'):
        file_path = os.path.join(directory_path, file_name)
        process_file(file_path)
        print(f"처리 완료: {file_name}")