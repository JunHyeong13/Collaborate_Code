# 경로 상에 파일이 있는지 확인하는 코드
import os

# 각 조건 범위 설정
numbers = [f"{i:02d}" for i in range(1, 27)]  # 01~26
types = ["jap", "straight", "hook_left", "hook_right", "uppercut_left", "uppercut_right"]
indices = [f"{i:03d}" for i in range(1, 6)]   # 001~005

# CSV가 있는 경로 지정
base_dir = "C:\\Users\\user\\Downloads\\TEPA\\Collaborate_Code"  # 실제 폴더 경로로 수정하세요.

# 모든 조합 탐색
for num in numbers:
    for t in types:
        for idx in indices:
            file_name = f"p{num}_{t}_{idx}_similarity_matrix.csv"
            file_path = os.path.join(base_dir, file_name)
            if not os.path.exists(file_path):
                print(f"{file_name} 파일이 없습니다")
