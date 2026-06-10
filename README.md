# DTW-based Boxer–Trainee Motion Similarity

## 소개

본 프로젝트는 멀티모달 데이터 중 120 fps 모션 캡처 데이터를 활용하여 Boxer의 기준 동작과 Trainee의 수행 동작 간 유사도를 정량적으로 분석합니다.

두 동작은 시작 시점, 종료 시점, 수행 속도 및 전체 프레임 수가 서로 다를 수 있습니다. 본 프로젝트는 Dynamic Time Warping(DTW)을 적용하여 서로 다른 시간축에 기록된 두 동작을 정렬한 뒤, 위치·회전·속도·가속도·관절 각도 정보를 비교합니다.

## 주요 기능

### DTW 기반 유사도 계산

FastDTW를 이용하여 Boxer와 Trainee의 동작 길이가 달라도 시간축상에서 유사한 움직임을 대응시킵니다.

DTW 거리는 다음과 같이 정규화됩니다.

```text
normalized_distance = DTW_distance / (length_boxer + length_trainee)
```

정규화된 거리는 다음 식을 통해 0과 1 사이의 유사도로 변환됩니다.

```text
similarity = exp(-k × normalized_distance)
```

기본값은 `k = 10`입니다.

* 1에 가까울수록 두 동작이 유사합니다.
* 0에 가까울수록 두 동작 간 차이가 큽니다.

현재 코드에는 sigmoid 함수가 아닌 지수 감쇠 함수가 적용되어 있습니다.

### 데이터 전처리

CSV 데이터를 불러온 뒤 다음 처리를 수행합니다.

* 비수치 데이터를 결측값으로 변환
* NaN 및 Inf 값을 0으로 치환
* 비정상 쿼터니언 교정
* 쿼터니언 정규화
* 프레임 간 쿼터니언 부호 연속성 유지
* Hip 관절 기준 위치 중심화
* Boxer와 Trainee의 시작 방향 정렬
* 어깨너비와 몸통 길이를 이용한 신체 크기 정규화

### 모션 특성 추출

각 관절에서 다음 특성을 추출합니다.

* Position: 관절의 3차원 위치
* Rotation: 관절의 쿼터니언 회전
* Velocity: 프레임 간 위치 변화량
* Acceleration: 프레임 간 속도 변화량
* Joint Angles: 주요 관절과 신체 부위의 각도

계산되는 주요 각도는 다음과 같습니다.

* 몸통 회전
* 좌우 어깨 각도
* 좌우 팔꿈치 각도
* 좌우 무릎 각도
* 좌우 발목 각도
* 목 굽힘 각도

### 핵심 동작 구간 검출

전체 관절 속도의 합을 이용하여 실제 동작이 수행된 구간을 자동으로 검출합니다.

```text
threshold = 최대 전체 속도 × 0.15
minimum segment length = 30 frames
```

검출된 활성 구간이 30프레임보다 짧으면 전체 프레임을 비교에 사용합니다.

### 특성별 유사도 계산

다음 특성의 유사도를 각각 계산합니다.

* Position similarity
* Rotation similarity
* Velocity similarity
* Acceleration similarity
* Joint-angle similarity

위치, 속도 및 가속도에는 Boxer와 Trainee 데이터를 함께 학습한 공통 스케일러가 적용됩니다.

```python
scaling="standard"
scaling="minmax"
scaling=None
```

### 신체 부위별 유사도 계산

다음 신체 부위별 유사도를 계산합니다.

* Left Arm
* Right Arm
* Left Leg
* Right Leg
* Core
* Head

각 부위의 관련 관절에 대해 위치, 회전, 속도, 가속도 및 관절 각도 DTW 유사도를 종합합니다.

### 전체 유사도 계산

전체 유사도는 각 특성별 점수에 가중치를 적용하여 계산합니다.

```python
default_feature_weights = {
    "position": 0.20,
    "rotation": 0.25,
    "velocity": 0.25,
    "acceleration": 0.10,
    "joint_angles": 0.20,
}
```

현재 실행 예시에서는 다음 가중치가 사용됩니다.

```python
custom_feature_weights = {
    "position": 0.0,
    "rotation": 0.0,
    "velocity": 1.0,
    "acceleration": 1.0,
    "joint_angles": 1.0,
}
```

가중치는 전체 합이 1이 되도록 자동 정규화됩니다. 따라서 현재 설정에서는 속도, 가속도 및 관절 각도가 동일한 비중으로 전체 점수에 반영됩니다.

### 배치 비교 및 결과 저장

하나의 Boxer 기준 동작을 여러 Trainee 동작과 비교할 수 있습니다.

* 폴더 내 모든 CSV 파일 비교
* 파일명 키워드 필터링
* 비교 파일 개수 제한
* `p02_Global`부터 `p26_Global`까지 자동 순회
* 그룹별 결과 CSV 저장
* 그룹 평균값 AVG 행 생성

저장되는 주요 항목은 다음과 같습니다.

```text
Head
Core
Right_Leg
Left_Leg
Right_Arm
Left_Arm
Acceleration
Velocity
Position
Joint Angle
Rotation
Overall
```

## 입력 데이터 형식

입력 파일은 CSV 형식이어야 하며 각 관절은 다음 컬럼 구조를 사용합니다.

```text
JointName.posX
JointName.posY
JointName.posZ
JointName.rotX
JointName.rotY
JointName.rotZ
JointName.rotW
```

예시:

```text
Hip.posX
Hip.posY
Hip.posZ
Hip.rotX
Hip.rotY
Hip.rotZ
Hip.rotW
Chest.posX
Chest.posY
Chest.posZ
...
```

`Hip` 관절의 position 데이터는 필수입니다.

## 설치 방법

Python 3.10 이상을 권장합니다.

```bash
pip install pandas numpy scipy fastdtw scikit-learn matplotlib pillow
```

프로젝트 내부에는 다음 시각화 모듈이 필요합니다.

```text
mocap.visualization
```

해당 모듈은 다음 함수를 제공해야 합니다.

```python
visualize_results
animate_3d_segments
export_joint_map_figure
```

## 사용 방법

### Boxer 기준 데이터와 Trainee 데이터 설정

```python
file1 = "/path/to/boxer_motion.csv"
file2 = "/path/to/trainee_group"
```

### 분석기 생성

```python
custom_feature_weights = {
    "position": 0.0,
    "rotation": 0.0,
    "velocity": 1.0,
    "acceleration": 1.0,
    "joint_angles": 1.0,
}

analyzer = MocapMotionAnalyzer(
    scaling="standard",
    feature_weights=custom_feature_weights,
    normalize_scale=True,
    scale_mode="combined",
)
```

### 두 동작 1:1 비교

```python
motion1 = analyzer.load_mocap_data(file1)
motion2 = analyzer.load_mocap_data(file2)

if motion1 is not None and motion2 is not None:
    similarity, details = analyzer.compare_motions(motion1, motion2)

    print(f"Overall similarity: {similarity:.4f}")
    print(details)
```

### 한 폴더의 여러 동작 비교

```python
result_df = save_similarity_matrix(
    file1_path=file1,
    file2_dir=file2,
    analyzer=analyzer,
    keyword="uppercut_left",
    limit=None,
    title="uppercut_left",
    output_csv_path="uppercut_left_similarity_matrix.csv",
)
```

### 여러 Trainee 그룹 비교

```python
results = save_similarity_across_groups(
    file1_path=file1,
    file2_path_or_base=file2,
    analyzer=analyzer,
    start=2,
    end=26,
    keyword="jap",
    limit=None,
    title="jap",
    output_dir=None,
)
```

### 결과 시각화

```python
analyzer.visualize_results(similarity, details)
```

### 3차원 동작 애니메이션 저장

```python
analyzer.animate_3d_segments(
    motion1,
    motion2,
    overlay=True,
    interval=40,
    save_path="output.gif",
)
```

## 분석 처리 과정

```text
1. Boxer 및 Trainee CSV 로드
2. 결측값과 비정상 데이터 보정
3. 관절 위치 및 회전 데이터 추출
4. Hip 기준 위치 중심화
5. 시작 방향 정렬
6. 신체 크기 정규화
7. 속도, 가속도 및 관절 각도 계산
8. 핵심 동작 구간 검출
9. Boxer 기준으로 Trainee의 크기와 방향 재정렬
10. 각 관절과 특성별 DTW 거리 계산
11. DTW 거리를 0~1 유사도로 변환
12. 특성별·신체 부위별·전체 유사도 계산
13. 결과 시각화 또는 CSV 저장
```

## 결과 해석 시 주의사항

계산된 유사도는 두 동작의 시계열 패턴이 얼마나 비슷한지를 나타냅니다. 복싱 기술의 절대적인 정확도나 수행 능력을 직접 판정하는 점수는 아닙니다.

점수는 관절 구성, 데이터 노이즈, 동작 구간 검출 기준, 특성 가중치, 스케일링 방식 및 DTW 계수에 영향을 받습니다. 서로 다른 실험 결과를 비교할 때는 동일한 전처리 조건과 가중치를 사용해야 합니다.
