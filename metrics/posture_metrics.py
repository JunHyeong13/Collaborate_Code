import argparse
from pathlib import Path
from typing import Tuple, Literal, Callable, Dict

import numpy as np
import pandas as pd

Plane = Literal['xy', 'xz', 'yz', 'xyz']

# -----------------------------
# Utilities
# -----------------------------

def _pair_distance(a: np.ndarray, b: np.ndarray, plane: Plane = 'xz') -> np.ndarray:
    """주어진 평면에서 두 3D 점 배열 (T,3) 간의 프레임별 거리를 계산합니다."""
    if plane == 'xyz':
        diff = a - b
    elif plane == 'xy':
        diff = a[:, [0, 1]] - b[:, [0, 1]]
    elif plane == 'yz':
        diff = a[:, [1, 2]] - b[:, [1, 2]]
    else:  # 'xz'
        diff = a[:, [0, 2]] - b[:, [0, 2]]
    return np.linalg.norm(diff, axis=1)


def _dtw_distance_1d(a: np.ndarray, b: np.ndarray) -> float:
    """1D 시퀀스의 DTW 거리를 계산합니다. 평균 경로 비용을 반환합니다."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    Na, Nb = len(a), len(b)
    
    D = np.full((Na + 1, Nb + 1), np.inf, dtype=np.float64)
    D[0, 0] = 0.0
    
    for i in range(1, Na + 1):
        for j in range(1, Nb + 1):
            c = abs(a[i - 1] - b[j - 1])
            D[i, j] = c + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    
    i, j = Na, Nb
    L = 0
    while i > 0 or j > 0:
        L += 1
        choices = [
            (D[i - 1, j] if i > 0 else np.inf, i - 1, j),
            (D[i, j - 1] if j > 0 else np.inf, i, j - 1),
            (D[i - 1, j - 1] if (i > 0 and j > 0) else np.inf, i - 1, j - 1),
        ]
        _, i, j = min(choices, key=lambda t: t[0])
    
    return float(D[Na, Nb] / max(L, 1))

# -----------------------------
# Core metrics
# -----------------------------

def _get_validated_data(df: pd.DataFrame, required_columns: list) -> pd.DataFrame:
    """필요한 컬럼이 있는지 확인하고 수치형 데이터로 변환합니다."""
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"필요한 컬럼 없음: {missing[:6]}{'...' if len(missing)>6 else ''}")
    return df[required_columns].apply(pd.to_numeric, errors='coerce').fillna(0.0)


def foot_to_shoulder_ratio(df: pd.DataFrame, plane: Plane = 'xz') -> np.ndarray:
    """비율 계산: 발간거리 / 어깨넓이 (기존 함수)"""
    required = [
        'LFoot.posX', 'LFoot.posY', 'LFoot.posZ', 'RFoot.posX', 'RFoot.posY', 'RFoot.posZ',
        'LShoulder.posX', 'LShoulder.posY', 'LShoulder.posZ', 'RShoulder.posX', 'RShoulder.posY', 'RShoulder.posZ',
    ]
    df_local = _get_validated_data(df, required)
    
    LFoot = df_local[['LFoot.posX', 'LFoot.posY', 'LFoot.posZ']].to_numpy()
    RFoot = df_local[['RFoot.posX', 'RFoot.posY', 'RFoot.posZ']].to_numpy()
    LShoulder = df_local[['LShoulder.posX', 'LShoulder.posY', 'LShoulder.posZ']].to_numpy()
    RShoulder = df_local[['RShoulder.posX', 'RShoulder.posY', 'RShoulder.posZ']].to_numpy()

    d_feet = _pair_distance(LFoot, RFoot, plane=plane)
    d_shoulders = _pair_distance(LShoulder, RShoulder, plane=plane)

    ratio = np.divide(d_feet, d_shoulders, out=np.zeros_like(d_feet), where=d_shoulders > 1e-8)
    return ratio


def stance_depth_width_ratio(df: pd.DataFrame, plane: Plane = 'xz') -> np.ndarray:
    """비율 계산: 스탠스 깊이 / 스탠스 너비"""
    required = ['LFoot.posX', 'LFoot.posZ', 'RFoot.posX', 'RFoot.posZ']
    df_local = _get_validated_data(df, required)

    d_width = np.abs(df_local['LFoot.posX'] - df_local['RFoot.posX'])
    d_depth = np.abs(df_local['LFoot.posZ'] - df_local['RFoot.posZ'])
    
    ratio = np.divide(d_depth, d_width, out=np.zeros_like(d_depth), where=d_width > 1e-8)
    return ratio


def _guard_ratio(df: pd.DataFrame, side: Literal['L', 'R']) -> np.ndarray:
    """내부 함수: 가드 높이 비율 계산 (좌/우 공통 로직)"""
    required = [
        f'{side}Hand.posY', f'{side}Shoulder.posY', 'Head.posY'
    ]
    df_local = _get_validated_data(df, required)
    
    dist_head_hand = np.abs(df_local['Head.posY'] - df_local[f'{side}Hand.posY'])
    dist_head_shoulder = np.abs(df_local['Head.posY'] - df_local[f'{side}Shoulder.posY'])

    ratio = np.divide(dist_head_hand, dist_head_shoulder, out=np.zeros_like(dist_head_hand), where=dist_head_shoulder > 1e-8)
    return ratio

def left_guard_ratio(df: pd.DataFrame, plane: Plane = 'xyz') -> np.ndarray:
    """비율 계산: (머리-왼손 수직거리) / (머리-왼어깨 수직거리)"""
    return _guard_ratio(df, 'L')

def right_guard_ratio(df: pd.DataFrame, plane: Plane = 'xyz') -> np.ndarray:
    """비율 계산: (머리-오른손 수직거리) / (머리-오른어깨 수직거리)"""
    return _guard_ratio(df, 'R')

def _hand_reach_ratio(df: pd.DataFrame, side: Literal['L', 'R']) -> np.ndarray:
    """내부 함수: 손 리치 비율 계산 (좌/우 공통 로직) - 수정된 버전"""
    required = [
        f'{side}Hand.posX', f'{side}Hand.posY', f'{side}Hand.posZ',
        'LShoulder.posX', 'LShoulder.posY', 'LShoulder.posZ',
        'RShoulder.posX', 'RShoulder.posY', 'RShoulder.posZ',
    ]
    df_local = _get_validated_data(df, required)

    Hand = df_local[[f'{side}Hand.posX', f'{side}Hand.posY', f'{side}Hand.posZ']].to_numpy()
    LShoulder = df_local[['LShoulder.posX', 'LShoulder.posY', 'LShoulder.posZ']].to_numpy()
    RShoulder = df_local[['RShoulder.posX', 'RShoulder.posY', 'RShoulder.posZ']].to_numpy()

    if side == 'L':
        Shoulder_for_reach = LShoulder
    else:  # side == 'R'
        Shoulder_for_reach = RShoulder

    dist_shoulder_hand = _pair_distance(Shoulder_for_reach, Hand, plane='xyz')
    dist_shoulders = _pair_distance(LShoulder, RShoulder, plane='xyz')
    
    ratio = np.divide(dist_shoulder_hand, dist_shoulders, out=np.zeros_like(dist_shoulder_hand), where=dist_shoulders > 1e-8)
    return ratio

def left_hand_reach_ratio(df: pd.DataFrame, plane: Plane = 'xyz') -> np.ndarray:
    """비율 계산: (어깨-왼손 거리) / (어깨너비)"""
    return _hand_reach_ratio(df, 'L')

def right_hand_reach_ratio(df: pd.DataFrame, plane: Plane = 'xyz') -> np.ndarray:
    """비율 계산: (어깨-오른손 거리) / (어깨너비)"""
    return _hand_reach_ratio(df, 'R')

def summarize_ratio(ratio: np.ndarray) -> Tuple[float, float, float]:
    """비율 시퀀스의 통계 요약을 반환합니다."""
    return float(np.mean(ratio)), float(np.median(ratio)), float(np.std(ratio))

# -----------------------------
# Main Execution Logic
# -----------------------------

def analyze_and_compare(
    metric_name: str, 
    metric_func: Callable, 
    df_a: pd.DataFrame, 
    df_b: pd.DataFrame | None, 
    plane: Plane
) -> float | None:
    """주어진 단일 지표에 대해 분석, 요약 및 DTW 비교를 수행하고 유사도 점수를 반환합니다."""
    print(f"\n\n{'='*50}\n분석 지표: {metric_name}\n{'='*50}")
    
    try:
        r_a = metric_func(df_a, plane=plane)
        mean_a, median_a, std_a = summarize_ratio(r_a)
        
        print('A 파일 요약:')
        print(f' - mean:   {mean_a:.6f}')
        print(f' - median: {median_a:.6f}')
        print(f' - std:    {std_a:.6f}')

        if df_b is not None:
            r_b = metric_func(df_b, plane=plane)
            mean_b, median_b, std_b = summarize_ratio(r_b)
            
            dist = _dtw_distance_1d(r_a, r_b)
            sim = 1.0 / (1.0 + dist)
            
            print('\nB 파일 및 비교 결과:')
            print(f' - B mean/median/std: {mean_b:.6f} / {median_b:.6f} / {std_b:.6f}')
            print(f' - DTW 거리 (A vs B): {dist:.6f}')
            print(f' - 유사도 (Similarity): {sim:.4f} ({sim*100:.2f}%)')
            return sim
        else:
            print('\n비교 파일을 제공하지 않아 단일 분석만 수행합니다.')

    except ValueError as e:
        print(f"'{metric_name}' 지표 계산 중 오류 발생. 건너뜁니다.")
        print(f"  (오류 원인: {e})")
    except Exception as e:
        print(f"예상치 못한 오류 발생: {e}")
    
    return None

def main():
    parser = argparse.ArgumentParser(description='복싱 자세 및 동작 분석기 (다중 지표)')
    parser.add_argument('--csv', type=str, default="./motion1.csv", help='분석할 CSV 파일 경로')
    parser.add_argument('--compare', type=str, default="./motion2.csv", help='비교할 두 번째 CSV 파일 경로')
    parser.add_argument('--plane', type=str, default='xz', choices=['xy', 'xz', 'yz', 'xyz'], help='거리 계산에 사용할 평면')
    parser.add_argument('--no-compare', action='store_true', help='비교 분석을 건너뜁니다.')
    args = parser.parse_args()

    csv_a = Path(args.csv)
    compare_path = None if args.no_compare else Path(args.compare)
    
    if not csv_a.exists():
        raise FileNotFoundError(f"분석 파일을 찾을 수 없습니다: {csv_a}")
    if compare_path and not compare_path.exists():
        raise FileNotFoundError(f"비교 파일을 찾을 수 없습니다: {compare_path}")

    print(f"분석 시작: [A] {csv_a.name} | [B] {compare_path.name if compare_path else '없음'} | 평면: {args.plane}")
    
    df_a = pd.read_csv(csv_a)
    df_b = pd.read_csv(compare_path) if compare_path else None

    metrics_to_run: Dict[str, Callable] = {
        "발 너비 / 어깨너비 비율": foot_to_shoulder_ratio,
        "스탠스 깊이 / 너비 비율": stance_depth_width_ratio,
        "왼손 가드 높이 비율": left_guard_ratio,
        "오른손 가드 높이 비율": right_guard_ratio,
        "뻗기 (L) 왼손 리치 / 어깨너비 비율": left_hand_reach_ratio,
        "뻗기 (R) 오른손 리치 / 어깨너비 비율": right_hand_reach_ratio,
    }

    all_scores = []
    
    for name, func in metrics_to_run.items():
        sim_score = analyze_and_compare(name, func, df_a, df_b, plane=args.plane)
        
        if sim_score is not None:
            all_scores.append(sim_score)

    if df_b is not None and all_scores:
        final_score = np.mean(all_scores)
        print(f"\n\n{'='*50}\n최종 종합 유사도 (A 대비 B)\n{'='*50}")
        print(f" - 분석된 지표 수: {len(all_scores)}개")
        print(f" - 종합 유사도 점수: {final_score:.4f} ({final_score*100:.2f}%)")

    print(f"\n\n{'='*50}\n모든 분석이 완료되었습니다.\n{'='*50}")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"\n치명적 오류가 발생했습니다: {e}")
