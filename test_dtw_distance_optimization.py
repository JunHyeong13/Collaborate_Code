#!/usr/bin/env python3
"""
DTW 거리 값 반환 및 성능 최적화 테스트 스크립트

이 스크립트는 다음을 테스트합니다:
1. 지수 함수 변환 제거 후 DTW 거리 값 직접 반환
2. 최적화된 DTW 성능 비교
3. Window size 제한 효과 확인
"""

import time
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# 현재 디렉토리를 Python 경로에 추가
sys.path.append(str(Path(__file__).parent))

from main_HCI_JJH import MocapMotionAnalyzer

def create_test_data(length=100):
    """테스트용 모션 데이터 생성"""
    np.random.seed(42)
    
    # 기본 조인트 데이터 생성
    joints = ['LShoulder', 'LUArm', 'LFArm', 'LHand', 'RShoulder', 'RUArm', 'RFArm', 'RHand']
    data = {'Frame': range(1, length + 1)}
    
    for joint in joints:
        # 위치 데이터 (x, y, z)
        for coord in ['x', 'y', 'z']:
            data[f'{joint}_{coord}'] = np.random.randn(length) * 10 + np.sin(np.linspace(0, 4*np.pi, length))
        
        # 쿼터니언 데이터 (w, x, y, z)
        quat_data = np.random.randn(length, 4)
        quat_data = quat_data / np.linalg.norm(quat_data, axis=1, keepdims=True)
        for i, comp in enumerate(['w', 'x', 'y', 'z']):
            data[f'{joint}_q{comp}'] = quat_data[:, i]
    
    return pd.DataFrame(data)

def test_dtw_distance_values():
    """DTW 거리 값 직접 반환 테스트"""
    print("=" * 60)
    print("1. DTW 거리 값 직접 반환 테스트")
    print("=" * 60)
    
    # 테스트 데이터 생성
    motion1 = create_test_data(50)
    motion2 = create_test_data(50)
    
    # 기본 analyzer (지수 변환 없음)
    analyzer = MocapMotionAnalyzer(scaling='standard', use_optimized_dtw=False)
    
    # 비교 실행
    similarity, details = analyzer.compare_motions(motion1, motion2)
    
    print(f"전체 유사도 (거리 값): {similarity:.6f}")
    print("\n개별 조인트 거리 값:")
    for joint in ['LShoulder', 'LUArm', 'LFArm', 'LHand']:
        if f'joint_{joint}' in details:
            print(f"  {joint}: {details[f'joint_{joint}']:.6f}")
    
    print("\n피처별 거리 값:")
    for feature in ['position', 'rotation', 'velocity', 'acceleration']:
        if feature in details:
            print(f"  {feature}: {details[feature]:.6f}")
    
    # 거리 값이 0~1 범위를 벗어날 수 있음을 확인
    print(f"\n거리 값 범위 확인:")
    print(f"  전체 유사도: {similarity}")
    print(f"  조인트 거리 값들: {[v for k, v in details.items() if k.startswith('joint_')][:5]}")
    print(f"  피처 거리 값들: {[v for k, v in details.items() if k in ['position', 'rotation', 'velocity', 'acceleration']]}")

def test_dtw_performance_optimization():
    """DTW 성능 최적화 테스트"""
    print("\n" + "=" * 60)
    print("2. DTW 성능 최적화 테스트")
    print("=" * 60)
    
    # 더 큰 테스트 데이터 생성
    motion1 = create_test_data(200)
    motion2 = create_test_data(200)
    
    # 기본 DTW
    analyzer_basic = MocapMotionAnalyzer(scaling='standard', use_optimized_dtw=False)
    
    # 최적화된 DTW (window size 제한)
    analyzer_optimized = MocapMotionAnalyzer(
        scaling='standard', 
        use_optimized_dtw=True, 
        dtw_window_size=20  # window size 제한
    )
    
    # 성능 측정
    print("기본 DTW 성능 측정...")
    start_time = time.time()
    similarity_basic, details_basic = analyzer_basic.compare_motions(motion1, motion2)
    basic_time = time.time() - start_time
    
    print("최적화된 DTW 성능 측정...")
    start_time = time.time()
    similarity_optimized, details_optimized = analyzer_optimized.compare_motions(motion1, motion2)
    optimized_time = time.time() - start_time
    
    # 결과 비교
    print(f"\n성능 비교 결과:")
    print(f"  기본 DTW 시간: {basic_time:.3f}초")
    print(f"  최적화된 DTW 시간: {optimized_time:.3f}초")
    print(f"  성능 개선: {((basic_time - optimized_time) / basic_time * 100):.1f}%")
    
    print(f"\n결과 값 비교:")
    print(f"  기본 DTW 거리: {similarity_basic:.6f}")
    print(f"  최적화된 DTW 거리: {similarity_optimized:.6f}")
    print(f"  차이: {abs(similarity_basic - similarity_optimized):.6f}")

def test_window_size_effect():
    """Window size 제한 효과 테스트"""
    print("\n" + "=" * 60)
    print("3. Window Size 제한 효과 테스트")
    print("=" * 60)
    
    motion1 = create_test_data(150)
    motion2 = create_test_data(150)
    
    window_sizes = [None, 10, 20, 50]
    
    for window_size in window_sizes:
        analyzer = MocapMotionAnalyzer(
            scaling='standard',
            use_optimized_dtw=True,
            dtw_window_size=window_size
        )
        
        start_time = time.time()
        similarity, _ = analyzer.compare_motions(motion1, motion2)
        elapsed_time = time.time() - start_time
        
        window_desc = "제한 없음" if window_size is None else f"{window_size}"
        print(f"Window Size {window_desc:>8}: {elapsed_time:.3f}초, 거리: {similarity:.6f}")

def test_dtaidistance_availability():
    """dtaidistance 라이브러리 가용성 테스트"""
    print("\n" + "=" * 60)
    print("4. dtaidistance 라이브러리 가용성 테스트")
    print("=" * 60)
    
    try:
        from dtaidistance import dtw
        print("✓ dtaidistance 라이브러리가 설치되어 있습니다.")
        print("  최적화된 DTW (pruning) 기능을 사용할 수 있습니다.")
        
        # 간단한 dtaidistance 테스트
        seq1 = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5]])
        seq2 = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5]])
        
        distance = dtw.distance(seq1, seq2, use_pruning=True)
        print(f"  dtaidistance 테스트 거리: {distance:.6f}")
        
    except ImportError:
        print("✗ dtaidistance 라이브러리가 설치되지 않았습니다.")
        print("  pip install dtaidistance 명령으로 설치하면 더 빠른 DTW 계산이 가능합니다.")
        print("  현재는 fastdtw를 사용한 기본 최적화만 적용됩니다.")

def main():
    """메인 테스트 실행"""
    print("DTW 거리 값 반환 및 성능 최적화 테스트 시작")
    print("=" * 60)
    
    try:
        # 1. DTW 거리 값 직접 반환 테스트
        test_dtw_distance_values()
        
        # 2. DTW 성능 최적화 테스트
        test_dtw_performance_optimization()
        
        # 3. Window size 제한 효과 테스트
        test_window_size_effect()
        
        # 4. dtaidistance 라이브러리 가용성 테스트
        test_dtaidistance_availability()
        
        print("\n" + "=" * 60)
        print("모든 테스트 완료!")
        print("=" * 60)
        
        print("\n주요 변경사항 요약:")
        print("1. ✓ 지수 함수 변환 제거 - DTW 거리 값 직접 반환")
        print("2. ✓ 최적화된 DTW 메서드 추가 (_dtw_similarity_optimized)")
        print("3. ✓ Window size 제한으로 계산 시간 단축")
        print("4. ✓ dtaidistance 라이브러리 지원 (선택적)")
        
        print("\n사용법:")
        print("# 기본 DTW (거리 값 반환)")
        print("analyzer = MocapMotionAnalyzer(use_optimized_dtw=False)")
        print("")
        print("# 최적화된 DTW (성능 개선)")
        print("analyzer = MocapMotionAnalyzer(use_optimized_dtw=True, dtw_window_size=20)")
        
    except Exception as e:
        print(f"테스트 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
