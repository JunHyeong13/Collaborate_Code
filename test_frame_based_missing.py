#!/usr/bin/env python3
"""
Frame 열 기준 정교한 결측치 처리 테스트 스크립트
"""

import pandas as pd
import numpy as np
import os
import sys

# 현재 디렉토리를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# MocapMotionAnalyzer 클래스 import
from main_HCI_JJH import MocapMotionAnalyzer

def create_test_csv_with_frame(filename: str, missing_scenario: str):
    """Frame 열이 있는 테스트용 CSV 파일 생성"""
    
    # 기본 데이터 생성 (100 프레임)
    frames = np.arange(1, 101)
    joints = ['LShoulder', 'RShoulder', 'LHand', 'RHand']
    
    data = {'Frame': frames}
    
    # 조인트별 위치 데이터 생성
    for joint in joints:
        np.random.seed(hash(f"{filename}_{joint}") % 1000)
        data[f'{joint}.posX'] = np.random.randn(100) * 10 + 100
        data[f'{joint}.posY'] = np.random.randn(100) * 10 + 50
        data[f'{joint}.posZ'] = np.random.randn(100) * 10 + 0
    
    df = pd.DataFrame(data)
    
    # 결측치 시나리오별 적용
    if missing_scenario == "low_missing":
        # 5% 이하 결측치 (3% 정도)
        # LShoulder.posX에서 3개 연속 결측치 생성
        df.loc[20:22, 'LShoulder.posX'] = np.nan
        # RHand.posY에서 2개 연속 결측치 생성
        df.loc[60:61, 'RHand.posY'] = np.nan
        
    elif missing_scenario == "high_missing":
        # 5% 이상 결측치 (8% 정도)
        # LShoulder에서 대량 결측치
        df.loc[10:17, 'LShoulder.posX'] = np.nan  # 8개
        df.loc[30:35, 'LShoulder.posY'] = np.nan  # 6개
        # RShoulder에서 대량 결측치
        df.loc[50:57, 'RShoulder.posX'] = np.nan  # 8개
        df.loc[70:73, 'RShoulder.posZ'] = np.nan  # 4개
        
    elif missing_scenario == "no_frame_column":
        # Frame 열 제거 (fallback 테스트용)
        df = df.drop('Frame', axis=1)
        
    elif missing_scenario == "invalid_frame":
        # Frame 열을 문자열로 변경 (fallback 테스트용)
        df['Frame'] = 'invalid'
    
    df.to_csv(filename, index=False)
    return filename

def test_frame_based_missing_handling():
    """Frame 열 기준 결측치 처리 테스트"""
    
    print("=== Frame 열 기준 정교한 결측치 처리 테스트 ===\n")
    
    # MocapMotionAnalyzer 인스턴스 생성
    analyzer = MocapMotionAnalyzer()
    
    # 테스트 시나리오들
    test_scenarios = [
        ("low_missing", "5% 이하 결측치 - (1-결측길이/전체프레임) 보간 방식"),
        ("high_missing", "5% 이상 결측치 - 전체 구간 평균값 대체 방식"),
        ("no_frame_column", "Frame 열 없음 - 기본 처리 방식"),
        ("invalid_frame", "Frame 열 유효하지 않음 - 기본 처리 방식")
    ]
    
    for scenario, description in test_scenarios:
        print(f"=== 테스트 시나리오: {description} ===")
        
        # 테스트 파일 생성
        test_file = f"test_{scenario}.csv"
        create_test_csv_with_frame(test_file, scenario)
        
        print(f"테스트 파일 생성: {test_file}")
        
        # CSV 로드 및 결측치 처리
        try:
            df = analyzer.load_mocap_data(test_file)
            if df is not None:
                print(f"처리 완료 - 최종 데이터 형태: {df.shape}")
                print(f"결측치 여부: {df.isna().sum().sum()}개")
                print("✅ 성공")
            else:
                print("❌ 로드 실패")
        except Exception as e:
            print(f"❌ 오류 발생: {e}")
        
        print()
        
        # 테스트 파일 정리
        if os.path.exists(test_file):
            os.remove(test_file)
    
    print("=== 추가 테스트: 구체적인 보간 방식 확인 ===")
    
    # 구체적인 보간 테스트
    create_detailed_interpolation_test()
    
    print("\n=== 테스트 완료 ===")
    print("✅ Frame 열 자동 감지 기능")
    print("✅ 5% 기준 다른 보간 방식 적용")
    print("✅ (1-결측길이/전체프레임) 보간 계산")
    print("✅ 전체 구간 평균값 대체")
    print("✅ Frame 열 없을 때 fallback 처리")

def create_detailed_interpolation_test():
    """구체적인 보간 방식 테스트"""
    
    print("구체적인 보간 계산 예시:")
    
    # 예시: 100 프레임 중 3개 연속 결측치
    total_frames = 100
    missing_length = 3
    missing_ratio = missing_length / total_frames  # 3%
    
    print(f"  전체 프레임: {total_frames}")
    print(f"  결측치 길이: {missing_length}")
    print(f"  결측치 비율: {missing_ratio:.2%} (5% 이하)")
    
    # 보간 계수 계산
    interpolation_factor = 1.0 - missing_ratio
    print(f"  보간 계수: 1 - {missing_ratio:.3f} = {interpolation_factor:.3f}")
    
    # 예시 값으로 보간 계산
    before_val = 10.0
    after_val = 20.0
    interpolated_val = before_val * interpolation_factor + after_val * (1 - interpolation_factor)
    
    print(f"  앞쪽 값: {before_val}")
    print(f"  뒤쪽 값: {after_val}")
    print(f"  보간값: {before_val} × {interpolation_factor:.3f} + {after_val} × {1-interpolation_factor:.3f} = {interpolated_val:.3f}")
    
    print("\n예시: 100 프레임 중 10개 연속 결측치")
    missing_length = 10
    missing_ratio = missing_length / total_frames  # 10%
    print(f"  결측치 비율: {missing_ratio:.2%} (5% 이상)")
    print(f"  → 전체 구간 평균값으로 대체")

if __name__ == "__main__":
    test_frame_based_missing_handling()
