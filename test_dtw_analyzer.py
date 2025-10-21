#!/usr/bin/env python3
"""
DTW Score 분석기 테스트 스크립트
"""

import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path

# 현재 디렉토리를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dtw_score_analyzer import DTWScoreAnalyzer

def create_test_data():
    """테스트용 CSV 파일 생성"""
    test_dir = Path("test_dtw_analysis")
    test_dir.mkdir(exist_ok=True)
    
    # 테스트 데이터 생성
    test_data = {
        'filename': [
            'p02_hook_left_pre_001',
            'p02_hook_left_pre_002', 
            'p02_hook_left_main_001',
            'p02_hook_left_main_002',
            'p02_hook_left_post_001',
            'p02_hook_left_post_002',
            'p04_hook_left_pre_001',
            'p04_hook_left_main_001',
            'p04_hook_left_post_001'
        ],
        'LShoulder': [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.15, 0.28, 0.42],
        'LUArm': [0.1, 0.12, 0.15, 0.18, 0.2, 0.22, 0.08, 0.14, 0.21],
        'LFArm': [0.3, 0.32, 0.35, 0.38, 0.4, 0.42, 0.28, 0.33, 0.41],
        'Overall': [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.45, 0.58, 0.68]
    }
    
    df = pd.DataFrame(test_data)
    
    # CSV 파일 저장
    test_file = test_dir / "test_hook_left_similarity_matrix.csv"
    df.to_csv(test_file, index=False)
    
    print(f"테스트 데이터 생성: {test_file}")
    print(f"데이터 형태: {df.shape}")
    print("테스트 데이터 미리보기:")
    print(df.head())
    
    return test_file, test_dir

def test_dtw_analyzer():
    """DTW Score 분석기 테스트"""
    
    print("=== DTW Score 분석기 테스트 ===\n")
    
    # 테스트 데이터 생성
    test_file, test_dir = create_test_data()
    
    # 분석기 생성 (테스트 디렉터리 사용)
    analyzer = DTWScoreAnalyzer(str(test_dir))
    
    try:
        # 1. CSV 파일 로드
        print("1. CSV 파일 로드 테스트")
        data_frames = analyzer.load_all_csv_files()
        print(f"로드된 파일 수: {len(data_frames)}개\n")
        
        # 2. 조건 추출 테스트
        print("2. 조건 추출 테스트")
        test_filenames = [
            'p02_hook_left_pre_001',
            'p02_hook_left_main_002', 
            'p02_hook_left_post_003'
        ]
        
        for filename in test_filenames:
            condition = analyzer.extract_condition_from_filename(filename)
            print(f"  {filename} → {condition}")
        print()
        
        # 3. 평균 계산 테스트
        print("3. pre, main, post 평균 계산 테스트")
        results = analyzer.analyze_pre_main_post_averages()
        
        print("\n계산된 결과:")
        for column, conditions in results.items():
            print(f"  {column}:")
            for condition, value in conditions.items():
                if not pd.isna(value):
                    print(f"    {condition}: {value:.6f}")
                else:
                    print(f"    {condition}: N/A")
        print()
        
        # 4. 요약 통계 테스트
        print("4. 요약 통계 테스트")
        analyzer.print_summary_statistics()
        print()
        
        # 5. 특정 조건 데이터 추출 테스트
        print("5. 특정 조건 데이터 추출 테스트")
        for condition in ['pre', 'main', 'post']:
            condition_data = analyzer.get_column_averages_by_condition(condition)
            print(f"  {condition.upper()} 데이터: {len(condition_data)}개 컬럼")
            for col, value in condition_data.items():
                print(f"    {col}: {value:.6f}")
        print()
        
        # 6. 조건별 비교 테스트
        print("6. 조건별 비교 테스트")
        comparison_df = analyzer.compare_conditions()
        print("조건별 비교 결과:")
        print(comparison_df)
        print()
        
        # 7. 결과 저장 테스트
        print("7. 결과 저장 테스트")
        output_file = analyzer.save_analysis_results()
        print(f"저장된 파일: {output_file}")
        
        # 저장된 파일 확인
        if os.path.exists(output_file):
            saved_df = pd.read_csv(output_file)
            print(f"저장된 데이터 형태: {saved_df.shape}")
            print("저장된 데이터 미리보기:")
            print(saved_df.head())
        
        print("\n=== 테스트 완료 ===")
        print("✅ CSV 파일 로드")
        print("✅ 조건 추출 (pre, main, post)")
        print("✅ 각 컬럼별 평균 계산")
        print("✅ 요약 통계 생성")
        print("✅ 특정 조건 데이터 추출")
        print("✅ 조건별 비교")
        print("✅ 결과 저장")
        
    except Exception as e:
        print(f"테스트 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 테스트 파일 정리
        import shutil
        if test_dir.exists():
            shutil.rmtree(test_dir)
            print(f"\n테스트 파일 정리 완료: {test_dir}")

def test_real_data():
    """실제 DTW Score 데이터로 테스트"""
    print("\n=== 실제 DTW Score 데이터 테스트 ===")
    
    dtw_score_dir = "/Users/jonabi/Downloads/TEPA/DTW_Score"
    
    if not os.path.exists(dtw_score_dir):
        print(f"DTW Score 디렉터리가 존재하지 않습니다: {dtw_score_dir}")
        return
    
    try:
        # 분석기 생성
        analyzer = DTWScoreAnalyzer(dtw_score_dir)
        
        # CSV 파일들 로드 (처음 5개 파일만)
        print("실제 데이터 로드 중...")
        data_frames = analyzer.load_all_csv_files()
        
        if len(data_frames) > 5:
            # 처음 5개 파일만 사용
            sample_files = dict(list(data_frames.items())[:5])
            analyzer.data_frames = sample_files
            print(f"테스트를 위해 처음 5개 파일만 사용합니다.")
        
        # 분석 실행
        print("\n분석 실행 중...")
        results = analyzer.analyze_pre_main_post_averages()
        
        # 요약 출력
        analyzer.print_summary_statistics()
        
        # 샘플 결과 출력
        print("\n=== 샘플 결과 (처음 5개 컬럼) ===")
        sample_results = dict(list(results.items())[:5])
        for column, conditions in sample_results.items():
            print(f"{column}:")
            for condition, value in conditions.items():
                if not pd.isna(value):
                    print(f"  {condition}: {value:.6f}")
        
        print("\n실제 데이터 테스트 완료!")
        
    except Exception as e:
        print(f"실제 데이터 테스트 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # 테스트 실행
    test_dtw_analyzer()
    
    # 실제 데이터 테스트 (선택사항)
    test_real_data()
