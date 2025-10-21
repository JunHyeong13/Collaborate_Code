#!/usr/bin/env python3
"""
DTW Score 개별 파일 분석기 테스트 스크립트
"""

import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path

# 현재 디렉토리를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dtw_score_individual_analyzer import DTWScoreIndividualAnalyzer

def create_test_individual_data():
    """개별 파일 분석용 테스트 데이터 생성"""
    test_dir = Path("test_individual_analysis")
    test_dir.mkdir(exist_ok=True)
    
    # 파일 1: hook_left_p02_Global_hook_left_001_similarity_matrix
    data1 = {
        'filename': [
            'p02_hook_left_pre_001',
            'p02_hook_left_pre_002', 
            'p02_hook_left_main_001',
            'p02_hook_left_main_002',
            'p02_hook_left_post_001',
            'p02_hook_left_post_002'
        ],
        'LShoulder': [0.2, 0.25, 0.3, 0.35, 0.4, 0.45],
        'LUArm': [0.1, 0.12, 0.15, 0.18, 0.2, 0.22],
        'LFArm': [0.3, 0.32, 0.35, 0.38, 0.4, 0.42],
        'Overall': [0.5, 0.55, 0.6, 0.65, 0.7, 0.75]
    }
    
    df1 = pd.DataFrame(data1)
    test_file1 = test_dir / "hook_left_p02_Global_hook_left_001_similarity_matrix.csv"
    df1.to_csv(test_file1, index=False)
    
    # 파일 2: hook_left_p04_Global_hook_left_002_similarity_matrix
    data2 = {
        'filename': [
            'p04_hook_left_pre_001',
            'p04_hook_left_pre_002', 
            'p04_hook_left_main_001',
            'p04_hook_left_main_002',
            'p04_hook_left_post_001'
        ],
        'LShoulder': [0.15, 0.18, 0.25, 0.28, 0.35],
        'LUArm': [0.08, 0.10, 0.12, 0.14, 0.16],
        'LFArm': [0.28, 0.30, 0.32, 0.34, 0.36],
        'Overall': [0.45, 0.50, 0.55, 0.60, 0.65]
    }
    
    df2 = pd.DataFrame(data2)
    test_file2 = test_dir / "hook_left_p04_Global_hook_left_002_similarity_matrix.csv"
    df2.to_csv(test_file2, index=False)
    
    # 파일 3: jap_p06_Global_jap_001_similarity_matrix
    data3 = {
        'filename': [
            'p06_jap_pre_001',
            'p06_jap_pre_002', 
            'p06_jap_main_001',
            'p06_jap_main_002',
            'p06_jap_post_001',
            'p06_jap_post_002',
            'p06_jap_post_003'
        ],
        'LShoulder': [0.3, 0.32, 0.4, 0.42, 0.5, 0.52, 0.54],
        'LUArm': [0.2, 0.22, 0.25, 0.27, 0.3, 0.32, 0.34],
        'LFArm': [0.4, 0.42, 0.45, 0.47, 0.5, 0.52, 0.54],
        'Overall': [0.6, 0.62, 0.7, 0.72, 0.8, 0.82, 0.84]
    }
    
    df3 = pd.DataFrame(data3)
    test_file3 = test_dir / "jap_p06_Global_jap_001_similarity_matrix.csv"
    df3.to_csv(test_file3, index=False)
    
    print(f"테스트 데이터 생성 완료:")
    print(f"  파일 1: {test_file1.name} ({len(df1)} 행)")
    print(f"  파일 2: {test_file2.name} ({len(df2)} 행)")
    print(f"  파일 3: {test_file3.name} ({len(df3)} 행)")
    
    return test_dir

def test_individual_analyzer():
    """DTW Score 개별 파일 분석기 테스트"""
    
    print("=== DTW Score 개별 파일 분석기 테스트 ===\n")
    
    # 테스트 데이터 생성
    test_dir = create_test_individual_data()
    
    # 분석기 생성 (테스트 디렉터리 사용)
    analyzer = DTWScoreIndividualAnalyzer(str(test_dir))
    
    try:
        # 1. 개별 파일 분석
        print("1. 각 파일별 개별 분석 테스트")
        results = analyzer.analyze_all_files_individual()
        print(f"분석된 파일 수: {len(results)}개\n")
        
        # 2. 개별 파일 결과 확인
        print("2. 개별 파일 결과 확인")
        for file_stem, file_results in results.items():
            print(f"\n파일: {file_stem}")
            print(f"  컬럼 수: {len(file_results)}개")
            
            # 각 조건별 데이터 수 확인
            for condition in ['pre', 'main', 'post']:
                count = sum(1 for col_results in file_results.values() 
                           if not pd.isna(col_results.get(condition, np.nan)))
                print(f"  {condition}: {count}개 컬럼에 데이터 있음")
            
            # 샘플 결과 출력 (처음 2개 컬럼만)
            sample_columns = list(file_results.keys())[:2]
            for column in sample_columns:
                print(f"  {column}:")
                for condition, value in file_results[column].items():
                    if not pd.isna(value):
                        print(f"    {condition}: {value:.6f}")
                    else:
                        print(f"    {condition}: N/A")
        
        # 3. 파일별 요약 통계
        print("\n3. 파일별 요약 통계")
        for file_stem in results.keys():
            analyzer.print_file_summary(file_stem)
        
        # 4. 전체 요약 통계
        print("\n4. 전체 요약 통계")
        analyzer.print_file_summary()
        
        # 5. 특정 컬럼 파일 간 비교
        print("\n5. 파일 간 비교 (LShoulder 컬럼)")
        comparison_df = analyzer.compare_files('LShoulder')
        print("LShoulder 컬럼 비교:")
        print(comparison_df)
        
        # 6. 결과 저장
        print("\n6. 결과 저장 테스트")
        output_dir = analyzer.save_individual_results()
        
        # 저장된 파일들 확인
        if os.path.exists(output_dir):
            saved_files = list(Path(output_dir).glob("*.csv"))
            print(f"저장된 파일 수: {len(saved_files)}개")
            for file_path in saved_files:
                print(f"  {file_path.name}")
        
        print("\n=== 테스트 완료 ===")
        print("✅ 각 파일별 개별 분석")
        print("✅ 파일 내 pre, main, post 조건별 평균 계산")
        print("✅ 개별 파일별 결과 저장")
        print("✅ 파일 간 비교 기능")
        print("✅ 요약 통계 생성")
        
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

def test_real_data_sample():
    """실제 DTW Score 데이터 샘플 테스트"""
    print("\n=== 실제 DTW Score 데이터 샘플 테스트 ===")
    
    dtw_score_dir = "/Users/jonabi/Downloads/TEPA/DTW_Score"
    
    if not os.path.exists(dtw_score_dir):
        print(f"DTW Score 디렉터리가 존재하지 않습니다: {dtw_score_dir}")
        return
    
    try:
        # 분석기 생성
        analyzer = DTWScoreIndividualAnalyzer(dtw_score_dir)
        
        # CSV 파일들 로드 (처음 3개 파일만)
        print("실제 데이터 로드 중...")
        data_frames = analyzer.load_all_csv_files()
        
        if len(data_frames) > 3:
            # 처음 3개 파일만 사용
            sample_files = dict(list(data_frames.items())[:3])
            analyzer.data_frames = sample_files
            print(f"테스트를 위해 처음 3개 파일만 사용합니다.")
        
        # 개별 분석 실행
        print("\n개별 분석 실행 중...")
        results = analyzer.analyze_all_files_individual()
        
        # 요약 출력
        analyzer.print_file_summary()
        
        # 샘플 결과 출력
        print("\n=== 샘플 결과 ===")
        for file_stem in list(results.keys())[:2]:  # 처음 2개 파일만
            print(f"\n파일: {file_stem}")
            file_results = analyzer.get_file_results(file_stem)
            
            # 처음 3개 컬럼만 출력
            sample_columns = list(file_results.keys())[:3]
            for column in sample_columns:
                print(f"  {column}:")
                for condition, value in file_results[column].items():
                    if not pd.isna(value):
                        print(f"    {condition}: {value:.6f}")
                    else:
                        print(f"    {condition}: N/A")
        
        print("\n실제 데이터 샘플 테스트 완료!")
        
    except Exception as e:
        print(f"실제 데이터 테스트 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # 테스트 실행
    test_individual_analyzer()
    
    # 실제 데이터 샘플 테스트 (선택사항)
    test_real_data_sample()
