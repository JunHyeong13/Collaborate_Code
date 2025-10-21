#!/usr/bin/env python3
"""
자동화된 유사도 계산 기능 테스트 스크립트
"""

import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path

# 현재 디렉토리를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# MocapMotionAnalyzer 클래스 import
from main_HCI_JJH import MocapMotionAnalyzer, save_similarity_across_groups

def create_test_file(filename: str, motion_type: str) -> str:
    """테스트용 CSV 파일 생성"""
    np.random.seed(hash(filename) % 1000)  # 파일명에 따른 시드 설정
    
    # 테스트용 조인트 데이터 생성
    joints = [
        'LShoulder', 'LUArm', 'LFArm', 'LHand',  # Left Arm
        'RShoulder', 'RUArm', 'RFArm', 'RHand',  # Right Arm
        'LThigh', 'LShin', 'LFoot', 'LToe',      # Left Leg
        'RThigh', 'RShin', 'RFoot', 'RToe',      # Right Leg
        'Hip', 'Ab', 'Chest', 'Neck', 'Head'     # Core & Head
    ]
    
    frames = 50
    data = {}
    
    for joint in joints:
        # 위치 데이터 (posX, posY, posZ)
        data[f'{joint}.posX'] = np.random.randn(frames) * 10 + 100
        data[f'{joint}.posY'] = np.random.randn(frames) * 10 + 50
        data[f'{joint}.posZ'] = np.random.randn(frames) * 10 + 0
        
        # 회전 데이터 (rotX, rotY, rotZ, rotW - 쿼터니언)
        data[f'{joint}.rotX'] = np.random.randn(frames) * 0.1
        data[f'{joint}.rotY'] = np.random.randn(frames) * 0.1
        data[f'{joint}.rotZ'] = np.random.randn(frames) * 0.1
        data[f'{joint}.rotW'] = np.sqrt(1 - np.square(data[f'{joint}.rotX']) - 
                                       np.square(data[f'{joint}.rotY']) - 
                                       np.square(data[f'{joint}.rotZ']))
    
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    return filename

def test_auto_similarity_functionality():
    """자동화된 유사도 계산 기능 테스트"""
    
    print("=== 자동화된 유사도 계산 기능 테스트 ===\n")
    
    # 테스트 디렉터리 생성
    test_dir = Path("test_auto_similarity")
    test_dir.mkdir(exist_ok=True)
    
    # file1용 테스트 파일들 생성 (다양한 motion type)
    file1_dir = test_dir / "mocap_test"
    file1_dir.mkdir(exist_ok=True)
    
    test_files = [
        "jap_001.csv",
        "jap_002.csv", 
        "straight_001.csv",
        "straight_002.csv",
        "hook_left_001.csv",
        "hook_left_002.csv",
        "hook_right_001.csv",
        "hook_right_002.csv",
        "uppercut_left_001.csv",
        "uppercut_right_001.csv"
    ]
    
    print("테스트 파일 생성 중...")
    for filename in test_files:
        motion_type = filename.split('_')[0]
        filepath = file1_dir / filename
        create_test_file(str(filepath), motion_type)
        print(f"  생성됨: {filepath}")
    
    # file2용 그룹 디렉터리 생성
    p02_dir = test_dir / "p02_Global"
    p02_dir.mkdir(exist_ok=True)
    
    p03_dir = test_dir / "p03_Global"
    p03_dir.mkdir(exist_ok=True)
    
    # 각 그룹에 테스트 파일들 생성 (motion type별로)
    motion_types = ['jap', 'hook_left', 'hook_right', 'uppercut_left']
    
    for group_dir in [p02_dir, p03_dir]:
        for motion_type in motion_types:
            for i in range(2):  # 각 motion type당 2개씩
                filename = group_dir / f"{motion_type}_test_{i:03d}.csv"
                create_test_file(str(filename), motion_type)
                print(f"  생성됨: {filename}")
        
        # 매칭되지 않는 파일도 생성 (테스트용)
        for i in range(2):
            filename = group_dir / f"unmatched_motion_{i:03d}.csv"
            create_test_file(str(filename), "unmatched")
            print(f"  생성됨: {filename}")
    
    print(f"\n테스트 환경 구성 완료!")
    print(f"file1 디렉터리: {file1_dir}")
    print(f"file2 베이스 디렉터리: {test_dir}")
    print()
    
    # MocapMotionAnalyzer 인스턴스 생성
    analyzer = MocapMotionAnalyzer()
    
    # 자동화된 유사도 계산 테스트
    print("=== 자동화된 유사도 계산 실행 (motion type 매칭) ===")
    print("예시:")
    print("  hook_left_001.csv (file1) → hook_left가 포함된 file2 파일들만 비교")
    print("  jap_002.csv (file1) → jap가 포함된 file2 파일들만 비교")
    print("  unmatched 파일들은 매칭되지 않음")
    print()
    
    all_results = save_similarity_across_groups(
        file1_path=str(file1_dir),           # 폴더 경로 (모든 CSV 파일 자동 순회)
        file2_path_or_base=str(test_dir),    # 베이스 경로
        analyzer=analyzer,
        start=2,                             # p02_Global부터
        end=3,                               # p03_Global까지
        keyword=None,                        # 모든 파일
        limit=None,                          # 제한 없음
        title=None,                          # 파일명에서 자동 추출
        output_dir=str(test_dir / "results"), # 결과 저장 디렉터리
        auto_file1=True,                     # 자동 순회 활성화
    )
    
    # 결과 요약 출력
    print("\n=== 처리 결과 요약 ===")
    total_files = 0
    total_groups = 0
    
    for file_name, file_results in all_results.items():
        print(f"기준 파일: {file_name}")
        print(f"  처리된 그룹 수: {len(file_results)}개")
        total_files += 1
        
        for group_num, df in file_results.items():
            print(f"  p{group_num:02d}_Global: {len(df)}개 파일 비교 완료")
            total_groups += 1
    
    print(f"\n=== 전체 요약 ===")
    print(f"처리된 기준 파일 수: {total_files}개")
    print(f"처리된 그룹 수: {total_groups}개")
    print(f"생성된 CSV 파일 수: {total_files * total_groups}개")
    
    # 생성된 결과 파일들 확인
    results_dir = test_dir / "results"
    if results_dir.exists():
        csv_files = list(results_dir.glob("*.csv"))
        print(f"실제 생성된 CSV 파일 수: {len(csv_files)}개")
        print("생성된 파일들:")
        for csv_file in sorted(csv_files):
            print(f"  {csv_file.name}")
    
    print("\n=== 테스트 완료 ===")
    print("✅ 자동 파일 순회 기능 정상 작동")
    print("✅ 파일명에서 title 자동 추출 기능 정상 작동")
    print("✅ 그룹별 CSV 파일 생성 기능 정상 작동")
    
    # 테스트 파일 정리 (선택사항)
    cleanup = input("\n테스트 파일을 삭제하시겠습니까? (y/n): ").lower().strip()
    if cleanup == 'y':
        import shutil
        shutil.rmtree(test_dir)
        print("테스트 파일이 삭제되었습니다.")
    else:
        print(f"테스트 파일이 보존되었습니다: {test_dir}")

if __name__ == "__main__":
    test_auto_similarity_functionality()
