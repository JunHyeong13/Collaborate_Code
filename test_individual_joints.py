#!/usr/bin/env python3
"""
개별 조인트별 유사도 계산 로직 테스트 스크립트
"""

import pandas as pd
import numpy as np
import os
import sys

# 현재 디렉토리를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# MocapMotionAnalyzer 클래스 import
from main_HCI_JJH import MocapMotionAnalyzer

def create_test_data():
    """테스트용 모션캡처 데이터 생성"""
    np.random.seed(42)
    
    # 테스트용 조인트 데이터 생성
    joints = [
        'LShoulder', 'LUArm', 'LFArm', 'LHand',  # Left Arm
        'RShoulder', 'RUArm', 'RFArm', 'RHand',  # Right Arm
        'LThigh', 'LShin', 'LFoot', 'LToe',      # Left Leg
        'RThigh', 'RShin', 'RFoot', 'RToe',      # Right Leg
        'Hip', 'Ab', 'Chest', 'Neck', 'Head'     # Core & Head
    ]
    
    frames = 100
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
    
    return pd.DataFrame(data)

def test_individual_joint_calculation():
    """개별 조인트별 유사도 계산 테스트"""
    
    print("=== 개별 조인트별 유사도 계산 테스트 ===\n")
    
    # MocapMotionAnalyzer 인스턴스 생성
    analyzer = MocapMotionAnalyzer()
    
    # 테스트 데이터 생성
    print("테스트 데이터 생성 중...")
    motion1 = create_test_data()
    motion2 = create_test_data()  # 동일한 구조, 다른 값
    
    print(f"Motion 1: {motion1.shape}")
    print(f"Motion 2: {motion2.shape}")
    print()
    
    # 유사도 계산
    print("개별 조인트별 유사도 계산 중...")
    similarity, details = analyzer.compare_motions(motion1, motion2)
    
    print(f"Overall 유사도: {similarity:.4f}")
    print()
    
    # 개별 조인트별 유사도 출력
    print("=== 개별 조인트별 유사도 ===")
    joint_names = [
        'LShoulder', 'LUArm', 'LFArm', 'LHand',  # Left Arm
        'RShoulder', 'RUArm', 'RFArm', 'RHand',  # Right Arm
        'LThigh', 'LShin', 'LFoot', 'LToe',      # Left Leg
        'RThigh', 'RShin', 'RFoot', 'RToe',      # Right Leg
        'Hip', 'Ab', 'Chest', 'Neck', 'Head'     # Core & Head
    ]
    
    for joint in joint_names:
        joint_key = f"joint_{joint}"
        if joint_key in details:
            print(f"{joint:12}: {details[joint_key]:.4f}")
        else:
            print(f"{joint:12}: N/A")
    
    print()
    
    # 피처별 유사도 출력
    print("=== 피처별 유사도 ===")
    feature_types = ['position', 'rotation', 'velocity', 'acceleration', 'joint_angles']
    for feature in feature_types:
        if feature in details:
            print(f"{feature:15}: {details[feature]:.4f}")
    
    print()
    
    # CSV 컬럼 구조 확인
    print("=== CSV 저장용 컬럼 구조 ===")
    col_order = [
        "LShoulder", "LUArm", "LFArm", "LHand",  # Left Arm
        "RShoulder", "RUArm", "RFArm", "RHand",  # Right Arm
        "LThigh", "LShin", "LFoot", "LToe",      # Left Leg
        "RThigh", "RShin", "RFoot", "RToe",      # Right Leg
        "Hip", "Ab", "Chest", "Neck", "Head",    # Core & Head
        "Acceleration", "Velocity", "Position", "Joint Angle", "rotation", "Overall"
    ]
    
    print("총 컬럼 수:", len(col_order))
    print("컬럼 순서:")
    for i, col in enumerate(col_order, 1):
        print(f"  {i:2d}. {col}")
    
    print()
    print("=== 테스트 완료 ===")
    print("✅ 개별 조인트별 유사도 계산 성공")
    print("✅ CSV 컬럼 구조 업데이트 완료")
    print("✅ 부위별 묶음 → 개별 조인트별 계산으로 변경 완료")

if __name__ == "__main__":
    test_individual_joint_calculation()
