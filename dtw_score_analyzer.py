#!/usr/bin/env python3
"""
DTW Score 분석기 - 각 컬럼별 pre, main, post 평균 계산
"""

import pandas as pd
import numpy as np
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple
import warnings

warnings.filterwarnings('ignore')

class DTWScoreAnalyzer:
    """DTW Score CSV 파일들을 분석하여 각 컬럼별 pre, main, post 평균을 계산하는 클래스"""
    
    def __init__(self, dtw_score_dir: str = "/Users/jonabi/Downloads/TEPA/DTW_Score"):
        self.dtw_score_dir = Path(dtw_score_dir)
        self.data_frames = {}
        self.analysis_results = {}
        
    def load_all_csv_files(self) -> Dict[str, pd.DataFrame]:
        """DTW_Score 디렉터리의 모든 CSV 파일을 로드"""
        print(f"DTW Score 디렉터리에서 CSV 파일들을 로드합니다: {self.dtw_score_dir}")
        
        if not self.dtw_score_dir.exists():
            raise FileNotFoundError(f"DTW Score 디렉터리가 존재하지 않습니다: {self.dtw_score_dir}")
        
        csv_files = list(self.dtw_score_dir.glob("*.csv"))
        print(f"발견된 CSV 파일 수: {len(csv_files)}개")
        
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                self.data_frames[csv_file.stem] = df
                print(f"  로드됨: {csv_file.name} ({len(df)} 행)")
            except Exception as e:
                print(f"  오류: {csv_file.name} - {e}")
        
        print(f"총 {len(self.data_frames)}개 파일 로드 완료")
        return self.data_frames
    
    def extract_condition_from_filename(self, filename: str) -> str:
        """파일명에서 조건(pre, main, post) 추출"""
        filename_lower = filename.lower()
        
        if 'pre' in filename_lower:
            return 'pre'
        elif 'main' in filename_lower:
            return 'main'
        elif 'post' in filename_lower:
            return 'post'
        else:
            return 'unknown'
    
    def analyze_pre_main_post_averages(self) -> Dict[str, Dict[str, float]]:
        """각 컬럼별로 pre, main, post 평균을 계산"""
        print("\n=== 각 컬럼별 pre, main, post 평균 계산 시작 ===")
        
        # 모든 데이터를 하나의 DataFrame으로 합치기
        all_data = []
        
        for file_stem, df in self.data_frames.items():
            # 첫 번째 컬럼이 파일명이므로, 각 행에 대해 조건 추출
            for idx, row in df.iterrows():
                filename = str(row.iloc[0])  # 첫 번째 컬럼의 파일명
                condition = self.extract_condition_from_filename(filename)
                
                # 데이터 부분만 추출 (첫 번째 컬럼 제외)
                data_row = row.iloc[1:].copy()
                data_row['condition'] = condition
                data_row['filename'] = filename
                data_row['file_stem'] = file_stem
                
                all_data.append(data_row)
        
        # 전체 데이터를 DataFrame으로 변환
        combined_df = pd.DataFrame(all_data)
        
        if combined_df.empty:
            print("분석할 데이터가 없습니다.")
            return {}
        
        print(f"전체 데이터 행 수: {len(combined_df)}")
        print(f"조건별 데이터 분포:")
        print(combined_df['condition'].value_counts())
        
        # 데이터 컬럼들 (condition, filename, file_stem 제외)
        data_columns = [col for col in combined_df.columns if col not in ['condition', 'filename', 'file_stem']]
        
        print(f"분석할 데이터 컬럼 수: {len(data_columns)}개")
        
        # 각 컬럼별로 pre, main, post 평균 계산
        results = {}
        
        for column in data_columns:
            column_results = {}
            
            for condition in ['pre', 'main', 'post']:
                condition_data = combined_df[combined_df['condition'] == condition][column]
                
                if len(condition_data) > 0:
                    # NaN 값 제외하고 평균 계산
                    valid_data = condition_data.dropna()
                    if len(valid_data) > 0:
                        avg_value = valid_data.mean()
                        column_results[condition] = avg_value
                        print(f"  {column} - {condition}: 평균 {avg_value:.6f} (데이터 수: {len(valid_data)}개)")
                    else:
                        column_results[condition] = np.nan
                        print(f"  {column} - {condition}: 유효한 데이터 없음")
                else:
                    column_results[condition] = np.nan
                    print(f"  {column} - {condition}: 데이터 없음")
            
            results[column] = column_results
        
        self.analysis_results = results
        print(f"\n분석 완료: {len(results)}개 컬럼 처리됨")
        
        return results
    
    def save_analysis_results(self, output_file: str = None) -> str:
        """분석 결과를 CSV 파일로 저장"""
        if not self.analysis_results:
            print("저장할 분석 결과가 없습니다. 먼저 analyze_pre_main_post_averages()를 실행하세요.")
            return ""
        
        if output_file is None:
            output_file = self.dtw_score_dir / "pre_main_post_analysis_results.csv"
        
        # 결과를 DataFrame으로 변환
        results_data = []
        
        for column, conditions in self.analysis_results.items():
            row = {'Column': column}
            row.update(conditions)
            results_data.append(row)
        
        results_df = pd.DataFrame(results_data)
        
        # CSV로 저장
        results_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        
        print(f"\n분석 결과가 저장되었습니다: {output_file}")
        print(f"저장된 데이터 형태: {results_df.shape}")
        
        return str(output_file)
    
    def print_summary_statistics(self):
        """분석 결과 요약 통계 출력"""
        if not self.analysis_results:
            print("분석 결과가 없습니다.")
            return
        
        print("\n=== 분석 결과 요약 통계 ===")
        
        # 전체 컬럼 수
        total_columns = len(self.analysis_results)
        print(f"총 컬럼 수: {total_columns}개")
        
        # 각 조건별 데이터 존재 여부
        for condition in ['pre', 'main', 'post']:
            columns_with_data = sum(1 for results in self.analysis_results.values() 
                                  if not pd.isna(results.get(condition, np.nan)))
            print(f"{condition.upper()} 데이터가 있는 컬럼 수: {columns_with_data}개")
        
        # 각 컬럼별 조건별 평균값 범위
        print("\n=== 각 조건별 평균값 범위 ===")
        for condition in ['pre', 'main', 'post']:
            values = []
            for results in self.analysis_results.values():
                val = results.get(condition, np.nan)
                if not pd.isna(val):
                    values.append(val)
            
            if values:
                print(f"{condition.upper()}: 최소 {min(values):.6f}, 최대 {max(values):.6f}, 평균 {np.mean(values):.6f}")
            else:
                print(f"{condition.upper()}: 데이터 없음")
    
    def get_column_averages_by_condition(self, condition: str) -> Dict[str, float]:
        """특정 조건(pre, main, post)의 모든 컬럼 평균값 반환"""
        if condition not in ['pre', 'main', 'post']:
            raise ValueError("condition은 'pre', 'main', 'post' 중 하나여야 합니다.")
        
        results = {}
        for column, conditions in self.analysis_results.items():
            value = conditions.get(condition, np.nan)
            if not pd.isna(value):
                results[column] = value
        
        return results
    
    def compare_conditions(self, column: str = None) -> pd.DataFrame:
        """조건별 비교 (특정 컬럼 또는 전체 컬럼)"""
        if column and column in self.analysis_results:
            # 특정 컬럼의 조건별 비교
            results = self.analysis_results[column]
            comparison_df = pd.DataFrame([results])
            comparison_df.index = [column]
        else:
            # 전체 컬럼의 조건별 비교
            comparison_data = []
            for col, results in self.analysis_results.items():
                row = {'Column': col}
                row.update(results)
                comparison_data.append(row)
            comparison_df = pd.DataFrame(comparison_data)
            comparison_df = comparison_df.set_index('Column')
        
        return comparison_df

def main():
    """메인 실행 함수"""
    print("DTW Score 분석기 시작")
    print("=" * 50)
    
    # 분석기 생성
    analyzer = DTWScoreAnalyzer()
    
    try:
        # 1. CSV 파일들 로드
        analyzer.load_all_csv_files()
        
        # 2. pre, main, post 평균 계산
        results = analyzer.analyze_pre_main_post_averages()
        
        # 3. 요약 통계 출력
        analyzer.print_summary_statistics()
        
        # 4. 결과 저장
        output_file = analyzer.save_analysis_results()
        
        # 5. 샘플 결과 출력
        print("\n=== 샘플 분석 결과 (처음 10개 컬럼) ===")
        sample_results = dict(list(results.items())[:10])
        for column, conditions in sample_results.items():
            print(f"{column}:")
            for condition, value in conditions.items():
                if not pd.isna(value):
                    print(f"  {condition}: {value:.6f}")
                else:
                    print(f"  {condition}: N/A")
        
        print(f"\n분석 완료! 결과 파일: {output_file}")
        
    except Exception as e:
        print(f"오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
