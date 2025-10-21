#!/usr/bin/env python3
"""
DTW Score 개별 파일 분석기 - 각 파일별로 pre, main, post 평균 계산
"""

import pandas as pd
import numpy as np
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple
import warnings

warnings.filterwarnings('ignore')

class DTWScoreIndividualAnalyzer:
    """각 DTW Score CSV 파일별로 개별적으로 pre, main, post 평균을 계산하는 클래스"""
    
    def __init__(self, dtw_score_dir: str = "/Users/jonabi/Downloads/TEPA/DTW_Score/"):
        self.dtw_score_dir = Path(dtw_score_dir)
        self.file_results = {}  # 각 파일별 결과 저장
        
    def load_all_csv_files(self) -> Dict[str, pd.DataFrame]:
        """DTW_Score 디렉터리의 모든 CSV 파일을 로드"""
        print(f"DTW Score 디렉터리에서 CSV 파일들을 로드합니다: {self.dtw_score_dir}")
        
        if not self.dtw_score_dir.exists():
            raise FileNotFoundError(f"DTW Score 디렉터리가 존재하지 않습니다: {self.dtw_score_dir}")
        
        csv_files = list(self.dtw_score_dir.glob("*.csv"))
        print(f"발견된 CSV 파일 수: {len(csv_files)}개")
        
        data_frames = {}
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                data_frames[csv_file.stem] = df
                print(f"  로드됨: {csv_file.name} ({len(df)} 행)")
            except Exception as e:
                print(f"  오류: {csv_file.name} - {e}")
        
        print(f"총 {len(data_frames)}개 파일 로드 완료")
        return data_frames
    
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
    
    def analyze_individual_file(self, file_stem: str, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """개별 파일의 pre, main, post 평균을 계산"""
        
        # 첫 번째 컬럼이 파일명이므로, 각 행에 대해 조건 추출
        condition_data = {'pre': [], 'main': [], 'post': []}
        
        for idx, row in df.iterrows():
            filename = str(row.iloc[0])  # 첫 번째 컬럼의 파일명
            condition = self.extract_condition_from_filename(filename)
            
            if condition in ['pre', 'main', 'post']:
                # 데이터 부분만 추출 (첫 번째 컬럼 제외)
                data_row = row.iloc[1:]
                condition_data[condition].append(data_row)
        
        # 각 조건별로 DataFrame 생성
        condition_dfs = {}
        for condition, rows in condition_data.items():
            if rows:
                condition_dfs[condition] = pd.DataFrame(rows)
            else:
                condition_dfs[condition] = pd.DataFrame()
        
        # 각 컬럼별로 조건별 평균 계산
        results = {}
        data_columns = [col for col in df.columns[1:]]  # 첫 번째 컬럼(파일명) 제외
        
        for column in data_columns:
            column_results = {}
            
            for condition in ['pre', 'main', 'post']:
                if condition in condition_dfs and not condition_dfs[condition].empty:
                    if column in condition_dfs[condition].columns:
                        # 해당 조건의 데이터에서 평균 계산
                        condition_values = condition_dfs[condition][column]
                        valid_values = condition_values.dropna()
                        
                        if len(valid_values) > 0:
                            avg_value = valid_values.mean()
                            column_results[condition] = avg_value
                        else:
                            column_results[condition] = np.nan
                    else:
                        column_results[condition] = np.nan
                else:
                    column_results[condition] = np.nan
            
            results[column] = column_results
        
        return results
    
    def analyze_all_files_individual(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        """모든 파일을 개별적으로 분석하여 각 파일별 pre, main, post 평균 계산"""
        print("\n=== 각 파일별 개별 분석 시작 ===")
        
        # 모든 CSV 파일 로드
        data_frames = self.load_all_csv_files()
        
        if not data_frames:
            print("분석할 파일이 없습니다.")
            return {}
        
        # 각 파일별로 개별 분석
        for file_stem, df in data_frames.items():
            print(f"\n분석 중: {file_stem}")
            
            # 파일별 조건 분포 확인
            conditions = []
            for idx, row in df.iterrows():
                filename = str(row.iloc[0])
                condition = self.extract_condition_from_filename(filename)
                conditions.append(condition)
            
            condition_counts = pd.Series(conditions).value_counts()
            print(f"  조건 분포: {dict(condition_counts)}")
            
            # 개별 파일 분석
            file_results = self.analyze_individual_file(file_stem, df)
            self.file_results[file_stem] = file_results
            
            # 결과 요약 출력
            print(f"  분석 완료 - {len(file_results)}개 컬럼 처리됨")
            
            # 샘플 결과 출력 (처음 3개 컬럼만)
            sample_columns = list(file_results.keys())[:3]
            for column in sample_columns:
                print(f"    {column}:")
                for condition, value in file_results[column].items():
                    if not pd.isna(value):
                        print(f"      {condition}: {value:.6f}")
                    else:
                        print(f"      {condition}: N/A")
        
        print(f"\n전체 분석 완료: {len(self.file_results)}개 파일 처리됨")
        return self.file_results
    
    def save_individual_results(self, output_dir: str = None) -> str:
        """각 파일별 개별 분석 결과를 CSV 파일로 저장"""
        if not self.file_results:
            print("저장할 분석 결과가 없습니다. 먼저 analyze_all_files_individual()를 실행하세요.")
            return ""
        
        if output_dir is None:
            output_dir = self.dtw_score_dir / "individual_analysis_results"
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 각 파일별로 개별 CSV 저장
        saved_files = []
        
        for file_stem, results in self.file_results.items():
            # 결과를 DataFrame으로 변환
            results_data = []
            
            for column, conditions in results.items():
                row = {'Column': column}
                row.update(conditions)
                results_data.append(row)
            
            results_df = pd.DataFrame(results_data)
            
            # 파일별 CSV 저장
            output_file = output_dir / f"{file_stem}_analysis.csv"
            results_df.to_csv(output_file, index=False, encoding='utf-8-sig')
            saved_files.append(str(output_file))
            
            print(f"저장됨: {output_file}")
        
        # 전체 결과를 하나의 파일로도 저장
        all_results_file = output_dir / "all_files_analysis_summary.csv"
        self._save_all_results_summary(all_results_file)
        saved_files.append(str(all_results_file))
        
        print(f"\n총 {len(saved_files)}개 파일 저장 완료")
        print(f"저장 디렉터리: {output_dir}")
        
        return str(output_dir)
    
    def _save_all_results_summary(self, output_file: Path):
        """모든 파일의 결과를 하나의 요약 파일로 저장"""
        summary_data = []
        
        for file_stem, results in self.file_results.items():
            for column, conditions in results.items():
                row = {
                    'File': file_stem,
                    'Column': column,
                    'Pre_Avg': conditions.get('pre', np.nan),
                    'Main_Avg': conditions.get('main', np.nan),
                    'Post_Avg': conditions.get('post', np.nan)
                }
                summary_data.append(row)
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        
        print(f"전체 요약 저장됨: {output_file}")
    
    def print_file_summary(self, file_stem: str = None):
        """특정 파일 또는 전체 파일의 요약 통계 출력"""
        if not self.file_results:
            print("분석 결과가 없습니다.")
            return
        
        if file_stem:
            # 특정 파일 요약
            if file_stem in self.file_results:
                print(f"\n=== {file_stem} 파일 요약 ===")
                results = self.file_results[file_stem]
                
                for condition in ['pre', 'main', 'post']:
                    values = []
                    for col_results in results.values():
                        val = col_results.get(condition, np.nan)
                        if not pd.isna(val):
                            values.append(val)
                    
                    if values:
                        print(f"{condition.upper()}: 최소 {min(values):.6f}, 최대 {max(values):.6f}, 평균 {np.mean(values):.6f}")
                    else:
                        print(f"{condition.upper()}: 데이터 없음")
            else:
                print(f"파일 '{file_stem}'을 찾을 수 없습니다.")
        else:
            # 전체 파일 요약
            print(f"\n=== 전체 파일 요약 ===")
            print(f"분석된 파일 수: {len(self.file_results)}개")
            
            # 각 조건별 전체 통계
            for condition in ['pre', 'main', 'post']:
                all_values = []
                files_with_data = 0
                
                for file_stem, results in self.file_results.items():
                    values = []
                    for col_results in results.values():
                        val = col_results.get(condition, np.nan)
                        if not pd.isna(val):
                            values.append(val)
                    
                    if values:
                        all_values.extend(values)
                        files_with_data += 1
                
                if all_values:
                    print(f"{condition.upper()}: {files_with_data}개 파일, 전체 평균 {np.mean(all_values):.6f}")
                else:
                    print(f"{condition.upper()}: 데이터 없음")
    
    def get_file_results(self, file_stem: str) -> Dict[str, Dict[str, float]]:
        """특정 파일의 분석 결과 반환"""
        return self.file_results.get(file_stem, {})
    
    def compare_files(self, column: str, file_stems: List[str] = None) -> pd.DataFrame:
        """여러 파일의 특정 컬럼 조건별 비교"""
        if file_stems is None:
            file_stems = list(self.file_results.keys())
        
        comparison_data = []
        
        for file_stem in file_stems:
            if file_stem in self.file_results:
                results = self.file_results[file_stem]
                if column in results:
                    row = {'File': file_stem}
                    row.update(results[column])
                    comparison_data.append(row)
        
        return pd.DataFrame(comparison_data)

def main():
    """메인 실행 함수"""
    print("DTW Score 개별 파일 분석기 시작")
    print("=" * 50)
    
    # 분석기 생성
    analyzer = DTWScoreIndividualAnalyzer()
    
    try:
        # 각 파일별 개별 분석 실행
        results = analyzer.analyze_all_files_individual()
        
        # 전체 요약 출력
        analyzer.print_file_summary()
        
        # 결과 저장
        output_dir = analyzer.save_individual_results()
        
        # 샘플 결과 출력 (처음 3개 파일)
        print("\n=== 샘플 분석 결과 (처음 3개 파일) ===")
        sample_files = list(results.keys())[:3]
        
        for file_stem in sample_files:
            print(f"\n파일: {file_stem}")
            file_results = analyzer.get_file_results(file_stem)
            
            # 처음 5개 컬럼만 출력
            sample_columns = list(file_results.keys())[:5]
            for column in sample_columns:
                print(f"  {column}:")
                for condition, value in file_results[column].items():
                    if not pd.isna(value):
                        print(f"    {condition}: {value:.6f}")
                    else:
                        print(f"    {condition}: N/A")
        
        print(f"\n분석 완료! 결과 디렉터리: {output_dir}")
        
    except Exception as e:
        print(f"오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
