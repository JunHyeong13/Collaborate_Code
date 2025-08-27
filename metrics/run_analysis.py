# run_analysis.py
# motion_analyzer 모듈을 사용하여 그룹별 파일들을 분석하고,
# gui_display 모듈을 통해 결과를 웹 UI로 시각화합니다.

from pathlib import Path
import pandas as pd
from collections import defaultdict
from motion_analyzer import analyze_motion_similarity
from gui_display import create_ui
from nicegui import ui

def main():
    """
    기준 CSV 파일과 여러 그룹의 대상 CSV 파일을 비교하고,
    각 그룹의 최종 및 세부 지표별 평균 유사도를 계산합니다.
    """
    # --- 설정: 여기에 파일 경로와 그룹을 직접 입력하세요 ---
    base_file_name = "expert.csv"
    plane = 'xyz'
    num = '08'
    analysis_groups = {
        "Pre-Training": [f"pre/p{num}_hook_left_pre_{i:03d}.csv" for i in range(1, 13)],
        "Main-Session": [f"main/p{num}_hook_left_main_{i:03d}.csv" for i in range(1, 13)],
        "Post-Training": [f"post/p{num}_hook_left_post_{i:03d}.csv" for i in range(1, 13)],
    }
    # -------------------------------------------

    base_path = Path(base_file_name)
    if not base_path.exists():
        print(f"오류: 기준 파일을 찾을 수 없습니다: {base_path}")
        return

    try:
        df_base = pd.read_csv(base_path)
        print(f"기준 파일 '{base_path.name}' 로딩 완료.")
    except Exception as e:
        print(f"오류: 기준 CSV 파일('{base_path.name}')을 읽는 중 문제가 발생했습니다: {e}")
        return

    final_summary_data = {}

    # --- 각 그룹별로 분석 수행 ---
    for group_name, file_list in analysis_groups.items():
        print(f"\n{'='*70}")
        print(f"'{group_name}' 그룹 분석 시작...")
        print(f"{'='*70}")

        group_final_scores = []
        group_individual_scores = defaultdict(list)
        
        target_paths = [Path(p) for p in file_list]
        valid_files_in_group = [p for p in target_paths if p.exists()]
        
        if not valid_files_in_group:
            print(f">>> '{group_name}' 그룹에서 분석할 유효한 파일이 없습니다.")
            continue

        for i, target_path in enumerate(valid_files_in_group):
            print(f"[{i+1}/{len(valid_files_in_group)}] '{target_path}' 분석 중...")
            try:
                df_target = pd.read_csv(target_path)
                analysis_results = analyze_motion_similarity(df_base, df_target, plane=plane)
                
                final_score = analysis_results.get("final_score", 0.0)
                group_final_scores.append(final_score)
                print(f"  -> 최종 유사도: {final_score:.2%}")

                for metric, score in analysis_results.get("individual_scores", {}).items():
                    if score is not None:
                        group_individual_scores[metric].append(score)
            except Exception as e:
                print(f"  -> 오류: '{target_path.name}' 파일 분석 중 문제 발생: {e}")

        # --- 그룹별 결과 요약 및 평균 계산 ---
        if group_final_scores:
            final_average = sum(group_final_scores) / len(group_final_scores)
            individual_averages = {
                metric: sum(scores) / len(scores)
                for metric, scores in group_individual_scores.items()
            }
            final_summary_data[group_name] = {
                "final_average": final_average,
                "individual_averages": individual_averages
            }
            print(f"\n>>> '{group_name}' 그룹 분석 완료 ({len(group_final_scores)}개 파일).")
            print(f"    - 최종 평균 유사도: {final_average:.2%}")
        else:
            print(f">>> '{group_name}' 그룹에서 분석을 완료한 파일이 없습니다.")

    print(f"\n\n{'='*70}\n모든 분석이 완료되었습니다. 결과를 웹 브라우저에 표시합니다.\n{'='*70}")
    
    create_ui(final_summary_data, base_path.name)
    ui.run(title=f"분석 결과: '{base_path.name}' 기준", reload=False, port=8081)


if __name__ == '__main__':
    main()
