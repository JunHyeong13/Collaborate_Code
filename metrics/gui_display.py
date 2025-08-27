# gui_display.py
# 분석 결과를 NiceGUI 웹 인터페이스로 표시하는 모듈

from nicegui import ui

def create_ui(summary_data, base_filename):
    """
    분석 결과를 NiceGUI 웹 인터페이스로 생성합니다.
    - 탭 1: 그룹별 최종 평균 유사도 막대 그래프
    - 탭 2: 그룹별 세부 지표 평균 유사도 테이블
    """
    if not summary_data:
        ui.label("표시할 분석 결과가 없습니다.").classes('text-h5 text-negative m-auto')
        return

    group_names = list(summary_data.keys())

    # --- UI 상단 헤더 ---
    with ui.header(elevated=True).style('background-color: #3874c8').classes('items-center justify-between'):
        ui.label(f"모션 분석 결과: '{base_filename}' 기준").classes('text-h6')

    # --- 탭 구성 ---
    with ui.tabs().classes('w-full') as tabs:
        one = ui.tab('종합 결과 요약')
        two = ui.tab('세부 지표별 평균')

    with ui.tab_panels(tabs, value=one).classes('w-full'):
        # --- 탭 1: 종합 결과 그래프 ---
        with ui.tab_panel(one):
            ui.label('그룹별 최종 평균 유사도').classes('text-h5 text-center my-4')
            
            # --- FIX: Updated chart options for ECharts compatibility ---
            chart_options = {
                'title': {'text': None},
                'xAxis': {
                    'type': 'category',
                    'data': group_names
                },
                'yAxis': {
                    'type': 'value',
                    'min': 0,
                    'max': 1,
                    'axisLabel': {
                        'formatter': '{value}'
                    }
                },
                'series': [{
                    'name': '평균 유사도',
                    'type': 'bar',
                    'data': [data['final_average'] for data in summary_data.values()],
                    'label': {
                        'show': True,
                        'position': 'top',
                        'formatter': 'function(params) { return (params.value * 100).toFixed(2) + "%"; }',
                        'color': 'black'
                    }
                }],
                'color': ['#ff9999', '#99ff99', '#66b3ff']
            }
            ui.echart(chart_options).classes('w-full h-96')

        # --- 탭 2: 세부 지표 결과 테이블 ---
        with ui.tab_panel(two):
            ui.label('그룹별 세부 지표 평균 유사도').classes('text-h5 text-center my-4')
            
            all_metrics = sorted(list(set(
                metric for data in summary_data.values() 
                for metric in data['individual_averages'].keys()
            )))

            columns = [{'name': 'metric', 'label': '분석 지표', 'field': 'metric', 'sortable': True, 'align': 'left'}]
            for name in group_names:
                columns.append({'name': name, 'label': name, 'field': name, 'sortable': True})

            rows = []
            for metric_name in all_metrics:
                row_data = {'metric': metric_name}
                for group_name in group_names:
                    score = summary_data[group_name]['individual_averages'].get(metric_name, 0.0)
                    row_data[group_name] = f"{score*100:.2f}%"
                rows.append(row_data)

            ui.aggrid({
                'columnDefs': columns,
                'rowData': rows,
                'domLayout': 'autoHeight',
            }).classes('w-full')

# --- 이 파일을 직접 실행할 경우를 위한 테스트 코드 ---
if __name__ in {"__main__", "__mp_main__"}:
    # 테스트용 더미 데이터 생성
    dummy_summary_data = {
        "Pre-Training": {
            "final_average": 0.657,
            "individual_averages": {
                "발 너비 / 어깨너비 비율": 0.71,
                "스탠스 깊이 / 너비 비율": 0.62,
                "왼손 가드 높이 비율": 0.68,
                "오른손 가드 높이 비율": 0.62,
                "뻗기 (L) 왼손 리치 / 어깨너비 비율": 0.65,
                "뻗기 (R) 오른손 리치 / 어깨너비 비율": 0.66,
            }
        },
        "Main-Session": {
            "final_average": 0.882,
            "individual_averages": {
                "발 너비 / 어깨너비 비율": 0.91,
                "스탠스 깊이 / 너비 비율": 0.85,
                "왼손 가드 높이 비율": 0.89,
                "오른손 가드 높이 비율": 0.88,
                "뻗기 (L) 왼손 리치 / 어깨너비 비율": 0.87,
                "뻗기 (R) 오른손 리치 / 어깨너비 비율": 0.90,
            }
        },
        "Post-Training": {
            "final_average": 0.925,
            "individual_averages": {
                "발 너비 / 어깨너비 비율": 0.96,
                "스탠스 깊이 / 너비 비율": 0.89,
                "왼손 가드 높이 비율": 0.94,
                "오른손 가드 높이 비율": 0.93,
                "뻗기 (L) 왼손 리치 / 어깨너비 비율": 0.91,
                "뻗기 (R) 오른손 리치 / 어깨너비 비율": 0.92,
            }
        }
    }
    
    # UI 생성 함수 호출
    create_ui(dummy_summary_data, "expert_dummy.csv")
    
    # NiceGUI 서버 실행
    ui.run(title="GUI 모듈 테스트", reload=False)
