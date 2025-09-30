# analysis/spss_rmanova_prep.py
import os
import re
import pandas as pd
from datetime import datetime

# ===== 사용자 설정 =====
directory_path = r'/Users/jonabi/Downloads/TEPA/Motion_weight_1.0'
output_dir = os.path.join(directory_path, 'DTW_for_spss')  # 기존 폴더 재사용
os.makedirs(output_dir, exist_ok=True)

columns_to_use = [
    'Head', 'Core', 'Right_Leg', 'Left_Leg', 'Right_Arm', 'Left_Arm',
    'Acceleration', 'Velocity', 'Position', 'Joint Angle', 'rotation', 'Overall'
]

# --- 조건 탐지 패턴 (대소문자 무시) ---
COND_PATTERNS = {
    'Pre': re.compile(r'pre', flags=re.I),
    'Main': re.compile(r'main', flags=re.I),
    'Post': re.compile(r'post', flags=re.I),
}

# --- 파일명 파서: subject, motion만 추출 (trial 제거) ---
# 예: p02_jab_001.csv / P26-straight.csv / p7.jab.12.csv
FNAME_RX = re.compile(
    r'(?P<subject>p?\d{1,3})[\._\-]+(?P<motion>[A-Za-z]+)',
    flags=re.I
)

# ---------- 정렬 유틸 (스칼라 키) ----------
def pad_numeric(s: str, width: int = 6) -> str:
    """
    'p2_jab_001' -> 'p000002_jab_000001'
    스칼라 문자열을 반환하여 pandas 정렬 시 안전하게 사용 가능.
    """
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return '{'  # 큰 문자로 채워 끝쪽으로 밀기
    s = str(s).lower()
    return re.sub(r'(\d+)', lambda m: f"{int(m.group(1)):0{width}d}", s)

def extract_subject_num(s) -> int:
    """
    'p02' -> 2, 'P7' -> 7, None/NaN -> 매우 큰 수(뒤로 밀기)
    """
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return 10**9
    m = re.search(r'(\d+)', str(s))
    return int(m.group(1)) if m else 10**9

def iter_csv_files(root: str):
    """하위 폴더 재귀 탐색 후, 경로 문자열을 pad_numeric으로 정렬."""
    all_paths = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.lower().endswith('.csv'):
                all_paths.append(os.path.join(dirpath, fn))
    all_paths.sort(key=lambda p: pad_numeric(p))
    return all_paths

# ---------- I/O & 전처리 ----------
def parse_identifiers(path_or_fname: str):
    """파일명에서 subject, motion만 파싱. 실패 시 None 처리."""
    stem = os.path.splitext(os.path.basename(path_or_fname))[0]
    m = FNAME_RX.search(stem)
    if m:
        subject = m.group('subject')
        motion = m.group('motion')
        return subject, motion, stem  # record_id = stem
    return None, None, stem

def robust_read_csv(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, header=0, index_col=0)
    except pd.errors.ParserError:
        df = pd.read_csv(path, header=0)
        df.index = df.iloc[:, 0].astype(str)
        df = df.iloc[:, 1:]
    df.columns = df.columns.str.strip()
    if isinstance(df.index, pd.MultiIndex):
        df.index = df.index.to_flat_index()
    df.index = df.index.map(lambda x: '' if pd.isna(x) else str(x))
    return df

def split_by_condition(df: pd.DataFrame) -> dict:
    out = {}
    idx = df.index.astype(str)
    for cond, rx in COND_PATTERNS.items():
        mask = idx.str.contains(rx)
        out[cond] = df.loc[mask] if mask.any() else pd.DataFrame(columns=df.columns)
    return out

def mean_by_condition(parts: dict, feature: str):
    vals = {}
    for cond in ['Pre', 'Main', 'Post']:
        part = parts.get(cond, pd.DataFrame())
        vals[cond] = float(part[feature].mean()) if (feature in part.columns and not part.empty) else float('nan')
    return vals

def build_long_row(subject, motion, record_id, feature, cond, value):
    return {
        'subject_id': subject,
        'motion': motion,
        'record_id': record_id,
        'feature': feature,
        'condition': cond,    # Pre/Main/Post
        'value': value
    }

def build_wide_row(subject, motion, record_id, feature, pre, main, post):
    d_mp = (main - pre) if pd.notna(main) and pd.notna(pre) else float('nan')   # Main - Pre
    d_pp = (post - pre) if pd.notna(post) and pd.notna(pre) else float('nan')   # Post - Pre
    d_pm = (post - main) if pd.notna(post) and pd.notna(main) else float('nan') # Post - Main
    return {
        'subject_id': subject,
        'motion': motion,
        'record_id': record_id,
        'feature': feature,
        'Pre': pre, 'Main': main, 'Post': post,
        'Delta_MinusPre': d_mp,
        'Delta_PostMinusPre': d_pp,
        'Delta_PostMinusMain': d_pm
    }

# ---------- 메인 ----------
def main():
    print("반복측정 분산분석(RM-ANOVA)용 데이터 정리를 시작합니다.")
    long_rows = []
    wide_rows_per_feature = {f: [] for f in columns_to_use}
    missing_report = []

    files = iter_csv_files(directory_path)
    if not files:
        print("CSV 파일이 없습니다. 경로를 확인해 주십시오.")
        return

    for fpath in files:
        try:
            df = robust_read_csv(fpath)
        except Exception as e:
            print(f"[경고] 파일을 읽는 중 오류가 발생했습니다: {fpath} -> {e}")
            continue

        subject, motion, record_id = parse_identifiers(fpath)
        parts = split_by_condition(df)

        for feature in columns_to_use:
            vals = mean_by_condition(parts, feature)
            pre, main, post = vals['Pre'], vals['Main'], vals['Post']

            # Long: Pre/Main/Post 3행
            for cond in ['Pre', 'Main', 'Post']:
                long_rows.append(
                    build_long_row(subject, motion, record_id, feature, cond, vals[cond])
                )

            # Wide: 1행
            wide_rows_per_feature[feature].append(
                build_wide_row(subject, motion, record_id, feature, pre, main, post)
            )

            # 결측 리포트
            miss = [c for c in ['Pre','Main','Post'] if pd.isna(vals[c])]
            if miss:
                missing_report.append(f"{fpath} | feature={feature} | missing={','.join(miss)}")

    # ----- 산출: Long -----
    long_df = pd.DataFrame(long_rows)

    # 타입 정규화
    long_df['subject_id'] = long_df['subject_id'].astype('string')
    long_df['motion']     = long_df['motion'].astype('string')
    long_df['feature']    = long_df['feature'].astype('string')
    long_df['record_id']  = long_df['record_id'].astype('string')

    # condition 순서 고정
    cond_order = pd.CategoricalDtype(categories=['Pre','Main','Post'], ordered=True)
    long_df['condition'] = long_df['condition'].astype(cond_order)

    # 보조 정렬 키(스칼라)
    long_df['_subj_num']   = long_df['subject_id'].map(extract_subject_num)
    long_df['_motion_key'] = long_df['motion'].map(pad_numeric)
    long_df['_feat_key']   = long_df['feature'].map(pad_numeric)
    long_df['_rec_key']    = long_df['record_id'].map(pad_numeric)

    long_df = long_df.sort_values(
        by=['_subj_num', '_motion_key', '_feat_key', 'condition', '_rec_key'],
        kind='mergesort'  # 안정 정렬
    ).reset_index(drop=True)

    # 보조 컬럼 제거
    long_df = long_df.drop(columns=['_subj_num','_motion_key','_feat_key','_rec_key'])

    long_out = os.path.join(output_dir, 'rmanova_long.csv')
    long_df.to_csv(long_out, index=False, encoding='utf-8-sig')
    print(f"[알림] Long 형식 저장: {long_out}")

    # ----- 산출: Wide (특징별 파일) -----
    wide_dir = os.path.join(output_dir, 'wide_per_feature')
    os.makedirs(wide_dir, exist_ok=True)

    for feature, rows in wide_rows_per_feature.items():
        wdf = pd.DataFrame(rows)

        wdf['subject_id'] = wdf['subject_id'].astype('string')
        wdf['motion']     = wdf['motion'].astype('string')
        wdf['feature']    = wdf['feature'].astype('string')
        wdf['record_id']  = wdf['record_id'].astype('string')

        wdf['_subj_num']   = wdf['subject_id'].map(extract_subject_num)
        wdf['_motion_key'] = wdf['motion'].map(pad_numeric)
        wdf['_feat_key']   = wdf['feature'].map(pad_numeric)
        wdf['_rec_key']    = wdf['record_id'].map(pad_numeric)

        wdf = wdf.sort_values(
            by=['_subj_num', '_motion_key', '_feat_key', '_rec_key'],
            kind='mergesort'
        ).reset_index(drop=True)

        wdf = wdf.drop(columns=['_subj_num','_motion_key','_feat_key','_rec_key'])

        wide_out = os.path.join(
            wide_dir,
            f'wide_feature={feature}.csv'.replace(' ', '_')
        )
        wdf.to_csv(wide_out, index=False, encoding='utf-8-sig')
        print(f"[알림] Wide 형식 저장: {wide_out}")

    # ----- 요약/결측 리포트 -----
    summary_path = os.path.join(output_dir, 'rmanova_summary.txt')
    with open(summary_path, 'w', encoding='utf-8') as fw:
        fw.write(f"Report generated at {datetime.now().isoformat()}\n")
        fw.write(f"Source directory: {directory_path}\n")
        fw.write(f"Total CSV files: {len(files)}\n")
        fw.write(f"Long rows: {len(long_df)}\n\n")

        if missing_report:
            fw.write("Missing condition means detected:\n")
            for line in missing_report[:5000]:
                fw.write(f"- {line}\n")
        else:
            fw.write("No missing condition means.\n")
    print(f"[알림] 요약 파일 저장: {summary_path}")
    print("처리가 완료되었습니다. 산출물을 확인해 주십시오.")

if __name__ == "__main__":
    main()
