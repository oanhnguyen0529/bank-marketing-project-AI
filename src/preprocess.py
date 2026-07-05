"""
src/preprocess.py

Pineline "Tiền xử lý" - bước 3 trong khung 8 bước
    3. Tiền xử lý — Xử lý thiếu dữ liệu, mã hóa biến phân loại (One-Hot/Ordinal/Target Encoding), chuẩn hóa/scale.

    Input:
    bank-additional-full.csv

    Yêu cầu xử lý:
    - Data Cleaning:
        df.dropna()             Missing value
        df.drop_duplicates()    Trùng lặp
        df.replace()            Đồng bộ hoá
    - Data Transformation:
        Min-Max scalling/ Z-score               Normalization
        One-Hot Encoding(str.get_dummies())     Encoding
        Ordinal/Target                          Encoding
        df.astype()                             Chỉnh sửa kiểu dữ liệu

    Output:
    data/processed/bank_final.csv + data/processed/version_info.json
"""

import json
from datetime import datetime

import pandas as pd

raw_path = '../data/raw/bank-additional-full.csv'
out_path = '../data/processed/bank-final.csv'
out_version_path = '../data/processed/version_info.json'

numeric_cols = [
    'age',
    'duration',
    'campaign',
    'pdays',
    'previous',
    'emp.var.rate',
    'cons.price.idx',
    'cons.conf.idx',
    'euribor3m',
    'nr.employed'
]

categorical_cols = [
    'job',
    'marital',
    'education',
    'default',
    'housing',
    'loan',
    'contact',
    'month',
    'day_of_week',
    'poutcome'
]


# --------------------------------------------------------------------------- #
# 1. LOAD
# --------------------------------------------------------------------------- #

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=';')
    df.columns = df.columns.str.strip()
    return df


# --------------------------------------------------------------------------- #
# 2. Data Cleaning
# --------------------------------------------------------------------------- #

def clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    - Xử lý trùng lặp: drop_duplicates()
    - Xử lý giá trị thiếu (missing values): 'unknown'
    """

    log = {}
    rows_before = len(df)

    # --- Trùng lặp ---
    df = df.drop_duplicates()
    log['duplicates_removed'] = rows_before - len(df)

    # --- Đổi tên cột cho nhất quán ---
    df = df.rename(columns={
        'y': 'subscribed',
        'cons.price.idx': 'cons_price_idx',
        'cons.conf.idx': 'cons_conf_idx',
        'emp.var.rate': 'emp_var_rate',
        'nr.employed': 'nr_employed',
    })

    # --- Chuẩn hoá yes/no -> 1/0 ---
    df['subscribed'] = df['subscribed'].replace({
        'yes': 1,
        'no': 0
    }).astype(int)

    # --- Missing categorical ---
    categorical_missing = df[categorical_cols].isna().sum().to_dict()

    for col in categorical_cols:
        if df[col].isna().any():
            mode_value = df[col].mode(dropna=True)[0]
            df[col] = df[col].fillna(mode_value)

    log['cate_missing_filled'] = {
        k: int(v)
        for k, v in categorical_missing.items()
        if v > 0
    }

    log['keep_unknown'] = {
        col: int((df[col] == 'unknown').sum())
        for col in categorical_cols
        if (df[col] == 'unknown').any()
    }

    # --- Numeric ---
    for col in numeric_cols:
        if col.replace('.', '_') in df.columns:
            col = col.replace('.', '_')

        df[col] = pd.to_numeric(df[col], errors='coerce')

    numeric_cols_clean = [c.replace('.', '_') for c in numeric_cols]

    numeric_missing = df[numeric_cols_clean].isna().sum().to_dict()

    for col in numeric_cols_clean:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    log['nume_missing_filled'] = {
        k: int(v)
        for k, v in numeric_missing.items()
        if v > 0
    }

    # --- Xoá dòng rỗng hoàn toàn ---
    rows_before_drop = len(df)
    df = df.dropna(how='all')
    log['fully_empty'] = rows_before_drop - len(df)

    # --- duration = 0 ---
    rows_before_duration = len(df)
    df = df[df['duration'] != 0]
    log['duration_new'] = rows_before_duration - len(df)

    # --- Ép kiểu ---
    df[numeric_cols_clean] = df[numeric_cols_clean].astype(float)
    df['age'] = df['age'].astype(int)

    df = df.reset_index(drop=True)

    log['rows_before'] = rows_before
    log['rows_after'] = len(df)

    return df, log


# --------------------------------------------------------------------------- #
# 3. Data Transformation
# --------------------------------------------------------------------------- #

def normalize_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """
    Biến đổi dữ liệu có 3 pp là
        - Chuẩn hoá min-max
        - Chuẩn hoá z-score
        - Chuẩn hoá theo tỷ lệ thập phân
    """

    df = df.copy()

    for col in cols:
        std = df[col].std()
        df[col] = 0.0 if std == 0 else (df[col] - df[col].mean()) / std

    return df


def encode_onehot(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """
    Mã hoá One-Hot bằng get_dummies
    drop_first=True để tránh đa cộng tuyến
    """
    return pd.get_dummies(df, columns=cols, drop_first=True)


# --------------------------------------------------------------------------- #
# 4. XUẤT FILE + GHI VERSION
# --------------------------------------------------------------------------- #

def export_p(
    df: pd.DataFrame,
    csv_path: str,
    version_path: str,
    clean_log: dict,
    extra_notes: str = ''
):
    df.to_csv(csv_path, index=False, encoding='utf-8')

    version_info = {
        'exported_at': datetime.now().isoformat(timespec='seconds'),
        'source_file': raw_path,
        'output_file': csv_path,
        'n_rows': len(df),
        'n_cols': df.shape[1],
        'columns': list(df.columns),
        'cleaning_log': clean_log,
        'notes': extra_notes,
    }

    with open(version_path, 'w', encoding='utf-8') as f:
        json.dump(version_info, f, ensure_ascii=False, indent=2)


# --------------------------------------------------------------------------- #
# MAIN
# --------------------------------------------------------------------------- #

def main():
    raw = load_data(raw_path)

    clean_df, clean_log = clean_data(raw)

    numeric_cols_clean = [
        c.replace('.', '_')
        for c in numeric_cols
        if c != 'duration'
    ]

    normalize_df = normalize_numeric(
        clean_df,
        numeric_cols_clean
    )

    final_df = encode_onehot(
        normalize_df,
        categorical_cols
    )

    export_p(
        final_df,
        out_path,
        out_version_path,
        clean_log,
        extra_notes='Numeric chuẩn hóa z-score'
    )

    print('\n=== CLEANING LOG ===')

    for k, v in clean_log.items():
        print(f'{k}: {v}')


if __name__ == '__main__':
    main()
