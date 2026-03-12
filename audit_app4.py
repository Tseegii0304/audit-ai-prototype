
"""
Audit AI Universal App vNext
- Universal parser registry for TB / Ledger / Journal / Part1 ready files
- Rich risk rules
- ML upgrades: IsolationForest + OneClassSVM + shallow autoencoder-like reconstruction + supervised comparison
- Dashboard + export
- Unused account handling in TB preparation
"""
import io
import re
import gzip
import warnings
from datetime import datetime
from difflib import SequenceMatcher

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from sklearn.ensemble import IsolationForest, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import OneClassSVM
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Audit AI Universal", page_icon="🔎", layout="wide")
st.title("🔎 Audit AI Universal App")
st.caption("Universal parser + risk engine + ML anomaly detection + export")

with st.sidebar:
    st.header("⚙️ Цэс")
    page = st.radio("Хэсэг сонгох", ["1️⃣ Өгөгдөл бэлтгэх", "2️⃣ Шинжилгээ"])
    st.markdown("---")
    st.subheader("ML тохиргоо")
    contamination = st.slider("Isolation contamination", 0.01, 0.25, 0.08, 0.01)
    n_estimators = st.slider("Random Forest trees", 100, 500, 200, 50)
    fast_mode = st.checkbox("Fast mode (том өгөгдөлд sample ашиглах)", value=True)
    max_txn_rows = st.number_input("Transaction шинжилгээнд дээд мөр", min_value=20000, max_value=500000, value=120000, step=10000)

STANDARD_LEDGER_COLUMNS = [
    "report_year", "account_code", "account_name", "transaction_no", "transaction_date",
    "journal_no", "document_no", "counterparty_name", "counterparty_id",
    "transaction_description", "debit_mnt", "credit_mnt", "balance_mnt", "month"
]

TB_SUMMARY_COLUMNS = [
    "account_code","account_name","opening_debit","opening_credit","opening_balance_signed",
    "turnover_debit","turnover_credit","turnover_net_signed",
    "closing_debit","closing_credit","closing_balance_signed","net_change_signed"
]

COL_SYNONYMS = {
    "account_code": ["дансны код","данс код","account code","account no","account number","gl code","gl account","acct no","acc code","код","данс"],
    "account_name": ["дансны нэр","данс нэр","account name","acc name","нэр"],
    "transaction_date": ["огноо","date","transaction date","txn date","огноо гүйлгээ","гүйлгээний огноо"],
    "debit_mnt": ["debit amount","debit","дебит дүн","дебит","дт","dt","dr amount"],
    "credit_mnt": ["credit amount","credit","кредит дүн","кредит","кт","ct","cr amount"],
    "balance_mnt": ["үлдэгдэл","balance","ending balance","closing balance"],
    "counterparty_name": ["харилцагч","байгууллагын нэр","харилцагчийн нэр","байгууллага","counterparty","partner","vendor","customer"],
    "transaction_description": ["гүйлгээний утга","гүйлгээний агуулга","гүйлгээний тайлбар","утга","тайлбар","description","memo","narration"],
    "journal_no": ["журнал","journal","journal no","журналын төрөл"],
    "document_no": ["баримт","баримт №","баримтын дугаар","баримт дугаар","дугаар","№","document","doc no","document no","voucher"],
    "amount_mnt": ["мөнгөн дүн","гүйлгээний дүн","дүн төг","дүн","amount","txn amount","amount mnt"],
    "debit_account": ["debit account","debit acc","дебет данс","дебет","дт данс","дт"],
    "credit_account": ["credit account","credit acc","кредит данс","кредит","кт данс","кт"],
}

READY_FILE_LABELS = {
    "tb_std": "📊 TB standardized",
    "ledger": "📄 Ledger",
    "part1": "📈 Part1",
    "raw_tb": "📗 Raw TB",
    "edt": "📘 Journal / EDT",
    "unknown": "❓ Unknown",
}

def safe_float(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return 0.0
    s = str(v).strip().replace(",", "")
    if s == "":
        return 0.0
    try:
        return float(s)
    except Exception:
        return 0.0

def smart_year_from_name(name: str) -> int:
    for y in range(2020, 2031):
        if str(y) in name:
            return y
    return datetime.now().year

def normalize_date(v, fallback_year=None):
    if v is None or str(v).strip() == "":
        return ""
    if isinstance(v, (pd.Timestamp, datetime)):
        return pd.to_datetime(v).strftime("%Y-%m-%d")
    s = str(v).strip()
    fmts = ["%Y-%m-%d", "%Y.%m.%d", "%d.%m.%Y", "%d-%m-%Y", "%y.%m.%d", "%y-%m-%d", "%Y/%m/%d", "%d/%m/%Y"]
    for fmt in fmts:
        try:
            d = datetime.strptime(s[:10], fmt)
            if d.year < 100 and fallback_year:
                d = d.replace(year=fallback_year)
            return d.strftime("%Y-%m-%d")
        except Exception:
            continue
    try:
        return pd.to_datetime(v).strftime("%Y-%m-%d")
    except Exception:
        return s[:10]

def month_from_date(s):
    return s[:7] if isinstance(s, str) and len(s) >= 7 else ""

def fuzzy_score(a: str, b: str) -> float:
    a = str(a).lower().strip()
    b = str(b).lower().strip()
    if not a or not b:
        return 0.0
    if a in b or b in a:
        return 1.0
    return SequenceMatcher(None, a, b).ratio()

def best_column(headers, synonyms):
    best_col, best = None, 0.0
    for h in headers:
        for s in synonyms:
            sc = fuzzy_score(h, s)
            if sc > best:
                best = sc
                best_col = h
    return best_col if best >= 0.55 else None

def auto_map_columns(columns):
    headers = [str(c).strip() for c in columns]
    mapping = {}
    for std_col, synonyms in COL_SYNONYMS.items():
        col = best_column(headers, synonyms)
        if col:
            mapping[std_col] = col
    return mapping

def account_category(code: str) -> str:
    return {
        "1": "Хөрөнгө","2": "Өр төлбөр","3": "Эздийн өмч","4": "Орлого",
        "5": "Орлого","6": "Зардал","7": "Зардал","8": "Зардал","9": "Түр / Бусад",
    }.get(str(code)[:1], "Тодорхойгүй")

def is_account_code(value: str) -> bool:
    s = str(value).strip()
    return bool(re.match(r"^\d{3}(-\d{2}-\d{2}-\d{3})?$", s) or re.match(r"^\d{5,12}$", s))

def prepare_download_xlsx(sheets: dict) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        for sheet_name, df in sheets.items():
            safe_name = re.sub(r"[\\/*?:\[\]]", "_", sheet_name)[:31]
            df.to_excel(writer, sheet_name=safe_name, index=False)
    buf.seek(0)
    return buf.getvalue()

def sample_if_needed(df, limit):
    if len(df) <= limit:
        return df.copy()
    return df.sample(limit, random_state=42).copy()

def read_csv_flexible(file_obj):
    encodings = ["utf-8", "utf-8-sig", "cp1251", "latin1"]
    seps = [None, ",", ";", "\t", "|"]
    raw = file_obj.read()
    file_obj.seek(0)
    last_err = None
    for enc in encodings:
        for sep in seps:
            try:
                if raw[:2] == b"\x1f\x8b":
                    content = gzip.decompress(raw).decode(enc, errors="ignore")
                    df = pd.read_csv(io.StringIO(content), sep=sep, engine="python")
                else:
                    df = pd.read_csv(io.BytesIO(raw), sep=sep, engine="python", encoding=enc)
                if df.shape[1] >= 2:
                    return df
            except Exception as e:
                last_err = e
    if last_err:
        raise last_err
    raise ValueError("CSV уншиж чадсангүй")

def excel_sheets(file_obj):
    raw = file_obj.read()
    file_obj.seek(0)
    try:
        xls = pd.ExcelFile(io.BytesIO(raw))
        return xls.sheet_names, raw
    except Exception:
        return [], raw


def detect_header_row(df_preview, min_hits=3):
    """
    Preview DataFrame-ээс аль мөр нь жинхэнэ header болохыг танина.
    """
    best_idx, best_hits = 0, 0
    nrows = min(15, len(df_preview))
    for i in range(nrows):
        row_vals = [str(v).strip().lower() for v in df_preview.iloc[i].tolist() if pd.notna(v)]
        hits = 0
        for vals in COL_SYNONYMS.values():
            for token in vals:
                if any(token in rv for rv in row_vals):
                    hits += 1
                    break
        if hits > best_hits:
            best_hits = hits
            best_idx = i
    return best_idx if best_hits >= min_hits else 0

def read_excel_smart(raw_bytes, sheet_name, nrows=None):
    """
    Sheet-ийг эхлээд headerгүй preview хийгээд,
    дараа нь жинхэнэ header мөрийг олж дахин уншина.
    """
    preview = pd.read_excel(io.BytesIO(raw_bytes), sheet_name=sheet_name, header=None, nrows=20)
    hdr = detect_header_row(preview, min_hits=2)
    return pd.read_excel(io.BytesIO(raw_bytes), sheet_name=sheet_name, header=hdr, nrows=nrows)


def detect_file_type(file_obj):
    name = file_obj.name.lower()
    year = smart_year_from_name(file_obj.name)

    # ready files first
    if name.endswith(".csv.gz") or name.endswith(".gz"):
        if "prototype_ledger" in name or "ledger" in name:
            return "ledger", year

    if name.endswith(".csv"):
        if "ledger" in name:
            return "ledger", year

    if name.endswith(".xlsx") or name.endswith(".xls"):
        if "tb_standardized" in name:
            return "tb_std", year
        if "prototype_part1" in name or "part1" in name:
            return "part1", year

    sheets, raw = excel_sheets(file_obj)
    if sheets:
        low_sheets = {str(s).lower() for s in sheets}
        if "04_risk_matrix" in low_sheets:
            return "part1", year
        if "02_account_summary" in low_sheets:
            return "tb_std", year

        try:
            for s in sheets[:10]:
                df = read_excel_smart(raw, s, nrows=40)
                headers = [str(c).strip().lower() for c in df.columns]
                mapping = auto_map_columns(df.columns)
                headers_join = " | ".join(headers)

                # 1) journal / edt
                if {"debit_account", "credit_account", "amount_mnt"} <= set(mapping.keys()):
                    return "edt", year

                if ("огноо" in headers_join and "дебет" in headers_join and "кредит" in headers_join):
                    return "edt", year

                if ("огноо" in headers_join and "мөнгөн дүн" in headers_join):
                    return "edt", year

                if {"transaction_date", "transaction_description"} & set(mapping.keys()):
                    if "amount_mnt" in mapping or "debit_mnt" in mapping or "credit_mnt" in mapping:
                        return "edt", year

                # 2) raw TB
                if "account_code" in mapping:
                    tb_like = {"opening_debit","opening_credit","turnover_debit","turnover_credit","closing_debit","closing_credit"}
                    hit_tb = len(tb_like & set(mapping.keys()))
                    if hit_tb >= 2 or len(df.columns) >= 8:
                        return "raw_tb", year

                # 3) first-column content fallback
                sample_text = " | ".join(
                    str(v).strip().lower()
                    for row in df.head(15).values.tolist()
                    for v in row[:5]
                    if pd.notna(v)
                )
                if "данс:" in sample_text or "ерөнхий журнал" in sample_text or "журнал" in sample_text:
                    return "edt", year

                # 4) account-code fallback
                for col in df.columns[:5]:
                    vals = df[col].dropna().astype(str).head(20).tolist()
                    if sum(is_account_code(v) for v in vals) >= 3:
                        if "огноо" in headers_join or "тайлбар" in headers_join or "утга" in headers_join:
                            return "edt", year
                        return "raw_tb", year

        except Exception:
            pass

    if name.endswith(".csv") or name.endswith(".gz") or name.endswith(".txt"):
        try:
            df = read_csv_flexible(file_obj)
            mapping = auto_map_columns(df.columns)
            if {"account_code", "debit_mnt", "credit_mnt"} & set(mapping.keys()):
                return "ledger", year
        except Exception:
            pass

    return "unknown", year

def process_raw_tb(file_obj, drop_unused=True):
    raw = file_obj.read()
    file_obj.seek(0)
    df = pd.read_excel(io.BytesIO(raw), sheet_name=0)
    if df.shape[1] < 8:
        df = pd.read_excel(io.BytesIO(raw), sheet_name=0, header=None)
    rows = []
    for _, row in df.iterrows():
        vals = row.tolist()
        if len(vals) < 8:
            continue
        code = None
        idx_code = None
        for i, v in enumerate(vals[:4]):
            if is_account_code(v):
                code = str(v).strip()
                idx_code = i
                break
        if not code:
            continue
        name = str(vals[idx_code + 1]).strip() if idx_code is not None and idx_code + 1 < len(vals) and vals[idx_code+1] is not None else ""
        nums = [safe_float(v) for v in vals]
        if len(nums) < 6:
            continue
        open_d, open_c, turn_d, turn_c, close_d, close_c = nums[-6:]
        rows.append({
            "account_code": code,"account_name": name,
            "opening_debit": open_d,"opening_credit": open_c,
            "turnover_debit": turn_d,"turnover_credit": turn_c,
            "closing_debit": close_d,"closing_credit": close_c,
        })
    tb = pd.DataFrame(rows).drop_duplicates(subset=["account_code"], keep="last")
    if tb.empty:
        return io.BytesIO(), pd.DataFrame(), pd.DataFrame()
    tb["opening_balance_signed"] = tb["opening_debit"] - tb["opening_credit"]
    tb["turnover_net_signed"] = tb["turnover_debit"] - tb["turnover_credit"]
    tb["closing_balance_signed"] = tb["closing_debit"] - tb["closing_credit"]
    tb["net_change_signed"] = tb["closing_balance_signed"] - tb["opening_balance_signed"]
    tb["used_in_year"] = ((tb["turnover_debit"].abs() + tb["turnover_credit"].abs() + tb["net_change_signed"].abs()) > 0).astype(int)
    unused = tb[tb["used_in_year"] == 0].copy()
    used = tb[tb["used_in_year"] == 1].copy() if drop_unused else tb.copy()
    summary = used[TB_SUMMARY_COLUMNS].copy()
    xlsx = prepare_download_xlsx({
        "01_TB_CLEAN": used[["account_code","account_name","opening_debit","opening_credit","turnover_debit","turnover_credit","closing_debit","closing_credit"]],
        "02_ACCOUNT_SUMMARY": summary,
        "03_UNUSED_ACCOUNTS": unused[["account_code","account_name","opening_debit","opening_credit","turnover_debit","turnover_credit","closing_debit","closing_credit"]] if not unused.empty else pd.DataFrame(columns=["account_code","account_name"])
    })
    return io.BytesIO(xlsx), summary, unused

def parse_standard_dans_sections(df, report_year):
    rows = []
    cur_code, cur_name = None, None
    for _, row in df.iterrows():
        vals = row.tolist()
        c0 = vals[0] if len(vals) else None
        if c0 is None:
            continue
        s = str(c0).strip()
        m = re.match(r"Данс:\s*\[([^\]]+)\]\s*(.*)", s)
        if m:
            cur_code = m.group(1).strip()
            cur_name = m.group(2).strip()
            continue
        if not cur_code:
            continue
        try:
            tx_no = int(float(c0))
        except Exception:
            continue
        tx_date = normalize_date(vals[1] if len(vals) > 1 else "", report_year)
        rows.append({
            "report_year": str(report_year),"account_code": cur_code,"account_name": cur_name,
            "transaction_no": str(tx_no),"transaction_date": tx_date,
            "journal_no": str(vals[5]).strip() if len(vals) > 5 and pd.notna(vals[5]) else "",
            "document_no": str(vals[6]).strip() if len(vals) > 6 and pd.notna(vals[6]) else "",
            "counterparty_name": str(vals[3]).strip() if len(vals) > 3 and pd.notna(vals[3]) else "",
            "counterparty_id": str(vals[4]).strip() if len(vals) > 4 and pd.notna(vals[4]) else "",
            "transaction_description": str(vals[7]).strip() if len(vals) > 7 and pd.notna(vals[7]) else "",
            "debit_mnt": safe_float(vals[9]) if len(vals) > 9 else 0.0,
            "credit_mnt": safe_float(vals[11]) if len(vals) > 11 else 0.0,
            "balance_mnt": safe_float(vals[13]) if len(vals) > 13 else 0.0,
            "month": month_from_date(tx_date),
        })
    return pd.DataFrame(rows, columns=STANDARD_LEDGER_COLUMNS)

def parse_dual_entry_table(df, report_year):
    mapping = auto_map_columns(df.columns)
    if not {"debit_account","credit_account","amount_mnt"} <= set(mapping.keys()):
        return pd.DataFrame(columns=STANDARD_LEDGER_COLUMNS)
    rows = []
    for idx, r in df.iterrows():
        debit_acc = str(r[mapping["debit_account"]]).strip() if pd.notna(r[mapping["debit_account"]]) else ""
        credit_acc = str(r[mapping["credit_account"]]).strip() if pd.notna(r[mapping["credit_account"]]) else ""
        amount = safe_float(r[mapping["amount_mnt"]])
        if not is_account_code(debit_acc) or not is_account_code(credit_acc) or amount == 0:
            continue
        tx_date = normalize_date(r[mapping["transaction_date"]], report_year) if "transaction_date" in mapping else ""
        tx_no = str(r[mapping["document_no"]]).strip() if "document_no" in mapping and pd.notna(r[mapping["document_no"]]) else str(idx + 1)
        cp = str(r[mapping["counterparty_name"]]).strip() if "counterparty_name" in mapping and pd.notna(r[mapping["counterparty_name"]]) else ""
        desc = str(r[mapping["transaction_description"]]).strip() if "transaction_description" in mapping and pd.notna(r[mapping["transaction_description"]]) else ""
        journal = str(r[mapping["journal_no"]]).strip() if "journal_no" in mapping and pd.notna(r[mapping["journal_no"]]) else ""
        rows.append({"report_year": str(report_year),"account_code": debit_acc,"account_name": "","transaction_no": tx_no,"transaction_date": tx_date,"journal_no": journal,"document_no": tx_no,"counterparty_name": cp,"counterparty_id": "","transaction_description": desc,"debit_mnt": amount,"credit_mnt": 0.0,"balance_mnt": 0.0,"month": month_from_date(tx_date)})
        rows.append({"report_year": str(report_year),"account_code": credit_acc,"account_name": "","transaction_no": tx_no,"transaction_date": tx_date,"journal_no": journal,"document_no": tx_no,"counterparty_name": cp,"counterparty_id": "","transaction_description": desc,"debit_mnt": 0.0,"credit_mnt": amount,"balance_mnt": 0.0,"month": month_from_date(tx_date)})
    return pd.DataFrame(rows, columns=STANDARD_LEDGER_COLUMNS)

def parse_rowwise_ledger_table(df, report_year):
    mapping = auto_map_columns(df.columns)
    if not {"account_code","transaction_date"} <= set(mapping.keys()):
        return pd.DataFrame(columns=STANDARD_LEDGER_COLUMNS)
    if "debit_mnt" not in mapping and "credit_mnt" not in mapping and "amount_mnt" not in mapping:
        return pd.DataFrame(columns=STANDARD_LEDGER_COLUMNS)
    rows = []
    for idx, r in df.iterrows():
        acc = str(r[mapping["account_code"]]).strip() if pd.notna(r[mapping["account_code"]]) else ""
        if not is_account_code(acc):
            continue
        debit = safe_float(r[mapping["debit_mnt"]]) if "debit_mnt" in mapping else 0.0
        credit = safe_float(r[mapping["credit_mnt"]]) if "credit_mnt" in mapping else 0.0
        amount = safe_float(r[mapping["amount_mnt"]]) if "amount_mnt" in mapping else 0.0
        if debit == 0 and credit == 0 and amount != 0:
            txt = str(r[mapping["transaction_description"]]).lower() if "transaction_description" in mapping and pd.notna(r[mapping["transaction_description"]]) else ""
            if "кредит" in txt or "credit" in txt or "кт" in txt:
                credit = amount
            else:
                debit = amount
        if debit == 0 and credit == 0:
            continue
        tx_date = normalize_date(r[mapping["transaction_date"]], report_year)
        rows.append({
            "report_year": str(report_year),"account_code": acc,
            "account_name": str(r[mapping["account_name"]]).strip() if "account_name" in mapping and pd.notna(r[mapping["account_name"]]) else "",
            "transaction_no": str(idx + 1),"transaction_date": tx_date,
            "journal_no": str(r[mapping["journal_no"]]).strip() if "journal_no" in mapping and pd.notna(r[mapping["journal_no"]]) else "",
            "document_no": str(r[mapping["document_no"]]).strip() if "document_no" in mapping and pd.notna(r[mapping["document_no"]]) else "",
            "counterparty_name": str(r[mapping["counterparty_name"]]).strip() if "counterparty_name" in mapping and pd.notna(r[mapping["counterparty_name"]]) else "",
            "counterparty_id": "",
            "transaction_description": str(r[mapping["transaction_description"]]).strip() if "transaction_description" in mapping and pd.notna(r[mapping["transaction_description"]]) else "",
            "debit_mnt": debit,"credit_mnt": credit,
            "balance_mnt": safe_float(r[mapping["balance_mnt"]]) if "balance_mnt" in mapping else 0.0,
            "month": month_from_date(tx_date),
        })
    return pd.DataFrame(rows, columns=STANDARD_LEDGER_COLUMNS)

PARSERS = [parse_dual_entry_table, parse_rowwise_ledger_table, parse_standard_dans_sections]

def process_edt(file_obj, report_year):
    sheets, raw = excel_sheets(file_obj)
    best_df = pd.DataFrame(columns=STANDARD_LEDGER_COLUMNS)
    best_meta = {"parser": "none", "sheet": "", "rows": 0}
    for s in sheets:
        try:
            for header in [0, None]:
                df = pd.read_excel(io.BytesIO(raw), sheet_name=s, header=header)
                if header is None:
                    df.columns = [f"col_{i}" for i in range(df.shape[1])]
                for parser in PARSERS:
                    parsed = parser(df, report_year)
                    if len(parsed) > best_meta["rows"]:
                        best_df = parsed
                        best_meta = {"parser": parser.__name__, "sheet": s, "rows": len(parsed)}
        except Exception:
            continue
    return best_df, best_meta

def load_tb(files):
    frames, stats, unused_map = [], {}, {}
    for f in files:
        year = smart_year_from_name(f.name)
        df = pd.read_excel(f, sheet_name="02_ACCOUNT_SUMMARY")
        for c in ["turnover_debit","turnover_credit","closing_debit","closing_credit","opening_debit","opening_credit","net_change_signed"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
        df["year"] = year
        frames.append(df)
        stats[year] = {"accounts": int(len(df)),"turnover_d": float(df["turnover_debit"].sum()) if "turnover_debit" in df.columns else 0.0,"turnover_c": float(df["turnover_credit"].sum()) if "turnover_credit" in df.columns else 0.0}
        try:
            unused_map[year] = pd.read_excel(f, sheet_name="03_UNUSED_ACCOUNTS")
        except Exception:
            unused_map[year] = pd.DataFrame(columns=["account_code","account_name"])
    return (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()), stats, unused_map

def read_ledger_ready(file_obj):
    raw = file_obj.read()
    file_obj.seek(0)
    if raw[:2] == b"\x1f\x8b":
        return pd.read_csv(io.BytesIO(gzip.decompress(raw)), dtype={"account_code": str})
    return pd.read_csv(io.BytesIO(raw), dtype={"account_code": str})

def load_ledger_stats(files, fast=True, sample_limit=120000):
    stats = {}
    all_frames = []
    for f in files:
        year = smart_year_from_name(f.name)
        df = read_ledger_ready(f)
        for c in ["debit_mnt","credit_mnt","balance_mnt"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
        if "month" not in df.columns and "transaction_date" in df.columns:
            df["month"] = df["transaction_date"].astype(str).str[:7]
        stats[year] = {"rows": int(len(df)),"accounts": int(df["account_code"].astype(str).nunique()) if "account_code" in df.columns else 0,"months": int(df["month"].nunique()) if "month" in df.columns else 0}
        all_frames.append(sample_if_needed(df, sample_limit) if fast else df)
    return stats, (pd.concat(all_frames, ignore_index=True) if all_frames else pd.DataFrame())

def load_part1(files):
    rm_all, mo_all = [], []
    for f in files:
        year = smart_year_from_name(f.name)
        try:
            rm = pd.read_excel(f, sheet_name="04_RISK_MATRIX")
            rm["year"] = year
            rm_all.append(rm)
        except Exception:
            pass
        try:
            mo = pd.read_excel(f, sheet_name="02_MONTHLY_SUMMARY")
            mo["year"] = year
            mo_all.append(mo)
        except Exception:
            pass
    return pd.concat(rm_all, ignore_index=True) if rm_all else pd.DataFrame(), pd.concat(mo_all, ignore_index=True) if mo_all else pd.DataFrame()

def generate_part1(df_led, year):
    df = df_led.copy()
    for c in ["debit_mnt","credit_mnt","balance_mnt"]:
        if c not in df.columns:
            df[c] = 0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    for c in ["transaction_date","month","account_code","counterparty_name","document_no","journal_no","transaction_description","account_name"]:
        if c not in df.columns:
            df[c] = ""
        df[c] = df[c].astype(str)
    if df["month"].eq("").all():
        df["month"] = df["transaction_date"].str[:7]
    df["amount"] = df["debit_mnt"].abs() + df["credit_mnt"].abs()
    monthly = df.groupby(["month","account_code"], dropna=False).agg(
        total_debit_mnt=("debit_mnt","sum"),
        total_credit_mnt=("credit_mnt","sum"),
        ending_balance_mnt=("balance_mnt","last"),
        transaction_count=("amount","count"),
    ).reset_index()
    monthly.insert(0, "report_year", str(year))
    acct = df.groupby("account_code", dropna=False).agg(
        total_debit_mnt=("debit_mnt","sum"),
        total_credit_mnt=("credit_mnt","sum"),
        closing_balance_mnt=("balance_mnt","last"),
        transaction_count=("amount","count"),
    ).reset_index()
    acct["account_name"] = df.groupby("account_code")["account_name"].first().reindex(acct["account_code"]).values
    acct.insert(0, "report_year", str(year))
    rm = df.groupby(["month","account_code","counterparty_name"], dropna=False).agg(
        transaction_count=("amount","count"),
        total_debit=("debit_mnt","sum"),
        total_credit=("credit_mnt","sum"),
        total_amount_mnt=("amount","sum"),
        max_amount=("amount","max"),
        distinct_docs=("document_no", pd.Series.nunique),
        distinct_journals=("journal_no", pd.Series.nunique),
        empty_desc_ratio=("transaction_description", lambda s: (s.astype(str).str.strip() == "").mean()),
    ).reset_index()
    rm["counterparty_name"] = rm["counterparty_name"].fillna("").astype(str)
    rm["account_category"] = rm["account_code"].astype(str).map(account_category)
    p75_amt = rm["total_amount_mnt"].quantile(0.75) if len(rm) else 0
    p90_amt = rm["max_amount"].quantile(0.90) if len(rm) else 0
    p75_freq = rm["transaction_count"].quantile(0.75) if len(rm) else 0
    rm["risk_flag_large_txn"] = (rm["total_amount_mnt"] > p75_amt).astype(int)
    rm["risk_flag_high_frequency"] = (rm["transaction_count"] > p75_freq).astype(int)
    rm["risk_flag_single_large_item"] = (rm["max_amount"] > p90_amt).astype(int)
    rm["risk_flag_one_sided_activity"] = (((rm["total_debit"] == 0) ^ (rm["total_credit"] == 0))).astype(int)
    rm["risk_flag_missing_counterparty"] = (rm["counterparty_name"].str.strip() == "").astype(int)
    rm["risk_flag_low_document_diversity"] = ((rm["transaction_count"] >= 3) & (rm["distinct_docs"] <= 1)).astype(int)
    rm["risk_flag_low_journal_diversity"] = ((rm["transaction_count"] >= 3) & (rm["distinct_journals"] <= 1)).astype(int)
    rm["risk_flag_empty_description"] = (rm["empty_desc_ratio"] >= 0.8).astype(int)
    flags = [c for c in rm.columns if c.startswith("risk_flag_")]
    rm["risk_score"] = rm[flags].sum(axis=1)
    rm["risk_level"] = pd.cut(rm["risk_score"], bins=[-1,1,3,5,99], labels=["🟢 Бага","🟡 Дунд","🟠 Өндөр","🔴 Маш өндөр"])
    xlsx = prepare_download_xlsx({"02_MONTHLY_SUMMARY": monthly,"03_ACCOUNT_SUMMARY": acct,"04_RISK_MATRIX": rm})
    return io.BytesIO(xlsx), monthly, acct, rm

def engineer_txn_features(df):
    d = df.copy()
    defaults = {"debit_mnt": 0.0, "credit_mnt": 0.0, "balance_mnt": 0.0,"account_code": "", "account_name": "", "counterparty_name": "","transaction_description": "", "transaction_date": "", "document_no": "", "journal_no": ""}
    for c, v in defaults.items():
        if c not in d.columns:
            d[c] = v
    for c in ["debit_mnt","credit_mnt","balance_mnt"]:
        d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0)
    for c in ["account_code","account_name","counterparty_name","transaction_description","transaction_date","document_no","journal_no"]:
        d[c] = d[c].astype(str).fillna("")
    d["amount"] = d["debit_mnt"].abs() + d["credit_mnt"].abs()
    d["log_amount"] = np.log1p(d["amount"])
    d["is_debit"] = (d["debit_mnt"] > 0).astype(int)
    d["account_prefix"] = d["account_code"].str[:3]
    try:
        d["acct_cat_num"] = LabelEncoder().fit_transform(d["account_prefix"].fillna("000"))
    except Exception:
        d["acct_cat_num"] = 0
    digits = d["amount"].apply(lambda x: int(str(int(abs(x)))[0]) if abs(x) >= 1 else 0)
    d["benford_digit"] = digits
    benford_expected = {1:0.301,2:0.176,3:0.125,4:0.097,5:0.079,6:0.067,7:0.058,8:0.051,9:0.046}
    actual = d[d["benford_digit"] > 0]["benford_digit"].value_counts(normalize=True)
    d["benford_dev"] = d["benford_digit"].map(lambda x: abs(actual.get(x, 0) - benford_expected.get(x, 0)) if x > 0 else 0)
    d["is_round"] = (((d["amount"] >= 1000) & (d["amount"] % 1000 == 0)) | ((d["amount"] >= 1000000) & (d["amount"] % 1000000 == 0))).astype(int)
    d["missing_counterparty"] = (d["counterparty_name"].str.strip() == "").astype(int)
    d["missing_document"] = (d["document_no"].str.strip() == "").astype(int)
    d["missing_description"] = (d["transaction_description"].str.strip() == "").astype(int)
    d["doc_amt_key"] = d["document_no"] + "|" + d["amount"].round(2).astype(str)
    d["dup_doc_amount"] = d["doc_amt_key"].map(d["doc_amt_key"].value_counts()).fillna(0).astype(int)
    d["dup_like"] = (d["dup_doc_amount"] > 1).astype(int)
    d["pair_key"] = d["account_code"] + "|" + d["counterparty_name"]
    pair_freq = d["pair_key"].value_counts()
    d["rare_pair"] = (d["pair_key"].map(pair_freq).fillna(0) <= 2).astype(int)
    d["cp_freq"] = d["counterparty_name"].map(d["counterparty_name"].value_counts()).fillna(0)
    d["rare_counterparty"] = (d["cp_freq"] <= 3).astype(int)
    d["day"] = pd.to_numeric(d["transaction_date"].str[8:10], errors="coerce").fillna(15)
    d["month_num"] = pd.to_numeric(d["transaction_date"].str[5:7], errors="coerce").fillna(6)
    d["is_month_end"] = (d["day"] >= 28).astype(int)
    d["is_year_end"] = (d["month_num"] == 12).astype(int)
    parsed = pd.to_datetime(d["transaction_date"], errors="coerce")
    d["weekday"] = parsed.dt.weekday.fillna(2)
    d["is_weekend"] = (d["weekday"] >= 5).astype(int)
    acct_stats = d.groupby("account_code")["amount"].agg(["mean","std"]).fillna(0)
    acct_stats.columns = ["acct_mean","acct_std"]
    d = d.merge(acct_stats, on="account_code", how="left")
    d["amt_zscore"] = np.where(d["acct_std"] > 0, (d["amount"] - d["acct_mean"]) / d["acct_std"], 0.0)
    d["amt_zscore"] = d["amt_zscore"].replace([np.inf, -np.inf], 0).fillna(0).clip(-10, 10)
    d["dir_mismatch"] = 0
    first = d["account_code"].str[:1]
    d.loc[(first == "1") & (d["credit_mnt"] > 0) & (d["debit_mnt"] == 0), "dir_mismatch"] = 1
    d.loc[(first == "2") & (d["debit_mnt"] > 0) & (d["credit_mnt"] == 0), "dir_mismatch"] = 1
    d.loc[(first.isin(["4","5"])) & (d["debit_mnt"] > 0) & (d["credit_mnt"] == 0), "dir_mismatch"] = 1
    d.loc[(first.isin(["6","7","8"])) & (d["credit_mnt"] > 0) & (d["debit_mnt"] == 0), "dir_mismatch"] = 1
    stop = {"данс","гүйлгээ","журнал","баримт","бусад","нийт","тоо","дүн","төлбөр"}
    def kw(text):
        words = re.findall(r"[а-яөүёА-ЯӨҮЁA-Za-z0-9]{3,}", str(text).lower())
        return {w for w in words if w not in stop}
    d["name_no_overlap"] = [int(len(kw(a) & kw(b)) == 0 and len(kw(a)) > 0 and len(kw(b)) > 0) for a, b in zip(d["account_name"], d["transaction_description"])]
    return d

def shallow_autoencoder_score(X_scaled):
    hidden = max(2, min(8, X_scaled.shape[1] // 2))
    ae = MLPRegressor(hidden_layer_sizes=(hidden,), activation="relu", max_iter=200, random_state=42)
    ae.fit(X_scaled, X_scaled)
    pred = ae.predict(X_scaled)
    return ((X_scaled - pred) ** 2).mean(axis=1)

def run_txn_models(df, contamination=0.08):
    d = engineer_txn_features(df)
    feats = ["log_amount","acct_cat_num","benford_dev","is_round","amt_zscore","rare_counterparty","rare_pair","missing_counterparty","missing_document","missing_description","dup_like","is_debit","is_month_end","is_year_end","is_weekend","dir_mismatch","name_no_overlap"]
    X = d[feats].fillna(0).replace([np.inf, -np.inf], 0).astype(float)
    Xs = StandardScaler().fit_transform(X)
    iso = IsolationForest(contamination=contamination, random_state=42, n_estimators=200)
    d["iforest_flag"] = (iso.fit_predict(X) == -1).astype(int)
    d["iforest_score"] = -iso.score_samples(X)
    try:
        svm = OneClassSVM(nu=min(max(contamination, 0.01), 0.2), kernel="rbf", gamma="scale")
        d["ocsvm_flag"] = (svm.fit_predict(Xs) == -1).astype(int)
    except Exception:
        d["ocsvm_flag"] = 0
    try:
        ae_err = shallow_autoencoder_score(Xs)
        thr = np.quantile(ae_err, 1 - contamination)
        d["autoenc_flag"] = (ae_err >= thr).astype(int)
        d["autoenc_score"] = ae_err
    except Exception:
        d["autoenc_flag"] = 0
        d["autoenc_score"] = 0.0
    d["rule_score"] = (d["dup_like"] * 2 + d["missing_counterparty"] + d["missing_document"] + d["rare_pair"] + d["rare_counterparty"] + d["is_round"] + d["is_month_end"] + (d["amt_zscore"].abs() > 3).astype(int) * 2 + d["dir_mismatch"] * 2 + d["name_no_overlap"])
    d["rule_flag"] = (d["rule_score"] >= 4).astype(int)
    d["ensemble_unsup"] = ((d[["iforest_flag","ocsvm_flag","autoenc_flag","rule_flag"]].sum(axis=1)) >= 2).astype(int)
    d["txn_risk"] = d[["iforest_flag","ocsvm_flag","autoenc_flag","rule_flag"]].sum(axis=1) + d["rule_score"]
    d["txn_risk_level"] = pd.cut(d["txn_risk"], bins=[-1,3,6,10,99], labels=["🟢 Бага","🟡 Дунд","🟠 Өндөр","🔴 Маш өндөр"])
    return d, feats

def run_tb_models(tb_all, n_estimators=200):
    df = tb_all.copy()
    if df.empty:
        return df, pd.DataFrame(), {}, None
    for c in ["turnover_debit","turnover_credit","closing_debit","closing_credit","opening_debit","opening_credit","net_change_signed","year"]:
        if c not in df.columns:
            df[c] = 0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    df["cat_code"] = df["account_code"].astype(str).str[:3]
    try:
        df["cat_num"] = LabelEncoder().fit_transform(df["cat_code"])
    except Exception:
        df["cat_num"] = 0
    df["log_turn_d"] = np.log1p(df["turnover_debit"].abs())
    df["log_turn_c"] = np.log1p(df["turnover_credit"].abs())
    df["log_close_d"] = np.log1p(df["closing_debit"].abs())
    df["log_close_c"] = np.log1p(df["closing_credit"].abs())
    df["turn_ratio"] = (df["turnover_debit"] / df["turnover_credit"].replace(0, np.nan)).replace([np.inf, -np.inf], 0).fillna(0)
    df["log_abs_change"] = np.log1p(df["net_change_signed"].abs())
    df["inactive_flag"] = ((df["turnover_debit"].abs() + df["turnover_credit"].abs()) == 0).astype(int)
    feats = ["cat_num","log_turn_d","log_turn_c","log_close_d","log_close_c","turn_ratio","log_abs_change","year","inactive_flag"]
    X = df[feats].fillna(0).astype(float)
    iso = IsolationForest(contamination=0.08, random_state=42, n_estimators=200)
    df["iso_anomaly"] = (iso.fit_predict(X) == -1).astype(int)
    zmax = np.abs(StandardScaler().fit_transform(X)).max(axis=1)
    df["zscore_anomaly"] = (zmax > 2.2).astype(int)
    p95 = np.quantile(np.abs(df["turn_ratio"]), 0.95) if len(df) else 0
    df["turn_anomaly"] = (np.abs(df["turn_ratio"]) > p95).astype(int)
    df["ensemble_anomaly"] = ((df["iso_anomaly"] + df["zscore_anomaly"] + df["turn_anomaly"]) >= 2).astype(int)
    y = df["ensemble_anomaly"].values
    results, fi, best_name = {}, pd.DataFrame(), None
    if len(np.unique(y)) > 1 and len(df) >= 50:
        cv_splits = min(5, max(2, int(np.bincount(y).min())))
        cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
        models = {
            "Random Forest": RandomForestClassifier(n_estimators=n_estimators, max_depth=10, random_state=42, class_weight="balanced"),
            "Gradient Boosting": GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, random_state=42),
            "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
        }
        for nm, mdl in models.items():
            yp = cross_val_predict(mdl, X, y, cv=cv, method="predict")
            ypr = cross_val_predict(mdl, X, y, cv=cv, method="predict_proba")[:, 1] if hasattr(mdl, "predict_proba") else yp.astype(float)
            results[nm] = {"precision": precision_score(y, yp, zero_division=0),"recall": recall_score(y, yp, zero_division=0),"f1": f1_score(y, yp, zero_division=0),"auc": roc_auc_score(y, ypr) if len(np.unique(ypr)) > 1 else np.nan}
        best_name = max(results, key=lambda k: results[k]["f1"])
        rf = models["Random Forest"].fit(X, y)
        fi = pd.DataFrame({"feature": feats, "importance": rf.feature_importances_}).sort_values("importance", ascending=False)
    return df, fi, results, best_name

def show_detection_table(detected_rows):
    if detected_rows:
        st.dataframe(pd.DataFrame(detected_rows), use_container_width=True, hide_index=True)

def dataframe_download_buttons(df, prefix):
    csv = df.to_csv(index=False).encode("utf-8-sig")
    xlsx = prepare_download_xlsx({prefix[:31]: df})
    c1, c2 = st.columns(2)
    c1.download_button(f"📥 {prefix}.csv", csv, file_name=f"{prefix}.csv", mime="text/csv")
    c2.download_button(f"📥 {prefix}.xlsx", xlsx, file_name=f"{prefix}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

if page.startswith("1"):
    st.header("1️⃣ Өгөгдөл бэлтгэх")
    st.info("Ямар ч accounting/export файл оруулж болно. Систем эхлээд төрлийг таньж, дараа нь хамгийн тохирох parser-ийг ашиглана.")
    drop_unused_accounts = st.checkbox("Тухайн жилд ашиглагдаагүй дансыг TB-ээс хасах", value=True)
    uploaded = st.file_uploader("Файлуудаа энд оруулна уу", type=["xlsx", "xls", "csv", "gz", "txt"], accept_multiple_files=True, key="prep_upload")
    if uploaded:
        show_detection_table([{"Файл": f.name, "Он": detect_file_type(f)[1], "Төрөл": READY_FILE_LABELS.get(detect_file_type(f)[0], detect_file_type(f)[0])} for f in uploaded])
        if st.button("⚙️ Хөрвүүлэлт эхлүүлэх", type="primary", use_container_width=True):
            outputs = []
            for f in uploaded:
                ftype, year = detect_file_type(f)
                if ftype == "raw_tb":
                    buf, tb_sum, unused = process_raw_tb(f, drop_unused=drop_unused_accounts)
                    outputs.append(("tb", year, f.name, buf.getvalue(), tb_sum, unused))
                elif ftype == "edt":
                    ledger_df, meta = process_edt(f, year)
                    if not ledger_df.empty:
                        p1_buf, mo, acct, rm = generate_part1(ledger_df, year)
                        outputs.append(("ledger_part1", year, f.name, (ledger_df.to_csv(index=False).encode("utf-8-sig"), p1_buf.getvalue(), meta), ledger_df, rm))
                    else:
                        outputs.append(("warn", year, f.name, None, pd.DataFrame(), pd.DataFrame()))
                else:
                    outputs.append(("info", year, f.name, None, pd.DataFrame(), pd.DataFrame()))
            st.session_state["prep_outputs"] = outputs
    if "prep_outputs" in st.session_state:
        st.markdown("---")
        st.subheader("📦 Хөрвүүлэлтийн үр дүн")
        for typ, year, name, payload, df1, df2 in st.session_state["prep_outputs"]:
            with st.expander(f"{year} — {name}", expanded=True):
                if typ == "tb":
                    st.success(f"TB хөрвүүлэлт амжилттай. Цэвэр данс: {len(df1):,}")
                    st.write("Ашиглагдаагүй дансны тоо:", int(len(df2)))
                    if not df2.empty:
                        st.dataframe(df2[["account_code","account_name"]], use_container_width=True, hide_index=True)
                    st.download_button(f"📥 TB_standardized_{year}.xlsx", payload, file_name=f"TB_standardized_{year}1231.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                elif typ == "ledger_part1":
                    ledger_csv, part1_xlsx, meta = payload
                    st.success(f"Ledger уншсан мөр: {len(df1):,}")
                    st.caption(f"Шилдэг parser: {meta['parser']} | sheet: {meta['sheet']} | rows: {meta['rows']:,}")
                    st.download_button(f"📥 prototype_ledger_{year}.csv", ledger_csv, file_name=f"prototype_ledger_{year}.csv", mime="text/csv")
                    st.download_button(f"📥 prototype_part1_{year}.xlsx", part1_xlsx, file_name=f"prototype_part1_{year}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                    st.dataframe(df1.head(20), use_container_width=True)
                elif typ == "warn":
                    st.warning("Гүйлгээ уншигдсангүй. Sheet/header форматыг шалгана уу.")
                else:
                    st.info("Энэ файл аль хэдийн ready-file эсвэл unknown байна.")

if page.startswith("2"):
    st.header("2️⃣ Шинжилгээ")
    all_files = st.file_uploader("Шинжилгээ хийх файлууд", type=["xlsx", "xls", "csv", "gz", "txt"], accept_multiple_files=True, key="analysis_upload")
    if all_files:
        detected = []
        tb_files, ledger_files, part1_files = [], [], []
        for f in all_files:
            ftype, year = detect_file_type(f)
            detected.append({"Файл": f.name, "Он": year, "Төрөл": READY_FILE_LABELS.get(ftype, ftype)})
            f.seek(0)
            if ftype == "tb_std":
                tb_files.append(f)
            elif ftype == "ledger":
                ledger_files.append(f)
            elif ftype == "part1":
                part1_files.append(f)
            elif ftype == "raw_tb":
                buf, _, _unused = process_raw_tb(f, drop_unused=True)
                buf.name = f"TB_standardized_{year}1231.xlsx"
                tb_files.append(buf)
            elif ftype == "edt":
                ledger_df, meta = process_edt(f, year)
                if not ledger_df.empty:
                    wrap = io.BytesIO(ledger_df.to_csv(index=False).encode("utf-8-sig"))
                    wrap.name = f"prototype_ledger_{year}.csv"
                    ledger_files.append(wrap)
                    p1_buf, _, _, _ = generate_part1(ledger_df, year)
                    p1_wrap = io.BytesIO(p1_buf.getvalue())
                    p1_wrap.name = f"prototype_part1_{year}.xlsx"
                    part1_files.append(p1_wrap)
                else:
                    st.warning(f"⚠️ {f.name} — ЕДТ гэж танигдсан боловч гүйлгээ уншигдсангүй.")
        show_detection_table(detected)
        tb_all, tb_stats, unused_map = load_tb(tb_files) if tb_files else (pd.DataFrame(), {}, {})
        led_stats, led_sample = load_ledger_stats(ledger_files, fast=fast_mode, sample_limit=max_txn_rows) if ledger_files else ({}, pd.DataFrame())
        rm_all, mo_all = load_part1(part1_files) if part1_files else (pd.DataFrame(), pd.DataFrame())
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("TB files", len(tb_files))
        c2.metric("Ledger files", len(ledger_files))
        c3.metric("Part1 files", len(part1_files))
        c4.metric("Unused accounts", int(sum(len(v) for v in unused_map.values())) if unused_map else 0)
        tabs = st.tabs(["📊 Ерөнхий тойм", "📚 TB шинжилгээ", "🔍 Transaction anomaly", "🧠 ML compare", "🔗 Risk matrix", "📈 Monthly trend", "🗂️ Ашиглагдаагүй данс", "📥 Export"])
        with tabs[0]:
            l, r = st.columns(2)
            if tb_stats:
                tb_stat_df = pd.DataFrame([{"year": y, **v} for y, v in tb_stats.items()]).sort_values("year")
                l.plotly_chart(px.bar(tb_stat_df, x="year", y="accounts", title="TB дахь дансны тоо"), use_container_width=True)
            if led_stats:
                led_stat_df = pd.DataFrame([{"year": y, **v} for y, v in led_stats.items()]).sort_values("year")
                r.plotly_chart(px.bar(led_stat_df, x="year", y="rows", title="Ledger мөрийн тоо"), use_container_width=True)
            if not rm_all.empty:
                risk_counts = rm_all["risk_level"].astype(str).value_counts().reset_index()
                risk_counts.columns = ["risk_level","count"]
                st.plotly_chart(px.pie(risk_counts, names="risk_level", values="count", title="Risk matrix тархалт"), use_container_width=True)
        with tabs[1]:
            if tb_all.empty:
                st.warning("TB файл оруулаагүй байна.")
            else:
                st.dataframe(tb_all.head(50), use_container_width=True)
                df_model, fi, model_res, best_name = run_tb_models(tb_all, n_estimators=n_estimators)
                c1, c2, c3 = st.columns(3)
                c1.metric("Нийт данс", len(df_model))
                c2.metric("Эрсдэлтэй данс", int(df_model["ensemble_anomaly"].sum()))
                c3.metric("Шилдэг model", best_name if best_name else "N/A")
                st.plotly_chart(px.scatter(df_model, x="log_turn_d", y="log_abs_change", color=df_model["ensemble_anomaly"].map({0:"Хэвийн",1:"Эрсдэлтэй"}), hover_data=["account_code","account_name","year"], title="TB anomaly scatter"), use_container_width=True)
                if not fi.empty:
                    st.plotly_chart(px.bar(fi, x="importance", y="feature", orientation="h", title="Feature importance"), use_container_width=True)
        with tabs[2]:
            if led_sample.empty:
                st.warning("Ledger файл оруулаагүй байна.")
            else:
                st.caption(f"Transaction шинжилгээнд ашигласан мөр: {len(led_sample):,} {'(sample)' if fast_mode else ''}")
                txn_df, txn_feats = run_txn_models(sample_if_needed(led_sample, max_txn_rows), contamination=contamination)
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Rows", len(txn_df))
                c2.metric("Ensemble flagged", int(txn_df["ensemble_unsup"].sum()))
                c3.metric("Rule flagged", int(txn_df["rule_flag"].sum()))
                c4.metric("Month-end flagged", int(txn_df["is_month_end"].sum()))
                plot_df = txn_df.sample(min(12000, len(txn_df)), random_state=42) if len(txn_df) > 12000 else txn_df
                st.plotly_chart(px.scatter(plot_df, x="log_amount", y="amt_zscore", color=plot_df["txn_risk_level"].astype(str), hover_data=["account_code","counterparty_name","document_no","transaction_description"], title="Transaction anomaly map"), use_container_width=True)
                risky = txn_df.sort_values(["txn_risk","iforest_score"], ascending=False).head(300)
                st.dataframe(risky[["transaction_date","account_code","account_name","counterparty_name","document_no","debit_mnt","credit_mnt","txn_risk","txn_risk_level","rule_score"]], use_container_width=True)
        with tabs[3]:
            if tb_all.empty:
                st.warning("TB data хэрэгтэй.")
            else:
                df_model, fi, model_res, best_name = run_tb_models(tb_all, n_estimators=n_estimators)
                if not model_res:
                    st.info("Supervised comparison хийхэд хангалттай variation алга.")
                else:
                    res_df = pd.DataFrame([{"model": k, **v} for k, v in model_res.items()])
                    st.dataframe(res_df, use_container_width=True, hide_index=True)
                    st.plotly_chart(px.bar(res_df.melt(id_vars="model", value_vars=["precision","recall","f1","auc"]), x="model", y="value", color="variable", barmode="group", title="Model metrics"), use_container_width=True)
        with tabs[4]:
            if rm_all.empty:
                st.warning("Part1 файл оруулаагүй байна.")
            else:
                risk_filter = st.multiselect("Risk level", sorted(rm_all["risk_level"].astype(str).dropna().unique().tolist()))
                rm_view = rm_all.copy()
                if risk_filter:
                    rm_view = rm_view[rm_view["risk_level"].astype(str).isin(risk_filter)]
                topn = st.slider("Хэдэн мөр харуулах", 20, 500, 100, 20, key="rm_topn")
                rm_view = rm_view.sort_values(["risk_score","total_amount_mnt"], ascending=False).head(topn)
                st.dataframe(rm_view, use_container_width=True)
                st.plotly_chart(px.scatter(rm_view, x="transaction_count", y="total_amount_mnt", size="risk_score", color=rm_view["risk_level"].astype(str), hover_data=["account_code","counterparty_name","month"], title="Risk matrix bubble"), use_container_width=True)
        with tabs[5]:
            if mo_all.empty:
                st.warning("Monthly summary байхгүй байна.")
            else:
                agg = mo_all.groupby(["year","month"], dropna=False).agg(total_debit_mnt=("total_debit_mnt","sum"), total_credit_mnt=("total_credit_mnt","sum"), transaction_count=("transaction_count","sum")).reset_index()
                st.plotly_chart(px.line(agg, x="month", y="total_debit_mnt", color="year", markers=True, title="Monthly debit trend"), use_container_width=True)
                st.plotly_chart(px.bar(agg, x="month", y="transaction_count", color="year", barmode="group", title="Monthly transaction count"), use_container_width=True)
        with tabs[6]:
            if not unused_map or all(v.empty for v in unused_map.values()):
                st.info("Ашиглагдаагүй дансны sheet байхгүй байна. Preparation дээр автоматаар үүсгэнэ.")
            else:
                years = sorted(unused_map.keys())
                yr = st.selectbox("Он", years)
                dfu = unused_map.get(yr, pd.DataFrame())
                st.write(f"{yr} оны ашиглагдаагүй данс: {len(dfu):,}")
                st.dataframe(dfu, use_container_width=True)
                if not dfu.empty:
                    dataframe_download_buttons(dfu, f"unused_accounts_{yr}")
        with tabs[7]:
            export_sheets = {}
            if not tb_all.empty:
                export_sheets["TB_ALL"] = tb_all
            if not led_sample.empty:
                txn_df, _ = run_txn_models(sample_if_needed(led_sample, max_txn_rows), contamination=contamination)
                export_sheets["TXN_RISK_TOP500"] = txn_df.sort_values(["txn_risk","iforest_score"], ascending=False).head(500)
            if not rm_all.empty:
                export_sheets["RISK_MATRIX_TOP500"] = rm_all.sort_values(["risk_score","total_amount_mnt"], ascending=False).head(500)
            if export_sheets:
                xlsx = prepare_download_xlsx(export_sheets)
                st.download_button("📥 audit_evidence_bundle.xlsx", xlsx, file_name="audit_evidence_bundle.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                for name, df in export_sheets.items():
                    with st.expander(name):
                        st.dataframe(df.head(50), use_container_width=True)
            else:
                st.info("Экспортлох өгөгдөл алга.")
