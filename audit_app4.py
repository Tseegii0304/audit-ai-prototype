"""
АУДИТЫН ХОУ ПРОТОТИП v3.2
ГҮЙЛГЭЭ_БАЛАНС + ЕДТ → TB + Ledger + Part1 → Шинжилгээ
pip install streamlit pandas numpy scikit-learn plotly openpyxl
streamlit run audit_app.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import IsolationForest, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import warnings, io, re, gzip
from datetime import datetime
warnings.filterwarnings('ignore')
st.set_page_config(page_title="Аудитын ХОУ v3.2", page_icon="🔍", layout="wide")
st.markdown('<style>.box-blue{background:#e3f2fd;padding:15px;border-radius:10px;border-left:4px solid #1565c0;margin:10px 0}.box-green{background:#e8f5e9;padding:15px;border-radius:10px;border-left:4px solid #4caf50;margin:10px 0}.box-orange{background:#fff3e0;padding:15px;border-radius:10px;border-left:4px solid #ff9800;margin:10px 0}</style>', unsafe_allow_html=True)
st.markdown('<h1 style="text-align:center;color:#1565c0">🔍 Аудитын ХОУ Прототип v3.2</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align:center;color:#666">ГҮЙЛГЭЭ_БАЛАНС + ЕДТ → TB + Ledger + Part1 → Шинжилгээ</p>', unsafe_allow_html=True)

with st.sidebar:
    st.header("📌 Цэс")
    page = st.radio("Алхам:", ["1️⃣ Өгөгдөл бэлтгэх", "2️⃣ Шинжилгээ"])
    st.markdown("---")
    st.markdown("**1️⃣ Өгөгдөл бэлтгэх:**\n- 📗 ГҮЙЛГЭЭ_БАЛАНС → TB\n- 📘 ЕДТ → Ledger + Part1\n\n**2️⃣ Шинжилгээ:**\n- TB + Ledger → ХОУ")

# ═══ ФУНКЦУУД ═══
def process_raw_tb(file_obj, year):
    import openpyxl
    wb = openpyxl.load_workbook(file_obj, read_only=True); ws = wb[wb.sheetnames[0]]
    rows = []
    for row in ws.iter_rows(values_only=True):
        c0 = row[0]
        if c0 is None: continue
        try: int(float(c0))
        except: continue
        code = str(row[1]).strip() if row[1] else ''
        if not code or not re.match(r'\d{3}-', code): continue
        name = str(row[2]).strip() if row[2] else ''
        def num(v):
            if v is None or v == '': return 0.0
            try: return float(v)
            except: return 0.0
        rows.append({'account_code':code,'account_name':name,'opening_debit':num(row[3]),'opening_credit':num(row[4]),
            'turnover_debit':num(row[5]),'turnover_credit':num(row[6]),'closing_debit':num(row[7]),'closing_credit':num(row[8])})
    wb.close()
    df = pd.DataFrame(rows)
    df['opening_balance_signed'] = df['opening_debit']-df['opening_credit']
    df['turnover_net_signed'] = df['turnover_debit']-df['turnover_credit']
    df['closing_balance_signed'] = df['closing_debit']-df['closing_credit']
    df['net_change_signed'] = df['closing_balance_signed']-df['opening_balance_signed']
    tb_clean = df[['account_code','account_name','opening_debit','opening_credit','turnover_debit','turnover_credit','closing_debit','closing_credit']].copy()
    tb_sum = df[['account_code','account_name','opening_debit','opening_credit','opening_balance_signed','turnover_debit','turnover_credit','turnover_net_signed','closing_debit','closing_credit','closing_balance_signed','net_change_signed']].copy()
    tt=df['turnover_debit']+df['turnover_credit']; ca=df['closing_balance_signed'].abs()
    df['turnover_total']=tt; df['closing_abs']=ca; p75t=tt.quantile(0.75); p75b=ca.quantile(0.75)
    df['risk_high_turn']=(tt>p75t).astype(int); df['risk_large_bal']=(ca>p75b).astype(int); df['risk_score']=df['risk_high_turn']+df['risk_large_bal']
    risk=df[['account_code','account_name','turnover_debit','turnover_credit','closing_debit','closing_credit','turnover_total','closing_abs','risk_high_turn','risk_large_bal','risk_score']].copy()
    checks=pd.DataFrame({'check':['Эргэлт Д=К','Эхний Д=К','Эцсийн Д=К','Данс'],
        'value':[df['turnover_debit'].sum()-df['turnover_credit'].sum(),df['opening_debit'].sum()-df['opening_credit'].sum(),df['closing_debit'].sum()-df['closing_credit'].sum(),len(df)],
        'status':['✓' if abs(df['turnover_debit'].sum()-df['turnover_credit'].sum())<1 else f"⚠{df['turnover_debit'].sum()-df['turnover_credit'].sum():,.0f}",
                  '~','~',f'{len(df)}']})
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine='openpyxl') as w:
        tb_clean.to_excel(w,sheet_name='01_TB_CLEAN',index=False); tb_sum.to_excel(w,sheet_name='02_ACCOUNT_SUMMARY',index=False)
        risk.to_excel(w,sheet_name='04_RISK_PROXY_TB',index=False); checks.to_excel(w,sheet_name='05_CHECKS',index=False)
    buf.seek(0)
    return buf, tb_sum, checks

ACCT_RE_B = re.compile(r'Данс:\s*\[([^\]]+)\]\s*(.*)')
ACCT_RE_P = re.compile(r'Данс:\s*(\d{3}-\d{2}-\d{2}-\d{3})\s+(.*)')
def parse_account(text):
    m = ACCT_RE_B.match(text)
    if m: return m.group(1).strip(), m.group(2).strip()
    m = ACCT_RE_P.match(text)
    if m: return m.group(1).strip(), m.group(2).strip()
    return None, None

def process_edt(file_obj, report_year):
    import openpyxl
    wb = openpyxl.load_workbook(file_obj, read_only=True); ws = wb[wb.sheetnames[0]]
    rows_out, cur_code, cur_name = [], None, None
    for row in ws.iter_rows(values_only=True):
        c0 = row[0]
        if c0 is None: continue
        s = str(c0).strip()
        if s.startswith('Данс:'):
            code, name = parse_account(s)
            if code: cur_code, cur_name = code, name
            continue
        if any(s.startswith(x) for x in ['Компани:','ЕРӨНХИЙ','Тайлант','Үүсгэсэн','Журнал:','№','Эцсийн','Дт -','Нийт','Эхний']) or s in ('Валютаар','Төгрөгөөр',''): continue
        try: tx_no = int(float(c0))
        except: continue
        if cur_code is None: continue
        td = row[1] if len(row)>1 else ''
        tx_date = td.strftime('%Y-%m-%d') if isinstance(td, datetime) else (str(td).strip() if td else '')
        rows_out.append({'report_year':str(report_year),'account_code':cur_code,'account_name':cur_name,'transaction_no':str(tx_no),'transaction_date':tx_date,
            'journal_no':str(row[5]).strip() if len(row)>5 and row[5] else '','document_no':str(row[6]).strip() if len(row)>6 and row[6] else '',
            'counterparty_name':str(row[3]).strip() if len(row)>3 and row[3] else '','counterparty_id':str(row[4]).strip() if len(row)>4 and row[4] else '',
            'transaction_description':str(row[7]).strip() if len(row)>7 and row[7] else '',
            'debit_mnt':float(row[9]) if len(row)>9 and row[9] is not None else 0.0,
            'credit_mnt':float(row[11]) if len(row)>11 and row[11] is not None else 0.0,
            'balance_mnt':float(row[13]) if len(row)>13 and row[13] is not None else 0.0,
            'month':tx_date[:7] if len(tx_date)>=7 else ''})
    wb.close()
    return pd.DataFrame(rows_out), len(rows_out)

def generate_part1(df_led, year):
    """Ledger-ээс prototype_part1 XLSX үүсгэх: Monthly Summary + Account Summary + Risk Matrix"""
    df = df_led.copy()
    df['debit_mnt'] = pd.to_numeric(df['debit_mnt'], errors='coerce').fillna(0)
    df['credit_mnt'] = pd.to_numeric(df['credit_mnt'], errors='coerce').fillna(0)
    df['balance_mnt'] = pd.to_numeric(df['balance_mnt'], errors='coerce').fillna(0)
    yr = str(year)
    # 02_MONTHLY_SUMMARY
    monthly = df.groupby(['month','account_code']).agg(
        total_debit_mnt=('debit_mnt','sum'), total_credit_mnt=('credit_mnt','sum'),
        ending_balance_mnt=('balance_mnt','last'), transaction_count=('debit_mnt','count')
    ).reset_index()
    monthly.insert(0, 'report_year', yr)
    # 03_ACCOUNT_SUMMARY
    anames = df.groupby('account_code')['account_name'].first()
    first_bal = df.sort_values('transaction_date').groupby('account_code').agg(
        fb=('balance_mnt','first'), fd=('debit_mnt','first'), fc=('credit_mnt','first')).reset_index()
    first_bal['opening_balance_mnt'] = first_bal['fb'] - first_bal['fd'] + first_bal['fc']
    acct = df.groupby('account_code').agg(
        total_debit_mnt=('debit_mnt','sum'), total_credit_mnt=('credit_mnt','sum'),
        closing_balance_mnt=('balance_mnt','last')).reset_index()
    acct = acct.merge(first_bal[['account_code','opening_balance_mnt']], on='account_code', how='left')
    acct['account_name'] = acct['account_code'].map(anames)
    acct['net_change_mnt'] = acct['closing_balance_mnt'] - acct['opening_balance_mnt']
    acct.insert(0, 'report_year', yr)
    acct = acct[['report_year','account_code','account_name','opening_balance_mnt','total_debit_mnt','total_credit_mnt','closing_balance_mnt','net_change_mnt']]
    # 04_RISK_MATRIX
    rm = df.groupby(['month','account_code','counterparty_name']).agg(
        transaction_count=('debit_mnt','count'),
        total_amount_mnt=('debit_mnt', lambda x: x.abs().sum() + df.loc[x.index,'credit_mnt'].abs().sum()),
        avg_transaction_mnt=('debit_mnt', lambda x: (x.abs().sum()+df.loc[x.index,'credit_mnt'].abs().sum())/len(x) if len(x)>0 else 0),
        max_transaction_mnt=('debit_mnt', lambda x: max(x.abs().max(), df.loc[x.index,'credit_mnt'].abs().max()))
    ).reset_index()
    rm.insert(0, 'report_year', yr)
    # Risk flags
    p75_amt = rm['total_amount_mnt'].quantile(0.75); p75_cnt = rm['transaction_count'].quantile(0.75)
    rm['risk_flag_large_txn'] = (rm['total_amount_mnt']>p75_amt).astype(int)
    rm['risk_flag_high_frequency'] = (rm['transaction_count']>p75_cnt).astype(int)
    rm['risk_score'] = rm['risk_flag_large_txn'] + rm['risk_flag_high_frequency']
    rm['account_category'] = rm['account_code'].str[:1].map({'1':'Хөрөнгө','2':'Өр','3':'Эздийн өмч','4':'Зардал','5':'Орлого','6':'Орлого','7':'Зардал','8':'Бусад','9':'Нэгдсэн'}).fillna('')
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine='openpyxl') as w:
        monthly.to_excel(w, sheet_name='02_MONTHLY_SUMMARY', index=False)
        acct.to_excel(w, sheet_name='03_ACCOUNT_SUMMARY', index=False)
        rm.to_excel(w, sheet_name='04_RISK_MATRIX', index=False)
    buf.seek(0)
    n_risk = len(rm[rm['risk_score']>0])
    return buf, monthly, acct, rm, n_risk

def read_ledger(f):
    raw = f.read(); f.seek(0)
    if raw[:2] == b'\x1f\x8b': return pd.read_csv(io.StringIO(gzip.decompress(raw).decode('utf-8')), dtype={'account_code':str})
    return pd.read_csv(io.BytesIO(raw), dtype={'account_code':str})

def load_tb(files):
    frames, stats = [], {}
    for f in files:
        year = next((y for y in range(2020,2030) if str(y) in f.name), 2025)
        df = pd.read_excel(f, sheet_name='02_ACCOUNT_SUMMARY'); df['year'] = year
        for c in ['turnover_debit','turnover_credit','closing_debit','closing_credit','opening_debit','opening_credit']:
            if c in df.columns: df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
        if 'net_change_signed' in df.columns: df['net_change_signed'] = pd.to_numeric(df['net_change_signed'], errors='coerce').fillna(0)
        stats[year] = {'accounts':len(df),'turnover_d':df['turnover_debit'].sum(),'turnover_c':df['turnover_credit'].sum()}
        frames.append(df)
    return pd.concat(frames, ignore_index=True), stats

def load_ledger_stats(files):
    stats = {}
    for f in files:
        year = next((y for y in range(2020,2030) if str(y) in f.name), 2025)
        df = read_ledger(f)
        for c in ['debit_mnt','credit_mnt']: df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
        mo = df.groupby('month').agg(rows=('debit_mnt','count'),debit=('debit_mnt','sum'),credit=('credit_mnt','sum'))
        stats[year] = {'rows':len(df),'accounts':df['account_code'].nunique(),'months':df['month'].nunique(),'debit':df['debit_mnt'].sum(),'credit':df['credit_mnt'].sum(),'monthly':mo}
        del df
    return stats

def run_ml(tb_all, cont, n_est):
    df = tb_all.copy(); df['cat_code'] = df['account_code'].astype(str).str[:3]
    le = LabelEncoder(); df['cat_num'] = le.fit_transform(df['cat_code'])
    df['log_turn_d']=np.log1p(df['turnover_debit'].abs()); df['log_turn_c']=np.log1p(df['turnover_credit'].abs())
    df['log_close_d']=np.log1p(df['closing_debit'].abs()); df['log_close_c']=np.log1p(df['closing_credit'].abs())
    df['turn_ratio']=(df['turnover_debit']/df['turnover_credit'].replace(0,np.nan)).fillna(0).replace([np.inf,-np.inf],0)
    if 'net_change_signed' in df.columns: df['log_abs_change']=np.log1p(df['net_change_signed'].abs())
    else: df['log_abs_change']=np.log1p((df['closing_debit']-df['opening_debit']).abs())
    feats=['cat_num','log_turn_d','log_turn_c','log_close_d','log_close_c','turn_ratio','log_abs_change','year']
    X=df[feats].fillna(0).replace([np.inf,-np.inf],0)
    iso=IsolationForest(contamination=cont,random_state=42,n_estimators=200); df['iso_anomaly']=(iso.fit_predict(X)==-1).astype(int)
    sc=StandardScaler(); df['zscore_anomaly']=(np.abs(sc.fit_transform(X)).max(axis=1)>2.0).astype(int)
    p95=df['turn_ratio'].quantile(0.95); df['turn_anomaly']=((df['turn_ratio']>p95)|(df['turn_ratio']<-p95)).astype(int)
    df['ensemble_anomaly']=((df['iso_anomaly']==1)|((df['zscore_anomaly']==1)&(df['turn_anomaly']==1))).astype(int)
    y=df['ensemble_anomaly'].values; cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
    models={'Random Forest':RandomForestClassifier(n_estimators=n_est,max_depth=10,random_state=42,class_weight='balanced'),
            'Gradient Boosting':GradientBoostingClassifier(n_estimators=150,max_depth=5,learning_rate=0.1,random_state=42),
            'Logistic Regression':LogisticRegression(max_iter=1000,random_state=42,class_weight='balanced')}
    res={}
    for nm,mdl in models.items():
        yp=cross_val_predict(mdl,X,y,cv=cv,method='predict'); ypr=cross_val_predict(mdl,X,y,cv=cv,method='predict_proba')[:,1]
        res[nm]={'pred':yp,'prob':ypr,'precision':precision_score(y,yp),'recall':recall_score(y,yp),'f1':f1_score(y,yp),'auc':roc_auc_score(y,ypr)}
    best=max(res,key=lambda k:res[k]['f1']); rf=models['Random Forest']; rf.fit(X,y)
    fi=pd.DataFrame({'feature':feats,'importance':rf.feature_importances_}).sort_values('importance',ascending=False)
    nt=len(df); ns=int(nt*0.20); at=df['turnover_debit'].abs()+df['turnover_credit'].abs()
    wt=at/at.sum(); wt=wt.fillna(1/nt); np.random.seed(42)
    ms=np.zeros(nt,dtype=int); ms[np.random.choice(nt,size=ns,replace=False,p=wt.values)]=1; ym=(ms&y).astype(int)
    return df,X,y,feats,res,best,fi,ym

# ═══════════════════════════════════════
# 1️⃣ ӨГӨГДӨЛ БЭЛТГЭХ
# ═══════════════════════════════════════
if page.startswith("1"):
    st.header("1️⃣ Өгөгдөл бэлтгэх")
    tb_tab, led_tab = st.tabs(["📗 ГҮЙЛГЭЭ_БАЛАНС → TB", "📘 ЕДТ → Ledger + Part1"])
    
    with tb_tab:
        st.markdown('<div class="box-orange"><b>ГҮЙЛГЭЭ_БАЛАНС Excel → TB_standardized XLSX</b></div>', unsafe_allow_html=True)
        raw_tb = st.file_uploader("ГҮЙЛГЭЭ_БАЛАНС файлууд", type=['xlsx'], accept_multiple_files=True, key='raw_tb')
        if st.button("⚙️ TB бэлтгэх", type="primary", use_container_width=True, key='btn_tb') and raw_tb:
            if 'tb_res' not in st.session_state: st.session_state.tb_res = {}
            for f in raw_tb:
                year = next((y for y in range(2020,2030) if str(y) in f.name), 2025)
                with st.spinner(f"{year}..."): buf,tb_s,chk = process_raw_tb(f,year); st.session_state.tb_res[year] = {'buf':buf.getvalue(),'tb':tb_s,'chk':chk}
                st.success(f"✅ {year}: {len(tb_s):,} данс, {tb_s['turnover_debit'].sum()/1e9:,.2f}T₮")
        if 'tb_res' in st.session_state and st.session_state.tb_res:
            st.markdown("---")
            for yr in sorted(st.session_state.tb_res.keys()):
                d = st.session_state.tb_res[yr]
                with st.expander(f"📅 {yr} — {len(d['tb']):,} данс", expanded=True):
                    c1,c2 = st.columns(2)
                    with c1: st.metric("Данс",f"{len(d['tb']):,}"); st.metric("Дебит",f"{d['tb']['turnover_debit'].sum()/1e9:,.2f}T₮"); st.metric("Кредит",f"{d['tb']['turnover_credit'].sum()/1e9:,.2f}T₮")
                    with c2: st.dataframe(d['chk'],use_container_width=True,hide_index=True)
                    st.download_button(f"📥 TB_standardized_{yr}1231.xlsx",d['buf'],f"TB_standardized_{yr}1231.xlsx","application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",key=f"dtb{yr}")
    
    with led_tab:
        st.markdown('<div class="box-blue"><b>ЕДТ Excel → Ledger CSV + prototype_part1 XLSX</b></div>', unsafe_allow_html=True)
        st.markdown("ЕДТ файлуудаас **2 файл** үүсгэнэ:\n- `prototype_ledger_YYYY.csv` — гүйлгээний дэлгэрэнгүй\n- `prototype_part1_YYYY.xlsx` — сарын нэгтгэл + эрсдэлийн матриц")
        c1,c2,c3 = st.columns(3)
        with c1: st.markdown("**2023**"); edt23 = st.file_uploader("ЕДТ 2023",type=['xlsx'],key='e23')
        with c2: st.markdown("**2024** (нэг/олон)"); edt24 = st.file_uploader("ЕДТ 2024",type=['xlsx'],key='e24',accept_multiple_files=True)
        with c3: st.markdown("**2025** (нэг/олон)"); edt25 = st.file_uploader("ЕДТ 2025",type=['xlsx'],key='e25',accept_multiple_files=True)
        if st.button("⚙️ Ledger + Part1 бэлтгэх", type="primary", use_container_width=True, key='btn_led'):
            if 'led_res' not in st.session_state: st.session_state.led_res = {}
            if edt23:
                with st.spinner("2023 ЕДТ уншиж байна..."): d23,c23 = process_edt(edt23,2023); st.session_state.led_res[2023] = d23
                st.success(f"✅ 2023: {c23:,} гүйлгээ, {d23['account_code'].nunique():,} данс")
            if edt24:
                with st.spinner("2024 ЕДТ уншиж байна..."): fr=[process_edt(f,2024)[0] for f in edt24]; d24=pd.concat(fr,ignore_index=True); st.session_state.led_res[2024]=d24
                st.success(f"✅ 2024: {len(d24):,} гүйлгээ, {d24['account_code'].nunique():,} данс")
            if edt25:
                with st.spinner("2025 ЕДТ уншиж байна..."): fr=[process_edt(f,2025)[0] for f in edt25]; d25=pd.concat(fr,ignore_index=True); st.session_state.led_res[2025]=d25
                st.success(f"✅ 2025: {len(d25):,} гүйлгээ, {d25['account_code'].nunique():,} данс")
        if 'led_res' in st.session_state and st.session_state.led_res:
            st.markdown("---"); st.subheader("📊 Үр дүн ба татах")
            cols_out=['report_year','account_code','account_name','transaction_no','transaction_date','journal_no','document_no','counterparty_name','counterparty_id','transaction_description','debit_mnt','credit_mnt','balance_mnt','month']
            for yr in sorted(st.session_state.led_res.keys()):
                dfy = st.session_state.led_res[yr]
                dfy['debit_mnt']=pd.to_numeric(dfy['debit_mnt'],errors='coerce').fillna(0)
                dfy['credit_mnt']=pd.to_numeric(dfy['credit_mnt'],errors='coerce').fillna(0)
                with st.expander(f"📅 {yr} — {len(dfy):,} гүйлгээ, {dfy['month'].nunique()} сар", expanded=True):
                    mo=dfy.groupby('month').agg(rows=('debit_mnt','count'),debit=('debit_mnt','sum'),credit=('credit_mnt','sum'))
                    mo['debit_T']=mo['debit']/1e9; mo['credit_T']=mo['credit']/1e9
                    st.dataframe(mo[['rows','debit_T','credit_T']],use_container_width=True)
                    # Part1 үүсгэх
                    with st.spinner(f"{yr} Part1 үүсгэж байна..."):
                        p1_buf, p1_mo, p1_acct, p1_rm, n_risk = generate_part1(dfy, yr)
                    c1,c2,c3 = st.columns(3)
                    c1.metric("Сарын нэгтгэл",f"{len(p1_mo):,} мөр")
                    c2.metric("Эрсдэлийн хос",f"{len(p1_rm):,}")
                    c3.metric("Эрсдэлтэй (score>0)",f"{n_risk:,}")
                    # Татах - Ledger CSV
                    st.download_button(f"📥 prototype_ledger_{yr}.csv ({len(dfy):,} мөр)",
                        dfy[cols_out].to_csv(index=False).encode('utf-8'),
                        f"prototype_ledger_{yr}.csv","text/csv",key=f"dled{yr}")
                    # Татах - Part1 XLSX
                    st.download_button(f"📥 prototype_part1_{yr}.xlsx ({len(p1_rm):,} хос)",
                        p1_buf.getvalue(),
                        f"prototype_part1_{yr}.xlsx","application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",key=f"dp1{yr}")
            st.markdown('<div class="box-green">✅ Бүгд бэлэн! → <b>2️⃣ Шинжилгээ</b></div>',unsafe_allow_html=True)

# ═══════════════════════════════════════
# 2️⃣ ШИНЖИЛГЭЭ
# ═══════════════════════════════════════
elif page.startswith("2"):
    st.header("2️⃣ ХОУ Шинжилгээ")
    st.markdown('<div class="box-blue"><b>TB + Ledger → 3 давхаргат шинжилгээ</b></div>',unsafe_allow_html=True)
    cl,cr = st.columns(2)
    with cl: tb_files = st.file_uploader("TB (.xlsx)",type=['xlsx'],accept_multiple_files=True,key='tb3')
    with cr: led_files2 = st.file_uploader("Ledger (.csv)",type=['csv','gz'],accept_multiple_files=True,key='led3')
    c1,c2 = st.columns(2)
    with c1: cont = st.slider("IF contamination",0.05,0.20,0.10,0.01)
    with c2: nest = st.slider("RF n_estimators",50,500,200,50)
    if st.button("🚀 Шинжилгээ",type="primary",use_container_width=True) and tb_files and led_files2:
        with st.spinner("TB..."): tb_all,tb_st = load_tb(tb_files)
        with st.spinner("Ledger..."): led_st = load_ledger_stats(led_files2)
        with st.spinner("🤖 ХОУ..."): df,X,y,feats,res,best,fi,ym = run_ml(tb_all,cont,nest)
        st.success(f"✅ {len(df):,} данс, {sum(d['rows'] for d in led_st.values()):,} гүйлгээ")
        yrs=sorted(tb_st.keys()); bp=res[best]['pred']
        tab1,tab2,tab3,tab4,tab5 = st.tabs(["📊 Нэгтгэл","🔴 Аномали","⚔️ ХОУ vs MUS","🧠 XAI","📋 Жагсаалт"])
        with tab1:
            m1,m2,m3,m4=st.columns(4); m1.metric("Данс",f"{len(df):,}"); m2.metric("Гүйлгээ",f"{sum(d['rows'] for d in led_st.values()):,}")
            m3.metric("Аномали",f"{df['ensemble_anomaly'].sum():,} ({df['ensemble_anomaly'].mean()*100:.1f}%)"); m4.metric("Шилдэг",f"{best} F1={res[best]['f1']:.4f}")
            fg=make_subplots(rows=1,cols=3,subplot_titles=("Данс","Эргэлт (T₮)","ЕДТ мөр")); cl3=['#2196F3','#4CAF50','#FF9800']
            for i,yv in enumerate(yrs):
                fg.add_trace(go.Bar(x=[str(yv)],y=[tb_st[yv]['accounts']],marker_color=cl3[i%3],showlegend=False),row=1,col=1)
                fg.add_trace(go.Bar(x=[str(yv)],y=[tb_st[yv]['turnover_d']/1e9],marker_color=cl3[i%3],showlegend=False),row=1,col=2)
                if yv in led_st: fg.add_trace(go.Bar(x=[str(yv)],y=[led_st[yv]['rows']],marker_color=cl3[i%3],showlegend=False),row=1,col=3)
            fg.update_layout(height=350); st.plotly_chart(fg,use_container_width=True)
        with tab2:
            mt={'Isolation Forest':'iso_anomaly','Z-score':'zscore_anomaly','Turn ratio':'turn_anomaly','ENSEMBLE':'ensemble_anomaly'}
            ad=[{'Арга':m,'Нийт':int(df[c].sum()),**{str(yv):f"{int(df.loc[df['year']==yv,c].sum())} ({df.loc[df['year']==yv,c].sum()/len(df[df['year']==yv])*100:.1f}%)" for yv in yrs}} for m,c in mt.items()]
            st.dataframe(pd.DataFrame(ad),use_container_width=True,hide_index=True)
            st.plotly_chart(px.scatter(df,x='log_turn_d',y='log_abs_change',color=df['ensemble_anomaly'].map({0:'Хэвийн',1:'Аномали'}),facet_col='year',opacity=0.5,color_discrete_map={'Хэвийн':'#90caf9','Аномали':'#c62828'},height=400),use_container_width=True)
        with tab3:
            st.dataframe(pd.DataFrame([{'Загвар':n,'Precision':f"{r['precision']:.4f}",'Recall':f"{r['recall']:.4f}",'F1':f"{r['f1']:.4f}",'AUC':f"{r['auc']:.4f}"} for n,r in res.items()]),use_container_width=True,hide_index=True)
            fg2=go.Figure()
            for n,r in res.items(): fpr,tpr,_=roc_curve(y,r['prob']); fg2.add_trace(go.Scatter(x=fpr,y=tpr,name=f"{n} (AUC={r['auc']:.4f})"))
            fg2.add_trace(go.Scatter(x=[0,1],y=[0,1],name='Random',line=dict(dash='dash',color='gray'))); fg2.update_layout(title='ROC',height=400); st.plotly_chart(fg2,use_container_width=True)
            dr=[]
            for yv in yrs:
                mk=(df['year']==yv).values; yt=y[mk]; nt2=yt.sum()
                if nt2>0: a2=1-(bp[mk]&yt).sum()/nt2; m2x=1-(ym[mk]&yt).sum()/nt2
                else: a2=m2x=0
                dr.append({'Жил':yv,'ХОУ':f"{a2:.4f}",'MUS 20%':f"{m2x:.4f}",'Сайжрал':f"{m2x-a2:.4f}"})
            st.dataframe(pd.DataFrame(dr),use_container_width=True,hide_index=True)
        with tab4:
            st.plotly_chart(px.bar(fi,x='importance',y='feature',orientation='h',color='importance',color_continuous_scale='Blues',title='Feature Importance').update_layout(height=400,yaxis={'categoryorder':'total ascending'}),use_container_width=True)
            fd={'log_abs_change':'Он дамнасан цэвэр өөрчлөлт','turn_ratio':'Дебит-кредит харьцаа','log_turn_d':'Баримт дебит','log_turn_c':'Баримт кредит','log_close_d':'Эцсийн дебит','log_close_c':'Эцсийн кредит','cat_num':'Дансны ангилал','year':'Жил'}
            for _,r in fi.iterrows(): st.markdown(f"**{r['feature']}** ({r['importance']:.4f}): {fd.get(r['feature'],'')}")
        with tab5:
            adf=df[df['ensemble_anomaly']==1][['year','account_code','account_name','turnover_debit','turnover_credit','turn_ratio','log_abs_change']].copy()
            yf=st.selectbox("Жил",['Бүгд']+[str(y2) for y2 in yrs])
            if yf!='Бүгд': adf=adf[adf['year']==int(yf)]
            st.write(f"Нийт: {len(adf)}"); st.dataframe(adf,use_container_width=True,hide_index=True,height=500)
            st.download_button("📥 CSV",adf.to_csv(index=False).encode('utf-8-sig'),"anomaly.csv","text/csv")
    elif not tb_files or not led_files2:
        st.info("👆 TB + Ledger upload. Бэлэн биш бол **1️⃣** руу очно.")
