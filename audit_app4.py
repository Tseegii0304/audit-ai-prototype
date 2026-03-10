"""
АУДИТЫН ХОУ ПРОТОТИП v3.3
TB + Ledger + Part1 (Risk Matrix) бүрэн шинжилгээ
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
st.set_page_config(page_title="Аудитын ХОУ v3.3", page_icon="🔍", layout="wide")
st.markdown('<style>.box-blue{background:#e3f2fd;padding:15px;border-radius:10px;border-left:4px solid #1565c0;margin:10px 0}.box-green{background:#e8f5e9;padding:15px;border-radius:10px;border-left:4px solid #4caf50;margin:10px 0}.box-orange{background:#fff3e0;padding:15px;border-radius:10px;border-left:4px solid #ff9800;margin:10px 0}</style>', unsafe_allow_html=True)
st.markdown('<h1 style="text-align:center;color:#1565c0">🔍 Аудитын ХОУ Прототип v3.3</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align:center;color:#666">TB + Ledger + Part1 → Бүрэн шинжилгээ</p>', unsafe_allow_html=True)

with st.sidebar:
    st.header("📌 Цэс")
    page = st.radio("Алхам:", ["1️⃣ Өгөгдөл бэлтгэх", "2️⃣ Шинжилгээ"])
    st.markdown("---")
    st.markdown("**1️⃣** ГҮЙЛГЭЭ_БАЛАНС→TB, ЕДТ→Ledger+Part1\n\n**2️⃣** TB+Ledger+Part1 → ХОУ шинжилгээ")

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
    df['opening_balance_signed']=df['opening_debit']-df['opening_credit']
    df['turnover_net_signed']=df['turnover_debit']-df['turnover_credit']
    df['closing_balance_signed']=df['closing_debit']-df['closing_credit']
    df['net_change_signed']=df['closing_balance_signed']-df['opening_balance_signed']
    tb_clean=df[['account_code','account_name','opening_debit','opening_credit','turnover_debit','turnover_credit','closing_debit','closing_credit']].copy()
    tb_sum=df[['account_code','account_name','opening_debit','opening_credit','opening_balance_signed','turnover_debit','turnover_credit','turnover_net_signed','closing_debit','closing_credit','closing_balance_signed','net_change_signed']].copy()
    tt=df['turnover_debit']+df['turnover_credit']; ca=df['closing_balance_signed'].abs()
    df['turnover_total']=tt; df['closing_abs']=ca; p75t=tt.quantile(0.75); p75b=ca.quantile(0.75)
    df['risk_high_turn']=(tt>p75t).astype(int); df['risk_large_bal']=(ca>p75b).astype(int); df['risk_score']=df['risk_high_turn']+df['risk_large_bal']
    risk=df[['account_code','account_name','turnover_debit','turnover_credit','closing_debit','closing_credit','turnover_total','closing_abs','risk_high_turn','risk_large_bal','risk_score']].copy()
    checks=pd.DataFrame({'check':['Эргэлт Д=К','Эхний Д=К','Эцсийн Д=К','Данс'],
        'value':[df['turnover_debit'].sum()-df['turnover_credit'].sum(),df['opening_debit'].sum()-df['opening_credit'].sum(),df['closing_debit'].sum()-df['closing_credit'].sum(),len(df)],
        'status':['✓' if abs(df['turnover_debit'].sum()-df['turnover_credit'].sum())<1 else '⚠','~','~',f'{len(df)}']})
    buf=io.BytesIO()
    with pd.ExcelWriter(buf,engine='openpyxl') as w:
        tb_clean.to_excel(w,sheet_name='01_TB_CLEAN',index=False); tb_sum.to_excel(w,sheet_name='02_ACCOUNT_SUMMARY',index=False)
        risk.to_excel(w,sheet_name='04_RISK_PROXY_TB',index=False); checks.to_excel(w,sheet_name='05_CHECKS',index=False)
    buf.seek(0)
    return buf, tb_sum, checks

ACCT_RE_B=re.compile(r'Данс:\s*\[([^\]]+)\]\s*(.*)')
ACCT_RE_P=re.compile(r'Данс:\s*(\d{3}-\d{2}-\d{2}-\d{3})\s+(.*)')
def parse_account(text):
    m=ACCT_RE_B.match(text)
    if m: return m.group(1).strip(),m.group(2).strip()
    m=ACCT_RE_P.match(text)
    if m: return m.group(1).strip(),m.group(2).strip()
    return None,None

def process_edt(file_obj, report_year):
    import openpyxl
    wb=openpyxl.load_workbook(file_obj,read_only=True); ws=wb[wb.sheetnames[0]]
    rows_out,cur_code,cur_name=[],None,None
    for row in ws.iter_rows(values_only=True):
        c0=row[0]
        if c0 is None: continue
        s=str(c0).strip()
        if s.startswith('Данс:'):
            code,name=parse_account(s)
            if code: cur_code,cur_name=code,name
            continue
        if any(s.startswith(x) for x in ['Компани:','ЕРӨНХИЙ','Тайлант','Үүсгэсэн','Журнал:','№','Эцсийн','Дт -','Нийт','Эхний']) or s in ('Валютаар','Төгрөгөөр',''): continue
        try: tx_no=int(float(c0))
        except: continue
        if cur_code is None: continue
        td=row[1] if len(row)>1 else ''
        tx_date=td.strftime('%Y-%m-%d') if isinstance(td,datetime) else (str(td).strip() if td else '')
        rows_out.append({'report_year':str(report_year),'account_code':cur_code,'account_name':cur_name,'transaction_no':str(tx_no),'transaction_date':tx_date,
            'journal_no':str(row[5]).strip() if len(row)>5 and row[5] else '','document_no':str(row[6]).strip() if len(row)>6 and row[6] else '',
            'counterparty_name':str(row[3]).strip() if len(row)>3 and row[3] else '','counterparty_id':str(row[4]).strip() if len(row)>4 and row[4] else '',
            'transaction_description':str(row[7]).strip() if len(row)>7 and row[7] else '',
            'debit_mnt':float(row[9]) if len(row)>9 and row[9] is not None else 0.0,
            'credit_mnt':float(row[11]) if len(row)>11 and row[11] is not None else 0.0,
            'balance_mnt':float(row[13]) if len(row)>13 and row[13] is not None else 0.0,
            'month':tx_date[:7] if len(tx_date)>=7 else ''})
    wb.close()
    return pd.DataFrame(rows_out),len(rows_out)

def generate_part1(df_led, year):
    df=df_led.copy(); yr=str(year)
    df['debit_mnt']=pd.to_numeric(df['debit_mnt'],errors='coerce').fillna(0)
    df['credit_mnt']=pd.to_numeric(df['credit_mnt'],errors='coerce').fillna(0)
    df['balance_mnt']=pd.to_numeric(df['balance_mnt'],errors='coerce').fillna(0)
    monthly=df.groupby(['month','account_code']).agg(total_debit_mnt=('debit_mnt','sum'),total_credit_mnt=('credit_mnt','sum'),ending_balance_mnt=('balance_mnt','last'),transaction_count=('debit_mnt','count')).reset_index()
    monthly.insert(0,'report_year',yr)
    anames=df.groupby('account_code')['account_name'].first()
    fb=df.sort_values('transaction_date').groupby('account_code').agg(fb=('balance_mnt','first'),fd=('debit_mnt','first'),fc=('credit_mnt','first')).reset_index()
    fb['opening_balance_mnt']=fb['fb']-fb['fd']+fb['fc']
    acct=df.groupby('account_code').agg(total_debit_mnt=('debit_mnt','sum'),total_credit_mnt=('credit_mnt','sum'),closing_balance_mnt=('balance_mnt','last')).reset_index()
    acct=acct.merge(fb[['account_code','opening_balance_mnt']],on='account_code',how='left')
    acct['account_name']=acct['account_code'].map(anames); acct['net_change_mnt']=acct['closing_balance_mnt']-acct['opening_balance_mnt']
    acct.insert(0,'report_year',yr)
    acct=acct[['report_year','account_code','account_name','opening_balance_mnt','total_debit_mnt','total_credit_mnt','closing_balance_mnt','net_change_mnt']]
    rm=df.groupby(['month','account_code','counterparty_name']).agg(
        transaction_count=('debit_mnt','count'),
        total_amount_mnt=('debit_mnt',lambda x: x.abs().sum()+df.loc[x.index,'credit_mnt'].abs().sum()),
        max_transaction_mnt=('debit_mnt',lambda x: max(x.abs().max(),df.loc[x.index,'credit_mnt'].abs().max()))
    ).reset_index()
    rm.insert(0,'report_year',yr)
    rm['avg_transaction_mnt']=rm['total_amount_mnt']/rm['transaction_count'].replace(0,1)
    p75a=rm['total_amount_mnt'].quantile(0.75); p75c=rm['transaction_count'].quantile(0.75)
    rm['risk_flag_large_txn']=(rm['total_amount_mnt']>p75a).astype(int)
    rm['risk_flag_high_frequency']=(rm['transaction_count']>p75c).astype(int)
    rm['risk_score']=rm['risk_flag_large_txn']+rm['risk_flag_high_frequency']
    rm['account_category']=rm['account_code'].str[:1].map({'1':'Хөрөнгө','2':'Өр','3':'Эздийн өмч','4':'Зардал','5':'Орлого','6':'Орлого','7':'Зардал'}).fillna('')
    buf=io.BytesIO()
    with pd.ExcelWriter(buf,engine='openpyxl') as w:
        monthly.to_excel(w,sheet_name='02_MONTHLY_SUMMARY',index=False)
        acct.to_excel(w,sheet_name='03_ACCOUNT_SUMMARY',index=False)
        rm.to_excel(w,sheet_name='04_RISK_MATRIX',index=False)
    buf.seek(0)
    return buf,monthly,acct,rm,len(rm[rm['risk_score']>0])

def read_ledger(f):
    raw=f.read(); f.seek(0)
    if raw[:2]==b'\x1f\x8b': return pd.read_csv(io.StringIO(gzip.decompress(raw).decode('utf-8')),dtype={'account_code':str})
    return pd.read_csv(io.BytesIO(raw),dtype={'account_code':str})

def load_tb(files):
    frames,stats=[],{}
    for f in files:
        year=next((y for y in range(2020,2030) if str(y) in f.name),2025)
        df=pd.read_excel(f,sheet_name='02_ACCOUNT_SUMMARY'); df['year']=year
        for c in ['turnover_debit','turnover_credit','closing_debit','closing_credit','opening_debit','opening_credit']:
            if c in df.columns: df[c]=pd.to_numeric(df[c],errors='coerce').fillna(0)
        if 'net_change_signed' in df.columns: df['net_change_signed']=pd.to_numeric(df['net_change_signed'],errors='coerce').fillna(0)
        stats[year]={'accounts':len(df),'turnover_d':df['turnover_debit'].sum(),'turnover_c':df['turnover_credit'].sum()}
        frames.append(df)
    return pd.concat(frames,ignore_index=True),stats

def load_ledger_stats(files):
    stats={}
    for f in files:
        year=next((y for y in range(2020,2030) if str(y) in f.name),2025)
        df=read_ledger(f)
        for c in ['debit_mnt','credit_mnt']: df[c]=pd.to_numeric(df[c],errors='coerce').fillna(0)
        mo=df.groupby('month').agg(rows=('debit_mnt','count'),debit=('debit_mnt','sum'),credit=('credit_mnt','sum'))
        stats[year]={'rows':len(df),'accounts':df['account_code'].nunique(),'months':df['month'].nunique(),'debit':df['debit_mnt'].sum(),'credit':df['credit_mnt'].sum(),'monthly':mo}
        del df
    return stats

def load_part1(files):
    all_rm,all_mo=[],[]
    for f in files:
        year=next((y for y in range(2020,2030) if str(y) in f.name),2025)
        try:
            rm=pd.read_excel(f,sheet_name='04_RISK_MATRIX'); rm['year']=year; all_rm.append(rm)
        except: pass
        try:
            mo=pd.read_excel(f,sheet_name='02_MONTHLY_SUMMARY'); mo['year']=year; all_mo.append(mo)
        except: pass
    rm_all=pd.concat(all_rm,ignore_index=True) if all_rm else pd.DataFrame()
    mo_all=pd.concat(all_mo,ignore_index=True) if all_mo else pd.DataFrame()
    return rm_all,mo_all

def run_ml(tb_all,cont,n_est):
    df=tb_all.copy(); df['cat_code']=df['account_code'].astype(str).str[:3]
    le=LabelEncoder(); df['cat_num']=le.fit_transform(df['cat_code'])
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
    tb_tab,led_tab=st.tabs(["📗 ГҮЙЛГЭЭ_БАЛАНС → TB","📘 ЕДТ → Ledger + Part1"])
    with tb_tab:
        st.markdown('<div class="box-orange"><b>ГҮЙЛГЭЭ_БАЛАНС Excel → TB_standardized XLSX</b></div>',unsafe_allow_html=True)
        raw_tb=st.file_uploader("ГҮЙЛГЭЭ_БАЛАНС файлууд",type=['xlsx'],accept_multiple_files=True,key='raw_tb')
        if st.button("⚙️ TB бэлтгэх",type="primary",use_container_width=True,key='btn_tb') and raw_tb:
            if 'tb_res' not in st.session_state: st.session_state.tb_res={}
            for f in raw_tb:
                year=next((y for y in range(2020,2030) if str(y) in f.name),2025)
                with st.spinner(f"{year}..."): buf,tb_s,chk=process_raw_tb(f,year); st.session_state.tb_res[year]={'buf':buf.getvalue(),'tb':tb_s,'chk':chk}
                st.success(f"✅ {year}: {len(tb_s):,} данс, {tb_s['turnover_debit'].sum()/1e9:,.2f}T₮")
        if 'tb_res' in st.session_state and st.session_state.tb_res:
            st.markdown("---")
            for yr in sorted(st.session_state.tb_res.keys()):
                d=st.session_state.tb_res[yr]
                with st.expander(f"📅 {yr} — {len(d['tb']):,} данс",expanded=True):
                    c1,c2=st.columns(2)
                    with c1: st.metric("Данс",f"{len(d['tb']):,}"); st.metric("Дебит",f"{d['tb']['turnover_debit'].sum()/1e9:,.2f}T₮")
                    with c2: st.dataframe(d['chk'],use_container_width=True,hide_index=True)
                    st.download_button(f"📥 TB_standardized_{yr}1231.xlsx",d['buf'],f"TB_standardized_{yr}1231.xlsx","application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",key=f"dtb{yr}")
    with led_tab:
        st.markdown('<div class="box-blue"><b>ЕДТ Excel → Ledger CSV + prototype_part1 XLSX</b></div>',unsafe_allow_html=True)
        c1,c2,c3=st.columns(3)
        with c1: st.markdown("**2023**"); edt23=st.file_uploader("ЕДТ 2023",type=['xlsx'],key='e23')
        with c2: st.markdown("**2024**"); edt24=st.file_uploader("ЕДТ 2024",type=['xlsx'],key='e24',accept_multiple_files=True)
        with c3: st.markdown("**2025**"); edt25=st.file_uploader("ЕДТ 2025",type=['xlsx'],key='e25',accept_multiple_files=True)
        if st.button("⚙️ Ledger + Part1 бэлтгэх",type="primary",use_container_width=True,key='btn_led'):
            if 'led_res' not in st.session_state: st.session_state.led_res={}
            if edt23:
                with st.spinner("2023..."): d23,c23=process_edt(edt23,2023); st.session_state.led_res[2023]=d23
                st.success(f"✅ 2023: {c23:,} гүйлгээ")
            if edt24:
                with st.spinner("2024..."): fr=[process_edt(f,2024)[0] for f in edt24]; d24=pd.concat(fr,ignore_index=True); st.session_state.led_res[2024]=d24
                st.success(f"✅ 2024: {len(d24):,} гүйлгээ")
            if edt25:
                with st.spinner("2025..."): fr=[process_edt(f,2025)[0] for f in edt25]; d25=pd.concat(fr,ignore_index=True); st.session_state.led_res[2025]=d25
                st.success(f"✅ 2025: {len(d25):,} гүйлгээ")
        if 'led_res' in st.session_state and st.session_state.led_res:
            st.markdown("---")
            cols_out=['report_year','account_code','account_name','transaction_no','transaction_date','journal_no','document_no','counterparty_name','counterparty_id','transaction_description','debit_mnt','credit_mnt','balance_mnt','month']
            for yr in sorted(st.session_state.led_res.keys()):
                dfy=st.session_state.led_res[yr]
                dfy['debit_mnt']=pd.to_numeric(dfy['debit_mnt'],errors='coerce').fillna(0); dfy['credit_mnt']=pd.to_numeric(dfy['credit_mnt'],errors='coerce').fillna(0)
                with st.expander(f"📅 {yr} — {len(dfy):,} гүйлгээ",expanded=True):
                    with st.spinner(f"Part1 {yr}..."): p1_buf,p1_mo,p1_acct,p1_rm,n_risk=generate_part1(dfy,yr)
                    c1,c2,c3=st.columns(3); c1.metric("Гүйлгээ",f"{len(dfy):,}"); c2.metric("Эрсдэлийн хос",f"{len(p1_rm):,}"); c3.metric("Эрсдэлтэй",f"{n_risk:,}")
                    import gzip as gz2
                    csv_bytes = dfy[cols_out].to_csv(index=False).encode('utf-8')
                    gz_bytes = gz2.compress(csv_bytes)
                    st.caption(f"CSV: {len(csv_bytes)/1e6:.0f}MB → GZ: {len(gz_bytes)/1e6:.0f}MB ({len(gz_bytes)/len(csv_bytes)*100:.0f}%)")
                    st.download_button(f"📥 prototype_ledger_{yr}.csv.gz (шахсан)",gz_bytes,f"prototype_ledger_{yr}.csv.gz","application/gzip",key=f"dled{yr}")
                    st.download_button(f"📥 prototype_part1_{yr}.xlsx",p1_buf.getvalue(),f"prototype_part1_{yr}.xlsx","application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",key=f"dp1{yr}")

# ═══════════════════════════════════════
# 2️⃣ ШИНЖИЛГЭЭ
# ═══════════════════════════════════════
elif page.startswith("2"):
    st.header("2️⃣ ХОУ Шинжилгээ")
    st.markdown('<div class="box-blue"><b>TB + Ledger + Part1 → Бүрэн шинжилгээ (7 tab)</b></div>',unsafe_allow_html=True)
    c1,c2,c3=st.columns(3)
    with c1: tb_files=st.file_uploader("TB (.xlsx)",type=['xlsx'],accept_multiple_files=True,key='tb3')
    with c2: led_files=st.file_uploader("Ledger (.csv)",type=['csv','gz'],accept_multiple_files=True,key='led3')
    with c3: p1_files=st.file_uploader("Part1 (.xlsx)",type=['xlsx'],accept_multiple_files=True,key='p13',help="prototype_part1_YYYY.xlsx")
    c1,c2=st.columns(2)
    with c1: cont=st.slider("IF contamination",0.05,0.20,0.10,0.01)
    with c2: nest=st.slider("RF n_estimators",50,500,200,50)
    
    if st.button("🚀 Шинжилгээ",type="primary",use_container_width=True) and tb_files and led_files:
        with st.spinner("TB..."): tb_all,tb_st=load_tb(tb_files)
        with st.spinner("Ledger..."): led_st=load_ledger_stats(led_files)
        rm_all,mo_all=pd.DataFrame(),pd.DataFrame()
        if p1_files:
            with st.spinner("Part1..."): rm_all,mo_all=load_part1(p1_files)
        with st.spinner("🤖 ХОУ..."): df,X,y,feats,res,best,fi,ym=run_ml(tb_all,cont,nest)
        st.success(f"✅ {len(df):,} данс, {sum(d['rows'] for d in led_st.values()):,} гүйлгээ" + (f", {len(rm_all):,} эрсдэлийн хос" if len(rm_all)>0 else ""))
        yrs=sorted(tb_st.keys()); bp=res[best]['pred']
        
        has_rm = len(rm_all)>0
        has_mo = len(mo_all)>0
        tabs = ["📊 Нэгтгэл","🔴 Аномали","⚔️ ХОУ vs MUS","🧠 XAI","📋 Жагсаалт"]
        if has_rm: tabs.append("🎯 Эрсдэлийн матриц")
        if has_mo: tabs.append("📈 Сарын чиг хандлага")
        all_tabs = st.tabs(tabs)
        
        # TAB 1: Нэгтгэл
        with all_tabs[0]:
            m1,m2,m3,m4=st.columns(4); m1.metric("Данс",f"{len(df):,}"); m2.metric("Гүйлгээ",f"{sum(d['rows'] for d in led_st.values()):,}")
            m3.metric("Аномали",f"{df['ensemble_anomaly'].sum():,} ({df['ensemble_anomaly'].mean()*100:.1f}%)"); m4.metric("Шилдэг",f"{best} F1={res[best]['f1']:.4f}")
            if has_rm:
                m5,m6=st.columns(2); m5.metric("Эрсдэлийн хос (нийт)",f"{len(rm_all):,}")
                m6.metric("Эрсдэлтэй хос (score>0)",f"{len(rm_all[rm_all['risk_score']>0]):,}")
            fg=make_subplots(rows=1,cols=3,subplot_titles=("Данс","Эргэлт (T₮)","ЕДТ мөр")); cl3=['#2196F3','#4CAF50','#FF9800']
            for i,yv in enumerate(yrs):
                fg.add_trace(go.Bar(x=[str(yv)],y=[tb_st[yv]['accounts']],marker_color=cl3[i%3],showlegend=False),row=1,col=1)
                fg.add_trace(go.Bar(x=[str(yv)],y=[tb_st[yv]['turnover_d']/1e9],marker_color=cl3[i%3],showlegend=False),row=1,col=2)
                if yv in led_st: fg.add_trace(go.Bar(x=[str(yv)],y=[led_st[yv]['rows']],marker_color=cl3[i%3],showlegend=False),row=1,col=3)
            fg.update_layout(height=350); st.plotly_chart(fg,use_container_width=True)
        
        # TAB 2: Аномали
        with all_tabs[1]:
            mt={'Isolation Forest':'iso_anomaly','Z-score':'zscore_anomaly','Turn ratio':'turn_anomaly','ENSEMBLE':'ensemble_anomaly'}
            ad=[{'Арга':m,'Нийт':int(df[c].sum()),**{str(yv):f"{int(df.loc[df['year']==yv,c].sum())} ({df.loc[df['year']==yv,c].sum()/len(df[df['year']==yv])*100:.1f}%)" for yv in yrs}} for m,c in mt.items()]
            st.dataframe(pd.DataFrame(ad),use_container_width=True,hide_index=True)
            st.plotly_chart(px.scatter(df,x='log_turn_d',y='log_abs_change',color=df['ensemble_anomaly'].map({0:'Хэвийн',1:'Аномали'}),facet_col='year',opacity=0.5,color_discrete_map={'Хэвийн':'#90caf9','Аномали':'#c62828'},height=400),use_container_width=True)
        
        # TAB 3: ХОУ vs MUS
        with all_tabs[2]:
            st.dataframe(pd.DataFrame([{'Загвар':n,'Precision':f"{r['precision']:.4f}",'Recall':f"{r['recall']:.4f}",'F1':f"{r['f1']:.4f}",'AUC':f"{r['auc']:.4f}"} for n,r in res.items()]),use_container_width=True,hide_index=True)
            fg2=go.Figure()
            for n,r in res.items(): fpr,tpr,_=roc_curve(y,r['prob']); fg2.add_trace(go.Scatter(x=fpr,y=tpr,name=f"{n} (AUC={r['auc']:.4f})"))
            fg2.add_trace(go.Scatter(x=[0,1],y=[0,1],name='Random',line=dict(dash='dash',color='gray'))); fg2.update_layout(title='ROC',height=400); st.plotly_chart(fg2,use_container_width=True)
            st.subheader("Detection Risk")
            dr=[]
            for yv in yrs:
                mk=(df['year']==yv).values; yt=y[mk]; nt2=yt.sum()
                if nt2>0: a2=1-(bp[mk]&yt).sum()/nt2; m2x=1-(ym[mk]&yt).sum()/nt2
                else: a2=m2x=0
                dr.append({'Жил':yv,'ХОУ':f"{a2:.4f}",'MUS 20%':f"{m2x:.4f}",'Сайжрал':f"{m2x-a2:.4f}"})
            st.dataframe(pd.DataFrame(dr),use_container_width=True,hide_index=True)
        
        # TAB 4: XAI
        with all_tabs[3]:
            st.plotly_chart(px.bar(fi,x='importance',y='feature',orientation='h',color='importance',color_continuous_scale='Blues',title='Feature Importance').update_layout(height=400,yaxis={'categoryorder':'total ascending'}),use_container_width=True)
            fd={'log_abs_change':'Он дамнасан цэвэр өөрчлөлт','turn_ratio':'Дебит-кредит харьцаа','log_turn_d':'Баримт дебит','log_turn_c':'Баримт кредит','log_close_d':'Эцсийн дебит','log_close_c':'Эцсийн кредит','cat_num':'Дансны ангилал','year':'Жил'}
            for _,r in fi.iterrows(): st.markdown(f"**{r['feature']}** ({r['importance']:.4f}): {fd.get(r['feature'],'')}")
        
        # TAB 5: Жагсаалт
        with all_tabs[4]:
            adf=df[df['ensemble_anomaly']==1][['year','account_code','account_name','turnover_debit','turnover_credit','turn_ratio','log_abs_change']].copy()
            yf=st.selectbox("Жил",['Бүгд']+[str(y2) for y2 in yrs])
            if yf!='Бүгд': adf=adf[adf['year']==int(yf)]
            st.write(f"Нийт: {len(adf)}"); st.dataframe(adf,use_container_width=True,hide_index=True,height=500)
            st.download_button("📥 CSV",adf.to_csv(index=False).encode('utf-8-sig'),"anomaly.csv","text/csv")
        
        # TAB 6: Эрсдэлийн матриц (Part1-ээс)
        if has_rm:
            with all_tabs[5]:
                st.subheader("🎯 Эрсдэлийн матриц (Харилцагч × Данс)")
                rm_all['risk_score']=pd.to_numeric(rm_all['risk_score'],errors='coerce').fillna(0)
                rm_all['total_amount_mnt']=pd.to_numeric(rm_all['total_amount_mnt'],errors='coerce').fillna(0)
                rm_all['transaction_count']=pd.to_numeric(rm_all['transaction_count'],errors='coerce').fillna(0)
                
                # Нэгтгэл
                rm_summary=[]
                for yv in sorted(rm_all['year'].unique()):
                    rmy=rm_all[rm_all['year']==yv]
                    rm_summary.append({'Жил':yv,'Нийт хос':f"{len(rmy):,}",'Эрсдэлтэй (score>0)':f"{len(rmy[rmy['risk_score']>0]):,}",
                        'Эрсдэлтэй %':f"{len(rmy[rmy['risk_score']>0])/len(rmy)*100:.1f}%",
                        'Нийт дүн (T₮)':f"{rmy['total_amount_mnt'].sum()/1e9:,.1f}"})
                st.dataframe(pd.DataFrame(rm_summary),use_container_width=True,hide_index=True)
                
                # Жилийн өсөлтийн график
                fig_rm=go.Figure()
                for yv in sorted(rm_all['year'].unique()):
                    rmy=rm_all[rm_all['year']==yv]
                    fig_rm.add_trace(go.Bar(x=['Нийт хос','Эрсдэлтэй'],y=[len(rmy),len(rmy[rmy['risk_score']>0])],name=str(yv)))
                fig_rm.update_layout(title='Эрсдэлийн хосын тоо (жил бүр)',barmode='group',height=350)
                st.plotly_chart(fig_rm,use_container_width=True)
                
                # Топ 20 эрсдэлтэй харилцагч
                st.subheader("Хамгийн эрсдэлтэй 20 харилцагч")
                top_cp=rm_all.groupby('counterparty_name').agg(
                    total_txn=('transaction_count','sum'),total_amt=('total_amount_mnt','sum'),
                    avg_risk=('risk_score','mean'),accounts=('account_code','nunique')
                ).sort_values('total_amt',ascending=False).head(20).reset_index()
                top_cp['total_amt']=top_cp['total_amt'].apply(lambda x:f"{x/1e9:,.2f}T₮")
                top_cp['avg_risk']=top_cp['avg_risk'].apply(lambda x:f"{x:.2f}")
                top_cp.columns=['Харилцагч','Гүйлгээний тоо','Нийт дүн','Дундаж эрсдэл','Дансны тоо']
                st.dataframe(top_cp,use_container_width=True,hide_index=True)
                
                # Дансны ангиллаар
                st.subheader("Дансны ангиллын эрсдэлийн хуваарилалт")
                if 'account_category' in rm_all.columns:
                    cat_risk=rm_all.groupby('account_category').agg(pairs=('risk_score','count'),risky=('risk_score',lambda x:(x>0).sum())).reset_index()
                    cat_risk['pct']=(cat_risk['risky']/cat_risk['pairs']*100).round(1)
                    fig_cat=px.bar(cat_risk,x='account_category',y='pct',color='pct',color_continuous_scale='Reds',title='Ангилал бүрийн эрсдэлтэй хосын хувь (%)')
                    st.plotly_chart(fig_cat,use_container_width=True)
        
        # TAB 7: Сарын чиг хандлага (Part1-ээс)
        if has_mo:
            with all_tabs[5 if not has_rm else 6]:
                st.subheader("📈 Сарын чиг хандлага")
                mo_all['total_debit_mnt']=pd.to_numeric(mo_all['total_debit_mnt'],errors='coerce').fillna(0)
                mo_all['total_credit_mnt']=pd.to_numeric(mo_all['total_credit_mnt'],errors='coerce').fillna(0)
                mo_all['transaction_count']=pd.to_numeric(mo_all['transaction_count'],errors='coerce').fillna(0)
                
                mo_agg=mo_all.groupby('month').agg(debit=('total_debit_mnt','sum'),credit=('total_credit_mnt','sum'),txn=('transaction_count','sum')).reset_index()
                mo_agg['debit_T']=mo_agg['debit']/1e9; mo_agg['credit_T']=mo_agg['credit']/1e9
                
                fig_mo=make_subplots(rows=2,cols=1,subplot_titles=("Дебит/Кредит эргэлт (тэрбум₮)","Гүйлгээний тоо"),shared_xaxes=True)
                fig_mo.add_trace(go.Scatter(x=mo_agg['month'],y=mo_agg['debit_T'],name='Дебит',line=dict(color='#1565c0')),row=1,col=1)
                fig_mo.add_trace(go.Scatter(x=mo_agg['month'],y=mo_agg['credit_T'],name='Кредит',line=dict(color='#e57373')),row=1,col=1)
                fig_mo.add_trace(go.Bar(x=mo_agg['month'],y=mo_agg['txn'],name='Гүйлгээ',marker_color='#4CAF50'),row=2,col=1)
                fig_mo.update_layout(height=600); st.plotly_chart(fig_mo,use_container_width=True)
                
                # 12 сарын дундажаас хазайсан сарууд
                if len(mo_agg)>=6:
                    avg_txn=mo_agg['txn'].mean(); std_txn=mo_agg['txn'].std()
                    outlier_months=mo_agg[mo_agg['txn']>avg_txn+2*std_txn]
                    if len(outlier_months)>0:
                        st.warning(f"⚠ Дунджаас 2σ-аас их гүйлгээтэй сарууд: {', '.join(outlier_months['month'].tolist())}")
    
    elif not tb_files or not led_files:
        st.info("👆 TB + Ledger upload. Part1 нэмбэл эрсдэлийн матриц + сарын чиг хандлагын шинжилгээ нэмэгдэнэ.")
