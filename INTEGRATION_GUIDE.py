"""
INTEGRATION_GUIDE.py
====================
audit_app.py дотор tab_descriptions.py модулийг хэрхэн нэмэх заавар.

Алхам 1: tab_descriptions.py файлыг audit_app.py-тай нэг хавтаст хадгална.
Алхам 2: audit_app.py-ийн эхэнд import нэмнэ.
Алхам 3: Tab бүрийн эхэнд тайлбар, доор нь дүгнэлт нэмнэ.

Доорх кодыг audit_app.py-ийн ТОХИРОХ ХЭСЭГТ хуулна.
"""

# ============================================================
# АЛХАМ 1: audit_app.py-ийн ЭХЭНД нэмэх (import хэсэг)
# ============================================================

# Одоо байгаа import-уудын доор:
from tab_descriptions import TabDescriptions
td = TabDescriptions()


# ============================================================
# АЛХАМ 2: TAB БҮРТ НЭМЭХ КОД
# ============================================================

# Таны одоогийн кодонд ойролцоогоор иймэрхүү бүтэц байгаа:
#
#   tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
#       "📊 Нэгтгэл", "🔍 Аномали", "⚖️ ХОУ vs MUS",
#       "🧠 XAI", "📋 Жагсаалт", "🔗 Эрсдэлийн матриц", "📈 Сарын trend"
#   ])


# ────────────────────────────────────────────
# TAB 1: Нэгтгэл
# ────────────────────────────────────────────
# with tab1:
#     # ⬇️ НЭМЭХ: Табын ЭХЭНД тайлбар
#     td.show_summary_description(
#         n_accounts=len(tb_combined),
#         n_transactions=ledger_stats.get('total_rows', 0),
#         n_risk_pairs=len(risk_matrix) if risk_matrix is not None else 0
#     )
#
#     # ... одоо байгаа metric card, bar chart кодууд ...
#
#     # ⬇️ НЭМЭХ: Графикуудын ДООР дүгнэлт
#     td.show_summary_interpretation()


# ────────────────────────────────────────────
# TAB 2: Аномали
# ────────────────────────────────────────────
# with tab2:
#     # ⬇️ НЭМЭХ: Табын ЭХЭНД тайлбар
#     td.show_anomaly_description()
#
#     # ... одоо байгаа scatter plot, metric card кодууд ...
#
#     # ⬇️ НЭМЭХ: Графикуудын ДООР дүгнэлт
#     td.show_anomaly_interpretation(
#         n_if=int(results['if_count']),
#         n_zscore=int(results['zscore_count']),
#         n_turn=int(results['turn_count']),
#         n_ensemble=int(results['ensemble_count'])
#     )


# ────────────────────────────────────────────
# TAB 3: ХОУ vs MUS
# ────────────────────────────────────────────
# with tab3:
#     # ⬇️ НЭМЭХ: Табын ЭХЭНД тайлбар
#     td.show_ai_vs_mus_description()
#
#     # ... одоо байгаа ROC Curve, хүснэгт кодууд ...
#
#     # ⬇️ НЭМЭХ: Графикуудын ДООР дүгнэлт
#     td.show_ai_vs_mus_interpretation(
#         rf_f1=0.9789,
#         rf_auc=0.9998,
#         dr_ai="1.2-3.1%",
#         dr_mus="39.5-52.7%",
#         mcnemar_chi2=357.81
#     )


# ────────────────────────────────────────────
# TAB 4: XAI
# ────────────────────────────────────────────
# with tab4:
#     # ⬇️ НЭМЭХ: Табын ЭХЭНД тайлбар
#     td.show_xai_description()
#
#     # ... одоо байгаа Feature Importance bar chart ...
#
#     # ⬇️ НЭМЭХ: Bar chart-ийн ДООР нарийвчилсан шинж чанарын тайлбар
#     # feature_importances нь загвараас гарах dict байна
#     # Жишээ: {'log_abs_change': 0.5769, 'log_turn_d': 0.1489, ...}
#     td.show_xai_feature_details(feature_importances=fi_dict)
#
#     # ⬇️ НЭМЭХ: Хамгийн ДООР нэгдсэн дүгнэлт
#     td.show_xai_interpretation()


# ────────────────────────────────────────────
# TAB 5: Жагсаалт
# ────────────────────────────────────────────
# with tab5:
#     # ⬇️ НЭМЭХ: Табын ЭХЭНД тайлбар
#     td.show_list_description()
#
#     # ... одоо байгаа dataframe, filter, CSV download кодууд ...
#
#     # ⬇️ НЭМЭХ: Хүснэгтийн ДООР дүгнэлт
#     td.show_list_interpretation(n_anomalies=len(anomaly_df))


# ────────────────────────────────────────────
# TAB 6: Эрсдэлийн матриц
# ────────────────────────────────────────────
# with tab6:
#     # ⬇️ НЭМЭХ: Табын ЭХЭНД тайлбар
#     td.show_risk_matrix_description()
#
#     # ... одоо байгаа risk matrix chart кодууд ...
#
#     # ⬇️ НЭМЭХ: Графикийн ДООР дүгнэлт
#     td.show_risk_matrix_interpretation(
#         n_pairs=len(risk_matrix),
#         top_counterparty="Хамгийн өндөр эрсдэлтэй харилцагчийн нэр"
#     )


# ────────────────────────────────────────────
# TAB 7: Сарын trend
# ────────────────────────────────────────────
# with tab7:
#     # ⬇️ НЭМЭХ: Табын ЭХЭНД тайлбар
#     td.show_monthly_trend_description()
#
#     # ... одоо байгаа line chart, bar chart кодууд ...
#
#     # ⬇️ НЭМЭХ: Графикуудын ДООР дүгнэлт
#     td.show_monthly_trend_interpretation()


# ────────────────────────────────────────────
# ХАМГИЙН ДООР: Dashboard footer (бүх tab-ийн гадна)
# ────────────────────────────────────────────
# td.show_dashboard_footer()


# ============================================================
# АЛХАМ 3: requirements.txt-д нэмэлт шаардлагагүй
# (streamlit, plotly зэрэг аль хэдийн суусан)
# ============================================================

# ============================================================
# АЛХАМ 4: Streamlit Cloud дахин deploy хийх
# ============================================================
# 1. tab_descriptions.py файлыг GitHub repo-д push хийх
# 2. audit_app.py-д дээрх өөрчлөлтүүдийг оруулж commit хийх
# 3. Streamlit Cloud автоматаар шинэчлэгдэнэ
