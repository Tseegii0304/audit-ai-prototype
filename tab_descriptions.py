"""
tab_descriptions.py
====================
Аудитын ХОУ прототип вэб програмын 7 tab-ын тайлбар, тодорхойлолт модуль.
audit_app.py-д import хийж, таб бүрийн эхэнд дуудна.

Хэрэглээ:
    from tab_descriptions import TabDescriptions
    td = TabDescriptions()
    # Tab 1 дотор:
    td.show_summary_description(n_accounts=9909, n_transactions=3329189, n_risk_pairs=440789)
"""

import streamlit as st


class TabDescriptions:
    """Dashboard-ийн 7 tab-ын тайлбар, тодорхойлолтыг удирдах класс."""

    # ─────────────────────────────────────────────────────────────────
    # TAB 1 — Нэгтгэл (Summary)
    # ─────────────────────────────────────────────────────────────────
    def show_summary_description(self, n_accounts=0, n_transactions=0, n_risk_pairs=0):
        """Tab 1: Нэгтгэл хэсгийн тайлбарыг харуулна."""
        st.markdown("""
        <div style="background: linear-gradient(135deg, #e8f4fd 0%, #f0f7ff 100%);
             padding: 20px; border-radius: 12px; border-left: 5px solid #1976D2;
             margin-bottom: 20px;">
            <h4 style="color: #1565C0; margin-top: 0;">📊 Нэгтгэл — Шинжилгээний ерөнхий тойм</h4>
            <p style="color: #333; font-size: 14px; line-height: 1.7; margin-bottom: 0;">
                Энэ таб нь шинжилгээнд хамрагдсан <b>бүх өгөгдлийн ерөнхий тоймыг</b>
                metric card болон bar chart хэлбэрээр харуулна. Уламжлалт MUS 20% түүвэрлэлт нь
                зөвхөн ~20%-ийг хамардаг бол ХОУ загвар нь <b>100% дансыг бүрэн хамарна</b>.
                ISA 300 «Аудитын ерөнхий стратеги» стандартын дагуу аудитын хүрээг
                тодорхойлоход суурь мэдээлэл болно.
            </p>
        </div>
        """, unsafe_allow_html=True)

        with st.expander("ℹ️ Metric card-уудын тайлбар", expanded=False):
            st.markdown(f"""
| Metric | Утга | Аудитын ач холбогдол |
|--------|------|---------------------|
| **Нийт данс** | {n_accounts:,} | Шинжилгээнд хамрагдсан бүх данс (MUS 20%-тай харьцуулахад 5 дахин) |
| **Нийт эргэлт** | Дебит + Кредит нийлбэр | Байгууллагын санхүүгийн үйл ажиллагааны цар хүрээ |
| **ЕДТ мөр** | {n_transactions:,} | Бие даасан гүйлгээний тоо (Бенфордын шинжилгээнд ашигласан) |
| **Эрсдэлийн хос** | {n_risk_pairs:,} | Харилцагч×Данс×Сар хосуудын тоо (Part1 файлаас) |

*Өмнөх жилтэй харьцуулсан өөрчлөлтийн хувийг ногоон/улаан өнгөөр тэмдэглэнэ.*
            """)

    def show_summary_interpretation(self, stats_dict=None):
        """Tab 1: График/metric-ийн доор дүгнэлт/тайлбарыг харуулна."""
        st.markdown("""
        ---
        <div style="background-color: #FFF8E1; padding: 15px; border-radius: 8px;
             border-left: 4px solid #FFA000;">
            <b>💡 Аудиторт зориулсан тайлбар:</b> Дээрх metric card-ууд нь шинжилгээнд
            хамрагдсан өгөгдлийн бүрэн бүтэн байдлыг харуулна. Хэрэв ямар нэг жилийн
            дансны тоо огцом буурсан/нэмэгдсэн бол дансны бүтцийн өөрчлөлт (нэгтгэл,
            хуваагдал, шинэ данс нээлт) байгаа эсэхийг нягталж шалгана.
        </div>
        """, unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────────
    # TAB 2 — Аномали (Anomaly Detection)
    # ─────────────────────────────────────────────────────────────────
    def show_anomaly_description(self):
        """Tab 2: Аномали илрүүлэлтийн тайлбарыг харуулна."""
        st.markdown("""
        <div style="background: linear-gradient(135deg, #fce4ec 0%, #fff5f5 100%);
             padding: 20px; border-radius: 12px; border-left: 5px solid #c62828;
             margin-bottom: 20px;">
            <h4 style="color: #b71c1c; margin-top: 0;">🔍 Аномали — Ensemble илрүүлэлтийн үр дүн</h4>
            <p style="color: #333; font-size: 14px; line-height: 1.7; margin-bottom: 0;">
                Гурван бие даасан алгоритм (Isolation Forest, Z-score, Turn ratio)-аар
                хэвийн бус дансуудыг илрүүлж, <b>ensemble санал нэгтгэлээр</b> нэгдсэн
                аномалийн шийдвэр гаргана. ISA 315 «Эрсдэлийг тодорхойлох» болон
                ISA 240 «Залилангийн эрсдэл» стандартуудтай нийцнэ.
            </p>
        </div>
        """, unsafe_allow_html=True)

        with st.expander("ℹ️ Гурван аргын тайлбар", expanded=False):
            st.markdown("""
| Арга | Зарчим | Аномали гэж тодорхойлох нөхцөл |
|------|--------|-------------------------------|
| **Isolation Forest** | Олон хэмжээст огторгуйд тусгаарлагдсан цэгүүдийг илрүүлнэ | `predict = -1` (contamination=0.10) |
| **Z-score** | Статистик хэвийн тархалтаас хазайлтыг хэмжинэ | `max(|z-scores|) > 2.0` |
| **Turn ratio** | Дебит/кредит харьцааны хэвийн бус хазайлтыг илрүүлнэ | `|turn_ratio| > P95` |
| **Ensemble** | Гурван аргын санал нэгтгэл | `IF=1 ЭСВЭЛ (Z=1 БА TR=1)` |

**Scatter plot тайлбар:** X тэнхлэг = `log_abs_change` (он дамнасан цэвэр өөрчлөлт),
Y тэнхлэг = `turn_ratio` (дебит-кредит харьцаа). 🔵 Цэнхэр = хэвийн, 🔴 Улаан = аномали.
Hover хийхэд дансны код, нэр, аномалийн төрлийг харуулна.
            """)

    def show_anomaly_interpretation(self, n_if=0, n_zscore=0, n_turn=0, n_ensemble=0):
        """Tab 2: Аномали илрүүлэлтийн доор дүгнэлт харуулна."""
        st.markdown(f"""
        ---
        <div style="background-color: #FFF8E1; padding: 15px; border-radius: 8px;
             border-left: 4px solid #FFA000;">
            <b>💡 Аудиторт зориулсан тайлбар:</b><br>
            • <b>Isolation Forest</b> {n_if} данс илрүүлсэн — олон хэмжээст огторгуйд бусад данснаас тусгаарлагдсан<br>
            • <b>Z-score</b> {n_zscore} данс илрүүлсэн — статистик хэвийн утгаас 2σ-оос их хазайсан<br>
            • <b>Turn ratio</b> {n_turn} данс илрүүлсэн — дебит/кредит харьцааны P95 давсан<br>
            • <b>Ensemble нийлбэр</b> {n_ensemble} аномали данс — цаашдын нарийвчилсан шалгалтад хамруулна<br><br>
            Scatter plot-ын баруун дээд буланд байрлах улаан цэгүүд нь хамгийн өндөр эрсдэлтэй
            дансууд бөгөөд он дамнасан огцом өөрчлөлт ба дебит-кредит тэнцвэрийн хазайлт хоёулаа өндөр.
        </div>
        """, unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────────
    # TAB 3 — ХОУ vs MUS (AI vs Traditional)
    # ─────────────────────────────────────────────────────────────────
    def show_ai_vs_mus_description(self):
        """Tab 3: ХОУ vs MUS харьцуулалтын тайлбарыг харуулна."""
        st.markdown("""
        <div style="background: linear-gradient(135deg, #e8f5e9 0%, #f1f8f1 100%);
             padding: 20px; border-radius: 12px; border-left: 5px solid #2e7d32;
             margin-bottom: 20px;">
            <h4 style="color: #1b5e20; margin-top: 0;">⚖️ ХОУ vs MUS — Загварын гүйцэтгэлийн харьцуулалт</h4>
            <p style="color: #333; font-size: 14px; line-height: 1.7; margin-bottom: 0;">
                Хиймэл оюуны гурван загвар (Random Forest, Gradient Boosting, Logistic Regression)-ын
                гүйцэтгэлийг уламжлалт <b>MUS 20% түүвэрлэлттэй</b> харьцуулна.
                ISA 200 Аудитын эрсдэлийн загвар (AR = IR × CR × <b>DR</b>)-ын хүрээнд
                илрүүлэлтийн эрсдэлийг хэрхэн бууруулсныг харуулна.
            </p>
        </div>
        """, unsafe_allow_html=True)

        with st.expander("ℹ️ Үнэлгээний хэмжүүрүүдийн тайлбар", expanded=False):
            st.markdown("""
| Хэмжүүр | Томьёо | Аудитын утга |
|----------|--------|-------------|
| **Precision** | TP / (TP+FP) | Аномали гэж тэмдэглэсэн данснуудаас хэдэн хувь нь үнэхээр аномали вэ |
| **Recall** | TP / (TP+FN) | Бүх аномали данснуудаас хэдэн хувийг зөв олсон бэ (хамгийн чухал!) |
| **F1-score** | 2×(P×R)/(P+R) | Precision ба Recall-ийн тэнцвэржүүлсэн дундаж |
| **AUC-ROC** | ROC муруйн доорх талбай | Загварын ялгах чадварын нэгдсэн хэмжүүр (1.0 = төгс) |
| **Detection Risk** | 1 − Recall | **Аудитын гол хэмжүүр** — материаллаг алдаа илрэлгүй үлдэх магадлал |

**ROC Curve:** X тэнхлэг = False Positive Rate, Y тэнхлэг = True Positive Rate.
Муруй зүүн дээд булан руу ойр байх тусам загвар сайн.
            """)

    def show_ai_vs_mus_interpretation(self, rf_f1=0, rf_auc=0, dr_ai="", dr_mus="", mcnemar_chi2=0):
        """Tab 3: Харьцуулалтын доор дүгнэлт харуулна."""
        st.markdown(f"""
        ---
        <div style="background-color: #E8F5E9; padding: 15px; border-radius: 8px;
             border-left: 4px solid #2E7D32;">
            <b>📋 Гүйцэтгэлийн дүгнэлт:</b><br>
            • Random Forest загвар: <b>F1={rf_f1}, AUC={rf_auc}</b><br>
            • Detection Risk: ХОУ <b>{dr_ai}</b> vs MUS <b>{dr_mus}</b> → <b>15-20 дахин бууруулсан</b><br>
            • McNemar тест: χ²={mcnemar_chi2}, <b>p<0.001</b> → статистикийн хувьд маш өндөр ач холбогдолтой<br>
            • Хугацаа: ХОУ <b>4.5 цаг</b> (100% данс) vs MUS <b>310-357 цаг</b> (20% данс) → <b>98.5-98.7% хэмнэлт</b><br><br>
            <i>Энэ нь H2 (загварын давуу тал) болон H3 (хугацааны хэмнэлт) таамаглалуудыг баталгаажуулна.</i>
        </div>
        """, unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────────
    # TAB 4 — XAI (Explainable AI)
    # ─────────────────────────────────────────────────────────────────
    def show_xai_description(self):
        """Tab 4: XAI тайлбарлагдах хиймэл оюуны тайлбарыг харуулна."""
        st.markdown("""
        <div style="background: linear-gradient(135deg, #f3e5f5 0%, #faf5fc 100%);
             padding: 20px; border-radius: 12px; border-left: 5px solid #7b1fa2;
             margin-bottom: 20px;">
            <h4 style="color: #6a1b9a; margin-top: 0;">🧠 XAI — Тайлбарлагдах хиймэл оюун</h4>
            <p style="color: #333; font-size: 14px; line-height: 1.7; margin-bottom: 0;">
                Загвар <b>яагаад тухайн дансыг аномали гэж үнэлсэн бэ?</b> гэсэн асуултад
                хариулна. Feature Importance шинжилгээ нь шинж чанар бүрийн загварын шийдвэрт
                үзүүлэх нөлөөг хэмжинэ. ISA 500 «Аудитын нотолгоо» стандартын дагуу
                аудиторын мэргэжлийн дүгнэлтийг дэмжих нотолгоо болно.
            </p>
        </div>
        """, unsafe_allow_html=True)

    # Feature-ийн нарийвчилсан тайлбарын толь бичиг
    FEATURE_DESCRIPTIONS = {
        "log_abs_change": {
            "mn_name": "Цэвэр өөрчлөлт (log)",
            "description": "Он дамнасан цэвэр өөрчлөлтийн логарифм. Дансны үлдэгдэл өмнөх "
                           "жилээс огцом өөрчлөгдсөн нь бүртгэлийн алдаа, ангиллын буруу "
                           "шилжүүлэг, эсвэл зориудын манипуляцийн шинж тэмдэг байж болно.",
            "isa_ref": "ISA 520 «Аналитик горим» — он дамнасан мэдэгдэхүйц өөрчлөлтийг заавал шинжилнэ",
            "example": "Жишээ: Авлагын данс 500 сая₮-өөс 2.5 тэрбум₮ болж 5 дахин өссөн",
            "icon": "📈"
        },
        "turn_ratio": {
            "mn_name": "Дебит/Кредит харьцаа",
            "description": "Дебит-кредит эргэлтийн харьцааны хазайлт. Хэвийн нөхцөлд данс "
                           "тодорхой дебит/кредит харьцаатай байдаг бөгөөд энэ харьцаа огцом "
                           "хазайсан нь нэг чиглэлтэй хэвийн бус гүйлгээ байгааг илтгэнэ.",
            "isa_ref": "ISA 240 «Залилан» — нэг чиглэлтэй гүйлгээний хэв маяг нь залилангийн индикатор",
            "example": "Жишээ: Зөвхөн дебит гүйлгээ ихсэх (зардлыг хэтрүүлэн бүртгэх)",
            "icon": "⚖️"
        },
        "log_turn_d": {
            "mn_name": "Баримт дебит (log)",
            "description": "Баримт дебит гүйлгээний нийт хэмжээний логарифм. Том хэмжээний "
                           "дебит гүйлгээтэй данс нь материаллаг алдааны магадлал өндөртэй.",
            "isa_ref": "ISA 320 «Материаллаг байдал» — гүйлгээний хэмжээ нь материаллаг түвшинд нөлөөлнө",
            "example": "Жишээ: Нэг дансаар 50 тэрбум₮-ийн дебит гүйлгээ гарсан",
            "icon": "📊"
        },
        "log_turn_c": {
            "mn_name": "Баримт кредит (log)",
            "description": "Баримт кредит гүйлгээний нийт хэмжээний логарифм. Дебиттэй хамтад "
                           "нь гүйлгээний нийт хэмжээг илэрхийлнэ.",
            "isa_ref": "ISA 320 «Материаллаг байдал»",
            "example": "Жишээ: Орлогын дансанд 30 тэрбум₮-ийн кредит бүртгэгдсэн",
            "icon": "📊"
        },
        "cat_num": {
            "mn_name": "Дансны ангилал",
            "description": "Дансны 3 оронтой ангиллын код (1xx=Хөрөнгө, 2xx=Өр, 3xx=Өмч, "
                           "4-5xx=Орлого, 6-8xx=Зардал, 9xx=Нэгдсэн). Тодорхой ангиллын "
                           "дансууд эрсдэлийн түвшин өөр.",
            "isa_ref": "ISA 315 «Эрсдэл» — дансны шинж чанараар эрсдэлийн түвшин ялгаатай",
            "example": "Жишээ: 1xx хөрөнгийн дансууд 38-44% эзэлж, хамгийн олон аномали агуулна",
            "icon": "🏷️"
        },
        "log_close_d": {
            "mn_name": "Эцсийн дебит (log)",
            "description": "Жилийн эцсийн дебит үлдэгдлийн логарифм. Тайлант хугацааны "
                           "эцсийн байдлыг илэрхийлнэ.",
            "isa_ref": "ISA 505 «Баталгаажуулалт» — эцсийн үлдэгдлийн баталгаажуулалт",
            "example": "Жишээ: Хөрөнгийн дансны жилийн эцсийн үлдэгдэл",
            "icon": "📋"
        },
        "log_close_c": {
            "mn_name": "Эцсийн кредит (log)",
            "description": "Жилийн эцсийн кредит үлдэгдлийн логарифм.",
            "isa_ref": "ISA 505 «Баталгаажуулалт»",
            "example": "Жишээ: Өр төлбөрийн дансны жилийн эцсийн үлдэгдэл",
            "icon": "📋"
        },
        "year": {
            "mn_name": "Тайлант жил",
            "description": "Тайлант хугацаа (2023/2024/2025). Хугацааны нөлөө бага боловч "
                           "тодорхой жилүүдэд эрсдэлийн түвшин өөрчлөгдөж болно.",
            "isa_ref": "ISA 315 — хугацааны хүчин зүйлийн нөлөө",
            "example": "Жишээ: 2025 онд шинэ бүртгэлийн бодлого нэвтрүүлснээр аномали нэмэгдсэн",
            "icon": "📅"
        },
    }

    def show_xai_feature_details(self, feature_importances=None):
        """Tab 4: Feature Importance-ийн нарийвчилсан тайлбарыг харуулна.

        Args:
            feature_importances: dict {feature_name: importance_value} эсвэл None
        """
        if feature_importances is None:
            feature_importances = {
                "log_abs_change": 0.5769,
                "log_turn_d": 0.1489,
                "log_turn_c": 0.1223,
                "cat_num": 0.0573,
                "log_close_c": 0.0522,
                "log_close_d": 0.0213,
                "turn_ratio": 0.0125,
                "year": 0.0086,
            }

        st.markdown("### 📖 Шинж чанар бүрийн нарийвчилсан тайлбар")

        # Эрэмбэлсэн дарааллаар харуулна
        sorted_features = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)

        for feat_name, importance in sorted_features:
            info = self.FEATURE_DESCRIPTIONS.get(feat_name, {})
            if not info:
                continue

            importance_pct = importance * 100
            # Нөлөөний түвшингээр өнгө сонгох
            if importance > 0.15:
                bar_color = "#c62828"  # Өндөр нөлөөтэй = улаан
                level = "🔴 Өндөр нөлөөтэй"
            elif importance > 0.05:
                bar_color = "#e65100"  # Дунд нөлөөтэй = улбар шар
                level = "🟠 Дунд нөлөөтэй"
            else:
                bar_color = "#2e7d32"  # Бага нөлөөтэй = ногоон
                level = "🟢 Бага нөлөөтэй"

            st.markdown(f"""
            <div style="background-color: #fafafa; padding: 15px; border-radius: 10px;
                 border: 1px solid #e0e0e0; margin-bottom: 12px;">
                <div style="display: flex; align-items: center; margin-bottom: 8px;">
                    <span style="font-size: 20px; margin-right: 8px;">{info.get('icon', '📊')}</span>
                    <b style="font-size: 16px; color: #333;">{info.get('mn_name', feat_name)}</b>
                    <code style="margin-left: 10px; background: #e8eaf6; padding: 2px 8px;
                          border-radius: 4px; font-size: 12px;">{feat_name}</code>
                    <span style="margin-left: auto; font-weight: bold; color: {bar_color};">
                        {importance:.4f} ({importance_pct:.1f}%) — {level}
                    </span>
                </div>
                <div style="background: #e0e0e0; border-radius: 4px; height: 8px; margin-bottom: 10px;">
                    <div style="background: {bar_color}; width: {min(importance_pct * 1.7, 100):.0f}%;
                         height: 8px; border-radius: 4px;"></div>
                </div>
                <p style="color: #555; font-size: 13px; margin: 4px 0;">
                    {info.get('description', '')}
                </p>
                <p style="color: #1565C0; font-size: 12px; margin: 4px 0;">
                    📌 <i>{info.get('isa_ref', '')}</i>
                </p>
                <p style="color: #777; font-size: 12px; margin: 4px 0 0 0;">
                    {info.get('example', '')}
                </p>
            </div>
            """, unsafe_allow_html=True)

    def show_xai_interpretation(self):
        """Tab 4: XAI-ийн нэгдсэн дүгнэлтийг харуулна."""
        st.markdown("""
        ---
        <div style="background-color: #F3E5F5; padding: 15px; border-radius: 8px;
             border-left: 4px solid #7B1FA2;">
            <b>💡 XAI-ийн аудитын утга учир:</b><br>
            Загвар нь «хар хайрцаг» биш — шинж чанар бүрийн нөлөөг тоогоор хэмжиж,
            аудитын стандартуудтай (ISA 240/315/520) нийцэж байгааг харуулна. Аудитор нь
            загварын шийдвэрт <b>бүрэн итгэх</b> бус, загварын санал болгосон чиглэлийг
            <b>мэргэжлийн хүрээнд үнэлэх</b> зарчмыг баримтална (ISA 500).<br><br>
            <i>340 аудиторын судалгаагаар «хэрэглэхэд хялбар байдал» (β=0.366) нь
            хүлээн зөвшөөрөхөд хамгийн хүчтэй нөлөөлөгч болсон — XAI тайлбар нь
            яг энэ хэрэгцээг хангаж байна.</i>
        </div>
        """, unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────────
    # TAB 5 — Жагсаалт (Anomaly List)
    # ─────────────────────────────────────────────────────────────────
    def show_list_description(self):
        """Tab 5: Аномали дансуудын жагсаалтын тайлбарыг харуулна."""
        st.markdown("""
        <div style="background: linear-gradient(135deg, #fff3e0 0%, #fffaf0 100%);
             padding: 20px; border-radius: 12px; border-left: 5px solid #e65100;
             margin-bottom: 20px;">
            <h4 style="color: #bf360c; margin-top: 0;">📋 Жагсаалт — Аномали дансуудын дэлгэрэнгүй</h4>
            <p style="color: #333; font-size: 14px; line-height: 1.7; margin-bottom: 0;">
                Ensemble загвараар аномали гэж тэмдэглэгдсэн <b>бүх дансуудын дэлгэрэнгүй
                жагсаалт</b>. ISA 330 «Аудиторын хариу арга хэмжээ» стандартын дагуу
                эрсдэлтэй дансуудад чиглэсэн нарийвчилсан шалгалтын хүрээг тодорхойлоход
                шууд ашиглагдана.
            </p>
        </div>
        """, unsafe_allow_html=True)

        with st.expander("ℹ️ Хүснэгтийн баганууд ба шүүлтүүрийн тайлбар", expanded=False):
            st.markdown("""
| Багана | Тайлбар |
|--------|---------|
| **Дансны код** | Дансны дугаарлалт (жишээ: 101-00-01-000) |
| **Дансны нэр** | Дансны бүтэн нэр |
| **Ангилал** | Хөрөнгө/Өр/Өмч/Орлого/Зардал |
| **Эхний үлдэгдэл** | Тайлант жилийн эхний үлдэгдэл |
| **Нийт дебит/кредит** | Тайлант жилийн нийт баримт гүйлгээ |
| **Эцсийн үлдэгдэл** | Тайлант жилийн эцсийн үлдэгдэл |
| **Аномалийн төрөл** | IF / Z-score / Turn ratio / Ensemble |

**Шүүлтүүрүүд:** Жилээр (2023/2024/2025), аномалийн төрлөөр шүүж болно.
**CSV татах:** «📥 CSV татах» товчоор аномали дансуудын жагсаалтыг татаж,
цаашдын баримт бичгийн шалгалтад ашиглана.
            """)

    def show_list_interpretation(self, n_anomalies=0):
        """Tab 5: Жагсаалтын доор дүгнэлт харуулна."""
        st.markdown(f"""
        ---
        <div style="background-color: #FFF3E0; padding: 15px; border-radius: 8px;
             border-left: 4px solid #E65100;">
            <b>💡 Аудиторт зориулсан зөвлөмж:</b><br>
            • Нийт <b>{n_anomalies}</b> аномали данс илэрсэн — эдгээрийг анхан шатны баримтаар шалгана<br>
            • Олон арга давхацсан (IF + Z-score + Turn ratio) дансуудыг <b>нэн тэргүүнд</b> шалгах<br>
            • CSV файлыг татаж, аудитын ажлын баримт бичигт хавсаргах<br>
            • Аномали биш ч материаллаг дүнтэй дансуудыг мөн нэмэлтээр шалгахыг зөвлөнө
        </div>
        """, unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────────
    # TAB 6 — Эрсдэлийн матриц (Risk Matrix)
    # ─────────────────────────────────────────────────────────────────
    def show_risk_matrix_description(self):
        """Tab 6: Эрсдэлийн матрицын тайлбарыг харуулна."""
        st.markdown("""
        <div style="background: linear-gradient(135deg, #e0f2f1 0%, #f0faf9 100%);
             padding: 20px; border-radius: 12px; border-left: 5px solid #00695c;
             margin-bottom: 20px;">
            <h4 style="color: #004d40; margin-top: 0;">🔗 Эрсдэлийн матриц — Харилцагч × Данс хос</h4>
            <p style="color: #333; font-size: 14px; line-height: 1.7; margin-bottom: 0;">
                Харилцагч тус бүрийн данс, сараар нэгтгэсэн <b>эрсдэлийн хосуудын</b>
                шинжилгээ. ISA 550 «Холбоотой этгээд» стандартын дагуу тодорхой
                харилцагчтай хэвийн бус хэмжээний, давтамжийн гүйлгээ байгааг илрүүлнэ.
                <i>Part1 XLSX файл шаардана.</i>
            </p>
        </div>
        """, unsafe_allow_html=True)

        with st.expander("ℹ️ Эрсдэлийн матрицын тайлбар", expanded=False):
            st.markdown("""
**Эрсдэлийн хос гэж юу вэ?**

Харилцагч × Данс × Сар гэсэн гурвалсан хослолоор эрсдэлийн оноог тооцно.
Жишээ: «Баянгол ХК — Авлагын данс — 2024/06 сар» = нэг эрсдэлийн хос.

**Топ 20 харилцагч:** Хамгийн өндөр эрсдэлтэй 20 харилцагчийг нийт эрсдэлийн
оноогоор эрэмбэлж bar chart-аар харуулна. Hover хийхэд нийт гүйлгээний дүн,
холбогдох дансуудын тоо, дундаж эрсдэлийн оноог нарийвчлан харуулна.

**Ач холбогдол:** Уламжлалт аудитаар 440,789+ хосын эрсдэлийг гараар
шалгах боломжгүй. ХОУ загвар автоматаар тооцож, анхаарал хандуулах
харилцагчдыг эрэмбэлнэ.
            """)

    def show_risk_matrix_interpretation(self, n_pairs=0, top_counterparty=""):
        """Tab 6: Эрсдэлийн матрицын доор дүгнэлт харуулна."""
        st.markdown(f"""
        ---
        <div style="background-color: #E0F2F1; padding: 15px; border-radius: 8px;
             border-left: 4px solid #00695C;">
            <b>💡 Аудиторт зориулсан зөвлөмж:</b><br>
            • Нийт <b>{n_pairs:,}</b> эрсдэлийн хос шинжлэгдсэн<br>
            • Топ 20 харилцагчаас эхлэн анхан шатны баримтыг нарийвчлан шалгах<br>
            • Нэг харилцагчтай олон дансаар, олон сараар давтагдсан гүйлгээнд анхаарах<br>
            • ISA 550-ийн дагуу холбоотой этгээдийн гүйлгээг тусгайлан шалгах
        </div>
        """, unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────────
    # TAB 7 — Сарын trend (Monthly Trend)
    # ─────────────────────────────────────────────────────────────────
    def show_monthly_trend_description(self):
        """Tab 7: Сарын чиг хандлагын тайлбарыг харуулна."""
        st.markdown("""
        <div style="background: linear-gradient(135deg, #e8eaf6 0%, #f5f5fc 100%);
             padding: 20px; border-radius: 12px; border-left: 5px solid #283593;
             margin-bottom: 20px;">
            <h4 style="color: #1a237e; margin-top: 0;">📈 Сарын trend — Цаг хугацааны шинжилгээ</h4>
            <p style="color: #333; font-size: 14px; line-height: 1.7; margin-bottom: 0;">
                Сар бүрийн гүйлгээний хэмжээ, тооны чиг хандлагыг цаг хугацааны цувааны
                графикаар харуулна. ISA 520 «Аналитик горим» стандартын дагуу хүлээгдэж буй
                утгатай харьцуулж, мэдэгдэхүйц зөрүүг тодорхойлно.
                <i>Part1 XLSX файл шаардана.</i>
            </p>
        </div>
        """, unsafe_allow_html=True)

        with st.expander("ℹ️ Сарын trend-ийн аудитын ач холбогдол", expanded=False):
            st.markdown("""
| Шинжилгээний зорилго | Жишээ | Аудитын хариу |
|----------------------|-------|---------------|
| **Улирлын хэвийн хэлбэлзэл** | 12-р сард гүйлгээ огцом өснө | Жилийн эцсийн тооцоо — хэвийн |
| **Хэвийн бус оргил (spike)** | 6-р сард гүйлгээ 3 дахин өсөв | Тусгай шалгалт шаардана |
| **Жил хоорондын харьцуулалт** | 2024 оны хэв маяг 2023-оос өөр | Шалтгааныг тодруулах |

**Line chart:** 3 жилийн (2023, 2024, 2025) сарын эргэлтийг нэг графикт давхарлана.
**Bar chart:** Сар бүрийн гүйлгээний мөрийн тоог харуулна.
            """)

    def show_monthly_trend_interpretation(self):
        """Tab 7: Сарын trend-ийн доор дүгнэлт харуулна."""
        st.markdown("""
        ---
        <div style="background-color: #E8EAF6; padding: 15px; border-radius: 8px;
             border-left: 4px solid #283593;">
            <b>💡 Аудиторт зориулсан зөвлөмж:</b><br>
            • Гурван жилийн хэв маягийг харьцуулж, <b>огцом өөрчлөлт</b> гарсан сарыг тодорхойлох<br>
            • 12-р сарын эцсийн гүйлгээний оргил нь хэвийн (жилийн эцсийн тооцоо)<br>
            • Хэвийн бус сарын оргилыг илрүүлбэл тухайн сарын гүйлгээг нарийвчлан шалгах<br>
            • Гүйлгээний тоо буурсан ч дүн нэмэгдсэн бол <b>том дүнтэй цөөн гүйлгээ</b>-д анхаарах
        </div>
        """, unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────────
    # НИЙТЛЭГ: Нэгдсэн тайлбар (Dashboard footer)
    # ─────────────────────────────────────────────────────────────────
    def show_dashboard_footer(self):
        """Dashboard-ийн хамгийн доод хэсэгт ISA нийцлийн тойм харуулна."""
        st.markdown("""
        ---
        <div style="background: linear-gradient(135deg, #263238 0%, #37474f 100%);
             padding: 20px; border-radius: 12px; color: white;">
            <h4 style="color: #80CBC4; margin-top: 0;">🏛️ Dashboard — ISA стандартын нийцлийн тойм</h4>
            <table style="color: #cfd8dc; font-size: 13px; width: 100%; border-collapse: collapse;">
                <tr style="border-bottom: 1px solid #455a64;">
                    <td style="padding: 6px;"><b>Tab 1. Нэгтгэл</b></td>
                    <td style="padding: 6px;">ISA 300 — Аудитын ерөнхий стратеги, хүрээг тодорхойлох</td>
                </tr>
                <tr style="border-bottom: 1px solid #455a64;">
                    <td style="padding: 6px;"><b>Tab 2. Аномали</b></td>
                    <td style="padding: 6px;">ISA 315 — Эрсдэлийг тодорхойлох, үнэлэх</td>
                </tr>
                <tr style="border-bottom: 1px solid #455a64;">
                    <td style="padding: 6px;"><b>Tab 3. ХОУ vs MUS</b></td>
                    <td style="padding: 6px;">ISA 200 — Аудитын эрсдэлийн загвар (DR бууруулалт)</td>
                </tr>
                <tr style="border-bottom: 1px solid #455a64;">
                    <td style="padding: 6px;"><b>Tab 4. XAI</b></td>
                    <td style="padding: 6px;">ISA 500 — Аудитын нотолгоо, мэргэжлийн дүгнэлтийг дэмжих</td>
                </tr>
                <tr style="border-bottom: 1px solid #455a64;">
                    <td style="padding: 6px;"><b>Tab 5. Жагсаалт</b></td>
                    <td style="padding: 6px;">ISA 330 — Эрсдэлд чиглэсэн нарийвчилсан шалгалт</td>
                </tr>
                <tr style="border-bottom: 1px solid #455a64;">
                    <td style="padding: 6px;"><b>Tab 6. Эрсдэлийн матриц</b></td>
                    <td style="padding: 6px;">ISA 550 — Холбоотой этгээдийн гүйлгээ</td>
                </tr>
                <tr>
                    <td style="padding: 6px;"><b>Tab 7. Сарын trend</b></td>
                    <td style="padding: 6px;">ISA 520 — Аналитик горим, цаг хугацааны шинжилгээ</td>
                </tr>
            </table>
        </div>
        """, unsafe_allow_html=True)
