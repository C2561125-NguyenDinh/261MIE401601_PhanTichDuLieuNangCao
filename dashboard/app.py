"""Dashboard giám sát hệ thống quyết định uplift X5 (Streamlit).

Chạy:  streamlit run dashboard/app.py
Dashboard chỉ đọc các tệp trong results/ do pipeline sinh ra, nên luôn khớp với báo cáo.
Trang được chọn ở thanh bên hoặc qua tham số URL ?page=1..5.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
TAB = RES / "tables"
GATE_COLORS = {"PASS": "#2E7D32", "FAIL": "#C62828"}
st.set_page_config(page_title="X5 Uplift – Giám sát", layout="wide")


@st.cache_data
def t(name):
    p = TAB / f"{name}.csv"
    return pd.read_csv(p, encoding="utf-8-sig") if p.exists() else None


def show(df, **kw):
    """Hiển thị bảng với số đã làm tròn: số lớn làm tròn đơn vị, số nhỏ giữ 4 chữ số."""
    d = df.copy()
    for c in d.select_dtypes("number").columns:
        d[c] = d[c].round(0) if d[c].abs().max() >= 100 else d[c].round(4)
    g = d.copy()
    if "Giá trị đo được" in g.columns:
        g["Giá trị đo được"] = pd.to_numeric(g["Giá trị đo được"], errors="ignore")
        g["Giá trị đo được"] = g["Giá trị đo được"].map(lambda v: f"{v:.4f}" if isinstance(v, float) else v)
    return g


@st.cache_data
def metrics():
    return json.loads((RES / "metrics.json").read_text(encoding="utf-8"))


def vn(x, d=0):
    return f"{x:,.{d}f}".replace(",", "X").replace(".", ",").replace("X", ".")


PAGES = ["1. Tổng quan và cổng phát hành", "2. Trôi dạt dữ liệu và mô hình", "3. Hiệu quả chính sách",
         "4. Uplift và giải thích", "5. Vận hành dịch vụ"]
qp = st.query_params.get("page")
default = int(qp) - 1 if qp and qp.isdigit() and 1 <= int(qp) <= len(PAGES) else 0
page = st.sidebar.radio("Trang", PAGES, index=default)
M = metrics()
st.sidebar.markdown(f"**Phiên bản mô hình:** {M.get('design', {}).get('model_version', 'v1.0.0')}")
st.sidebar.caption("Nguồn: results/ của pipeline X5 uplift")

if page == PAGES[0]:
    st.title("Tổng quan hệ thống quyết định – chiến dịch SMS giữ chân khách hàng")
    cal = t("t12_calibration_results"); g = t("t48_release_gates"); pl = t("t28_policy_comparison_locked"); po = t("t28_policy_comparison_oot")
    ate = t("t17_ate_estimates")
    c = st.columns(5)
    oot_cal = cal[(cal["Tập"].str.contains("Out")) & (cal["Hiệu chỉnh"] == M["forecast"]["calibration_selected"])].iloc[0]
    c[0].metric("AUC dự báo (OOT)", vn(oot_cal["AUC"], 4))
    c[1].metric("ECE (OOT)", vn(oot_cal["ECE"], 4), delta="vượt ngưỡng 0,03" if oot_cal["ECE"] > 0.03 else "đạt", delta_color="inverse")
    c[2].metric("Lợi nhuận MILP – Locked (RUB)", vn(pl.set_index("Chính sách").loc["MILP (SCIP)", "Lợi nhuận tăng thêm DR (RUB)"]))
    c[3].metric("Lợi nhuận MILP – OOT (RUB)", vn(po.set_index("Chính sách").loc["MILP (SCIP)", "Lợi nhuận tăng thêm DR (RUB)"]))
    c[4].metric("Cổng phát hành đạt", f"{(g['Kết quả'] == 'PASS').sum()}/{len(g)}")
    st.subheader("Cổng phát hành")
    st.dataframe(show(g).style.map(lambda v: f"color: {GATE_COLORS.get(v, 'black')}; font-weight: bold" if v in GATE_COLORS else "", subset=["Kết quả"]),
                 use_container_width=True, hide_index=True)
    st.info("Khuyến nghị: triển khai dạng thí nghiệm có nhóm đối chứng ngẫu nhiên; chưa bật tự động hoàn toàn khi còn cổng FAIL.")

elif page == PAGES[1]:
    st.title("Trôi dạt dữ liệu, mô hình và chính sách")
    psi = t("t44_psi_drift")
    col = st.selectbox("So sánh", ["PSI Train→OOT cohort", "PSI S1→S3 (thời gian)", "PSI Train→Locked"])
    d = psi.sort_values(col, ascending=False).head(15)
    fig = px.bar(d, x=col, y="Đặc trưng", orientation="h", color=col, color_continuous_scale="Reds")
    fig.add_vline(x=0.10, line_dash="dash", annotation_text="cảnh báo 0,10")
    fig.add_vline(x=0.25, line_dash="dash", line_color="red", annotation_text="báo động 0,25")
    fig.update_layout(height=430, yaxis={"autorange": "reversed"})
    st.plotly_chart(fig, use_container_width=True)
    c1, c2 = st.columns(2)
    sp = t("tx30_snapshot_performance")
    if sp is not None:
        f2 = go.Figure()
        f2.add_bar(x=sp["Snapshot"], y=sp["ECE"], name="ECE")
        f2.add_hline(y=0.03, line_dash="dash", annotation_text="ngưỡng ECE 0,03")
        f2.update_layout(title="ECE theo snapshot thời gian", height=330)
        c1.plotly_chart(f2, use_container_width=True)
    ld = t("t46_label_delay")
    if ld is not None:
        f3 = px.line(ld, x="Số ngày đã chờ nhãn", y=["ECE đo được", "AUC đo được"], markers=True, title="Độ trễ nhãn: chỉ số đo được theo số ngày chờ")
        f3.update_layout(height=330)
        c2.plotly_chart(f3, use_container_width=True)
    ev = t("tx42_evidently_drift")
    if ev is not None:
        st.subheader("Kiểm định trôi dạt Evidently (Train → OOT)")
        st.dataframe(show(ev), use_container_width=True, hide_index=True, height=260)

elif page == PAGES[2]:
    st.title("Hiệu quả chính sách phân bổ (ước lượng kép vững)")
    which = st.radio("Tập đánh giá", ["Locked Test", "OOT"], horizontal=True)
    d = t("t28_policy_comparison_locked" if which == "Locked Test" else "t28_policy_comparison_oot")
    d = d[d["Chính sách"] != "Không gửi (baseline)"]
    fig = go.Figure(go.Bar(x=d["Chính sách"], y=d["Lợi nhuận tăng thêm DR (RUB)"],
                           error_y=dict(type="data", symmetric=False,
                                        array=d["KTC95 trên"] - d["Lợi nhuận tăng thêm DR (RUB)"],
                                        arrayminus=d["Lợi nhuận tăng thêm DR (RUB)"] - d["KTC95 dưới"]),
                           marker_color=["#1565C0" if k else "#9E9E9E" for k in d["Khả thi"]]))
    fig.add_hline(y=0, line_color="black")
    fig.update_layout(height=420, yaxis_title="Lợi nhuận tăng thêm (RUB)", title="Xanh: khả thi với ràng buộc; xám: vượt ngân sách hoặc ràng buộc")
    st.plotly_chart(fig, use_container_width=True)
    pt = t("tx40_policy_tree_eval")
    if pt is not None:
        st.subheader("Cây chính sách so với MILP")
        st.dataframe(show(pt[pt["Tập"] == which]), use_container_width=True, hide_index=True)
    pw = t("tx32_power_analysis")
    if pw is not None:
        st.caption("Lực thống kê: " + "; ".join(f"{r['So sánh']} – lực {vn(100 * r['Lực tại ước lượng điểm'], 1)}%" for _, r in pw[pw['Tập'] == which].iterrows()))

elif page == PAGES[3]:
    st.title("Uplift, tính không đồng nhất và giải thích nhân quả")
    q = t("t19_uplift_learners")
    ec = t("tx35_econml_causal_forest")
    piv = q.pivot(index="Learner", columns="Tập", values="Qini_norm")
    if ec is not None:
        piv.loc["CausalForestDML (EconML)"] = ec.set_index("Tập")["Qini chuẩn hóa (CausalForestDML)"]
    piv = piv[["val", "locked", "oot"]].reset_index().melt(id_vars="Learner", var_name="Tập", value_name="Qini chuẩn hóa")
    st.plotly_chart(px.bar(piv, x="Learner", y="Qini chuẩn hóa", color="Tập", barmode="group", height=380), use_container_width=True)
    c1, c2 = st.columns(2)
    sh = t("tx37_shap_cate")
    if sh is not None:
        f = px.bar(sh.head(10), x="Tỷ trọng SHAP tuyệt đối TB", y="Diễn giải", orientation="h", title="SHAP: đặc trưng dẫn dắt CATE")
        f.update_layout(yaxis={"autorange": "reversed"}, height=380)
        c1.plotly_chart(f, use_container_width=True)
    het = t("t22_segment_heterogeneity")
    dim = c2.selectbox("Chiều phân khúc", het["Chiều phân khúc"].unique())
    h = het[het["Chiều phân khúc"] == dim]
    f2 = go.Figure(go.Scatter(x=h["Nhóm"], y=h["Uplift quan sát"], mode="markers",
                              error_y=dict(type="data", array=1.96 * h["SE"]), marker=dict(size=10)))
    f2.add_hline(y=0, line_color="black"); f2.update_layout(title=f"Uplift quan sát theo {dim.lower()}", height=380)
    c2.plotly_chart(f2, use_container_width=True)

else:
    st.title("Vận hành: độ trễ, định tuyến và dịch vụ")
    lat = t("t43_latency"); api = t("tx44_fastapi_latency"); tests = t("tx43_fastapi_tests"); act = t("t35_action_distribution")
    c1, c2 = st.columns(2)
    f = px.bar(lat[lat["Bước"] != "Đầu cuối (E2E)"], x="Bước", y=["P50 (ms)", "P95 (ms)", "P99 (ms)"], barmode="group",
               title="Độ trễ theo bước của luồng chấm điểm")
    c1.plotly_chart(f, use_container_width=True)
    a = act[act["Tập"] == "locked"]
    c2.plotly_chart(px.pie(a, names="Hành động", values="Số khách", title="Phân bổ hành động (Locked Test)", hole=0.45), use_container_width=True)
    if tests is not None:
        st.subheader("Kiểm thử dịch vụ FastAPI")
        st.dataframe(show(tests), use_container_width=True, hide_index=True)
    if api is not None:
        st.dataframe(show(api), use_container_width=True, hide_index=True)
