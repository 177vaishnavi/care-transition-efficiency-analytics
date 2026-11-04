import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Care Transition Efficiency Dashboard", layout="wide")

@st.cache_data
def load():
    d = pd.read_csv("uac_clean.csv", parse_dates=["date"])
    d = d[["date", "apprehended", "cbp_custody", "transferred", "hhs_care", "discharged"]]
    d = d.sort_values("date").reset_index(drop=True)
    d["transfer_eff"] = d["transferred"] / d["cbp_custody"]
    d["discharge_eff"] = d["discharged"] / d["hhs_care"]
    d["hhs_change"] = d["hhs_care"].diff()
    d["month"] = d["date"].dt.to_period("M").astype(str)
    return d

df_all = load()

# ---------------- Sidebar ----------------
st.sidebar.header("Filters")
min_d, max_d = df_all["date"].min().date(), df_all["date"].max().date()
rng = st.sidebar.date_input("Date range", value=(min_d, max_d), min_value=min_d, max_value=max_d)
if len(rng) != 2:
    st.info("Select both a start date and an end date.")
    st.stop()
start, end = rng

metrics = st.sidebar.multiselect("Ratio metrics to show",
                                 ["Transfer Efficiency", "Discharge Effectiveness"],
                                 default=["Transfer Efficiency", "Discharge Effectiveness"])
window = st.sidebar.slider("Rolling window (reports)", 1, 30, 14)

st.sidebar.subheader("Alert thresholds")
t_te = st.sidebar.slider("Min transfer efficiency", 0.0, 1.0, 0.5, 0.05)
t_de = st.sidebar.slider("Min discharge effectiveness", 0.0, 0.05, 0.02, 0.005)
t_bl = st.sidebar.slider("Max avg HHS load growth per report", 0, 100, 20)

df = df_all[(df_all["date"].dt.date >= start) & (df_all["date"].dt.date <= end)].copy()
if len(df) < 15:
    st.warning("Please select a longer date range (at least 15 reports).")
    st.stop()

# ---------------- Title + KPIs ----------------
st.title("Care Transition Efficiency & Placement Outcome Analytics")
st.caption("HHS Unaccompanied Alien Children Program | CBP custody -> HHS care -> sponsor placement")

cv = df["discharged"].std() / df["discharged"].mean()
throughput = df["discharged"].sum() / max(df["apprehended"].sum(), 1)

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Transfer Efficiency", f"{df['transfer_eff'].mean():.2f}")
k2.metric("Discharge Effectiveness", f"{df['discharge_eff'].mean():.3f}")
k3.metric("Pipeline Throughput", f"{throughput:.2f}")
k4.metric("Backlog Rate (avg HHS change/report)", f"{df['hhs_change'].mean():.1f}")
k5.metric("Outcome Stability Score", f"{1 / (1 + cv):.2f}")

# ---------------- Alerts (latest window in range) ----------------
st.subheader(f"Alerts (last {window} reports in selected range)")
recent = df.tail(window)
r_te, r_de, r_bl = recent["transfer_eff"].mean(), recent["discharge_eff"].mean(), recent["hhs_change"].mean()
a1, a2, a3 = st.columns(3)
(a1.error if r_te < t_te else a1.success)(f"Transfer efficiency {r_te:.2f} (min {t_te:.2f})")
(a2.error if r_de < t_de else a2.success)(f"Discharge effectiveness {r_de:.3f} (min {t_de:.3f})")
(a3.error if r_bl > t_bl else a3.success)(f"HHS load change {r_bl:.1f}/report (max {t_bl})")

# ---------------- Tabs ----------------
tab1, tab2, tab3, tab4 = st.tabs(["Pipeline Flow", "Efficiency", "Bottlenecks", "Outcome Trends"])

with tab1:
    c1, c2 = st.columns(2)
    stage = pd.DataFrame({
        "Stage": ["Apprehended (CBP intake)", "Transferred to HHS", "Discharged to sponsors"],
        "Children": [df["apprehended"].sum(), df["transferred"].sum(), df["discharged"].sum()]})
    c1.plotly_chart(px.bar(stage, x="Stage", y="Children", color="Stage",
                           title="Pipeline stage volumes (total in range)"))
    load_fig = go.Figure()
    load_fig.add_trace(go.Scatter(x=df["date"], y=df["cbp_custody"], name="In CBP custody"))
    load_fig.add_trace(go.Scatter(x=df["date"], y=df["hhs_care"], name="In HHS care", yaxis="y2"))
    load_fig.update_layout(title="Care load by stage", yaxis=dict(title="CBP custody"),
                           yaxis2=dict(title="HHS care", overlaying="y", side="right"))
    c2.plotly_chart(load_fig)

with tab2:
    cols = st.columns(len(metrics)) if metrics else []
    if "Transfer Efficiency" in metrics:
        f = go.Figure(go.Scatter(x=df["date"], y=df["transfer_eff"].rolling(window).mean(), name="Transfer Efficiency"))
        f.add_hline(y=t_te, line_dash="dash", line_color="red", annotation_text="threshold")
        f.update_layout(title="Transfer Efficiency Ratio (transfers / CBP custody)")
        cols[metrics.index("Transfer Efficiency")].plotly_chart(f)
    if "Discharge Effectiveness" in metrics:
        f = go.Figure(go.Scatter(x=df["date"], y=df["discharge_eff"].rolling(window).mean(),
                                 name="Discharge Effectiveness", line=dict(color="green")))
        f.add_hline(y=t_de, line_dash="dash", line_color="red", annotation_text="threshold")
        f.update_layout(title="Discharge Effectiveness (discharges / HHS care)")
        cols[metrics.index("Discharge Effectiveness")].plotly_chart(f)
    if not metrics:
        st.info("Select at least one metric in the sidebar.")

with tab3:
    b1, b2 = st.columns(2)
    flow = go.Figure()
    flow.add_trace(go.Scatter(x=df["date"], y=df["transferred"].rolling(window).mean(), name="Transferred in"))
    flow.add_trace(go.Scatter(x=df["date"], y=df["discharged"].rolling(window).mean(), name="Discharged out"))
    flow.update_layout(title="Inflow vs exits (rolling average)")
    b1.plotly_chart(flow)

    monthly = df.groupby("month")["hhs_change"].mean().reset_index()
    monthly["direction"] = np.where(monthly["hhs_change"] > 0, "Growing", "Shrinking")
    b2.plotly_chart(px.bar(monthly, x="month", y="hhs_change", color="direction",
                           color_discrete_map={"Growing": "crimson", "Shrinking": "seagreen"},
                           title="Avg change in HHS care per report, by month"))
    hot = monthly[monthly["hhs_change"] > t_bl]
    if len(hot):
        st.warning("Months above the backlog threshold: " + ", ".join(hot["month"]))
    else:
        st.success("No month exceeds the backlog threshold in this range.")

with tab4:
    o1, o2 = st.columns(2)
    r7 = df["discharged"].rolling(7).mean()
    prev = r7.shift(7)
    flag = ((r7 / prev - 1) * 100 < -30) & (prev >= 50)
    f = go.Figure(go.Scatter(x=df["date"], y=r7, name="Discharges (7-report avg)", line=dict(color="green")))
    f.add_trace(go.Scatter(x=df.loc[flag, "date"], y=r7[flag], mode="markers",
                           marker=dict(color="red", size=7), name="Sudden drop"))
    f.update_layout(title="Discharges with sudden drops flagged")
    o1.plotly_chart(f)

    cvroll = df["discharged"].rolling(30).std() / df["discharged"].rolling(30).mean()
    o2.plotly_chart(px.line(x=df["date"], y=cvroll,
                            labels={"x": "date", "y": "CV"},
                            title="Variability of discharges (30-report CV, higher = less stable)"))
    st.caption(f"{int(flag.sum())} reports flagged as sudden drops in this range.")

st.caption("Note: HHS reports mostly Mon-Thu and Sunday (no Saturday, almost no Friday). "
           "Efficiency ratios are used for comparisons because they are unaffected by gaps.")
