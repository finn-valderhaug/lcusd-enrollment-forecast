"""
LCUSD Enrollment Outlook - Streamlit dashboard (Phase 3).

Run from the project folder:   streamlit run app.py
Needs the CSVs produced by phase1_data_pipeline.py and phase2_tune.py (or phase2_model.py).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import phase2_model as pm

PROC = Path("data/processed")
st.set_page_config(page_title="LCUSD Enrollment Outlook", layout="wide")


@st.cache_data
def load():
    m, ungr = pm.load_matrix()
    d = {"m": m, "ungr": ungr}
    for name in ["history_totals", "forecast_totals", "forecast_by_grade", "backtest"]:
        d[name] = pd.read_csv(PROC / f"{name}.csv")
    acs = PROC / "acs_91011.csv"
    d["acs"] = pd.read_csv(acs) if acs.exists() else None
    cfg = PROC / "model_config.json"
    d["cfg"] = json.loads(cfg.read_text()) if cfg.exists() else None
    return d


try:
    d = load()
except FileNotFoundError as e:
    st.error(f"Missing data file: {e.filename}. Run phase1_data_pipeline.py, then phase2_tune.py.")
    st.stop()

m, ungr = d["m"], d["ungr"]
hist, ft, fg, bt = d["history_totals"], d["forecast_totals"], d["forecast_by_grade"], d["backtest"]
last_y, last_n = int(hist.year.iloc[-1]), float(hist.total.iloc[-1])
first_y, first_n = int(hist.year.iloc[0]), float(hist.total.iloc[0])
cagr = (last_n / first_n) ** (1 / (last_y - first_y)) - 1
end = ft.iloc[-1]


def scenario(window, kchg, horizon=5, excl=(2021,)):
    r = pm.csr_table(m).tail(window)
    keep = r.drop(index=[y for y in r.index if y in excl])
    csr = (keep if len(keep) >= 3 else r).mean()
    k0 = m["K"].tail(3).mean()
    cur = m.iloc[-1].copy()
    rows = []
    for h in range(1, horizon + 1):
        nxt = {"K": k0 * (1 + kchg / 100) ** h}
        for i, g in enumerate(pm.GRADES[1:], 1):
            nxt[g] = cur[pm.GRADES[i - 1]] * csr[g]
        cur = pd.Series(nxt)[pm.GRADES]
        rows.append(cur)
    out = pd.DataFrame(rows, index=range(last_y + 1, last_y + 1 + horizon))
    return out.sum(axis=1) + float(ungr.iloc[-1])


st.title("La Cañada Unified School District: Enrollment Outlook")
st.caption(f"Source: California Department of Education Census Day enrollment, {first_y}-{last_y}; "
           f"US Census ACS 5-year (ZCTA 91011). Forecast: cohort-survival model with Monte Carlo bands.")

c1, c2, c3, c4 = st.columns(4)
c1.metric(f"Enrollment {last_y}", f"{int(last_n):,}")
c2.metric(f"Annual change {first_y}-{last_y}", f"{cagr * 100:+.2f}%")
c3.metric(f"Forecast median {int(end.year)}", f"{int(end['median']):,}",
          f"{(end['median'] / last_n - 1) * 100:+.1f}% vs {last_y}")
c4.metric("80% range", f"{int(end.p10):,} - {int(end.p90):,}")

tabs = st.tabs(["Overview", "By grade", "Scenario lab", "Funding impact", "Demographics", "Model & accuracy"])

# ---------------------------------------------------------------- Overview
with tabs[0]:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist.year, y=hist.total, name="Actual", mode="lines+markers",
                             line=dict(width=3)))
    fig.add_trace(go.Scatter(x=ft.year, y=ft.p90, line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=ft.year, y=ft.p10, fill="tonexty", line=dict(width=0),
                             name="80% interval (cohort model)", fillcolor="rgba(100,100,255,0.2)"))
    fig.add_trace(go.Scatter(x=ft.year, y=ft["median"], name="Cohort model median",
                             mode="lines+markers", line=dict(dash="dash")))
    fig.add_trace(go.Scatter(x=[last_y] + list(ft.year), y=[last_n] * (len(ft) + 1),
                             name="Flat baseline (no change)", line=dict(dash="dot", color="gray")))
    fig.update_layout(height=480, yaxis_title="Students (TK-12)", xaxis_title="School year (ending)",
                      hovermode="x unified", legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig, use_container_width=True)
    st.info("The cohort model assumes kindergarten entry holds near its recent average. Kindergarten entry "
            "has been falling, so treat the upper part of the range as conditional on that assumption. "
            "Test it in the **Scenario lab**.")

# ---------------------------------------------------------------- By grade
with tabs[1]:
    left, right = st.columns(2)
    bands = pd.DataFrame({
        "K-5": m[["K", "1", "2", "3", "4", "5"]].sum(axis=1),
        "6-8": m[["6", "7", "8"]].sum(axis=1),
        "9-12": m[["9", "10", "11", "12"]].sum(axis=1)}).reset_index()
    bl = bands.melt(id_vars="year", var_name="Band", value_name="Students")
    left.plotly_chart(px.area(bl, x="year", y="Students", color="Band", title="Enrollment by grade band"),
                      use_container_width=True)
    heat = px.imshow(m.T.values, x=m.index.astype(int), y=m.columns, aspect="auto",
                     color_continuous_scale="Blues", labels=dict(x="Year", y="Grade", color="Students"),
                     title="Students by grade and year")
    right.plotly_chart(heat, use_container_width=True)

    yr = st.select_slider("Forecast year", options=sorted(fg.year.unique()), value=int(fg.year.max()))
    sel = fg[fg.year == yr].copy()
    sel["now"] = [m[g].iloc[-1] for g in sel.grade]
    f2 = go.Figure()
    f2.add_trace(go.Bar(x=sel.grade, y=sel.now, name=f"{last_y} actual"))
    f2.add_trace(go.Bar(x=sel.grade, y=sel["median"], name=f"{yr} forecast (median)",
                        error_y=dict(type="data", symmetric=False,
                                     array=sel.p90 - sel["median"], arrayminus=sel["median"] - sel.p10)))
    f2.update_layout(barmode="group", title=f"Grade-level forecast for {yr} (bars: 10th-90th percentile)",
                     xaxis_title="Grade", yaxis_title="Students")
    st.plotly_chart(f2, use_container_width=True)

# ---------------------------------------------------------------- Scenario lab
with tabs[2]:
    st.markdown("Change the assumptions and see the projection update. Everything else follows from the "
                "historical grade-to-grade survival ratios.")
    s1, s2 = st.columns(2)
    window = s1.slider("Survival-ratio window (most recent transitions)", 2, 11, 11)
    kchg = s2.slider("Annual change in kindergarten entry (%)", -6.0, 6.0, 0.0, 0.5)
    sc = scenario(window, kchg)
    presets = {"Kindergarten -3%/yr": scenario(window, -3), "Kindergarten flat": scenario(window, 0),
               "Kindergarten +2%/yr": scenario(window, 2)}
    f3 = go.Figure()
    f3.add_trace(go.Scatter(x=hist.year, y=hist.total, name="Actual", line=dict(width=3)))
    for name, s in presets.items():
        f3.add_trace(go.Scatter(x=s.index, y=s.values, name=name, line=dict(dash="dot")))
    f3.add_trace(go.Scatter(x=sc.index, y=sc.values, name="Your scenario", mode="lines+markers",
                            line=dict(width=4)))
    f3.update_layout(height=450, yaxis_title="Students (TK-12)", hovermode="x unified")
    st.plotly_chart(f3, use_container_width=True)
    out = pd.DataFrame({"Year": sc.index, "Students": sc.round(0).astype(int).values,
                        f"vs {last_y}": (sc.values / last_n - 1) * 100})
    st.dataframe(out.style.format({f"vs {last_y}": "{:+.1f}%"}), hide_index=True)

# ---------------------------------------------------------------- Funding
with tabs[3]:
    st.markdown("Translate enrollment changes into dollars. **This is a simplified illustration**: enter "
                "LCUSD's actual revenue per student from its adopted budget or the CDE "
                "current-expense-of-education data.")
    f1, f2c = st.columns(2)
    model = f1.radio("Funding model", ["Enrollment-driven (LCFF-style)", "Fixed local revenue (basic-aid style)"])
    ppr = f2c.number_input("Revenue per student, $ (illustrative placeholder)", 5000, 50000, 18000, 500)
    base_rev = last_n * ppr
    rows = []
    for _, r in ft.iterrows():
        for lab, n in [("p10", r.p10), ("median", r["median"]), ("p90", r.p90)]:
            if model.startswith("Enrollment"):
                rev = n * ppr
                rows.append(dict(year=int(r.year), case=lab, revenue_m=rev / 1e6,
                                 change_m=(rev - base_rev) / 1e6, per_student=ppr))
            else:
                rows.append(dict(year=int(r.year), case=lab, revenue_m=base_rev / 1e6, change_m=0.0,
                                 per_student=base_rev / n))
    fr = pd.DataFrame(rows)
    if model.startswith("Enrollment"):
        figf = px.line(fr, x="year", y="change_m", color="case",
                       title=f"Revenue change vs {last_y} ($M), by enrollment case",
                       labels={"change_m": "$ millions"})
        st.plotly_chart(figf, use_container_width=True)
        mid = fr[fr.case == "median"].iloc[-1]
        st.metric(f"Median-case revenue change by {int(mid.year)}", f"${mid.change_m:+.1f}M")
    else:
        figf = px.line(fr, x="year", y="per_student", color="case",
                       title="Revenue per student if local revenue stays fixed ($)",
                       labels={"per_student": "$ per student"})
        st.plotly_chart(figf, use_container_width=True)
        mid = fr[fr.case == "median"].iloc[-1]
        st.metric(f"Median-case revenue per student, {int(mid.year)}", f"${mid.per_student:,.0f}",
                  f"{(mid.per_student / ppr - 1) * 100:+.1f}% vs {last_y}")
    st.caption("Real funding rules (average daily attendance, multi-year averaging, supplemental grants) are "
               "more complex than this model.")

# ---------------------------------------------------------------- Demographics
with tabs[4]:
    acs = d["acs"]
    if acs is None:
        st.warning("No ACS file found. Run phase1_data_pipeline.py with a Census API key.")
    else:
        st.markdown("Neighborhood indicators for ZIP/ZCTA 91011 (ACS 5-year estimates). ZCTA boundaries do "
                    "not match the school district exactly, and 5-year estimates are smoothed.")
        cols = [c for c in ["school_age_5_17", "pop_u5", "pop_under18", "total_pop"] if c in acs.columns]
        a1, a2 = st.columns(2)
        a1.plotly_chart(px.line(acs, x="acs_year", y=cols, markers=True, title="Population by age group"),
                        use_container_width=True)
        cols2 = [c for c in ["median_home_value", "median_hh_income"] if c in acs.columns]
        a2.plotly_chart(px.line(acs, x="acs_year", y=cols2, markers=True, title="Home value and income ($)"),
                        use_container_width=True)
        both = acs.merge(hist, left_on="acs_year", right_on="year")
        if len(both) > 4 and "school_age_5_17" in both:
            corr = both["school_age_5_17"].corr(both["total"])
            st.metric("Correlation: ACS school-age population vs. district enrollment", f"{corr:.2f}")

# ---------------------------------------------------------------- Model
with tabs[5]:
    s = bt.groupby("horizon")[["model_ape", "naive_ape"]].mean().round(2).reset_index()
    s.columns = ["Years ahead", "Cohort model error (%)", "Flat baseline error (%)"]
    st.subheader("Out-of-sample accuracy (rolling-origin backtest)")
    st.dataframe(s, hide_index=True)
    h1 = bt[bt.horizon == 1]
    f4 = go.Figure()
    f4.add_trace(go.Scatter(x=h1.target, y=h1.actual, name="Actual", mode="lines+markers"))
    f4.add_trace(go.Scatter(x=h1.target, y=h1.model, name="1-year-ahead cohort forecast", mode="lines+markers"))
    f4.update_layout(title="One-year-ahead forecasts vs actual", yaxis_title="Students")
    st.plotly_chart(f4, use_container_width=True)
    if d["cfg"]:
        st.write("Selected settings:", d["cfg"])
    st.subheader("Method and limitations")
    st.markdown(
        "- **Cohort survival:** each grade's forecast = prior grade x its historical survival ratio "
        "(captures migration, private-school flows and retention).\n"
        "- **Kindergarten** has no prior cohort and is the main driver of uncertainty. TK is combined with K "
        "so the series is consistent across years.\n"
        "- **Uncertainty:** 5,000 Monte Carlo runs resampling historical years.\n"
        "- **Accuracy:** for this stable district the flat baseline is hard to beat in total; the cohort model "
        "adds grade-level detail and uncertainty bands.\n"
        "- **Settings were chosen on the same backtest reported here**, so accuracy is mildly optimistic "
        "(only ~18 overlapping forecasts).\n"
        "- The 2020-21 pandemic transition is excluded from survival ratios; ZCTA 91011 is not the district boundary.")
    