
"""
Phase 2: Cohort-survival enrollment forecast for LCUSD.

Method
  * Cohort survival ratio (CSR): E[grade g, year t] / E[grade g-1, year t-1]. This captures
    net migration + private-school flows + retention for each grade transition.
  * Kindergarten (TK+K combined for a consistent series) has no cohort behind it, so it is
    forecast with a damped linear trend on the last 8 years.
  * Monte Carlo: each simulated year resamples a historical year's full vector of CSRs (keeps
    the cross-grade correlation) and draws K entry from trend +/- residual SD -> 10th/50th/90th
    percentile bands.
  * Rolling-origin backtest vs a naive "no change" forecast gives out-of-sample error.

Inputs : data/processed/lcusd_enrollment_by_grade.csv   (from phase1_data_pipeline.py)
Outputs: data/processed/forecast_totals.csv, forecast_by_grade.csv, backtest.csv,
         history_totals.csv
Run    : python phase2_model.py
"""
import numpy as np
import pandas as pd
from pathlib import Path

PROC = Path("data/processed")
GRADES = ["K"] + [str(i) for i in range(1, 13)]
HORIZON = 5          # forecast years
CSR_WINDOW = 6       # most recent grade transitions used
EXCLUDE = {2021}     # pandemic transition (2020 -> 2021), dropped from CSR pool if enough remain
K_TREND_YEARS = 8
K_DAMP = 0.8         # trend momentum decays 20% per year
N_SIMS = 5000
SEED = 340


def load_matrix():
    df = pd.read_csv(PROC / "lcusd_enrollment_by_grade.csv")
    df["grade"] = df["grade"].astype(str).replace({"TK": "K"})      # combine TK with K
    m = df.pivot_table(index="year", columns="grade", values="enrollment", aggfunc="sum").fillna(0)
    ungr = m["UNGR"] if "UNGR" in m.columns else pd.Series(0.0, index=m.index)
    return m[GRADES].astype(float), ungr.astype(float)


def csr_table(m):
    prev = m.shift(1)
    r = pd.DataFrame({g: m[g] / prev[GRADES[i - 1]] for i, g in enumerate(GRADES) if i > 0})
    return r.replace([np.inf, -np.inf], np.nan).dropna()


def csr_pool(m, window=CSR_WINDOW, exclude=EXCLUDE):
    r = csr_table(m).tail(window)
    keep = r.drop(index=[y for y in r.index if y in exclude])
    return keep if len(keep) >= 3 else r


def k_trend(k, horizon, years=K_TREND_YEARS, damp=K_DAMP):
    y = k.tail(years).values
    x = np.arange(len(y))
    slope, intercept = np.polyfit(x, y, 1)
    resid_sd = float(np.std(y - (intercept + slope * x), ddof=2)) if len(y) > 2 else 0.0
    level = intercept + slope * (len(y) - 1)                    # fitted (smoothed) last value
    steps = np.cumsum(damp ** np.arange(1, horizon + 1))
    return np.maximum(level + slope * steps, 0.0), resid_sd


def project(m, horizon=HORIZON):
    """Deterministic projection using mean CSRs."""
    csr = csr_pool(m).mean()
    kpath, _ = k_trend(m["K"], horizon)
    cur = m.iloc[-1].copy()
    rows = []
    for h in range(horizon):
        nxt = {"K": kpath[h]}
        for i, g in enumerate(GRADES[1:], 1):
            nxt[g] = cur[GRADES[i - 1]] * csr[g]
        cur = pd.Series(nxt)[GRADES]
        rows.append(cur)
    return pd.DataFrame(rows, index=range(int(m.index[-1]) + 1, int(m.index[-1]) + 1 + horizon))


def simulate(m, horizon=HORIZON, n=N_SIMS, seed=SEED):
    rng = np.random.default_rng(seed)
    pool = csr_pool(m).values                                   # (T, 12)
    kpath, ksd = k_trend(m["K"], horizon)
    cur = np.tile(m.iloc[-1].values, (n, 1))
    sims = np.zeros((n, horizon, len(GRADES)))
    for h in range(horizon):
        r = pool[rng.integers(0, len(pool), size=n)]            # one historical year's CSR vector per sim
        nxt = np.empty_like(cur)
        nxt[:, 0] = np.maximum(rng.normal(kpath[h], ksd, size=n), 0)
        nxt[:, 1:] = cur[:, :-1] * r
        sims[:, h, :] = nxt
        cur = nxt
    return sims


def backtest(m, max_h=3, min_hist=5):
    rows = []
    last = int(m.index[-1])
    for o in m.index:
        o = int(o)
        if len(m.loc[:o]) < min_hist or o >= last:
            continue
        hist = m.loc[:o]
        proj = project(hist, horizon=min(max_h, last - o))
        naive = hist.iloc[-1].sum()
        for y in proj.index:
            actual = m.loc[y].sum()
            rows.append(dict(origin=o, target=y, horizon=y - o, actual=actual,
                             model=proj.loc[y].sum(), naive=naive))
    bt = pd.DataFrame(rows)
    bt["model_ape"] = (bt.model - bt.actual).abs() / bt.actual * 100
    bt["naive_ape"] = (bt.naive - bt.actual).abs() / bt.actual * 100
    return bt


def main():
    m, ungr = load_matrix()
    print(f"History: {int(m.index[0])}-{int(m.index[-1])} ({len(m)} years), "
          f"{int(m.iloc[-1].sum() + ungr.iloc[-1]):,} students in {int(m.index[-1])}")

    # history totals (graded K-12 + ungraded) for the dashboard
    hist = pd.DataFrame({"year": m.index.astype(int), "total": (m.sum(axis=1) + ungr).values})
    hist.to_csv(PROC / "history_totals.csv", index=False)

    # backtest
    bt = backtest(m)
    bt.to_csv(PROC / "backtest.csv", index=False)
    summ = bt.groupby("horizon")[["model_ape", "naive_ape"]].mean().round(2)
    summ["n"] = bt.groupby("horizon").size()
    print("\nRolling-origin backtest, mean absolute % error of K-12 total (lower is better):")
    print(summ.rename(columns={"model_ape": "cohort model", "naive_ape": "naive (no change)"}))

    # forecast
    horizon = HORIZON
    years = list(range(int(m.index[-1]) + 1, int(m.index[-1]) + 1 + horizon))
    det = project(m, horizon)
    sims = simulate(m, horizon)
    tot = sims.sum(axis=2) + ungr.iloc[-1]
    q = np.percentile(tot, [10, 50, 90], axis=0)
    ft = pd.DataFrame({"year": years, "p10": q[0], "median": q[1], "p90": q[2],
                       "deterministic": det.sum(axis=1).values + ungr.iloc[-1]}).round(0)
    ft.to_csv(PROC / "forecast_totals.csv", index=False)

    gq = np.percentile(sims, [10, 50, 90], axis=0)               # (3, horizon, grades)
    fg = pd.DataFrame([dict(year=y, grade=g, p10=gq[0, h, i], median=gq[1, h, i], p90=gq[2, h, i])
                       for h, y in enumerate(years) for i, g in enumerate(GRADES)]).round(1)
    fg.to_csv(PROC / "forecast_by_grade.csv", index=False)

    print("\nForecast (K-12 total incl. ungraded):")
    print(ft.to_string(index=False))
    ratios = csr_pool(m).mean().round(3)
    print("\nMean cohort survival ratios used:", ratios.to_dict())
    print(f"\nSaved forecast files to {PROC.resolve()}")


if __name__ == "__main__":
    main()