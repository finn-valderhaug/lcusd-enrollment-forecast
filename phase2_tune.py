"""
Phase 2b: choose the cohort-model settings by backtest, then produce the final forecast.

Grid: CSR averaging window (2,3,4,6,8,all transitions) x kindergarten method (damped trend vs
mean of last 3 years). Each combo is scored by rolling-origin out-of-sample error (horizons 1-3).
The best combo is applied, phase2_model.main() is rerun, and the choice is saved.

Run after phase2_model.py is in the same folder:  python phase2_tune.py
"""
import json
import numpy as np
import pandas as pd
import phase2_model as pm

orig_k = pm.k_trend


def make_pool(window):
    def pool(m, window=window, exclude=pm.EXCLUDE):
        r = pm.csr_table(m).tail(window)
        keep = r.drop(index=[y for y in r.index if y in exclude])
        return keep if len(keep) >= 3 else r
    return pool


def make_k(mode):
    if mode == "trend":
        return orig_k

    def k(kser, horizon, **kw):
        return np.full(horizon, kser.tail(3).mean()), float(kser.tail(8).std(ddof=1))
    return k


m, ungr = pm.load_matrix()
rows = []
for w in [2, 3, 4, 6, 8, 99]:
    for kmode in ["trend", "mean3"]:
        pm.csr_pool, pm.k_trend = make_pool(w), make_k(kmode)
        bt = pm.backtest(m)
        signed = ((bt.model - bt.actual) / bt.actual * 100).mean()
        rows.append(dict(window=("all" if w == 99 else w), k_method=kmode,
                         mape=bt.model_ape.mean(), mape_h1=bt[bt.horizon == 1].model_ape.mean(),
                         mape_h3=bt[bt.horizon == 3].model_ape.mean(), bias_pct=signed,
                         _w=w))
naive = bt.naive_ape.mean()
res = pd.DataFrame(rows).sort_values("mape").reset_index(drop=True)
print(f"Backtest MAPE, horizons 1-3 pooled ({len(bt)} forecasts). Naive no-change = {naive:.2f}%")
print(res.drop(columns="_w").round(2).to_string(index=False))

best = res.iloc[0]
pm.csr_pool, pm.k_trend = make_pool(int(best["_w"])), make_k(best["k_method"])
cfg = dict(csr_window=str(best["window"]), k_method=best["k_method"],
           backtest_mape=round(float(best["mape"]), 2), naive_mape=round(float(naive), 2),
           selected_on="rolling-origin backtest, horizons 1-3")
(pm.PROC / "model_config.json").write_text(json.dumps(cfg, indent=2))
print(f"\nSelected: window={best['window']}, K method={best['k_method']}  "
      f"(MAPE {best['mape']:.2f}% vs naive {naive:.2f}%)\n" + "=" * 60)
pm.main()