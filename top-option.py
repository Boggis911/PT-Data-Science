{
"run\_meta": {
"version": "1.2",
"anchor\_start\_utc": "2025-08-25T12:43:26Z",
"horizon\_weeks": 10,
"week\_length\_days": 7,
"mu\_units": "simple\_return\_per\_week",
"sigma\_units": "stddev\_of\_simple\_return\_per\_week"
},
"tickers": \[
{
"ticker": "ACHR",
"current\_price": 8.95,
"weeks": \[
{ "week\_index": 1, "background": { "mu\_b": 0.0015, "sigma\_b": 0.07072, "skew\_b": -0.6, "student\_t\_df": 8 } },
{ "week\_index": 2, "background": { "mu\_b": 0.001611, "sigma\_b": 0.070385, "skew\_b": -0.6, "student\_t\_df": 8 } },
{ "week\_index": 3, "background": { "mu\_b": 0.001722, "sigma\_b": 0.07005, "skew\_b": -0.6, "student\_t\_df": 8 } },
{ "week\_index": 4, "background": { "mu\_b": 0.001833, "sigma\_b": 0.069714, "skew\_b": -0.6, "student\_t\_df": 8 } },
{ "week\_index": 5, "background": { "mu\_b": 0.001944, "sigma\_b": 0.069379, "skew\_b": -0.6, "student\_t\_df": 8 } },
{ "week\_index": 6, "background": { "mu\_b": 0.002056, "sigma\_b": 0.069044, "skew\_b": -0.6, "student\_t\_df": 8 } },
{ "week\_index": 7, "background": { "mu\_b": 0.002167, "sigma\_b": 0.068367, "skew\_b": -0.6, "student\_t\_df": 8 } },
{ "week\_index": 8, "background": { "mu\_b": 0.002278, "sigma\_b": 0.067974, "skew\_b": -0.6, "student\_t\_df": 8 } },
{ "week\_index": 9, "background": { "mu\_b": 0.002389, "sigma\_b": 0.067581, "skew\_b": -0.6, "student\_t\_df": 8 } },
{ "week\_index": 10, "background": { "mu\_b": 0.0025, "sigma\_b": 0.067189, "skew\_b": -0.6, "student\_t\_df": 8 } }
],
"events": \[
{ "id": "needham\_conf", "date": "2025-09-02T12:00:00Z", "type": "normal", "p\_event": 0.35, "mu": 0.0, "sigma": 0.02 },
{ "id": "db\_aviation\_forum", "date": "2025-09-03T12:00:00Z", "type": "normal", "p\_event": 0.30, "mu": 0.0, "sigma": 0.018 },
{
"id": "cpi\_aug2025",
"date": "2025-09-11T12:30:00Z",
"type": "trinomial",
"p\_event": 1.0,
"scenarios": \[
{ "label": "hot", "prob": 0.35, "mu": -0.03, "sigma": 0.025 },
{ "label": "inline", "prob": 0.45, "mu": 0.0, "sigma": 0.015 },
{ "label": "cool", "prob": 0.20, "mu": 0.02, "sigma": 0.02 }
]
},
{
"id": "fomc\_sep2025",
"date": "2025-09-17T18:00:00Z",
"type": "trinomial",
"p\_event": 1.0,
"scenarios": \[
{ "label": "cut\_25bp", "prob": 0.70, "mu": 0.008, "sigma": 0.015 },
{ "label": "hold", "prob": 0.20, "mu": -0.02, "sigma": 0.02 },
{ "label": "cut\_50bp", "prob": 0.10, "mu": 0.015, "sigma": 0.018 }
]
},
{
"id": "us\_shutdown\_risk",
"date": "2025-10-01T00:00:00Z",
"type": "binary",
"p\_event": 0.35,
"scenarios": \[
{ "label": "shutdown", "prob": 1.0, "mu": -0.025, "sigma": 0.02 }
]
}
],
"quality\_score": 0.78,
"quality\_flag": false
},
{
"ticker": "ANET",
"current\_price": 136.55,
"weeks": \[
{ "week\_index": 1, "background": { "mu\_b": 0.0035, "sigma\_b": 0.04715, "skew\_b": -0.3, "student\_t\_df": 10 } },
{ "week\_index": 2, "background": { "mu\_b": 0.003611, "sigma\_b": 0.046911, "skew\_b": -0.3, "student\_t\_df": 10 } },
{ "week\_index": 3, "background": { "mu\_b": 0.003722, "sigma\_b": 0.046672, "skew\_b": -0.3, "student\_t\_df": 10 } },
{ "week\_index": 4, "background": { "mu\_b": 0.003833, "sigma\_b": 0.046433, "skew\_b": -0.3, "student\_t\_df": 10 } },
{ "week\_index": 5, "background": { "mu\_b": 0.003944, "sigma\_b": 0.046195, "skew\_b": -0.3, "student\_t\_df": 10 } },
{ "week\_index": 6, "background": { "mu\_b": 0.004056, "sigma\_b": 0.045956, "skew\_b": -0.3, "student\_t\_df": 10 } },
{ "week\_index": 7, "background": { "mu\_b": 0.004167, "sigma\_b": 0.045717, "skew\_b": -0.3, "student\_t\_df": 10 } },
{ "week\_index": 8, "background": { "mu\_b": 0.004278, "sigma\_b": 0.045478, "skew\_b": -0.3, "student\_t\_df": 10 } },
{ "week\_index": 9, "background": { "mu\_b": 0.004389, "sigma\_b": 0.045239, "skew\_b": -0.3, "student\_t\_df": 10 } },
{ "week\_index": 10, "background": { "mu\_b": 0.0045, "sigma\_b": 0.045, "skew\_b": -0.3, "student\_t\_df": 10 } }
],
"events": \[
{
"id": "analyst\_day\_2025",
"date": "2025-09-11T22:00:00Z",
"type": "trinomial",
"p\_event": 1.0,
"scenarios": \[
{ "label": "bullish\_guide", "prob": 0.50, "mu": 0.04, "sigma": 0.03 },
{ "label": "inline", "prob": 0.35, "mu": 0.005, "sigma": 0.02 },
{ "label": "disappoint", "prob": 0.15, "mu": -0.05, "sigma": 0.035 }
]
},
{
"id": "cpi\_aug2025",
"date": "2025-09-11T12:30:00Z",
"type": "trinomial",
"p\_event": 1.0,
"scenarios": \[
{ "label": "hot", "prob": 0.35, "mu": -0.015, "sigma": 0.015 },
{ "label": "inline", "prob": 0.45, "mu": 0.0, "sigma": 0.01 },
{ "label": "cool", "prob": 0.20, "mu": 0.012, "sigma": 0.012 }
]
},
{
"id": "fomc\_sep2025",
"date": "2025-09-17T18:00:00Z",
"type": "trinomial",
"p\_event": 1.0,
"scenarios": \[
{ "label": "cut\_25bp", "prob": 0.70, "mu": 0.006, "sigma": 0.012 },
{ "label": "hold", "prob": 0.20, "mu": -0.012, "sigma": 0.015 },
{ "label": "cut\_50bp", "prob": 0.10, "mu": 0.012, "sigma": 0.015 }
]
},
{
"id": "us\_shutdown\_risk",
"date": "2025-10-01T00:00:00Z",
"type": "binary",
"p\_event": 0.35,
"scenarios": \[
{ "label": "shutdown", "prob": 1.0, "mu": -0.012, "sigma": 0.01 }
]
}
],
"quality\_score": 0.84,
"quality\_flag": false
},
{
"ticker": "CCJ",
"current\_price": 77.39,
"weeks": \[
{ "week\_index": 1, "background": { "mu\_b": 0.0045, "sigma\_b": 0.05186, "skew\_b": 0.2, "student\_t\_df": 11 } },
{ "week\_index": 2, "background": { "mu\_b": 0.0045, "sigma\_b": 0.051604, "skew\_b": 0.2, "student\_t\_df": 11 } },
{ "week\_index": 3, "background": { "mu\_b": 0.0045, "sigma\_b": 0.051348, "skew\_b": 0.2, "student\_t\_df": 11 } },
{ "week\_index": 4, "background": { "mu\_b": 0.0045, "sigma\_b": 0.051093, "skew\_b": 0.2, "student\_t\_df": 11 } },
{ "week\_index": 5, "background": { "mu\_b": 0.0045, "sigma\_b": 0.050837, "skew\_b": 0.2, "student\_t\_df": 11 } },
{ "week\_index": 6, "background": { "mu\_b": 0.0045, "sigma\_b": 0.050581, "skew\_b": 0.2, "student\_t\_df": 11 } },
{ "week\_index": 7, "background": { "mu\_b": 0.0045, "sigma\_b": 0.050325, "skew\_b": 0.2, "student\_t\_df": 11 } },
{ "week\_index": 8, "background": { "mu\_b": 0.0045, "sigma\_b": 0.050069, "skew\_b": 0.2, "student\_t\_df": 11 } },
{ "week\_index": 9, "background": { "mu\_b": 0.0045, "sigma\_b": 0.049814, "skew\_b": 0.2, "student\_t\_df": 11 } },
{ "week\_index": 10, "background": { "mu\_b": 0.0045, "sigma\_b": 0.049558, "skew\_b": 0.2, "student\_t\_df": 11 } }
],
"events": \[
{ "id": "nei\_uranium\_seminar", "date": "2025-10-27T14:00:00Z", "type": "normal", "p\_event": 0.8, "mu": 0.003, "sigma": 0.012 },
{ "id": "cpi\_aug2025", "date": "2025-09-11T12:30:00Z", "type": "normal", "p\_event": 1.0, "mu": 0.0, "sigma": 0.01 },
{
"id": "fomc\_sep2025",
"date": "2025-09-17T18:00:00Z",
"type": "trinomial",
"p\_event": 1.0,
"scenarios": \[
{ "label": "cut\_25bp", "prob": 0.70, "mu": 0.004, "sigma": 0.01 },
{ "label": "hold", "prob": 0.20, "mu": -0.01, "sigma": 0.012 },
{ "label": "cut\_50bp", "prob": 0.10, "mu": 0.008, "sigma": 0.012 }
]
},
{
"id": "us\_shutdown\_risk",
"date": "2025-10-01T00:00:00Z",
"type": "binary",
"p\_event": 0.35,
"scenarios": \[
{ "label": "shutdown", "prob": 1.0, "mu": -0.01, "sigma": 0.008 }
]
}
],
"quality\_score": 0.81,
"quality\_flag": false
},
{
"ticker": "CTRA",
"current\_price": 24.44,
"weeks": \[
{ "week\_index": 1, "background": { "mu\_b": 0.0015, "sigma\_b": 0.03536, "skew\_b": -0.1, "student\_t\_df": 10 } },
{ "week\_index": 2, "background": { "mu\_b": 0.001611, "sigma\_b": 0.035127, "skew\_b": -0.1, "student\_t\_df": 10 } },
{ "week\_index": 3, "background": { "mu\_b": 0.001722, "sigma\_b": 0.034894, "skew\_b": -0.1, "student\_t\_df": 10 } },
{ "week\_index": 4, "background": { "mu\_b": 0.001833, "sigma\_b": 0.034661, "skew\_b": -0.1, "student\_t\_df": 10 } },
{ "week\_index": 5, "background": { "mu\_b": 0.001944, "sigma\_b": 0.034428, "skew\_b": -0.1, "student\_t\_df": 10 } },
{ "week\_index": 6, "background": { "mu\_b": 0.002056, "sigma\_b": 0.034195, "skew\_b": -0.1, "student\_t\_df": 10 } },
{ "week\_index": 7, "background": { "mu\_b": 0.002167, "sigma\_b": 0.033962, "skew\_b": -0.1, "student\_t\_df": 10 } },
{ "week\_index": 8, "background": { "mu\_b": 0.002278, "sigma\_b": 0.033728, "skew\_b": -0.1, "student\_t\_df": 10 } },
{ "week\_index": 9, "background": { "mu\_b": 0.002389, "sigma\_b": 0.033495, "skew\_b": -0.1, "student\_t\_df": 10 } },
{ "week\_index": 10, "background": { "mu\_b": 0.0024, "sigma\_b": 0.033262, "skew\_b": -0.1, "student\_t\_df": 10 } }
],
"events": \[
{
"id": "opec\_plus\_meeting",
"date": "2025-09-07T00:00:00Z",
"type": "trinomial",
"p\_event": 1.0,
"scenarios": \[
{ "label": "addl\_cuts", "prob": 0.25, "mu": 0.04, "sigma": 0.03 },
{ "label": "status\_quo", "prob": 0.60, "mu": 0.0, "sigma": 0.015 },
{ "label": "increase", "prob": 0.15, "mu": -0.03, "sigma": 0.025 }
]
},
{
"id": "q3\_earnings",
"date": "2025-10-30T20:30:00Z",
"type": "trinomial",
"p\_event": 1.0,
"scenarios": \[
{ "label": "beat\_raise", "prob": 0.30, "mu": 0.05, "sigma": 0.04 },
{ "label": "inline", "prob": 0.50, "mu": 0.01, "sigma": 0.02 },
{ "label": "miss", "prob": 0.20, "mu": -0.05, "sigma": 0.04 }
]
},
{ "id": "cpi\_aug2025", "date": "2025-09-11T12:30:00Z", "type": "normal", "p\_event": 1.0, "mu": 0.0, "sigma": 0.008 },
{
"id": "fomc\_sep2025",
"date": "2025-09-17T18:00:00Z",
"type": "trinomial",
"p\_event": 1.0,
"scenarios": \[
{ "label": "cut\_25bp", "prob": 0.70, "mu": 0.003, "sigma": 0.01 },
{ "label": "hold", "prob": 0.20, "mu": -0.01, "sigma": 0.012 },
{ "label": "cut\_50bp", "prob": 0.10, "mu": 0.006, "sigma": 0.012 }
]
},
{
"id": "us\_shutdown\_risk",
"date": "2025-10-01T00:00:00Z",
"type": "binary",
"p\_event": 0.35,
"scenarios": \[
{ "label": "shutdown", "prob": 1.0, "mu": -0.012, "sigma": 0.01 }
]
}
],
"quality\_score": 0.82,
"quality\_flag": false
},
{
"ticker": "DKNG",
"current\_price": 47.98,
"weeks": \[
{ "week\_index": 1, "background": { "mu\_b": 0.003, "sigma\_b": 0.04008, "skew\_b": 0.1, "student\_t\_df": 9 } },
{ "week\_index": 2, "background": { "mu\_b": 0.003222, "sigma\_b": 0.039882, "skew\_b": 0.1, "student\_t\_df": 9 } },
{ "week\_index": 3, "background": { "mu\_b": 0.003444, "sigma\_b": 0.039683, "skew\_b": 0.1, "student\_t\_df": 9 } },
{ "week\_index": 4, "background": { "mu\_b": 0.003667, "sigma\_b": 0.039484, "skew\_b": 0.1, "student\_t\_df": 9 } },
{ "week\_index": 5, "background": { "mu\_b": 0.003889, "sigma\_b": 0.039286, "skew\_b": 0.1, "student\_t\_df": 9 } },
{ "week\_index": 6, "background": { "mu\_b": 0.004111, "sigma\_b": 0.039087, "skew\_b": 0.1, "student\_t\_df": 9 } },
{ "week\_index": 7, "background": { "mu\_b": 0.004333, "sigma\_b": 0.038888, "skew\_b": 0.1, "student\_t\_df": 9 } },
{ "week\_index": 8, "background": { "mu\_b": 0.004556, "sigma\_b": 0.03869, "skew\_b": 0.1, "student\_t\_df": 9 } },
{ "week\_index": 9, "background": { "mu\_b": 0.004778, "sigma\_b": 0.038491, "skew\_b": 0.1, "student\_t\_df": 9 } },
{ "week\_index": 10, "background": { "mu\_b": 0.005, "sigma\_b": 0.038292, "skew\_b": 0.1, "student\_t\_df": 9 } }
],
"events": \[
{
"id": "q3\_earnings",
"date": "2025-10-30T20:15:00Z",
"type": "trinomial",
"p\_event": 1.0,
"scenarios": \[
{ "label": "beat\_raise", "prob": 0.40, "mu": 0.08, "sigma": 0.06 },
{ "label": "inline", "prob": 0.40, "mu": 0.015, "sigma": 0.03 },
{ "label": "miss", "prob": 0.20, "mu": -0.07, "sigma": 0.05 }
]
},
{
"id": "cpi\_aug2025",
"date": "2025-09-11T12:30:00Z",
"type": "trinomial",
"p\_event": 1.0,
"scenarios": \[
{ "label": "hot", "prob": 0.35, "mu": -0.018, "sigma": 0.018 },
{ "label": "inline", "prob": 0.45, "mu": 0.0, "sigma": 0.012 },
{ "label": "cool", "prob": 0.20, "mu": 0.014, "sigma": 0.014 }
]
},
{
"id": "fomc\_sep2025",
"date": "2025-09-17T18:00:00Z",
"type": "trinomial",
"p\_event": 1.0,
"scenarios": \[
{ "label": "cut\_25bp", "prob": 0.70, "mu": 0.006, "sigma": 0.012 },
{ "label": "hold", "prob": 0.20, "mu": -0.015, "sigma": 0.015 },
{ "label": "cut\_50bp", "prob": 0.10, "mu": 0.012, "sigma": 0.015 }
]
},
{
"id": "us\_shutdown\_risk",
"date": "2025-10-01T00:00:00Z",
"type": "binary",
"p\_event": 0.35,
"scenarios": \[
{ "label": "shutdown", "prob": 1.0, "mu": -0.015, "sigma": 0.012 }
]
}
],
"quality\_score": 0.83,
"quality\_flag": false
}
]
}






# (Colab) bootstrap if needed:
# !pip install yfinance pandas numpy scipy matplotlib --quiet

import json, numpy as np, pandas as pd, matplotlib.pyplot as plt
from datetime import datetime, timezone, timedelta
from scipy.stats import skewnorm, t as student_t, norm
import re, unicodedata, math
import yfinance as yf
from IPython.display import display

# ----- Controls (minimal) -----
N_WEEKS            = 10          # hold N weeks, sell at start of week N+1
STRIKES_PER_EXPIRY = 10
N_SIMS_DEFAULT     = 50_000

# Small plotting defaults (used only if you later add charts)
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["figure.dpi"] = 110

# ---------- Sanitizers & parsing ----------
DASH_CLASS = r"[\u2010\u2011\u2012\u2013\u2014\u2015\u2212\uFE58\uFE63\uFF0D]"

def _sanitize_iso(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(DASH_CLASS, "-", s).replace("Z", "+00:00").strip()
    return s

def parse_utc(ts: str) -> datetime:
    dt = datetime.fromisoformat(_sanitize_iso(ts))
    if dt.tzinfo is None: dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)

def clean_expiry_for_yf(d: str) -> str:
    return _sanitize_iso(d)[:10]

# ---------- Business-day math (weekends only) ----------
def subtract_business_days(dt_utc: datetime, n: int) -> datetime:
    """Subtract n business days (Mon–Fri) from a timezone-aware datetime."""
    assert dt_utc.tzinfo is not None
    d = dt_utc
    step = -1 if n >= 0 else 1
    remaining = abs(n)
    while remaining > 0:
        d = d + timedelta(days=step)
        if d.weekday() < 5:  # Mon..Fri
            remaining -= 1
    return d

# ---------- Week utils ----------
def week_bounds(anchor_start_utc, week_index, week_len_days=7):
    start = parse_utc(anchor_start_utc) + timedelta(days=week_len_days*(week_index-1))
    end   = start + timedelta(days=week_len_days)
    return start, end

def interval_overlap_fraction(a_start, a_end, b_start, b_end) -> float:
    """Return overlap(a, b) / (a_end - a_start)."""
    start = max(a_start, b_start); end = min(a_end, b_end)
    if end <= start: return 0.0
    total = (a_end - a_start).total_seconds()
    return ((end - start).total_seconds() / total) if total > 0 else 0.0

# ---------- Market data ----------
def get_spot(ticker, fallback=None):
    S0 = None
    try:
        tk = yf.Ticker(ticker)
        fi = getattr(tk, "fast_info", {}) or {}
        S0 = fi.get("last_price") or fi.get("last_close")
        if S0 is None:
            hist = tk.history(period="1d")
            if not hist.empty:
                S0 = float(hist["Close"].iloc[-1])
    except Exception:
        S0 = None
    return float(S0) if S0 is not None else (float(fallback) if fallback is not None else None)

# ---------- Option chain helpers ----------
def fetch_calls_df(ticker, expiry_yyyy_mm_dd):
    tk = yf.Ticker(ticker)
    try:
        chain = tk.option_chain(expiry_yyyy_mm_dd).calls
    except Exception as ex:
        raise RuntimeError(f"Chain fetch failed {ticker}@{expiry_yyyy_mm_dd}: {ex}")
    if chain is None or chain.empty:
        raise RuntimeError(f"No calls for {ticker}@{expiry_yyyy_mm_dd}")
    return chain

def mid_from_row(row):
    bid = float(row.get("bid", np.nan)); ask = float(row.get("ask", np.nan)); last = float(row.get("lastPrice", np.nan))
    mids = [x for x in (bid, ask) if np.isfinite(x) and x > 0]
    mid = np.mean(mids) if len(mids) == 2 else (mids[0] if len(mids) == 1 else np.nan)
    return float(last) if (np.isfinite(last) and last > 0) else (float(mid) if np.isfinite(mid) and mid > 0 else np.nan)

# ---------- Background sampler (scaled for partial weeks) ----------
def _skewnorm_loc_for_mean(a, target_mean, scale):
    delta = a / np.sqrt(1 + a*a)
    return target_mean - scale * delta * np.sqrt(2/np.pi)

def sample_background_returns(n, mu_b, sigma_b, skew_b, df):
    loc_b = _skewnorm_loc_for_mean(skew_b, mu_b, sigma_b)
    r_skew = skewnorm.rvs(a=skew_b, loc=loc_b, scale=sigma_b, size=n)
    r_t = student_t.rvs(df=df, size=n) * (sigma_b/5.0)
    r = r_skew + r_t
    m, s = float(r.mean()), float(r.std(ddof=0))
    return mu_b + (r - m) * (sigma_b / s) if s > 0 else np.full(n, mu_b, float)

# ---------- Events over arbitrary interval ----------
def sample_event_jumps_in_interval(n, events, interval_start, interval_end):
    if not events: return np.zeros(n)
    jumps = np.zeros(n)
    for ev in events:
        typ = str(ev.get("type","")).lower()
        if typ not in ("normal","trinomial"):
            continue
        ev_dt = parse_utc(ev["date"])
        if not (interval_start <= ev_dt < interval_end):
            continue
        p = float(ev.get("p_event", 0.0))
        if p <= 0:
            continue
        fire = np.random.rand(n) < p
        m = int(fire.sum())
        if m == 0:
            continue
        if typ == "normal":
            mu, sig = float(ev["mu"]), float(ev["sigma"])
            jumps[fire] += norm.rvs(loc=mu, scale=sig, size=m)
        else:
            scen = ev["scenarios"]
            probs = np.array([s["prob"] for s in scen], float); probs /= probs.sum()
            choice = np.random.choice(len(scen), size=m, p=probs)
            idx_fire = np.where(fire)[0]
            for i, s in enumerate(scen):
                pick = idx_fire[choice==i]
                if pick.size:
                    jumps[pick] += norm.rvs(loc=float(s["mu"]), scale=float(s["sigma"]), size=pick.size)
    return jumps

# ---------- Background builder ----------
def build_week_backgrounds(forecast, ticker, weeks_needed):
    tk = next((x for x in forecast["tickers"] if x["ticker"].upper()==ticker.upper()), None)
    if tk is None: raise ValueError(f"{ticker} not in forecast JSON.")
    # Fill default weeks if missing/empty
    wk_list = tk.get("weeks") or []
    if not wk_list:
        wk_list = DEFAULT_WEEKS_TEMPLATE
        tk["weeks"] = wk_list
    wk_defs = {int(w["week_index"]): w["background"] for w in wk_list}
    last, out = {}, []
    for k in range(1, weeks_needed+1):
        last.update(wk_defs.get(k, {}))
        need = {"mu_b","sigma_b","skew_b","df"}
        if not need.issubset(last.keys()):
            missing = need - set(last.keys()); raise ValueError(f"Week {k} missing fields: {missing}")
        out.append({"week_index":k, "mu_b":float(last["mu_b"]), "sigma_b":float(last["sigma_b"]),
                    "skew_b":float(last["skew_b"]), "df":int(last["df"])})
    return out, tk.get("events", [])

# ---------- Simulate to arbitrary target datetime (handles partial week scaling) ----------
def simulate_prices_to_target(forecast, ticker, S0, target_utc, n=N_SIMS_DEFAULT, seed=None):
    if seed is not None: np.random.seed(seed)
    meta = forecast["run_meta"]
    anchor = parse_utc(meta["anchor_start_utc"])
    if target_utc <= anchor:
        return np.full(n, float(S0), float)  # nothing to evolve (or future anchor)
    week_len = int(meta.get("week_length_days", 7))
    # figure out how many whole week slots to cover
    weeks = 0
    while True:
        weeks += 1
        _, w_end = week_bounds(meta["anchor_start_utc"], weeks, week_len)
        if w_end >= target_utc: break
    bgs, events = build_week_backgrounds(forecast, ticker, weeks)
    S = np.full(n, float(S0), float)
    for w in bgs:
        ws, we = week_bounds(meta["anchor_start_utc"], w["week_index"], week_len)
        seg_end = min(we, target_utc)
        f = interval_overlap_fraction(ws, we, ws, seg_end)  # fraction of this week used
        if f <= 0:
            continue
        mu_scaled = w["mu_b"] * f
        sigma_scaled = w["sigma_b"] * math.sqrt(f)
        r_bg = sample_background_returns(n, mu_scaled, sigma_scaled, w["skew_b"], w["df"])
        r_ev = sample_event_jumps_in_interval(n, events, ws, seg_end)
        S *= (1.0 + r_bg + r_ev)
        if seg_end >= target_utc: break
    return S

# ---------- Utility: choose OTM strikes in [S0, 2*S0], up to M ----------
def pick_otm_strikes(S0, strikes, m=10, upper_mult=2.0):
    strikes = np.asarray(strikes, float)
    mask = (strikes >= S0) & (strikes <= upper_mult * S0)
    cand = np.sort(strikes[mask])
    if cand.size == 0:
        return np.array([], float)
    return cand[:m]

# ===== Cell 3 (updated labels & ROI as %) =====
forecast = json.loads(FORECAST_JSON)
now_utc = datetime.now(timezone.utc)

def week_distance(d1, d2):
    return (d2 - d1).total_seconds() / (7*24*3600)

def summarize_ticker_weeks_cumulative(forecast, ticker):
    """Cumulative (compounded) μ and σ up to W1, W2, W4, W8."""
    tk = next((x for x in forecast["tickers"] if x["ticker"].upper()==ticker.upper()), None)
    wk_list = (tk.get("weeks") or DEFAULT_WEEKS_TEMPLATE)
    wk_defs = {int(w["week_index"]): w["background"] for w in wk_list}

    targets = [1,2,4,8]
    rows = []
    mu_prod = 1.0
    var_sum = 0.0
    last_idx = 0
    for t in targets:
        for idx in range(last_idx+1, t+1):
            bg = wk_defs.get(idx, wk_defs.get(last_idx, {}))
            mu_i = float(bg.get("mu_b", 0.0))
            sigma_i = float(bg.get("sigma_b", 0.0))
            mu_prod *= (1.0 + mu_i)
            var_sum += sigma_i**2
        rows.append({"Horizon": f"W{t}", "μ_cum": mu_prod - 1.0, "σ_cum": np.sqrt(var_sum)})
        last_idx = t
    return pd.DataFrame(rows)

def _roi_bg(val):
    if pd.isna(val): return ""
    if val < 0:   return "background-color: #ffcccc"
    if val < 0.5: return "background-color: #fff3b0"
    return "background-color: #c6f6c6"

rows = []
for tk in forecast["tickers"]:
    ticker = tk["ticker"].upper()
    spot_ref = float(tk.get("spot_ref", np.nan))
    S0 = get_spot(ticker, fallback=spot_ref)
    if not np.isfinite(S0):
        print(f"[skip] {ticker}: no price"); continue

    # Header with compounded weekly summary
    print(f"\n=== {ticker} ===   Current price ≈ {S0:.2f}")
    cum_df = summarize_ticker_weeks_cumulative(forecast, ticker)
    display(cum_df.style.format({"μ_cum":"{:.2%}","σ_cum":"{:.2%}"}))

    # Expiries: keep ONLY those whose calendar week is N+1, then sell at start of that week
    try:
        all_expiries = yf.Ticker(ticker).options or []
    except Exception as ex:
        print(f"[skip] {ticker}: options list failed: {ex}")
        continue

    wk_len = int(forecast["run_meta"].get("week_length_days", 7))
    found_any = False

    for e_str in all_expiries:
        # parse expiry datetime (UTC end-of-day)
        try:
            exp_dt = datetime.fromisoformat(e_str + "T23:59:59+00:00").astimezone(timezone.utc)
        except Exception:
            continue

        # find which forecast week contains this expiry
        w_idx = 1
        while True:
            ws, we = week_bounds(forecast["run_meta"]["anchor_start_utc"], w_idx, wk_len)
            if we >= exp_dt:
                break
            w_idx += 1

        # enforce: expiry must be in week N+1
        if w_idx != N_WEEKS + 1:
            continue

        found_any = True

        # sell at start of week N+1
        t_sell = ws
        if t_sell <= now_utc:
            continue  # skip if sell time not in future

        # Simulate exactly N weeks to t_sell
        S_eval = simulate_prices_to_target(forecast, ticker, S0, t_sell, n=N_SIMS_DEFAULT)

        # Fetch calls and build mid prices
        try:
            calls = fetch_calls_df(ticker, e_str)
        except Exception as ex:
            print(f"[skip] {ticker}@{e_str}: {ex}")
            continue

        strikes = calls["strike"].astype(float).to_numpy()
        premiums = calls.apply(mid_from_row, axis=1).to_numpy()

        # Pick OTM strikes and align premiums
        K_sel = pick_otm_strikes(S0, strikes, m=STRIKES_PER_EXPIRY, upper_mult=2.0)
        if K_sel.size == 0:
            continue

        prem_map = dict(zip(strikes, premiums))
        prem_sel = np.array([prem_map.get(k, np.nan) for k in K_sel], float)
        valid = np.isfinite(prem_sel) & (prem_sel > 0)
        if not valid.any():
            continue
        K_sel = K_sel[valid]
        prem_sel = prem_sel[valid]

        # Expected intrinsic at t_sell as a proxy for sale price
        payoff = np.maximum(S_eval[:, None] - K_sel[None, :], 0.0)
        exp_payoff = payoff.mean(axis=0)

        # ROI versus today's premium for the N+1 expiry
        roi = (exp_payoff - prem_sel) / prem_sel
        prob_itm = (S_eval[:, None] >= K_sel[None, :]).mean(axis=0)

        # Scoring: drop <10% ITM, else ROI*(ProbITM+0.20)
        score = np.where(prob_itm < 0.10, 0.0, roi * (prob_itm + 0.20))

        for k, p, ep, r, pi, sc in zip(K_sel, prem_sel, exp_payoff, roi, prob_itm, score):
            rows.append({
                "Ticker": ticker,
                "Expiry": e_str,
                "Eval_Date": t_sell.strftime("%Y-%m-%d"),
                "Strike": float(k),
                "Premium": float(p),
                "Expected Payoff": float(ep),
                "ROI": float(r),        # decimal; displayed as %
                "Prob ITM": float(pi),  # decimal; displayed as %
                "Score": float(sc),
            })

    if not found_any:
        print(f"[info] {ticker}: no expiries in week N+1")

# Rank & show Top 5 (ROI displayed as %)
if rows:
    out = pd.DataFrame(rows)
    out.sort_values(by="Score", ascending=False, inplace=True)
    top5 = out.head(5).reset_index(drop=True)
    display(
        top5.style
            .format({
                "Strike": "{:.2f}",
                "Premium": "{:.2f}",
                "Expected Payoff": "{:.2f}",
                "ROI": "{:.2%}",
                "Prob ITM": "{:.1%}",
                "Score": "{:.3f}",
            })
            .map(_roi_bg, subset=["ROI"])
    )

    # Winner deep-dive: specs table (ROI as %) + underlying distribution with strike
    best = top5.iloc[0]
    tk_best, exp_best, k_best, prem_best = best["Ticker"], best["Expiry"], best["Strike"], best["Premium"]

    # compute t_sell = start of week that contains the expiry (this is week N+1)
    exp_dt = datetime.fromisoformat(exp_best + "T23:59:59+00:00").astimezone(timezone.utc)
    wk_len = int(forecast["run_meta"].get("week_length_days", 7))
    w_idx = 1
    while True:
        ws, we = week_bounds(forecast["run_meta"]["anchor_start_utc"], w_idx, wk_len)
        if we >= exp_dt:
            break
        w_idx += 1
    t_sell = ws  # start of week N+1

    S0_best = get_spot(tk_best)
    S_eval = simulate_prices_to_target(forecast, tk_best, S0_best, t_sell, n=N_SIMS_DEFAULT)

    payoff = np.maximum(S_eval - k_best, 0.0)
    roi_samples = (payoff - prem_best) / prem_best
    pcts = np.percentile(roi_samples, [75,90,95,98])

    spec_df = pd.DataFrame([{
        "Ticker": tk_best,
        "Expiry": exp_best,
        "Strike": k_best,
        "Premium": prem_best,
        "Expected Payoff": payoff.mean(),
        "Mean ROI": roi_samples.mean(),
        "Prob ITM": (S_eval >= k_best).mean(),
        "ROI@P75": pcts[0],
        "ROI@P90": pcts[1],
        "ROI@P95": pcts[2],
        "ROI@P98": pcts[3],
    }])
    display(spec_df.style.format({
        "Strike":"{:.2f}","Premium":"{:.2f}",
        "Expected Payoff":"{:.2f}",
        "Mean ROI":"{:.2%}","Prob ITM":"{:.1%}",
        "ROI@P75":"{:.2%}","ROI@P90":"{:.2%}",
        "ROI@P95":"{:.2%}","ROI@P98":"{:.2%}",
    }).map(_roi_bg, subset=["Mean ROI"]))

    fig, ax = plt.subplots()
    ax.hist(S_eval, bins=300, density=True, alpha=0.6, label="Underlying @ sell (start of week N+1)")
    ax.axvline(k_best, color="red", linestyle="--", linewidth=2, label=f"Strike={k_best:.2f}")
    if np.isfinite(S0_best):
        ax.axvline(S0_best, linestyle="--", linewidth=2, label=f"Current price now={S0_best:.2f}")
    ax.set_title(f"{tk_best} underlying distribution @ {t_sell.date()} | Exp {exp_best}")
    ax.set_xlabel("Underlying Price"); ax.set_ylabel("Density")
    ax.legend(); ax.grid(alpha=0.3)
    plt.show()
else:
    print("No candidates found.")
