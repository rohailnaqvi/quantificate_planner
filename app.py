# app.py — Quantificate Personal Investment Planner — Guide, Explore, Plan and Execute
# Header: compact two-column hero (logo + centered title), auto light/dark logos & light favicon
# Explore: full-width, Step-1-like visual layout for series checkboxes
# Plan: buffered group constraints with safe, deferred apply (no widget-state errors)

__version__ = "Quantificate PIP v1 (2025-10-03)"

import os, base64
import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
import altair as alt
from pandas.tseries.offsets import DateOffset
from typing import Tuple, Optional

# ---------- Optional optimizer ----------
try:
    from scipy.optimize import minimize
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False

# ---------- Brand assets ----------
CANDIDATE_ROOTS = [
    os.environ.get("QUANTIFICATE_ASSETS", "").strip(),
    r"C:\Users\rohai\OneDrive\Documents\Quantificate.ca\quantificate_brand_assets_structured_v4",
    os.path.join(os.path.dirname(__file__), "quantificate_brand_assets_structured_v4"),
    "quantificate_brand_assets_structured_v4",
]
def first_existing_dir(paths):
    for p in paths:
        if p and os.path.isdir(p):
            return p
    return ""
ASSETS_ROOT = first_existing_dir(CANDIDATE_ROOTS)

def find_one(possible_names):
    if not ASSETS_ROOT:
        return ""
    subfolders = ["logo", "logos", ""]
    for sub in subfolders:
        for name in possible_names:
            p = os.path.join(ASSETS_ROOT, sub, name)
            if os.path.exists(p):
                return p
    for root, _, files in os.walk(ASSETS_ROOT):
        for f in files:
            if f.lower() in [n.lower() for n in possible_names]:
                return os.path.join(root, f)
    return ""

LOGO_PRIMARY_LIGHT = find_one(["logo_light.svg","logo_light.png","logo.svg","logo.png"])
LOGO_LIGHT_TRANSPARENT = find_one(["logo_light_on_transparent.svg","logo_light_on_transparent.png"])
LOGO_DARK = find_one(["logo_dark.svg","logo_dark_on_transparent.svg","logo_dark.png","logo_dark_on_transparent.png"])

def find_favicon_light():
    if not ASSETS_ROOT:
        return ""
    favs = [
        os.path.join(ASSETS_ROOT, "favicon", "favicon_light.ico"),
        os.path.join(ASSETS_ROOT, "favicon_light.ico"),
        os.path.join(ASSETS_ROOT, "favicon", "favicon_light.png"),
        os.path.join(ASSETS_ROOT, "favicon_light.png"),
    ]
    for p in favs:
        if os.path.exists(p): return p
    return ""
FAVICON_PATH = find_favicon_light()

BRAND = {"ink":"#0F172A","teal":"#14B8A6","royal":"#6D28D9","gold":"#F59E0B","sky":"#0284C7","gray":"#94A3B8","white":"#FFFFFF"}
ALTAIR_BRAND_PALETTE = [BRAND["teal"], BRAND["royal"], BRAND["sky"], BRAND["gold"], "#111827", BRAND["gray"], "#e5e7eb", "#1f2937"]

def file_to_data_uri(path: str) -> str:
    if not path or not os.path.exists(path): return ""
    mime = "image/svg+xml" if path.lower().endswith(".svg") else "image/png"
    b64 = base64.b64encode(open(path,"rb").read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"

DATAURI_LOGO_LIGHT  = file_to_data_uri(LOGO_PRIMARY_LIGHT)
DATAURI_LOGO_DARK_T = file_to_data_uri(LOGO_LIGHT_TRANSPARENT) or file_to_data_uri(LOGO_DARK)

# ---------- Page & styling ----------
st.set_page_config(
    page_title="Quantificate — Personal Investment Planner",
    page_icon=FAVICON_PATH if os.path.exists(FAVICON_PATH) else None,
    layout="wide",
)
st.markdown(
    f"""
<style>
  :root {{
    --ink:{BRAND["ink"]}; --teal:{BRAND["teal"]}; --royal:{BRAND["royal"]};
    --gold:{BRAND["gold"]}; --sky:{BRAND["sky"]}; --gray:{BRAND["gray"]};
    --white:{BRAND["white"]};
  }}
  .block-container {{padding-top:.6rem; padding-bottom:2.2rem; max-width:1300px;}}
  h1,h2,h3 {{letter-spacing:.2px; color:var(--ink);}}

  .stButton>button {{
    border-radius:10px; border:1px solid color-mix(in srgb, var(--teal) 40%, white);
    color:var(--ink); background:color-mix(in srgb, var(--teal) 8%, white);
  }}
  .stButton>button:hover {{ border-color:var(--teal); background:color-mix(in srgb, var(--teal) 16%, white); }}
  .stDataFrame thead tr th {{ background:color-mix(in srgb, var(--teal) 8%, white) !important; }}

  /* Compact hero (logo left @60%, title centered) */
  .brand-hero {{
    display:grid; grid-template-columns:300px 1fr; gap:24px; align-items:center;
    margin:.4rem 0 1.0rem 0;
  }}
  .brand-logo {{ width:60%; max-width:260px; height:auto; display:block; }}
  .brand-title {{ text-align:center; }}
  .brand-title h1 {{ margin:0; font-size:2.2rem; line-height:1.15; }}
  .brand-sub {{ margin-top:.25rem; color:#475569; font-size:1.05rem; }}
  .logo-light {{ display:none; }} .logo-dark {{ display:none; }}
  @media (prefers-color-scheme: light){{ .logo-light {{display:block;}} }}
  @media (prefers-color-scheme: dark){{ 
    .logo-dark {{display:block;}} 
    .brand-title h1 {{color:#e5e7eb;}} .brand-sub {{color:#cbd5e1;}}
  }}

  .pill {{display:inline-block;padding:.2rem .55rem;border-radius:9999px;font-size:.8rem;font-weight:600}}
  .ok  {{background: color-mix(in srgb, var(--teal) 12%, white); color:#065f46; border:1px solid color-mix(in srgb, var(--teal) 48%, white)}}
  .warn{{background: color-mix(in srgb, var(--gold) 12%, white); color:#7c2d12; border:1px solid color-mix(in srgb, var(--gold) 48%, white)}}
  .err {{background:#fee2e2; color:#7f1d1d; border:1px solid #fecaca}}
</style>
""",
    unsafe_allow_html=True,
)

# ---------- Brand header ----------
if DATAURI_LOGO_LIGHT or DATAURI_LOGO_DARK_T:
    st.markdown(
        f"""
<div class="brand-hero">
  <div>{f'<img class="brand-logo logo-light" src="{DATAURI_LOGO_LIGHT}">' if DATAURI_LOGO_LIGHT else ''}{f'<img class="brand-logo logo-dark" src="{DATAURI_LOGO_DARK_T}">' if DATAURI_LOGO_DARK_T else ''}</div>
  <div class="brand-title">
    <h1>Personal Investment Planner</h1>
    <div class="brand-sub">Guide, Explore, Plan and Execute</div>
    <div style="margin-top:6px; font-size:.85rem; color:#64748b;">Build: Quantificate PIP v1 ({__version__.split()[-1]})</div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )
else:
    st.title("Personal Investment Planner")
    st.caption(f"Build: {__version__}")

# ---------- Universe ----------
INDEX_TICKERS = {
    "S&P 500": "^GSPC", "Dow Jones Industrial Average":"^DJI", "NYSE Composite":"^NYA",
    "Russell 1000":"^RUI", "Russell 2000":"^RUT", "Russell 3000":"^RUA",
    "NASDAQ Composite":"^IXIC", "Wilshire 5000":"^W5000",
    "SPDR Gold Shares (GLD)":"GLD", "iShares Silver Trust (SLV)":"SLV",
    "Bitcoin (BTC-USD)":"BTC-USD", "Ethereum (ETH-USD)":"ETH-USD",
}
GROUPS = {
    "Equity indices":["S&P 500","Dow Jones Industrial Average","NYSE Composite","Russell 1000","Russell 2000","Russell 3000","NASDAQ Composite","Wilshire 5000"],
    "Precious metals":["SPDR Gold Shares (GLD)","iShares Silver Trust (SLV)"],
    "Crypto":["Bitcoin (BTC-USD)","Ethereum (ETH-USD)"],
}
ALL = list(INDEX_TICKERS.keys())

ANCHORS_YEARS = [0,1,5,10,15,20]
ANCHOR_LABELS = {0:"Latest",1:"−1y",5:"−5y",10:"−10y",15:"−15y",20:"−20y"}

# ---------- Helpers ----------
def nearest(idx: pd.DatetimeIndex, target: pd.Timestamp) -> pd.Timestamp:
    if len(idx)==0: return pd.NaT
    if target <= idx[0]: return idx[0]
    if target >= idx[-1]: return idx[-1]
    return idx[idx.get_indexer([target], method="nearest")[0]]

def fmt_pct(x, signed=False): 
    if pd.isna(x): return ""
    s=f"{x*100:.2f}%"; return f"+{s}" if signed and x>=0 else s
def fmt_pct_abs(x): return "" if pd.isna(x) else f"{x*100:.2f}%"
def fmt_ratio(x): return "" if pd.isna(x) else f"{x:.2f}"
def fmt_int_with_commas(x): 
    if pd.isna(x): return ""
    try: return f"{int(round(x)):,}"
    except: return ""
def fmt_years(x): return "" if pd.isna(x) else f"{x:.2f}"
def as_float(x): 
    try: return float(x)
    except: return np.nan
def parse_money(s, fallback=100000):
    try: return max(int(float(str(s).replace(",","").replace("$","").strip())),1)
    except: return fallback
def window_label(T:int)->str: return "−10y" if T<=10 else ("−15y" if T<=15 else ("−20y" if T<=20 else "Max"))
def normalize_weights_in_state(enabled_list):
    if not enabled_list: return
    weights = st.session_state.get("weights", {})
    total = sum(max(0.0, float(weights.get(n,0.0))) for n in enabled_list)
    if total<=0: return
    f=100.0/total
    for n in enabled_list:
        st.session_state["weights"][n]=float(max(0.0,float(weights.get(n,0.0)))*f)

# ---------- Data ----------
@st.cache_data(ttl=24*60*60, show_spinner=True)
def fetch_history(tkr: str)->pd.DataFrame:
    df = yf.download(tkr, period="max", interval="1d", auto_adjust=False, progress=False)
    if df.empty: return df
    df = df[~df["Close"].isna()].copy(); df.sort_index(inplace=True); return df

@st.cache_data(ttl=24*60*60, show_spinner=False)
def fetch_histories()->dict[str,pd.DataFrame]:
    return {name: fetch_history(tkr) for name,tkr in INDEX_TICKERS.items()}

@st.cache_data(ttl=24*60*60, show_spinner=False)
def fetch_rf_history(code: str)->pd.DataFrame:
    df = yf.download(code, period="max", interval="1d", auto_adjust=False, progress=False)
    if df.empty: return df
    return df[~df["Close"].isna()].copy().sort_index()

def get_rf_for_horizon(h:int)->float:
    try:
        tnx, tyx = fetch_rf_history("^TNX"), fetch_rf_history("^TYX")
        if h<=10 and tnx is not None and not tnx.empty: rf = float(tnx["Close"].iloc[-1])/1000.0
        elif tyx is not None and not tyx.empty:       rf = float(tyx["Close"].iloc[-1])/1000.0
        else: rf = 0.0
        return float(rf) if np.isfinite(rf) else 0.0
    except: return 0.0

with st.spinner("Loading market histories…"):
    H = fetch_histories()

valid_latest = [df.index[-1] for df in H.values() if not df.empty]
REF_LATEST = max(valid_latest) if valid_latest else pd.NaT
TARGET_DATES = {y:(REF_LATEST - DateOffset(years=y)).date() if pd.notna(REF_LATEST) else None for y in ANCHORS_YEARS}

def get_value_on_or_near(df:pd.DataFrame, target_date:pd.Timestamp):
    if df.empty: return (pd.NaT, float("nan"))
    actual_date = nearest(df.index, pd.Timestamp(target_date))
    return (actual_date, float(df.loc[actual_date,"Close"]))

def window_start_for_label(label:str)->Optional[pd.Timestamp]:
    if label in ANCHOR_LABELS.values():
        y = {v:k for k,v in ANCHOR_LABELS.items()}[label]
        return pd.Timestamp(TARGET_DATES[y]) if TARGET_DATES[y] is not None else None
    if label=="2000-01-01": return pd.Timestamp("2000-01-01")
    if label=="Max":
        firsts = [df.index[0] for df in H.values() if not df.empty]
        return max(firsts) if firsts else None
    return None

# ======================================================================
# ====== Master/child checkbox helpers (top-level, used in tabs) =======
# ======================================================================

def toggle_hist_group(master_key: str, names: list[str]):
    """Explore tab: master 'Show all' toggles child checkboxes."""
    flag = bool(st.session_state.get(master_key, False))
    for n in names:
        st.session_state["hist_show"][n] = flag

def render_explore_group(grp: str, names: list[str], key_prefix: str):
    """Explore tab group renderer with working master checkbox."""
    hL, hR = st.columns([0.7, 0.3])
    with hL:
        st.markdown(f"*{grp}*")
    with hR:
        master_key = f"{key_prefix}_master_{grp}"
        # On first load, reflect current children
        if master_key not in st.session_state:
            st.session_state[master_key] = all(st.session_state["hist_show"].get(n, True) for n in names)
        # Master drives children via callback
        st.checkbox("Show all", key=master_key, on_change=toggle_hist_group, args=(master_key, names))
        # Optional visual cue (no state mutation)
        status_all_on = all(st.session_state["hist_show"].get(n, True) for n in names)
        st.caption("All selected" if status_all_on else "Some hidden")

    # Child checkboxes (2-column grid)
    cols = st.columns(2)
    for i, name in enumerate(names):
        with cols[i % 2]:
            st.session_state["hist_show"][name] = st.checkbox(
                name,
                value=st.session_state["hist_show"].get(name, True),
                key=f"{key_prefix}_{name}",
            )

def toggle_plan_group(master_key: str, names: list[str]):
    """Plan tab: master 'Show all' toggles child checkboxes."""
    flag = bool(st.session_state.get(master_key, False))
    for n in names:
        st.session_state["enabled_assets"][n] = flag

def render_plan_group(grp: str, names: list[str], master_key: str):
    """Plan tab group renderer with working master checkbox."""
    hL, hR = st.columns([0.7, 0.3])
    with hL:
        st.markdown(f"*{grp}*")
    with hR:
        # On first load, reflect current children
        if master_key not in st.session_state:
            st.session_state[master_key] = all(st.session_state["enabled_assets"].get(n, True) for n in names)
        # Master drives children via callback
        st.checkbox("Show all", key=master_key, on_change=toggle_plan_group, args=(master_key, names))
        # Optional visual cue (no state mutation)
        status_all_on = all(st.session_state["enabled_assets"].get(n, True) for n in names)
        st.caption("All selected" if status_all_on else "Some hidden")

    # Child checkboxes (2-column grid)
    cols = st.columns(2)
    for i, n in enumerate(names):
        with cols[i % 2]:
            st.session_state["enabled_assets"][n] = st.checkbox(
                n,
                value=st.session_state["enabled_assets"].get(n, True),
                key=f"en_{n}",
            )

# =========================================================
# =========================  TABS  ========================
# =========================================================
tabs = st.tabs(["👋 Welcome","🔎 Explore (History)","🧭 Plan (Projections)","📘 Guide: Index Definitions & ETFs"])

# =========================================================
# ======================  WELCOME  ========================
# =========================================================
with tabs[0]:
    st.subheader("Welcome to Quantificate")
    st.write(
        """
**This app is for anyone who’s ever thought “I should invest” and then immediately felt stuck.**  
We start with broad market building blocks (indexes & core assets), see how they behaved, then plan a simple allocation you can understand.

1. **Explore (History):** View how indices, metals, and crypto moved over time.
2. **Plan (Projections):** Choose weights, horizon, and (optionally) optimize for Sharpe.
3. **Guide:** Plain-English definitions + popular ETF examples.

*Educational sandbox — not investment advice.*
"""
    )

# =========================================================
# ================  EXPLORE (HISTORY)  ====================
# =========================================================
with tabs[1]:
    st.subheader("Growth of a Custom Investment")

    # Controls (form — columns created INSIDE the form)
    with st.form("history_controls", clear_on_submit=False):
        c_ctrl = st.columns([0.28, 0.22, 0.22, 0.18, 0.10])
        with c_ctrl[0]:
            start_amount_text = st.text_input("Starting amount ($)", value="1,000")
        with c_ctrl[1]:
            lookback = st.selectbox("Lookback window", ["1y","5y","10y","15y","20y","Since 2000-01-01"], index=4)
        with c_ctrl[2]:
            freq = st.selectbox("Display frequency", ["Weekly","Monthly"], index=1)
        with c_ctrl[4]:
            go_hist = st.form_submit_button("Update")

    # FULL-WIDTH series checklist
    st.markdown("**Series to show (chart only):**")
    if "hist_show" not in st.session_state:
        st.session_state["hist_show"] = {k: True for k in ALL}

    eA, eB, eC = st.columns(3)
    with eA: render_explore_group("Equity indices", GROUPS["Equity indices"], "histA")
    with eB: render_explore_group("Precious metals", GROUPS["Precious metals"], "histB")
    with eC: render_explore_group("Crypto", GROUPS["Crypto"], "histC")

    st.divider()

    # Compute chart series when submitted
    if ("hist_df" not in st.session_state) or go_hist:
        def parse_money_local(s, fallback=1000):
            try: return max(int(float(str(s).replace(",","").replace("$","").strip())),1)
            except: return fallback
        start_amount = parse_money_local(start_amount_text, 1000)
        lb_years = {"1y":1,"5y":5,"10y":10,"15y":15,"20y":20}.get(lookback,None)
        start_override = None if lb_years is not None else pd.Timestamp("2000-01-01")

        def resample(val_series, full_idx, fr):
            rule = "W-FRI" if fr=="Weekly" else "M"
            rs = val_series.resample(rule).last().dropna()
            rs.loc[full_idx[-1]] = float(val_series.iloc[-1])
            return rs[~rs.index.duplicated(keep="last")].sort_index()

        rows = []
        start_target = (start_override if start_override is not None
                        else (None if pd.isna(REF_LATEST) else pd.Timestamp((REF_LATEST - DateOffset(years=lb_years)).date())))
        for name, df in H.items():
            if df.empty or start_target is None: continue
            s_idx = nearest(df.index, start_target)
            if pd.isna(s_idx) or s_idx >= df.index[-1]: continue
            seg = df.loc[s_idx:, "Close"].copy()
            if seg.empty: continue
            base = float(seg.iloc[0])
            if base <= 0 or pd.isna(base): continue
            dollars = start_amount * (seg / base)
            disp = resample(dollars, df.index, freq)
            n = len(disp.index)
            rows.append(pd.DataFrame({"Date":disp.index.to_numpy(), "Value":np.asarray(disp.to_numpy()).reshape(n,), "Series":[name]*n}))
        st.session_state["hist_df"] = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["Date","Value","Series"])
        st.session_state["hist_note"] = f"Lookback: {lookback} • Frequency: {freq} • Start: ${int(start_amount):,}"

    # Chart (full width)
    hist_df = st.session_state.get("hist_df", pd.DataFrame())
    if hist_df.empty:
        st.info("Not enough data to draw the chart yet.")
    else:
        show_series = [k for k, v in st.session_state["hist_show"].items() if v]
        view = hist_df[hist_df["Series"].isin(show_series)].copy()
        if view.empty:
            st.info("All series hidden. Enable at least one above.")
        else:
            view["Date"] = pd.to_datetime(view["Date"])
            time_fmt = "%b %Y" if any(x in st.session_state.get("hist_note","") for x in ["1y","5y","10y"]) else "%Y"
            hover = alt.selection_point(fields=["Series"], on="mouseover", empty="all")
            chart = alt.Chart(view).mark_line().encode(
                x=alt.X("Date:T", axis=alt.Axis(format=time_fmt), title="Date"),
                y=alt.Y("Value:Q", axis=alt.Axis(format=",.0f"), title="Value ($)"),
                color=alt.Color("Series:N", title="Series", scale=alt.Scale(range=ALTAIR_BRAND_PALETTE)),
                opacity=alt.condition(hover, alt.value(1.0), alt.value(0.25)),
                tooltip=[alt.Tooltip("Date:T"), "Series:N", alt.Tooltip("Value:Q", format=",.0f")],
            ).add_selection(hover).properties(height=420)
            st.altair_chart(chart, use_container_width=True)
            st.caption(st.session_state.get("hist_note",""))

    # ------------------- Tables -------------------
    def build_uniform_price_table() -> pd.DataFrame:
        rows = []
        for name, df in H.items():
            if df.empty: continue
            row = {"Index": name}
            for y in ANCHORS_YEARS:
                t = pd.Timestamp(TARGET_DATES[y]) if TARGET_DATES[y] is not None else pd.NaT
                if pd.isna(t): row[ANCHOR_LABELS[y]] = ""; continue
                actual_date, val = get_value_on_or_near(df, t)
                txt = fmt_int_with_commas(val)
                if pd.notna(actual_date) and actual_date.date() != t.date():
                    txt = f"{txt} ({actual_date.date().isoformat()})"
                row[ANCHOR_LABELS[y]] = txt
            actual_date, val = get_value_on_or_near(df, pd.Timestamp("2000-01-01"))
            if pd.isna(actual_date): row["2000-01-01"] = ""
            else:
                txt = fmt_int_with_commas(val)
                if actual_date.date() != pd.Timestamp("2000-01-01").date():
                    txt = f"{txt} ({actual_date.date().isoformat()})"
                row["2000-01-01"] = txt
            row["Max"] = fmt_int_with_commas(float(df.loc[df.index[0], "Close"]))
            rows.append(row)
        table = pd.DataFrame(rows)
        if table.empty: return table
        desired = ["Index"] + [ANCHOR_LABELS[y] for y in ANCHORS_YEARS] + ["2000-01-01"] + ["Max"]
        desired = [c for c in desired if c in table.columns]
        table = table[desired]
        non_index_cols = [c for c in table.columns if c != "Index"]
        mask = table[non_index_cols].replace("", pd.NA).isna().all(axis=1)
        return table[~mask].reset_index(drop=True)

    def build_elapsed_years_table() -> pd.DataFrame:
        rows = []
        for name, df in H.items():
            if df.empty: continue
            idx_latest = df.index[-1]
            row = {"Index": name}
            for y in ANCHORS_YEARS:
                t = pd.Timestamp(TARGET_DATES[y]) if TARGET_DATES[y] is not None else pd.NaT
                if pd.isna(t): row[ANCHOR_LABELS[y]] = ""; continue
                actual_date, _ = get_value_on_or_near(df, t)
                if pd.notna(actual_date):
                    years = (idx_latest - actual_date).days / 365.25
                    val = fmt_years(years)
                    if actual_date.date() != t.date(): val = f"{val} ({actual_date.date().isoformat()})"
                else: val = ""
                row[ANCHOR_LABELS[y]] = val
            actual_date, _ = get_value_on_or_near(df, pd.Timestamp("2000-01-01"))
            if pd.notna(actual_date):
                years = (idx_latest - actual_date).days / 365.25
                val = fmt_years(years)
                if actual_date.date() != pd.Timestamp("2000-01-01").date():
                    val = f"{val} ({actual_date.date().isoformat()})"
            else: val = ""
            row["2000-01-01"] = val
            years_max = (idx_latest - df.index[0]).days / 365.25
            row["Max"] = fmt_years(years_max)
            rows.append(row)
        table = pd.DataFrame(rows)
        if table.empty: return table
        desired = ["Index"] + [ANCHOR_LABELS[y] for y in ANCHORS_YEARS] + ["2000-01-01"] + ["Max"]
        desired = [c for c in desired if c in table.columns]
        table = table[desired]
        non_index_cols = [c for c in table.columns if c != "Index"]
        mask = table[non_index_cols].replace("", pd.NA).isna().all(axis=1)
        return table[~mask].reset_index(drop=True)

    def build_pct_change_table() -> pd.DataFrame:
        rows = []
        for name, df in H.items():
            if df.empty: continue
            idx_latest = df.index[-1]
            latest_val = float(df.loc[idx_latest, "Close"])
            row = {"Index": name}
            for y in ANCHORS_YEARS:
                t = pd.Timestamp(TARGET_DATES[y]) if TARGET_DATES[y] is not None else pd.NaT
                if pd.isna(t): row[ANCHOR_LABELS[y]] = ""; continue
                if y == 0: row[ANCHOR_LABELS[y]] = "+0.00%"; continue
                actual_date, anchor_val = get_value_on_or_near(df, t)
                if pd.isna(anchor_val) or anchor_val == 0:
                    row[ANCHOR_LABELS[y]] = ""; continue
                pct = (latest_val / anchor_val - 1)
                txt = fmt_pct(pct, signed=True)
                if pd.notna(actual_date) and actual_date.date() != t.date():
                    txt = f"{txt} ({actual_date.date().isoformat()})"
                row[ANCHOR_LABELS[y]] = txt
            actual_date, anchor_val = get_value_on_or_near(df, pd.Timestamp("2000-01-01"))
            if pd.isna(anchor_val) or anchor_val == 0: row["2000-01-01"] = ""
            else:
                pct = (latest_val / anchor_val - 1)
                txt = fmt_pct(pct, signed=True)
                if pd.notna(actual_date) and actual_date.date() != pd.Timestamp("2000-01-01").date():
                    txt = f"{txt} ({actual_date.date().isoformat()})"
                row["2000-01-01"] = txt
            row["Max"] = fmt_pct((latest_val / float(df.loc[df.index[0], "Close"]) - 1), signed=True)
            rows.append(row)
        table = pd.DataFrame(rows)
        if table.empty: return table
        desired = ["Index"] + [ANCHOR_LABELS[y] for y in ANCHORS_YEARS] + ["2000-01-01"] + ["Max"]
        desired = [c for c in desired if c in table.columns]
        table = table[desired]
        non_index_cols = [c for c in table.columns if c != "Index"]
        mask = table[non_index_cols].replace("", pd.NA).isna().all(axis=1)
        return table[~mask].reset_index(drop=True)

    def build_cagr_table_numeric() -> pd.DataFrame:
        rows = []
        for name, df in H.items():
            if df.empty: continue
            idx_latest = df.index[-1]
            latest_val = float(df.loc[idx_latest, "Close"])
            row_num = {"Index": name}
            for y in ANCHORS_YEARS:
                t = pd.Timestamp(TARGET_DATES[y]) if TARGET_DATES[y] is not None else pd.NaT
                if y == 0 or pd.isna(t):
                    row_num[ANCHOR_LABELS[y]] = np.nan; continue
                actual_date, anchor_val = get_value_on_or_near(df, t)
                if pd.isna(anchor_val) or anchor_val <= 0:
                    row_num[ANCHOR_LABELS[y]] = np.nan; continue
                years = (idx_latest - actual_date).days / 365.25
                if years <= 0: row_num[ANCHOR_LABELS[y]] = np.nan; continue
                cagr = (latest_val / anchor_val) ** (1 / years) - 1
                row_num[ANCHOR_LABELS[y]] = cagr
            actual_date, anchor_val = get_value_on_or_near(df, pd.Timestamp("2000-01-01"))
            if pd.isna(anchor_val) or anchor_val <= 0:
                row_num["2000-01-01"] = np.nan
            else:
                years = (idx_latest - actual_date).days / 365.25
                row_num["2000-01-01"] = (latest_val / anchor_val) ** (1 / years) - 1 if years > 0 else np.nan
            years_max = (idx_latest - df.index[0]).days / 365.25
            row_num["Max"] = (latest_val / float(df.loc[df.index[0], "Close"])) ** (1 / years_max) - 1 if years_max > 0 else np.nan
            rows.append(row_num)
        num = pd.DataFrame(rows)
        if num.empty: return num
        desired = ["Index"] + [ANCHOR_LABELS[y] for y in ANCHORS_YEARS if y != 0] + ["2000-01-01"] + ["Max"]
        desired = [c for c in desired if c in num.columns]
        return num[desired].reset_index(drop=True)

    def build_cagr_table_display(num_df: pd.DataFrame) -> pd.DataFrame:
        disp = num_df.copy()
        for c in disp.columns:
            if c == "Index": continue
            disp[c] = disp[c].apply(lambda v: fmt_pct(v, signed=True) if pd.notna(v) else "")
        return disp

    def build_period_stdev_table(return_raw=False):
        rows, raw_sigmas = [], []
        labels = [ANCHOR_LABELS[y] for y in ANCHORS_YEARS if y != 0] + ["2000-01-01"] + ["Max"]
        for name, df in H.items():
            if df.empty: continue
            idx_latest = df.index[-1]
            row, raw_row = {"Index": name}, {"Index": name}
            for lab in labels:
                start_ts = window_start_for_label(lab)
                if start_ts is None:
                    row[lab] = ""; raw_row[lab] = np.nan; continue
                start = nearest(df.index, pd.Timestamp(start_ts))
                if pd.isna(start) or start >= idx_latest:
                    row[lab] = ""; raw_row[lab] = np.nan; continue
                win = df.loc[start:idx_latest, "Close"].copy()
                rets = win.pct_change().dropna()
                if len(rets) < 2:
                    row[lab] = ""; raw_row[lab] = np.nan; continue
                sigma = float(rets.std())
                row[lab] = fmt_pct_abs(sigma)
                raw_row[lab] = sigma
            rows.append(row); raw_sigmas.append(raw_row)
        table = pd.DataFrame(rows); raw = pd.DataFrame(raw_sigmas)
        labels_order = ["Index"] + labels
        if not table.empty: table = table[labels_order]
        if return_raw: return table, (raw[labels_order] if not raw.empty else raw)
        return table

    def build_annualized_vol_tables(raw_daily_sigma: pd.DataFrame):
        if raw_daily_sigma is None or raw_daily_sigma.empty:
            return pd.DataFrame(columns=["Index"]), pd.DataFrame(columns=["Index"])
        raw_df = raw_daily_sigma.copy()
        cols = [c for c in raw_df.columns if c != "Index"]
        raw_df[cols] = raw_df[cols] * np.sqrt(252.0)
        display_df = raw_df.copy()
        for c in cols:
            display_df[c] = display_df[c].apply(lambda v: fmt_pct_abs(v) if pd.notna(v) else "")
        return display_df, raw_df

    def scale_yield_decimal(series: pd.Series, code: str) -> pd.Series:
        if code in ("^TNX","^FVX","^TYX"): return series / 1000.0
        elif code == "^IRX": return series / 100.0
        else: return series / 100.0

    rf_TNX = fetch_rf_history("^TNX")
    rf_TYX = fetch_rf_history("^TYX")
    rf_FVX = fetch_rf_history("^FVX")
    rf_IRX = fetch_rf_history("^IRX")

    def rf_near(df_rf: pd.DataFrame, target: pd.Timestamp) -> float:
        if df_rf.empty: return np.nan
        ts = nearest(df_rf.index, target)
        if pd.isna(ts): return np.nan
        return float(df_rf.loc[ts, "Close"])

    def rf_snapshot_for_window(start_date: pd.Timestamp, window_label: str) -> float:
        if window_label == "−1y":
            v = rf_near(rf_IRX, start_date); return float(scale_yield_decimal(pd.Series([v]), "^IRX").iloc[0]) if not np.isnan(v) else np.nan
        elif window_label == "−5y":
            v = rf_near(rf_FVX, start_date); return float(scale_yield_decimal(pd.Series([v]), "^FVX").iloc[0]) if not np.isnan(v) else np.nan
        elif window_label == "−10y":
            v = rf_near(rf_TNX, start_date); return float(scale_yield_decimal(pd.Series([v]), "^TNX").iloc[0]) if not np.isnan(v) else np.nan
        elif window_label in ("−15y","−20y","2000-01-01","Max"):
            v = rf_near(rf_TYX, start_date); return float(scale_yield_decimal(pd.Series([v]), "^TYX").iloc[0]) if not np.isnan(v) else np.nan
        else: return np.nan

    def build_rf_snapshot_table():
        labels = [ANCHOR_LABELS[y] for y in ANCHORS_YEARS if y != 0] + ["2000-01-01"] + ["Max"]
        row = {"Index": "Risk-free snapshot (at window start)"}
        for lab in labels:
            start_ts = window_start_for_label(lab)
            if start_ts is None: row[lab] = ""; continue
            rf = rf_snapshot_for_window(pd.Timestamp(start_ts), lab)
            row[lab] = fmt_pct_abs(rf) if pd.notna(rf) else ""
        return pd.DataFrame([row])[["Index"] + labels]

    def build_sharpe_table_optionA(cagr_num: pd.DataFrame, ann_vol_raw: pd.DataFrame,
                                   rf_snap: pd.DataFrame) -> pd.DataFrame:
        labels = [ANCHOR_LABELS[y] for y in ANCHORS_YEARS if y != 0] + ["2000-01-01"] + ["Max"]
        cagr = cagr_num.set_index("Index"); sig_ann = ann_vol_raw.set_index("Index")
        rfA = rf_snap.set_index("Index").iloc[0]
        def parse_pct_str_to_decimal(s: str) -> float:
            if not isinstance(s, str) or s.strip() == "": return np.nan
            return float(s.replace("%", "")) / 100.0
        rfA_dec = rfA.apply(parse_pct_str_to_decimal)
        rows_A = []
        for idx in cagr.index:
            rowA = {"Index": idx}
            for lab in labels:
                r = as_float(cagr.at[idx, lab]) if lab in cagr.columns else np.nan
                s = as_float(sig_ann.at[idx, lab]) if lab in sig_ann.columns else np.nan
                rf_a = as_float(rfA_dec.get(lab, np.nan))
                valA = (r - rf_a) / s if (np.isfinite(r) and np.isfinite(rf_a) and np.isfinite(s) and s != 0) else np.nan
                rowA[lab] = fmt_ratio(valA) if pd.notna(valA) else ""
            rows_A.append(rowA)
        tabA = pd.DataFrame(rows_A)[["Index"] + labels]
        return tabA

    st.subheader("Historical Analysis — Tables")
    with st.expander("Section 1 — Index/Asset Historical Returns Analysis", expanded=False):
        if pd.notna(REF_LATEST):
            st.markdown(
                f"**Reference latest (common):** `{REF_LATEST.date().isoformat()}`  |  "
                f"**Dynamic targets:** " + " • ".join([f"{ANCHOR_LABELS[y]} → `{TARGET_DATES[y].isoformat()}`" for y in ANCHORS_YEARS if TARGET_DATES[y] is not None]) +
                f"  |  **Fixed target:** `2000-01-01`"
            )
        st.subheader("Table 1: Prices"); st.dataframe(build_uniform_price_table(), use_container_width=True, hide_index=True)
        st.subheader("Table 2: Years elapsed"); st.dataframe(build_elapsed_years_table(), use_container_width=True, hide_index=True)
        st.subheader("Table 3: % Change"); st.dataframe(build_pct_change_table(), use_container_width=True, hide_index=True)
        cagr_num_cached = build_cagr_table_numeric()
        st.subheader("Table 4: CAGR"); st.dataframe(build_cagr_table_display(cagr_num_cached), use_container_width=True, hide_index=True)

    with st.expander("Section 2 — Historical Risk Assessment (Volatility)", expanded=False):
        st.subheader("Table 5: Period-specific volatility (daily std dev over window, NOT annualized)")
        stdev_table_display, stdev_table_raw = build_period_stdev_table(return_raw=True)
        st.dataframe(stdev_table_display, use_container_width=True, hide_index=True)
        st.caption("Volatility is the standard deviation of daily simple returns within each window (not annualized).")
        st.subheader("Table 5b: Annualized volatility (daily std × √252)")
        annualized_vol_table_display, annualized_vol_raw = build_annualized_vol_tables(stdev_table_raw)
        st.dataframe(annualized_vol_table_display, use_container_width=True, hide_index=True)
        st.caption("Annualized volatility derived from Table 5’s daily stdevs by multiplying by √252. Used for Sharpe.")

    with st.expander("Section 3 — Historical Risk Adjusted Returns (Primary Method)", expanded=False):
        st.subheader("Table 6: Risk-free rate — snapshot at window start (per horizon)")
        rf_snapshot_table = build_rf_snapshot_table()
        st.dataframe(rf_snapshot_table, use_container_width=True, hide_index=True)
        st.caption("RF snapshot per window start uses Yahoo rates: 1y → IRX, 5y → FVX, 10y → TNX, 15y/20y/2000/Max → TYX.")
        st.subheader("Table 7: Sharpe Ratio — Option A (CAGR − RF snapshot) / Annualized Volatility (from 5b)")
        sharpe_A_table = build_sharpe_table_optionA(
            cagr_num=cagr_num_cached, ann_vol_raw=annualized_vol_raw, rf_snap=rf_snapshot_table
        )
        st.dataframe(sharpe_A_table, use_container_width=True, hide_index=True)

# =========================================================
# ================   PLAN (PROJECTIONS)  ==================
# =========================================================
with tabs[2]:
    st.subheader("Plan: Build a Simple Forward Projection")
    st.caption("Set allocations, choose a horizon, then Calculate or Optimize. Constraints only apply on 'Apply constraints'.")

    # Step 1 — Choose assets
    st.markdown("**Step 1 — Choose assets to include**")
    if "enabled_assets" not in st.session_state:
        st.session_state["enabled_assets"] = {k: True for k in ALL}

    cA, cB, cC = st.columns(3)
    with cA:
        render_plan_group("Equity indices", GROUPS["Equity indices"], "master_Equity indices")
    with cB:
        render_plan_group("Precious metals", GROUPS["Precious metals"], "master_Precious metals")
    with cC:
        render_plan_group("Crypto", GROUPS["Crypto"], "master_Crypto")

    enabled_list = [k for k, v in st.session_state["enabled_assets"].items() if v]

    # Step 2 — Amount, horizon, weights
    st.markdown("**Step 2 — Set total, horizon & weights**")
    if "weights" not in st.session_state:
        eq = round(100 / len(ALL), 2)
        st.session_state["weights"] = {k: eq for k in ALL}
    c_top = st.columns([1.2, 1])
    with c_top[0]:
        inv_text = st.text_input("Total investment ($)", value="1,000")
    with c_top[1]:
        horizon = st.selectbox("Horizon (years)", list(range(5, 26)), index=10)

    st.markdown("*Weights (% of portfolio)*")
    cA, cB, cC = st.columns(3)
    def weight_box(names, col):
        with col:
            for n in names:
                st.session_state["weights"][n] = st.number_input(
                    n, min_value=0.0, max_value=100.0, step=0.25,
                    value=float(st.session_state["weights"][n]),
                    key=f"w_{n}", disabled=(n not in enabled_list)
                )
    weight_box(GROUPS["Equity indices"], cA); weight_box(GROUPS["Precious metals"], cB); weight_box(GROUPS["Crypto"], cC)

    w_sum = sum(float(st.session_state["weights"].get(n, 0.0)) for n in enabled_list) if enabled_list else 0.0
    sum_col, _ = st.columns([0.45, 0.55])
    if abs(w_sum - 100.0) <= 0.01:
        sum_col.markdown(f"<span class='pill ok'>Total weights (enabled): {w_sum:.2f}%</span>", unsafe_allow_html=True)
    elif w_sum < 100.0:
        sum_col.markdown(f"<span class='pill warn'>Total weights (enabled): {w_sum:.2f}% — add {100.0 - w_sum:.2f}%</span>", unsafe_allow_html=True)
    else:
        sum_col.markdown(f"<span class='pill err'>Total weights (enabled): {w_sum:.2f}% — reduce {w_sum - 100.0:.2f}%</span>", unsafe_allow_html=True)

    # ---------- Advanced — Constraints (deferred apply) ----------
    with st.expander("Advanced (optional) — Weight constraints for Optimizer", expanded=False):
        if "min_on" not in st.session_state: st.session_state["min_on"] = False
        if "max_on" not in st.session_state: st.session_state["max_on"] = False
        c1, c2 = st.columns(2)
        with c1: st.session_state["min_on"] = st.checkbox("Enable minimums", value=st.session_state["min_on"])
        with c2: st.session_state["max_on"] = st.checkbox("Enable caps", value=st.session_state["max_on"])

        if "group_apply_toggle" not in st.session_state: st.session_state["group_apply_toggle"] = True
        st.session_state["group_apply_toggle"] = st.checkbox(
            "Enable master group constraint inputs (and apply to per-asset ON SUBMIT)",
            value=st.session_state["group_apply_toggle"]
        )
        group_inputs_enabled = bool(st.session_state["group_apply_toggle"])

        if "buf_min" not in st.session_state: st.session_state["buf_min"] = {k: 0.0 for k in ALL}
        if "buf_max" not in st.session_state: st.session_state["buf_max"] = {k: 100.0 for k in ALL}

        # ======= HANDLE PENDING COPY (BEFORE WIDGETS) =======
        if st.session_state.get("_constraints_apply_pending", False):
            if st.session_state.get("group_apply_toggle", True):
                p_gmin = st.session_state.get("_pending_group_min", {})
                p_gmax = st.session_state.get("_pending_group_max", {})
                for grp, names in GROUPS.items():
                    gmin = float(p_gmin.get(grp, 0.0))
                    gmax = float(p_gmax.get(grp, 100.0))
                    for n in names:
                        if not st.session_state["enabled_assets"].get(n, False): continue
                        if st.session_state.get("min_on", False):
                            st.session_state["buf_min"][n] = gmin
                            st.session_state[f"cmin_{n}"] = gmin
                        if st.session_state.get("max_on", False):
                            st.session_state["buf_max"][n] = gmax
                            st.session_state[f"cmax_{n}"] = gmax
            st.session_state["_constraints_apply_pending"] = False
            st.session_state.pop("_pending_group_min", None)
            st.session_state.pop("_pending_group_max", None)
        # ======= END PENDING HANDLER =======

        existing_gmin = {g: 0.0 for g in GROUPS.keys()}
        existing_gmax = {g: 100.0 for g in GROUPS.keys()}
        existing_gmin.update(st.session_state.get("group_min", {}))
        existing_gmax.update(st.session_state.get("group_max", {}))

        with st.form("group_constraints_form", clear_on_submit=False):
            st.subheader("Group-level constraints (buffered)")
            gA, gB, gC = st.columns(3)
            gmin_temp, gmax_temp = {}, {}
            for grp, col in zip(GROUPS.keys(), [gA, gB, gC]):
                with col:
                    active_grp = any(st.session_state["enabled_assets"].get(n, False) for n in GROUPS[grp])
                    disable_min = (not group_inputs_enabled) or (not st.session_state["min_on"]) or (not active_grp)
                    disable_max = (not group_inputs_enabled) or (not st.session_state["max_on"]) or (not active_grp)
                    gmin_val = st.number_input(
                        f"{grp} — Master Min %",
                        min_value=0.0, max_value=100.0, step=0.5,
                        value=float(existing_gmin.get(grp, 0.0)),
                        key=f"gmin_temp_{grp}", disabled=disable_min
                    )
                    gmax_val = st.number_input(
                        f"{grp} — Master Max %",
                        min_value=0.0, max_value=100.0, step=0.5,
                        value=float(existing_gmax.get(grp, 100.0)),
                        key=f"gmax_temp_{grp}", disabled=disable_max
                    )
                    gmin_temp[grp] = float(gmin_val); gmax_temp[grp] = float(gmax_val)

            st.divider()
            st.subheader("Per-asset constraints (the optimizer only reads these)")
            g1, g2, g3 = st.columns(3)
            for names, col in zip(GROUPS.values(), [g1, g2, g3]):
                with col:
                    for n in names:
                        active_asset = st.session_state["enabled_assets"].get(n, False)
                        r = st.columns(2)
                        with r[0]:
                            vmin = st.number_input(
                                f"Min % — {n}", min_value=0.0, max_value=100.0, step=0.5,
                                value=float(st.session_state.get(f"cmin_{n}", st.session_state["buf_min"].get(n, 0.0))),
                                key=f"cmin_{n}", disabled=(not active_asset or not st.session_state["min_on"]),
                            )
                            st.session_state["buf_min"][n] = float(vmin)
                        with r[1]:
                            vmax = st.number_input(
                                f"Max % — {n}", min_value=0.0, max_value=100.0, step=0.5,
                                value=float(st.session_state.get(f"cmax_{n}", st.session_state["buf_max"].get(n, 100.0))),
                                key=f"cmax_{n}", disabled=(not active_asset or not st.session_state["max_on"]),
                            )
                            st.session_state["buf_max"][n] = float(vmax)
            applied = st.form_submit_button("Apply constraints")

        if applied:
            st.session_state["group_min"] = dict(gmin_temp)
            st.session_state["group_max"] = dict(gmax_temp)
            st.session_state["_constraints_apply_pending"] = True
            st.session_state["_pending_group_min"] = dict(gmin_temp)
            st.session_state["_pending_group_max"] = dict(gmax_temp)
            st.rerun()

    # Buttons
    b1, b2 = st.columns([0.5, 0.5])
    do_calc = b1.button("Calculate")
    do_opt  = b2.button("Optimize for Sharpe")

    # ---------- Builders shared with Plan ----------
    @st.cache_data(ttl=24*60*60, show_spinner=False)
    def get_cagr_table() -> pd.DataFrame:
        return build_cagr_table_numeric()

    def build_vol_and_corr(enabled: list[str], T: int) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
        if pd.isna(REF_LATEST) or T is None:
            return pd.DataFrame(columns=["Index","AnnVol"]), pd.DataFrame(), ""
        start = pd.Timestamp((REF_LATEST - DateOffset(years=int(T))).date())
        end = REF_LATEST
        series = {}
        for n in enabled:
            df = H.get(n, pd.DataFrame())
            if df.empty: continue
            s = nearest(df.index, start)
            if pd.isna(s) or s >= df.index[-1]: continue
            r = df.loc[s:end, "Close"].pct_change().dropna()
            if len(r) >= 3:
                r = r.copy(); r.name = n; series[n] = r
        if len(series) < 1:
            return pd.DataFrame(columns=["Index","AnnVol"]), pd.DataFrame(), "Not enough overlapping data."
        R = pd.concat(series.values(), axis=1, join="inner")
        R.columns = list(series.keys())
        if R.shape[0] < 3:
            return pd.DataFrame(columns=["Index","AnnVol"]), pd.DataFrame(), "Too few overlapping observations."
        daily_sigma = R.std(axis=0)
        ann_vol_df = pd.DataFrame({"Index": R.columns, "AnnVol": (daily_sigma.to_numpy() * np.sqrt(252))})
        ann_vol_df = ann_vol_df.groupby("Index", as_index=False)["AnnVol"].mean()
        corr = R.corr().round(2); corr.index = corr.index.astype(str); corr.columns = corr.columns.astype(str)
        note = f"Daily returns from {R.index[0].date()} to {R.index[-1].date()} • n={R.shape[0]} days."
        return ann_vol_df, corr, note

    # ---------- Optimize ----------
    if do_opt:
        normalize_weights_in_state(enabled_list)
        cagr_num = get_cagr_table().set_index("Index")
        label = window_label(horizon)
        ann_vol, corr, note = build_vol_and_corr(enabled_list, horizon)

        def g(i: str) -> float:
            return as_float(cagr_num.at[i, label]) if (i in cagr_num.index and label in cagr_num.columns) else np.nan
        def s_(i: str) -> float:
            try: return float(ann_vol.loc[ann_vol["Index"] == i, "AnnVol"].values[0])
            except Exception: return np.nan

        if not SCIPY_OK:
            st.warning("SciPy not available: cannot optimize. Install `scipy`.")
        else:
            usable = [i for i in enabled_list if np.isfinite(g(i)) and np.isfinite(s_(i))]
            if len(usable) < 2 or corr.empty:
                st.error("Optimization needs at least two enabled assets with data.")
            else:
                mu = np.array([g(i) for i in usable], dtype=float).reshape(-1)
                sig = np.array([s_(i) for i in usable], dtype=float).reshape(-1)
                Rm = corr.loc[usable, usable].to_numpy(dtype=float)
                np.fill_diagonal(Rm, 1.0); Rm = np.nan_to_num(Rm, nan=0.0)
                D = np.diag(sig); cov = D @ Rm @ D; cov = (cov + cov.T) / 2.0
                rf = get_rf_for_horizon(horizon)

                lb = np.array([float(st.session_state.get(f"cmin_{i}", st.session_state['buf_min'].get(i, 0.0))) / 100.0 for i in usable], dtype=float)
                ub = np.array([float(st.session_state.get(f"cmax_{i}", st.session_state['buf_max'].get(i, 100.0))) / 100.0 for i in usable], dtype=float)
                if not bool(st.session_state.get("min_on", False)): lb = np.zeros_like(lb)
                if not bool(st.session_state.get("max_on", False)): ub = np.ones_like(ub)
                if np.any(lb > ub):
                    st.error("Per-asset constraints infeasible (a min exceeds its max). Adjust and Apply constraints.")
                else:
                    mu_ex = mu - rf
                    span = np.maximum(ub - lb, 0.0)
                    w0 = lb + (span / max(span.sum(), 1e-12))
                    w0 = np.clip(w0, lb, ub)
                    ssum = w0.sum(); w0 = (w0 / ssum) if ssum > 0 else np.ones_like(mu) / len(mu)
                    def neg_sharpe(w):
                        w = np.asarray(w, dtype=float).reshape(-1)
                        num = float(w @ mu_ex); den = float(np.sqrt(max(w @ cov @ w, 1e-16)))
                        return -num / den
                    cons = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
                    bnds = tuple((float(lo), float(hi)) for lo, hi in zip(lb, ub))
                    res = minimize(neg_sharpe, w0, method="SLSQP", bounds=bnds, constraints=cons, options={"maxiter": 800})
                    w = np.clip(res.x if res.success else w0, lb, ub)
                    ssum = w.sum(); w = (w / ssum) if ssum > 0 else np.ones_like(mu) / len(mu)
                    for i, name in enumerate(usable):
                        st.session_state["weights"][name] = float(w[i] * 100.0)
                    st.success("Optimized weights applied. Updating table & chart…")
                    do_calc = True

    # ---------- Projection state init ----------
    if "proj_lines" not in st.session_state: st.session_state["proj_lines"] = pd.DataFrame()
    if "proj_table_display" not in st.session_state: st.session_state["proj_table_display"] = pd.DataFrame()
    if "proj_flags" not in st.session_state: st.session_state["proj_flags"] = {}
    if "corr_view" not in st.session_state: st.session_state["corr_view"] = pd.DataFrame()
    if "corr_note" not in st.session_state: st.session_state["corr_note"] = ""
    if "legend_series" not in st.session_state: st.session_state["legend_series"] = ["Portfolio"]

    # ---------- Calculate ----------
    if do_calc:
        normalize_weights_in_state(enabled_list)
        invest = parse_money(inv_text, 1000)
        label = window_label(horizon)
        rf = get_rf_for_horizon(horizon)

        cagr_num = get_cagr_table().set_index("Index")
        ann_vol, corr, note = build_vol_and_corr([a for a in enabled_list], horizon)

        rows = []
        for n in ALL:
            if n not in enabled_list:
                rows.append({"Index": n, "Allocation %": 0.0, "Allocation $": 0.0,
                             "RF": rf, "CAGR": np.nan, "AnnVol": np.nan, "Sharpe": np.nan,
                             "Future %": np.nan, "Future $": np.nan})
                continue
            w = float(st.session_state["weights"][n]) / 100.0
            dollars = invest * w
            r = as_float(cagr_num.at[n, label]) if (n in cagr_num.index and label in cagr_num.columns) else np.nan
            rows.append({"Index": n, "Allocation %": w*100.0, "Allocation $": dollars, "RF": rf, "CAGR": r, "AnnVol": np.nan})

        table = pd.DataFrame(rows)
        if "AnnVol" not in table.columns: table["AnnVol"] = np.nan
        if not ann_vol.empty:
            table = table.merge(ann_vol.rename(columns={"AnnVol": "AnnVol_new"}), on="Index", how="left")
            table["AnnVol"] = pd.to_numeric(table["AnnVol"], errors="coerce")
            table["AnnVol_new"] = pd.to_numeric(table["AnnVol_new"], errors="coerce")
            table["AnnVol"] = table["AnnVol_new"].combine_first(table["AnnVol"])
            table = table.drop(columns=["AnnVol_new"])
        for col in ["Allocation %","Allocation $","RF","CAGR","AnnVol"]:
            table[col] = pd.to_numeric(table[col], errors="coerce")
        table["Sharpe"] = ((table["CAGR"] - table["RF"]) / table["AnnVol"]).replace([np.inf, -np.inf], np.nan)
        table["Future %"] = np.where(table["CAGR"].notna(), (1 + table["CAGR"]) ** horizon * 100.0, np.nan)
        table["Future $"] = table["Allocation $"] * np.where(table["CAGR"].notna(), (1 + table["CAGR"]) ** horizon, np.nan)

        en = table[table["Allocation %"] > 0].copy()
        wv = (en["Allocation %"] / 100.0).to_numpy(float)
        rv = en["CAGR"].to_numpy(float)
        port_factor = float(np.sum(wv * (1 + rv) ** horizon)) if (len(wv) > 0 and np.all(np.isfinite(rv))) else np.nan
        port_cagr = port_factor ** (1 / horizon) - 1 if np.isfinite(port_factor) else np.nan

        port_sigma = np.nan
        if (not corr.empty) and (not en.empty):
            names = [i for i in en["Index"] if i in corr.index]
            if len(names) >= 1:
                wv2 = np.array([float(st.session_state["weights"][i]) / 100.0 for i in names], float)
                av_map = dict(zip(ann_vol["Index"], ann_vol["AnnVol"])) if not ann_vol.empty else {}
                sv = np.array([as_float(av_map.get(i, np.nan)) for i in names], float)
                m = np.isfinite(wv2) & np.isfinite(sv)
                if m.sum() >= 1:
                    Rm = corr.loc[names, names].to_numpy(float)
                    Rm = Rm[np.ix_(m, m)]
                    np.fill_diagonal(Rm, 1.0)
                    D = np.diag(sv[m]); cov = D @ Rm @ D
                    port_sigma = float(np.sqrt(max(wv2[m] @ cov @ wv2[m], 0.0)))
        port_sharpe = (port_cagr - rf) / port_sigma if (np.isfinite(port_cagr) and np.isfinite(port_sigma) and port_sigma > 0) else np.nan

        totals = pd.DataFrame([{
            "Index": "Portfolio", "Allocation %": en["Allocation %"].sum(), "Allocation $": invest,
            "RF": rf, "CAGR": port_cagr, "AnnVol": port_sigma, "Sharpe": port_sharpe,
            "Future %": (port_factor * 100.0) if np.isfinite(port_factor) else np.nan,
            "Future $": en["Future $"].sum(skipna=True),
        }])
        display = pd.concat([table, totals], ignore_index=True)

        show_tbl = display.copy()
        show_tbl["Allocation %"] = show_tbl["Allocation %"].map(lambda v: f"{v:.2f}%")
        show_tbl["Allocation $"] = show_tbl["Allocation $"].map(lambda v: f"{int(round(v)):,}")
        show_tbl["RF"] = show_tbl["RF"].map(lambda v: "" if pd.isna(v) else fmt_pct(v))
        show_tbl["CAGR"] = show_tbl["CAGR"].map(lambda v: "" if pd.isna(v) else fmt_pct(v))
        show_tbl["AnnVol"] = show_tbl["AnnVol"].map(lambda v: "" if pd.isna(v) else fmt_pct_abs(v))
        def fmt_sharpe(v):
            try:
                vf = float(v); return f"{vf:.2f}" if np.isfinite(vf) else "—"
            except Exception:
                return "—"
        show_tbl["Sharpe"] = show_tbl["Sharpe"].map(fmt_sharpe)
        show_tbl["Future %"] = show_tbl["Future %"].map(lambda v: "" if pd.isna(v) else f"{v:.2f}%")
        show_tbl["Future $"] = show_tbl["Future $"].map(lambda v: "" if pd.isna(v) else f"{int(round(v)):,}")

        st.session_state["proj_table_display"] = show_tbl
        st.session_state["corr_view"] = corr
        st.session_state["corr_note"] = note

        # Projection lines
        proj_rows = []
        base = pd.Timestamp.today().normalize() if pd.isna(REF_LATEST) else REF_LATEST.normalize()
        timeline = [base + DateOffset(years=t) for t in range(0, horizon + 1)]
        if np.isfinite(port_cagr):
            for t, dt in enumerate(timeline):
                proj_rows.append({"Series": "Portfolio", "Date": dt, "Value": invest * ((1 + port_cagr) ** t)})

        eligible = display.loc[
            (display["Index"] != "Portfolio") &
            (pd.to_numeric(display["Allocation %"], errors="coerce") > 0) &
            (display["CAGR"].notna()),
            ["Index", "CAGR"]
        ]
        for _, r_ in eligible.iterrows():
            for t, dt in enumerate(timeline):
                proj_rows.append({"Series": r_["Index"], "Date": dt, "Value": invest * ((1 + r_["CAGR"]) ** t)})
        lines_df = pd.DataFrame(proj_rows) if proj_rows else pd.DataFrame(columns=["Series", "Date", "Value"])
        if not lines_df.empty: lines_df["Date"] = pd.to_datetime(lines_df["Date"])
        st.session_state["proj_lines"] = lines_df
        legend_series = ["Portfolio"] + sorted(enabled_list)
        st.session_state["legend_series"] = legend_series
        for s in legend_series: st.session_state["proj_flags"].setdefault(s, True)
        for k in list(st.session_state["proj_flags"].keys()):
            if k not in legend_series: del st.session_state["proj_flags"][k]

    # ---------- Projection chart ----------
    st.subheader("Projection: Portfolio vs Selected Assets (CAGR-based)")
    lines_df = st.session_state.get("proj_lines", pd.DataFrame())
    legend_series = st.session_state.get("legend_series", ["Portfolio"])
    for s in legend_series: st.session_state["proj_flags"].setdefault(s, True)
    left_plot, right_legend = st.columns([0.72, 0.28])
    series = sorted(legend_series, key=lambda s: (s != "Portfolio", s))
    color_map = {s: ALTAIR_BRAND_PALETTE[i % len(ALTAIR_BRAND_PALETTE)] for i, s in enumerate(series)}
    with right_legend:
        st.markdown("**Legend (toggle lines):**")
        for s in series:
            row = st.columns([0.18, 0.82])
            with row[0]:
                current = st.session_state["proj_flags"].get(s, True)
                checked = st.checkbox("", value=current, key=f"pl_{s}")
                st.session_state["proj_flags"][s] = bool(checked)
            with row[1]:
                sw = f"<span style='display:inline-block;width:12px;height:12px;background:{color_map[s]};border-radius:2px;margin-right:8px;vertical-align:middle;'></span><span style='vertical-align:middle;'>{s}</span>"
                st.markdown(sw, unsafe_allow_html=True)
        st.caption("Legend lists all assets selected in Step 1.")
    with left_plot:
        if lines_df.empty:
            st.info("Run **Calculate** (or **Optimize**) to generate the projection.")
        else:
            visible = [s for s in series if st.session_state["proj_flags"].get(s, True)]
            plot = lines_df[lines_df["Series"].isin(visible)].copy()
            if plot.empty:
                st.info("No lines to draw. Enable Portfolio or assets with non-zero weight & valid CAGR.")
            else:
                hover = alt.selection_point(fields=["Series"], on="mouseover", empty="all")
                scale = alt.Scale(domain=list(color_map.keys()), range=list(color_map.values()))
                baseC = alt.Chart(plot).properties(height=420)
                lines = baseC.mark_line(point=True).encode(
                    x=alt.X("Date:T", axis=alt.Axis(format="%Y"), title="Year"),
                    y=alt.Y("Value:Q", axis=alt.Axis(format=",.0f"), title="Projected Value ($)"),
                    color=alt.Color("Series:N", scale=scale, legend=None),
                    opacity=alt.condition(hover, alt.value(1.0), alt.value(0.3)),
                    tooltip=[alt.Tooltip("Date:T"), "Series:N", alt.Tooltip("Value:Q", format=",.0f")],
                ).add_selection(hover)
                if "Portfolio" in visible and "Portfolio" in plot["Series"].unique():
                    thick = baseC.transform_filter(alt.FieldEqualPredicate(field="Series", equal="Portfolio")) \
                                .mark_line(point=True, strokeWidth=4) \
                                .encode(x=alt.X("Date:T"), y=alt.Y("Value:Q"), color=alt.Color("Series:N", scale=scale, legend=None))
                    chart = lines + thick
                else:
                    chart = lines
                st.altair_chart(chart, use_container_width=True)

    # ---------- Projection Table ----------
    st.subheader("Projection Table")
    show_tbl = st.session_state.get("proj_table_display", pd.DataFrame())
    if show_tbl.empty:
        st.info("Press **Calculate** (or **Optimize**) to compute the table.")
    else:
        st.dataframe(show_tbl, use_container_width=True, hide_index=True)
        st.caption("RF uses latest ^TNX (≤10y horizon) or ^TYX (>10y). Sharpe = (CAGR − RF) / AnnVol.")

    # ---------- Correlation ----------
    st.subheader("Correlation matrix (daily returns)")
    cdf = st.session_state.get("corr_view", pd.DataFrame())
    cnote = st.session_state.get("corr_note", "")
    if cdf.empty:
        st.info("Not available yet. Press **Calculate** (or **Optimize**) after picking assets/horizon.")
    else:
        st.dataframe(cdf, use_container_width=True)
        if cnote: st.caption(cnote)

# =========================================================
# ==============  GUIDE (DEFINITIONS & ETFs)  =============
# =========================================================
with tabs[3]:
    st.subheader("What are these indexes & assets? How do I invest in them?")
    st.caption("Plain-English descriptions and example ETFs you can research further.")
    guide_rows = [
        ("S&P 500", "Tracks ~500 of the largest U.S. companies — a broad ‘big-company’ snapshot.", "SPY, IVV, VOO"),
        ("Dow Jones Industrial Average", "30 major U.S. companies, price-weighted.", "DIA"),
        ("NASDAQ Composite", "Thousands of NASDAQ stocks; tech-tilted.", "ONEQ (Composite proxy); NASDAQ-100: QQQ"),
        ("Russell 1000", "Large & mid-cap U.S. stocks.", "IWB"),
        ("Russell 2000", "Small-cap U.S. stocks.", "IWM"),
        ("Russell 3000", "Almost the whole U.S. market.", "IWV; total-market: VTI, SCHB, ITOT"),
        ("NYSE Composite", "All common stocks on the NYSE.", "No pure-play ETF; total-market funds like VTI can proxy"),
        ("Wilshire 5000", "‘Total U.S. market’ concept.", "No direct ETF; common proxies: VTI, ITOT"),
        ("Gold (via GLD)", "Gold bullion exposure via ETF.", "GLD (alt: IAU)"),
        ("Silver (via SLV)", "Silver bullion exposure via ETF.", "SLV (alt: SIVR)"),
        ("Bitcoin", "Original cryptocurrency; high volatility.", "Spot BTC ETFs (e.g., IBIT, FBTC)*"),
        ("Ethereum", "Smart-contract platform, #2 crypto.", "Spot ETH ETFs (e.g., ETHA, EETH)*"),
    ]
    st.table(pd.DataFrame(guide_rows, columns=["Asset/Index", "What it tracks (plain English)", "Popular ETFs (examples)"]))
    st.caption("*ETF availability depends on your country/broker. Educational only, not a recommendation.*")
