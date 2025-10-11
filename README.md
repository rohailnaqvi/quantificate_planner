# Quantificate — Personal Investment Planner

An educational Streamlit app to **Guide, Explore, Plan and Execute** a simple, diversified investment approach:
- **Explore (History):** compare historical index & asset behavior
- **Plan (Projections):** set allocations, horizon, and (optionally) optimize for Sharpe
- **Guide:** plain-English definitions and example ETFs to research further

> **Note:** This app is for learning and exploration only. It is **not** investment advice.

---

## Live demo (Streamlit Cloud)

If you deploy on Streamlit Community Cloud, commits to `main` auto-redeploy your app.

---

## Features

- Clean brand header with **auto light/dark logos** and light favicon
- Full-width **Explore** chart with master + per-asset toggles (no recompute on each click)
- **Historical tables:** prices, elapsed years, % change, CAGR, volatility, Sharpe
- **Plan (Projections):**
  - Choose assets and weights; see portfolio projection line vs assets
  - **Optimizer** (Sharpe) using historical stats (requires `scipy`)
  - **Constraints UI** with group → per-asset **deferred apply** (fast, no lag)
  - Live “weights sum” pill (green/yellow/red)
  - Correlation matrix for chosen horizon

---

## Repo structure

