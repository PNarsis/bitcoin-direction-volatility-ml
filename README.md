# Bitcoin Next-Day Candle Direction Classifier

A binary classification pipeline that predicts whether tomorrow's BTC-USD daily candle will
close **green** (`Close > Open`) or **red**, using technical indicators from Bitcoin itself,
cross-market signals (gold, oil, US Dollar Index, S&P 500, Nasdaq, VIX, 10-Year Treasury
yield, Ethereum), crypto-specific sentiment (Funding Rate, Fear & Greed Index), and a set of
experimental "Smart Money Concepts" liquidity proxies. Three gradient-boosting models —
**XGBoost**, **CatBoost**, and **LightGBM** — are trained and compared. A bonus section also
tackles a more tractable target: next-day **volatility regime**.

## ⚠️ Disclaimer

This is an **educational / portfolio project**, not a trading system and not financial
advice. Daily crypto price direction is extremely hard to predict from public market data
alone; even professional quant funds rarely sustain accuracy much above 55-60%. The goal here
is to demonstrate a **correct, leakage-free machine learning pipeline** for financial time
series — honest baselines, time-aware validation, and transparent limitations — not to claim
a profitable trading edge.

Some features (the "Smart Money Concepts" liquidity proxies — sweep detection, distance to
recent highs/lows, equal-highs count) are based on **informal, interpretive trading concepts**
without a single agreed mathematical definition, unlike RSI/MACD/Bollinger Bands. They are
included as experimental proxies, clearly separated in the code from standard indicators, and
should be read with that caveat.

## Pipeline

1. **Data collection** — daily OHLCV for BTC-USD and 8 correlated assets via `yfinance` (no
   API key required), plus Binance Funding Rate and the Fear & Greed Index (both free, public,
   no key required).
2. **Feature engineering** — returns, volatility, moving averages, RSI, MACD, Bollinger Bands,
   volume change, and day-of-week from BTC itself; lagged daily returns from the cross-market
   assets; lagged funding rate / fear-greed sentiment; experimental SMC liquidity proxies.
   Everything is lagged specifically to avoid any timezone/market-close ambiguity leaking
   future information.
3. **Leakage-safe target** — `target(t) = 1` if `Close(t+1) - Open(t+1) > 0`, computed so that
   every feature at row `t` only uses information available up to day `t`.
4. **Time-aware split** — chronological train/test split (no random shuffling) plus
   `TimeSeriesSplit` cross-validation.
5. **Baseline** — majority-class accuracy, reported alongside every model so results are
   interpretable rather than misleading.
6. **Model training & comparison** — XGBoost, CatBoost, LightGBM, evaluated on accuracy,
   precision, recall, F1, and ROC-AUC.
7. **Feature importance** — per-model top-15 feature plots.
8. **Simple backtest** — naive long/cash strategy vs Buy & Hold, with Sharpe ratio (ignores
   fees/slippage/taxes — illustrative only).
9. **Bonus: volatility regime prediction** — a parallel, more tractable target based on the
   well-documented "volatility clustering" phenomenon, using a causal trailing-median
   threshold and the same feature set and models.

## Project structure

```
.
├── notebooks/
│   └── btc_direction_prediction.ipynb
├── data/                      # raw CSVs are saved here on first run
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

Then open `notebooks/btc_direction_prediction.ipynb` and run all cells. No API keys are
needed — all data comes from Yahoo Finance, Binance's public futures API, and alternative.me.

## Limitations & future work

- No on-chain data (exchange netflows, whale activity) or order-book microstructure.
- Daily granularity only; higher-frequency data may carry more signal.
- No walk-forward retraining loop yet — a real deployment would need periodic retraining and
  drift monitoring.
- Backtest is illustrative only (no fees, slippage, or taxes modeled).
- Funding rate history is only available from ~September 2019 onward (Binance perpetuals
  launch), so earlier rows may have missing sentiment features.

Contributions and suggestions are welcome.
