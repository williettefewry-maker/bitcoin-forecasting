# ₿ Bitcoin Price Forecaster

A machine learning web app that forecasts Bitcoin prices using a Random Forest + XGBoost ensemble, enriched with external market data from Gold, the S&P 500, and the FTSE 100.

Built with [Streamlit](https://streamlit.io) and deployed interactively in the browser.

---

## What it does

- Fetches 5 years of daily Bitcoin price data and three external market indicators
- Engineers over 60 features including moving averages, RSI, MACD, Bollinger Bands, lag returns, and cross-asset correlations
- Trains a Random Forest and an XGBoost model on the feature set
- Combines both models into a weighted ensemble (adjustable via sidebar slider)
- Displays an N-day price forecast alongside historical actuals and test-set predictions
- Shows model performance metrics: MAE, RMSE, and R²
- Visualises external market performance, rolling correlations, a correlation heatmap, and feature importances

---

## Data sources

All data is fetched via [yfinance](https://pypi.org/project/yfinance/) from Yahoo Finance.

| Asset | Ticker | Use |
|---|---|---|
| Bitcoin | `BTC-USD` | Target variable (closing price) |
| Gold Futures | `GC=F` | External feature |
| S&P 500 | `^GSPC` | External feature |
| FTSE 100 | `^FTSE` | External feature |

5 years of daily OHLCV data is downloaded on app load and cached for 1 hour. External market prices are forward-filled on Bitcoin's calendar to handle weekend and market-closure gaps.

---

## Feature engineering

Features are computed from the Bitcoin closing price and the three external assets:

**Bitcoin technical indicators**
- Simple moving averages: MA7, MA14, MA21, MA50, MA200
- Exponential moving averages: EMA7, EMA14, EMA21, EMA50, EMA200
- Bollinger Bands (20-day): upper, lower, mid, width
- RSI (14-day)
- MACD and signal line (12/26/9)
- Lag prices: 1, 2, 3, 5, 7, 14 days
- Lag returns: 1, 2, 3, 5, 7, 14 days
- Volume 7-day MA and volume ratio
- Calendar features: day of week, month

**External asset features** (per asset: Gold, S&P 500, FTSE 100)
- Price level
- Lagged prices: 1, 2, 5, 14 days
- Lagged returns: 1, 2, 5, 14 days
- 7-day and 21-day moving averages
- 30-day rolling correlation with Bitcoin

---

## Models

### Random Forest
`sklearn.ensemble.RandomForestRegressor` with configurable number of trees (default 200). Features are scaled with `StandardScaler` before training.

### XGBoost
`xgboost.XGBRegressor` with:
- Learning rate: 0.05
- Max depth: 6
- Subsample: 0.8
- Column sample by tree: 0.8

### Ensemble
Final prediction is a weighted average of both models:

```
prediction = rf_weight × RF_pred + (1 − rf_weight) × XGB_pred
```

The RF weight is adjustable in the sidebar (0.0–1.0). The complement is applied automatically to XGBoost.

---

## App features

| Section | Description |
|---|---|
| Metrics row | Current price, MAE, RMSE, R², N-day forecast with % delta |
| Individual model scores | Expandable table showing MAE and R² for RF and XGB separately |
| Forecast chart | Full price history, test predictions, and ensemble N-day forecast |
| External markets | Normalised performance of BTC, Gold, S&P 500, FTSE 100 (base = 100) |
| Rolling correlation | 30-day rolling correlation between Bitcoin and each external asset |
| Correlation heatmap | Daily return correlations across all four assets |
| Feature importances | Top 20 features for RF and XGBoost shown in separate tabs |
| Moving averages | BTC price with MA7, MA21, MA50, MA200 overlaid |
| RSI | 14-day RSI with overbought (70) and oversold (30) bands |

---

## Running locally

### 1. Clone the repository

```bash
git clone https://github.com/williettefewry-maker/bitcoin-forecasting.git
cd bitcoin-forecasting
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the app

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## Requirements

```
streamlit
yfinance
pandas
numpy
plotly
scikit-learn
xgboost
```

---

## Disclaimer

This app is for educational purposes only and does not constitute financial advice. Past price patterns are not a reliable indicator of future performance.
