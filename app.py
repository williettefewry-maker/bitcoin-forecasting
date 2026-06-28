import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from datetime import datetime, timedelta

st.set_page_config(page_title="Bitcoin Forecaster", page_icon="₿", layout="wide")

st.title("₿ Bitcoin Price Forecaster")
st.markdown("Powered by Random Forest | Data via Yahoo Finance")

# --- Sidebar ---
st.sidebar.header("Settings")
forecast_days = st.sidebar.slider("Forecast horizon (days)", 1, 30, 7)
n_estimators = st.sidebar.slider("RF trees", 50, 500, 200, step=50)
train_split = st.sidebar.slider("Train/test split (%)", 60, 90, 80)

EXTERNAL_TICKERS = {
    "Gold": "GC=F",
    "S&P 500": "^GSPC",
    "FTSE 100": "^FTSE",
}

@st.cache_data(ttl=3600)
def fetch_data():
    end = datetime.today()
    start = end - timedelta(days=5 * 365)

    btc = yf.download("BTC-USD", start=start, end=end, auto_adjust=True)
    btc.columns = btc.columns.get_level_values(0)

    ext = {}
    for name, ticker in EXTERNAL_TICKERS.items():
        raw = yf.download(ticker, start=start, end=end, auto_adjust=True)
        raw.columns = raw.columns.get_level_values(0)
        ext[name] = raw["Close"].rename(name)

    ext_df = pd.DataFrame(ext)
    # Forward-fill market closures so BTC dates always have a value
    ext_df = ext_df.reindex(btc.index).ffill().bfill()

    return btc, ext_df

def engineer_features(btc, ext_df):
    df = btc.copy()
    close = df["Close"]

    # Moving averages
    for w in [7, 14, 21, 50, 200]:
        df[f"MA{w}"] = close.rolling(w).mean()
        df[f"EMA{w}"] = close.ewm(span=w, adjust=False).mean()

    # Bollinger Bands (20-day)
    df["BB_mid"] = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    df["BB_upper"] = df["BB_mid"] + 2 * std20
    df["BB_lower"] = df["BB_mid"] - 2 * std20
    df["BB_width"] = df["BB_upper"] - df["BB_lower"]

    # RSI (14-day)
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    # Lag features
    for lag in [1, 2, 3, 5, 7, 14]:
        df[f"lag_{lag}"] = close.shift(lag)
        df[f"return_{lag}d"] = close.pct_change(lag)

    # Volume features
    df["volume_MA7"] = df["Volume"].rolling(7).mean()
    df["volume_ratio"] = df["Volume"] / df["volume_MA7"]

    # Calendar
    df["day_of_week"] = df.index.dayofweek
    df["month"] = df.index.month

    # --- External market features ---
    for name in EXTERNAL_TICKERS:
        series = ext_df[name]
        col = name.lower().replace(" ", "_").replace("&", "and")
        df[f"{col}_price"] = series
        for lag in [1, 2, 5, 14]:
            df[f"{col}_return_{lag}d"] = series.pct_change(lag)
            df[f"{col}_lag_{lag}"] = series.shift(lag)
        df[f"{col}_ma7"] = series.rolling(7).mean()
        df[f"{col}_ma21"] = series.rolling(21).mean()
        # 30-day rolling correlation with BTC
        df[f"{col}_corr30"] = close.rolling(30).corr(series)

    df.dropna(inplace=True)
    return df

def build_and_forecast(df, forecast_days, n_estimators, train_split_pct):
    feature_cols = [c for c in df.columns if c not in ["Open", "High", "Low", "Close", "Volume"]]
    X = df[feature_cols].values
    y = df["Close"].values

    split = int(len(X) * train_split_pct / 100)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    model = RandomForestRegressor(n_estimators=n_estimators, random_state=42, n_jobs=-1)
    model.fit(X_train_sc, y_train)

    y_pred = model.predict(X_test_sc)

    # Iterative future forecast
    last_row = df[feature_cols].iloc[-1].values.copy()
    future_preds = []
    future_dates = []

    for i in range(forecast_days):
        x_scaled = scaler.transform(last_row.reshape(1, -1))
        pred = model.predict(x_scaled)[0]
        future_preds.append(pred)
        future_dates.append(df.index[-1] + timedelta(days=i + 1))

        new_row = last_row.copy()
        for lag_name in [c for c in feature_cols if c.startswith("lag_")]:
            lag_val = int(lag_name.split("_")[1])
            col_idx = feature_cols.index(lag_name)
            if lag_val == 1:
                new_row[col_idx] = pred
            else:
                prev = f"lag_{lag_val - 1}"
                if prev in feature_cols:
                    new_row[col_idx] = last_row[feature_cols.index(prev)]
        last_row = new_row

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    return (
        df.index[split:], y_test, y_pred,
        future_dates, future_preds,
        mae, rmse,
        model, feature_cols
    )

# --- Load & process ---
with st.spinner("Fetching BTC, Gold, S&P 500 and FTSE 100 data..."):
    raw, ext_df = fetch_data()

with st.spinner("Engineering features & training model..."):
    df = engineer_features(raw, ext_df)
    test_dates, y_test, y_pred, future_dates, future_preds, mae, rmse, model, feature_cols = build_and_forecast(
        df, forecast_days, n_estimators, train_split
    )

# --- Metrics ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("Current Price", f"${raw['Close'].iloc[-1]:,.0f}")
col2.metric("MAE", f"${mae:,.0f}")
col3.metric("RMSE", f"${rmse:,.0f}")
col4.metric(f"{forecast_days}-day Forecast", f"${future_preds[-1]:,.0f}",
            delta=f"{((future_preds[-1] / raw['Close'].iloc[-1]) - 1) * 100:.2f}%")

st.divider()

# --- Main forecast chart ---
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=df.index, y=df["Close"],
    name="Actual Price", line=dict(color="#F7931A", width=1.5)
))
fig.add_trace(go.Scatter(
    x=list(test_dates), y=list(y_pred),
    name="Test Predictions", line=dict(color="#00BFFF", width=1.5, dash="dot")
))
fig.add_trace(go.Scatter(
    x=future_dates, y=future_preds,
    name=f"{forecast_days}-day Forecast",
    line=dict(color="#00FF88", width=2.5),
    mode="lines+markers", marker=dict(size=5)
))
fig.add_vrect(
    x0=future_dates[0], x1=future_dates[-1],
    fillcolor="rgba(0,255,136,0.05)", line_width=0
)
fig.update_layout(
    title="Bitcoin Price: Actual vs Predicted vs Forecast",
    xaxis_title="Date", yaxis_title="Price (USD)",
    template="plotly_dark", height=550,
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
    hovermode="x unified"
)
st.plotly_chart(fig, use_container_width=True)

# --- External markets chart ---
st.subheader("External Market Indicators")

ext_colors = {"Gold": "#FFD700", "S&P 500": "#00BFFF", "FTSE 100": "#FF69B4"}
norm_ext = ext_df.div(ext_df.iloc[0]) * 100
norm_btc = (raw["Close"] / raw["Close"].iloc[0]) * 100

fig_ext = go.Figure()
fig_ext.add_trace(go.Scatter(
    x=raw.index, y=norm_btc,
    name="Bitcoin", line=dict(color="#F7931A", width=1.5)
))
for name, color in ext_colors.items():
    fig_ext.add_trace(go.Scatter(
        x=norm_ext.index, y=norm_ext[name],
        name=name, line=dict(color=color, width=1.2)
    ))
fig_ext.update_layout(
    title="Normalised Performance (Base = 100)",
    xaxis_title="Date", yaxis_title="Indexed Price",
    template="plotly_dark", height=400,
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
    hovermode="x unified"
)
st.plotly_chart(fig_ext, use_container_width=True)

# --- Rolling correlations ---
st.subheader("30-Day Rolling Correlation with Bitcoin")
corr_colors = {"Gold": "#FFD700", "S&P 500": "#00BFFF", "FTSE 100": "#FF69B4"}
fig_corr = go.Figure()
for name, color in corr_colors.items():
    col = name.lower().replace(" ", "_").replace("&", "and")
    fig_corr.add_trace(go.Scatter(
        x=df.index, y=df[f"{col}_corr30"],
        name=name, line=dict(color=color, width=1.2)
    ))
fig_corr.add_hline(y=0, line_dash="dash", line_color="white", line_width=0.5)
fig_corr.update_layout(
    template="plotly_dark", height=300,
    yaxis=dict(range=[-1, 1], title="Correlation"),
    hovermode="x unified"
)
st.plotly_chart(fig_corr, use_container_width=True)

# --- Feature importance ---
st.subheader("Top 15 Feature Importances")
importance_df = (
    pd.DataFrame({"feature": feature_cols, "importance": model.feature_importances_})
    .sort_values("importance", ascending=True)
    .tail(15)
)
fig2 = go.Figure(go.Bar(
    x=importance_df["importance"],
    y=importance_df["feature"],
    orientation="h",
    marker_color="#F7931A"
))
fig2.update_layout(template="plotly_dark", height=420, margin=dict(l=10))
st.plotly_chart(fig2, use_container_width=True)

# --- Moving averages ---
st.subheader("Moving Averages")
fig3 = go.Figure()
fig3.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Price", line=dict(color="#F7931A", width=1)))
for ma, color in [("MA7", "#FFD700"), ("MA21", "#00BFFF"), ("MA50", "#FF69B4"), ("MA200", "#98FB98")]:
    fig3.add_trace(go.Scatter(x=df.index, y=df[ma], name=ma, line=dict(width=1.2, color=color)))
fig3.update_layout(template="plotly_dark", height=380, hovermode="x unified")
st.plotly_chart(fig3, use_container_width=True)

# --- RSI ---
st.subheader("RSI (14-day)")
fig4 = go.Figure()
fig4.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI", line=dict(color="#00BFFF")))
fig4.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
fig4.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
fig4.update_layout(template="plotly_dark", height=280, yaxis=dict(range=[0, 100]))
st.plotly_chart(fig4, use_container_width=True)

st.caption("Disclaimer: This is for educational purposes only and not financial advice.")
