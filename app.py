import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
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

@st.cache_data(ttl=3600)
def fetch_data():
    end = datetime.today()
    start = end - timedelta(days=5 * 365)
    df = yf.download("BTC-USD", start=start, end=end, auto_adjust=True)
    df.columns = df.columns.get_level_values(0)
    return df

def engineer_features(df):
    df = df.copy()
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

    temp_df = df.copy()
    for i in range(forecast_days):
        x_scaled = scaler.transform(last_row.reshape(1, -1))
        pred = model.predict(x_scaled)[0]
        future_preds.append(pred)
        future_dates.append(temp_df.index[-1] + timedelta(days=1))

        # Roll a synthetic next row: shift lags and update MAs naively
        new_close = pred
        new_row = last_row.copy()
        lag_map = {f"lag_{l}": i for i, l in enumerate([1, 2, 3, 5, 7, 14])}
        for lag_name, col_idx in [(c, feature_cols.index(c)) for c in feature_cols if c.startswith("lag_")]:
            lag_val = int(lag_name.split("_")[1])
            if lag_val == 1:
                new_row[col_idx] = new_close
            else:
                prev_lag = f"lag_{lag_val - 1}"
                if prev_lag in feature_cols:
                    new_row[col_idx] = last_row[feature_cols.index(prev_lag)]
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
with st.spinner("Fetching 5 years of BTC-USD data..."):
    raw = fetch_data()

with st.spinner("Engineering features & training model..."):
    df = engineer_features(raw)
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

# --- Main chart ---
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
    mode="lines+markers",
    marker=dict(size=5)
))

# Shade forecast region
fig.add_vrect(
    x0=future_dates[0], x1=future_dates[-1],
    fillcolor="rgba(0,255,136,0.05)", line_width=0
)

fig.update_layout(
    title="Bitcoin Price: Actual vs Predicted vs Forecast",
    xaxis_title="Date",
    yaxis_title="Price (USD)",
    template="plotly_dark",
    height=550,
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
    hovermode="x unified"
)

st.plotly_chart(fig, use_container_width=True)

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

# --- Moving average chart ---
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
