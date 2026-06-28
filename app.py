import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from datetime import datetime, timedelta

st.set_page_config(page_title="Bitcoin Forecaster", page_icon="₿", layout="wide")

st.title("₿ Bitcoin Price Forecaster")
st.markdown("Powered by Random Forest + XGBoost Ensemble | Data via Yahoo Finance")

# --- Sidebar ---
st.sidebar.header("Settings")
forecast_days = st.sidebar.slider("Forecast horizon (days)", 1, 30, 7)
n_estimators = st.sidebar.slider("RF trees", 50, 500, 200, step=50)
train_split = st.sidebar.slider("Train/test split (%)", 60, 90, 80)

st.sidebar.divider()
st.sidebar.subheader("Ensemble Weights")
rf_weight = st.sidebar.slider("Random Forest weight", 0.0, 1.0, 0.5, step=0.05)
xgb_weight = round(1.0 - rf_weight, 2)
st.sidebar.caption(f"XGBoost weight: **{xgb_weight}**")

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
    ext_df = ext_df.reindex(btc.index).ffill().bfill()

    return btc, ext_df

def engineer_features(btc, ext_df):
    df = btc.copy()
    close = df["Close"]

    for w in [7, 14, 21, 50, 200]:
        df[f"MA{w}"] = close.rolling(w).mean()
        df[f"EMA{w}"] = close.ewm(span=w, adjust=False).mean()

    df["BB_mid"] = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    df["BB_upper"] = df["BB_mid"] + 2 * std20
    df["BB_lower"] = df["BB_mid"] - 2 * std20
    df["BB_width"] = df["BB_upper"] - df["BB_lower"]

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    for lag in [1, 2, 3, 5, 7, 14]:
        df[f"lag_{lag}"] = close.shift(lag)
        df[f"return_{lag}d"] = close.pct_change(lag)

    df["volume_MA7"] = df["Volume"].rolling(7).mean()
    df["volume_ratio"] = df["Volume"] / df["volume_MA7"]

    df["day_of_week"] = df.index.dayofweek
    df["month"] = df.index.month

    for name in EXTERNAL_TICKERS:
        series = ext_df[name]
        col = name.lower().replace(" ", "_").replace("&", "and")
        df[f"{col}_price"] = series
        for lag in [1, 2, 5, 14]:
            df[f"{col}_return_{lag}d"] = series.pct_change(lag)
            df[f"{col}_lag_{lag}"] = series.shift(lag)
        df[f"{col}_ma7"] = series.rolling(7).mean()
        df[f"{col}_ma21"] = series.rolling(21).mean()
        df[f"{col}_corr30"] = close.rolling(30).corr(series)

    df.dropna(inplace=True)
    return df

def build_and_forecast(df, forecast_days, n_estimators, train_split_pct, rf_weight):
    feature_cols = [c for c in df.columns if c not in ["Open", "High", "Low", "Close", "Volume"]]
    X = df[feature_cols].values
    y = df["Close"].values

    split = int(len(X) * train_split_pct / 100)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    rf = RandomForestRegressor(n_estimators=n_estimators, random_state=42, n_jobs=-1)
    rf.fit(X_train_sc, y_train)

    xgb = XGBRegressor(n_estimators=n_estimators, random_state=42,
                        learning_rate=0.05, max_depth=6,
                        subsample=0.8, colsample_bytree=0.8,
                        verbosity=0)
    xgb.fit(X_train_sc, y_train)

    rf_pred = rf.predict(X_test_sc)
    xgb_pred = xgb.predict(X_test_sc)
    xgb_weight = 1.0 - rf_weight
    y_pred = rf_weight * rf_pred + xgb_weight * xgb_pred

    # Iterative future forecast
    last_row = df[feature_cols].iloc[-1].values.copy()
    future_preds, future_preds_rf, future_preds_xgb = [], [], []
    future_dates = []

    for i in range(forecast_days):
        x_scaled = scaler.transform(last_row.reshape(1, -1))
        p_rf = rf.predict(x_scaled)[0]
        p_xgb = xgb.predict(x_scaled)[0]
        p_ens = rf_weight * p_rf + xgb_weight * p_xgb
        future_preds_rf.append(p_rf)
        future_preds_xgb.append(p_xgb)
        future_preds.append(p_ens)
        future_dates.append(df.index[-1] + timedelta(days=i + 1))

        new_row = last_row.copy()
        for lag_name in [c for c in feature_cols if c.startswith("lag_")]:
            lag_val = int(lag_name.split("_")[1])
            col_idx = feature_cols.index(lag_name)
            if lag_val == 1:
                new_row[col_idx] = p_ens
            else:
                prev = f"lag_{lag_val - 1}"
                if prev in feature_cols:
                    new_row[col_idx] = last_row[feature_cols.index(prev)]
        last_row = new_row

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    # Per-model test scores
    scores = {
        "RF":  {"mae": mean_absolute_error(y_test, rf_pred),
                "r2":  r2_score(y_test, rf_pred)},
        "XGB": {"mae": mean_absolute_error(y_test, xgb_pred),
                "r2":  r2_score(y_test, xgb_pred)},
    }

    return (
        df.index[split:], y_test, y_pred, rf_pred, xgb_pred,
        future_dates, future_preds, future_preds_rf, future_preds_xgb,
        mae, rmse, r2, scores, rf, xgb, feature_cols
    )

# --- Load & process ---
with st.spinner("Fetching BTC, Gold, S&P 500 and FTSE 100 data..."):
    raw, ext_df = fetch_data()

with st.spinner("Engineering features & training RF + XGBoost ensemble..."):
    df = engineer_features(raw, ext_df)
    (test_dates, y_test, y_pred, rf_pred, xgb_pred,
     future_dates, future_preds, future_preds_rf, future_preds_xgb,
     mae, rmse, r2, scores, rf_model, xgb_model, feature_cols) = build_and_forecast(
        df, forecast_days, n_estimators, train_split, rf_weight
    )

# --- Metrics ---
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Current Price", f"${raw['Close'].iloc[-1]:,.0f}")
col2.metric("MAE", f"${mae:,.0f}")
col3.metric("RMSE", f"${rmse:,.0f}")
col4.metric("R²", f"{r2:.4f}")
col5.metric(f"{forecast_days}-day Forecast", f"${future_preds[-1]:,.0f}",
            delta=f"{((future_preds[-1] / raw['Close'].iloc[-1]) - 1) * 100:.2f}%")

# Per-model score table
with st.expander("Individual model scores"):
    score_df = pd.DataFrame(scores).T
    score_df.columns = ["MAE ($)", "R²"]
    score_df["MAE ($)"] = score_df["MAE ($)"].map(lambda x: f"${x:,.0f}")
    score_df["R²"] = score_df["R²"].map(lambda x: f"{x:.4f}")
    st.dataframe(score_df, use_container_width=True)

st.divider()

# --- Main forecast chart ---
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=df.index, y=df["Close"],
    name="Actual Price", line=dict(color="#F7931A", width=1.5)
))
fig.add_trace(go.Scatter(
    x=list(test_dates), y=list(y_pred),
    name="Ensemble (test)", line=dict(color="#00BFFF", width=1.5, dash="dot")
))
fig.add_trace(go.Scatter(
    x=future_dates, y=future_preds,
    name=f"Ensemble {forecast_days}-day Forecast",
    line=dict(color="#00FF88", width=2.5),
    mode="lines+markers", marker=dict(size=5)
))
fig.add_trace(go.Scatter(
    x=future_dates, y=future_preds_rf,
    name="RF Forecast", line=dict(color="#FFD700", width=1.2, dash="dash"), visible="legendonly"
))
fig.add_trace(go.Scatter(
    x=future_dates, y=future_preds_xgb,
    name="XGB Forecast", line=dict(color="#FF69B4", width=1.2, dash="dash"), visible="legendonly"
))
fig.add_vrect(
    x0=future_dates[0], x1=future_dates[-1],
    fillcolor="rgba(0,255,136,0.05)", line_width=0
)
fig.update_layout(
    title="Bitcoin Price: Actual vs Predicted vs Ensemble Forecast",
    xaxis_title="Date", yaxis_title="Price (USD)",
    template="plotly_dark", height=550,
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
    hovermode="x unified"
)
st.plotly_chart(fig, use_container_width=True)

# --- External markets chart ---
st.subheader("External Market Indicators")
norm_ext = ext_df.div(ext_df.iloc[0]) * 100
norm_btc = (raw["Close"] / raw["Close"].iloc[0]) * 100
ext_colors = {"Gold": "#FFD700", "S&P 500": "#00BFFF", "FTSE 100": "#FF69B4"}

fig_ext = go.Figure()
fig_ext.add_trace(go.Scatter(x=raw.index, y=norm_btc, name="Bitcoin", line=dict(color="#F7931A", width=1.5)))
for name, color in ext_colors.items():
    fig_ext.add_trace(go.Scatter(x=norm_ext.index, y=norm_ext[name], name=name, line=dict(color=color, width=1.2)))
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
fig_corr = go.Figure()
for name, color in ext_colors.items():
    col = name.lower().replace(" ", "_").replace("&", "and")
    fig_corr.add_trace(go.Scatter(x=df.index, y=df[f"{col}_corr30"], name=name, line=dict(color=color, width=1.2)))
fig_corr.add_hline(y=0, line_dash="dash", line_color="white", line_width=0.5)
fig_corr.update_layout(template="plotly_dark", height=300, yaxis=dict(range=[-1, 1], title="Correlation"), hovermode="x unified")
st.plotly_chart(fig_corr, use_container_width=True)

# --- Correlation heatmap ---
st.subheader("Correlation Heatmap")
price_data = pd.concat([
    raw["Close"].rename("Bitcoin"),
    ext_df["Gold"],
    ext_df["S&P 500"],
    ext_df["FTSE 100"],
], axis=1).dropna()
returns = price_data.pct_change().dropna()
corr_matrix = returns.corr()
labels = corr_matrix.columns.tolist()

fig_heat = go.Figure(go.Heatmap(
    z=corr_matrix.values,
    x=labels, y=labels,
    colorscale="RdBu",
    zmid=0, zmin=-1, zmax=1,
    text=np.round(corr_matrix.values, 2),
    texttemplate="%{text}",
    textfont=dict(size=14),
    hoverongaps=False,
))
fig_heat.update_layout(
    template="plotly_dark", height=420,
    xaxis=dict(side="bottom"),
    margin=dict(l=10, r=10, t=30, b=10),
)
st.plotly_chart(fig_heat, use_container_width=True)

# --- Feature importance ---
st.subheader("Top 20 Feature Importances")
tab_rf, tab_xgb = st.tabs(["Random Forest", "XGBoost"])

for tab, model, color in [(tab_rf, rf_model, "#F7931A"), (tab_xgb, xgb_model, "#FF69B4")]:
    with tab:
        imp_df = (
            pd.DataFrame({"feature": feature_cols, "importance": model.feature_importances_})
            .sort_values("importance", ascending=True)
            .tail(20)
        )
        fig_imp = go.Figure(go.Bar(x=imp_df["importance"], y=imp_df["feature"], orientation="h", marker_color=color))
        fig_imp.update_layout(template="plotly_dark", height=420, margin=dict(l=10))
        st.plotly_chart(fig_imp, use_container_width=True)

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
