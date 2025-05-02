# Combined Screener + Optimizer: Alpha Financial Nordic (Streamlit Replit App)
# Uses manually selected Schwab ETFs to build and display Efficient Frontier

import pandas as pd
from pypfopt import EfficientFrontier, risk_models, expected_returns, plotting
from pypfopt.objective_functions import L2_reg
import yfinance as yf
import time
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("Alpha's Ultimate Portfolio Optimizer")

# ===== TICKER SELECTION =====
st.subheader("Step 1: Select Schwab-Compatible ETFs")
default_tickers = ["SCHB", "SCHG", "SCHF", "SCHZ", "SCYB", "SCHI", "SCHH"]
TICKERS = st.multiselect("Select ETFs", default_tickers, default=default_tickers)
if not TICKERS:
    st.error("Please select at least one ETF.")
    st.stop()
st.write(f"Loaded {len(TICKERS)} ETFs for optimization")

# ===== FETCH HISTORICAL PRICES =====
st.subheader("Step 2: Downloading Historical Prices")
price_data = {}
min_tickers_required = 3
progress = st.progress(0)
for i, ticker in enumerate(TICKERS):
    try:
        df = yf.download(ticker, period="5y", interval="1mo")[['Adj Close']].rename(columns={'Adj Close': ticker})
        if not df.empty and len(df) >= 12:
            price_data[ticker] = df
            st.success(f"Downloaded: {ticker}")
        else:
            st.warning(f"No data for {ticker}")
        time.sleep(1)
    except Exception as e:
        st.warning(f"Failed to download {ticker}: {e}")
    progress.progress((i + 1) / len(TICKERS))

if len(price_data) < min_tickers_required:
    st.error(f"Insufficient data: Only {len(price_data)}/{min_tickers_required} tickers returned valid data.")
    st.stop()

prices = pd.concat(price_data.values(), axis=1, join="outer").ffill()
prices.index = pd.to_datetime(prices.index)
prices = prices.sort_index()

if len(prices) < 12:
    st.error("Insufficient data points for optimization (minimum 12 months required).")
    st.stop()

# ===== CALCULATE RETURNS & COVARIANCE =====
st.subheader("Step 3: Optimizing Portfolio")
returns = prices.pct_change().dropna()
mu = expected_returns.mean_historical_return(prices, frequency=12)
S = risk_models.sample_cov(prices, frequency=12)
ef = EfficientFrontier(mu, S, weight_bounds=(0, 1))
gamma = st.slider("L2 Regularization Strength", 0.0, 2.0, 1.0)
ef.add_objective(L2_reg, gamma=gamma)
weights = ef.max_sharpe()
cleaned_weights = ef.clean_weights()
expected_ret, volatility, sharpe = ef.portfolio_performance()

# ===== DISPLAY RESULTS =====
st.write("### Optimal Portfolio Weights")
weights_df = pd.DataFrame.from_dict(cleaned_weights, orient='index', columns=['Weight'])
st.dataframe(weights_df.style.format("{:.2%}"))

csv = weights_df.to_csv().encode('utf-8')
st.download_button("Download Portfolio Weights", csv, "portfolio_weights.csv", "text/csv")

st.metric("Expected Annual Return", f"{expected_ret:.2%}")
st.metric("Annual Volatility", f"{volatility:.2%}")
st.metric("Sharpe Ratio", f"{sharpe:.2f}")

# ===== PLOT FRONTIER =====
st.write("### Efficient Frontier")
fig, ax = plt.subplots(figsize=(6, 4))
plotting.plot_efficient_frontier(ef, ax=ax, show_assets=True)
ax.set_xlabel("Volatility")
ax.set_ylabel("Return")
ax.set_title("Efficient Frontier (Alpha’s Optimized ETFs)")
ax.grid(True)
st.pyplot(fig)
plt.close(fig)
