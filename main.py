import cvxpy as cp
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import matplotlib.pyplot as plt

# Streamlit app title
st.title("Alpha's Ultimate Portfolios")

# Default ETF tickers (from your project summary)
default_tickers = ["SCHB", "SCHG", "SCHF", "SCHZ", "SCHH"]

# Fetch data using yfinance
st.write("Fetching ETF data...")
try:
    df = yf.download(default_tickers, period="2y", interval="1mo")['Adj Close']
    if df.empty or df.isna().all().all():
        raise ValueError("No data retrieved for the selected ETFs.")
except Exception as e:
    st.error(f"Error fetching ETF data: {e}")
    st.stop()

# Calculate expected returns and covariance
returns = df.pct_change().dropna()
if returns.empty:
    st.error("No valid returns data after processing. Please check the ETFs or data period.")
    st.stop()

mu = returns.mean() * 252  # Annualized expected returns
S = returns.cov() * 252    # Annualized covariance matrix

# Portfolio optimization with cvxpy
st.write("Optimizing portfolio...")
n = len(mu)
w = cp.Variable(n)  # Weights
gamma = 0.1  # Risk aversion parameter (adjust as needed)

# Objective: Maximize return - risk (mean-variance optimization)
objective = cp.Maximize(mu @ w - gamma * cp.quad_form(w, S))
constraints = [cp.sum(w) == 1, w >= 0]  # Sum of weights = 1, no shorting
problem = cp.Problem(objective, constraints)

# Solve the optimization problem
try:
    problem.solve()
    weights = w.value
except Exception as e:
    st.error(f"Optimization failed: {e}")
    st.stop()

# Check if optimization succeeded
if weights is None or not np.isfinite(weights).all():
    st.error("Optimization failed to find a solution. Try adjusting the gamma parameter or data.")
    st.stop()

# Clean weights (similar to pypfopt's clean_weights)
weights = np.maximum(weights, 0)  # Ensure no negative weights
weights /= np.sum(weights)  # Normalize to sum to 1
weights_dict = {ticker: round(weight, 4) for ticker, weight in zip(default_tickers, weights) if weight > 0.0001}

# Display optimized weights
st.write("### Optimized Portfolio Weights")
st.write(weights_dict)

# Plot the weights
st.write("### Portfolio Allocation Chart")
plt.figure(figsize=(10, 6))
plt.bar(weights_dict.keys(), weights_dict.values(), color='skyblue')
plt.title("Optimized Portfolio Weights")
plt.xlabel("ETFs")
plt.ylabel("Weights")
plt.xticks(rotation=45)
st.pyplot(plt)