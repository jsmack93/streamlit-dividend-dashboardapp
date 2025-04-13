# -*- coding: utf-8 -*-
"""financial4.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1Fl8p0bWuOMC6tyqgbknA7JzLEVkN_Z2L
"""

#!pip install streamlit

import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# Set the page title
st.title("Financial Dashboard")

# Sidebar input: prompt the user to enter a ticker symbol
ticker_symbol = st.sidebar.text_input("Enter the ticker symbol (e.g., AAPL)", value="AAPL")

if ticker_symbol:
    # Retrieve data for the entered ticker using yfinance
    ticker_data = yf.Ticker(ticker_symbol)
    info = ticker_data.info

    # ---------------------------
    # 1. Company Overview
    # ---------------------------
    st.subheader("Company Overview")
    if 'longBusinessSummary' in info:
        st.write(info['longBusinessSummary'])
    else:
        st.write("No overview available.")

    # ---------------------------
    # 2. Dividend History (Last 10 Entries)
    # ---------------------------
    st.subheader("Dividend History (Last 10 Entries)")
    dividends = ticker_data.dividends
    if dividends.empty:
        st.write("No dividend data available for this ticker.")
    else:
        # To avoid clutter, show only the last 10 entries if there are more than 10
        data_to_plot = dividends.tail(10) if len(dividends) > 10 else dividends
        st.write(data_to_plot)

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(data_to_plot.index.astype(str), data_to_plot)  # convert dates to string for clarity
        ax.set_title("Dividend History")
        ax.set_xlabel("Date")
        ax.set_ylabel("Dividend ($)")
        plt.xticks(rotation=45)
        st.pyplot(fig)

    # ---------------------------
    # 3. Price History (Last 1 Year)
    # ---------------------------
    st.subheader("Price History (Last 1 Year)")
    price_history = ticker_data.history(period="1y")
    if price_history.empty:
        st.write("No price data available for this ticker.")
    else:
        st.write(price_history[['Close']].head())  # display first few rows for preview
        fig_price, ax_price = plt.subplots(figsize=(10, 4))
        ax_price.plot(price_history.index, price_history['Close'], label="Closing Price")
        ax_price.set_title("Price History (Last 1 Year)")
        ax_price.set_xlabel("Date")
        ax_price.set_ylabel("Price ($)")
        ax_price.legend()
        st.pyplot(fig_price)

    # ---------------------------
    # 4. Key Financial Metrics
    # ---------------------------
    st.subheader("Key Financial Metrics")
    trailing_eps = info.get('trailingEps', None)
    dividend_rate = info.get('dividendRate', None)
    dividend_yield = info.get('dividendYield', None)

    # Calculate a simple dividend payout ratio (Dividend Rate / Trailing EPS) if possible
    if trailing_eps and trailing_eps != 0 and dividend_rate:
        dividend_payout_ratio = dividend_rate / trailing_eps
    else:
        dividend_payout_ratio = None

    st.write("**Trailing EPS:**", trailing_eps if trailing_eps is not None else "N/A")
    st.write("**Dividend Rate:**", dividend_rate if dividend_rate is not None else "N/A")
    st.write("**Dividend Yield:**", f"{dividend_yield:.2%}" if dividend_yield is not None else "N/A")
    if dividend_payout_ratio is not None:
        st.write("**Dividend Payout Ratio:**", round(dividend_payout_ratio, 2))
    else:
        st.write("Dividend payout ratio could not be calculated due to missing data.")

import yfinance as yf
import pprint

def compute_altman_z(ticker: str):
    """
    Compute the Altman Z-Score for the given ticker using data from Yahoo Finance.
    
    The formula used (for manufacturing companies) is:
    
      Z = 1.2*(Working Capital/Total Assets) +
          1.4*(Retained Earnings/Total Assets) +
          3.3*(EBIT/Total Assets) +
          0.6*(Market Value of Equity/Total Liabilities) +
          (Sales/Total Assets)
    
    If any essential metric (Total Assets, Total Liabilities, or Market Value of Equity) is missing,
    the function will print an error and return None.
    
    Returns a dictionary containing the calculated ratios, the Z‑Score, and the classification.
    """
    # Retrieve data for the ticker
    t = yf.Ticker(ticker)
    bs = t.balance_sheet     # Balance Sheet
    fs = t.financials        # Income Statement
    info = t.info            # Company info

    # Ensure the balance sheet and financials have data
    try:
        bs_col = bs.columns[0]
    except Exception:
        print("Balance sheet data not available.")
        return None
        
    try:
        fs_col = fs.columns[0]
    except Exception:
        print("Financial statement data not available.")
        return None

    # Extract key balance sheet items
    try:
        total_assets = bs.loc['Total Assets'][bs_col]
    except Exception:
        total_assets = None

    try:
        total_liabilities = bs.loc['Total Liab'][bs_col]
    except Exception:
        total_liabilities = None

    # Current Assets (try "Total Current Assets" and fallback to "Current Assets")
    if 'Total Current Assets' in bs.index:
        current_assets = bs.loc['Total Current Assets'][bs_col]
    elif 'Current Assets' in bs.index:
        current_assets = bs.loc['Current Assets'][bs_col]
    else:
        current_assets = None

    # Current Liabilities (try "Total Current Liabilities" and fallback to "Current Liabilities")
    if 'Total Current Liabilities' in bs.index:
        current_liabilities = bs.loc['Total Current Liabilities'][bs_col]
    elif 'Current Liabilities' in bs.index:
        current_liabilities = bs.loc['Current Liabilities'][bs_col]
    else:
        current_liabilities = None

    working_capital = current_assets - current_liabilities if (current_assets is not None and current_liabilities is not None) else None

    try:
        retained_earnings = bs.loc['Retained Earnings'][bs_col]
    except Exception:
        retained_earnings = None

    # Extract key income statement items
    try:
        ebit = fs.loc['Operating Income'][fs_col]
    except Exception:
        ebit = None

    try:
        sales = fs.loc['Total Revenue'][fs_col]
    except Exception:
        sales = None

    # Calculate Market Value of Equity using the regular market price and shares outstanding
    try:
        share_price = info.get('regularMarketPrice', None)
        shares_outstanding = info.get('sharesOutstanding', None)
        market_value_of_equity = share_price * shares_outstanding if (share_price and shares_outstanding) else None
    except Exception:
        market_value_of_equity = None

    # If any essential data is missing, exit
    if total_assets is None or total_liabilities is None or market_value_of_equity is None:
        print("Essential data not available for ticker", ticker)
        return None

    # Calculate each component ratio.
    ratio1 = (working_capital / total_assets) if working_capital is not None else 0.0
    ratio2 = (retained_earnings / total_assets) if retained_earnings is not None else 0.0
    ratio3 = (ebit / total_assets) if ebit is not None else 0.0
    ratio4 = (market_value_of_equity / total_liabilities) if total_liabilities != 0 else 0.0
    ratio5 = (sales / total_assets) if sales is not None else 0.0

    # Compute the Altman Z‑Score using the coefficients:
    z_score = 1.2 * ratio1 + 1.4 * ratio2 + 3.3 * ratio3 + 0.6 * ratio4 + ratio5

    # Classify the company based on the Z‑Score:
    if z_score > 2.99:
        classification = "Safe"
    elif z_score >= 1.81:
        classification = "Grey Zone"
    else:
        classification = "Distressed"

    return {
        "Ticker": ticker,
        "Total Assets": total_assets,
        "Total Liabilities": total_liabilities,
        "Working Capital": working_capital,
        "Retained Earnings": retained_earnings,
        "EBIT": ebit,
        "Sales": sales,
        "Market Value of Equity": market_value_of_equity,
        "Ratio1 (WC/TA)": ratio1,
        "Ratio2 (Retained Earnings/TA)": ratio2,
        "Ratio3 (EBIT/TA)": ratio3,
        "Ratio4 (MVE/TL)": ratio4,
        "Ratio5 (Sales/TA)": ratio5,
        "Altman Z-Score": z_score,
        "Classification": classification
    }

if __name__ == "__main__":
    ticker_input = input("Enter ticker symbol: ").strip()
    result = compute_altman_z(ticker_input)
    if result:
        print("\n=== Altman Z‑Score Analysis ===")
        pprint.pprint(result)

