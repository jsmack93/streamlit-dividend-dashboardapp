import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------------
# DIVIDEND DASHBOARD FUNCTION
# ------------------------------------

def display_dividend_dashboard(ticker: str):
    t_obj = yf.Ticker(ticker)
    info = t_obj.info

    st.subheader("Company Overview")
    if 'longBusinessSummary' in info:
        st.write(info['longBusinessSummary'])
    else:
        st.write("No overview available.")

    st.subheader("Dividend History (Last 10 Entries)")
    dividends = t_obj.dividends
    if dividends.empty:
        st.write("No dividend data available for this ticker.")
    else:
        data_to_plot = dividends.tail(10)
        st.write(data_to_plot)
        fig_div, ax_div = plt.subplots(figsize=(10, 4))
        ax_div.bar(data_to_plot.index, data_to_plot)
        ax_div.set_title("Dividend History (Last 10 Entries)")
        ax_div.set_xlabel("Date")
        ax_div.set_ylabel("Dividend ($)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig_div)

    st.subheader("Price History (Last 1 Year)")
    price_history = t_obj.history(period="1y")
    if price_history.empty:
        st.write("No price data available for this ticker.")
    else:
        st.write(price_history[['Close']].head())
        fig_price, ax_price = plt.subplots(figsize=(10, 4))
        ax_price.plot(price_history.index, price_history['Close'], label="Closing Price")
        ax_price.set_title("Price History (Last 1 Year)")
        ax_price.set_xlabel("Date")
        ax_price.set_ylabel("Price ($)")
        ax_price.legend()
        st.pyplot(fig_price)

    st.subheader("Key Financial Metrics")
    trailing_eps = info.get('trailingEps')
    dividend_rate = info.get('dividendRate')
    dividend_yield = info.get('dividendYield')
    if trailing_eps and trailing_eps != 0 and dividend_rate:
        payout = dividend_rate / trailing_eps
    else:
        payout = None

    st.write("Trailing EPS:", trailing_eps if trailing_eps is not None else "N/A")
    st.write("Dividend Rate:", dividend_rate if dividend_rate is not None else "N/A")
    st.write("Dividend Yield:", dividend_yield if dividend_yield is not None else "N/A")
    st.write("Payout Ratio:", round(payout,2) if payout is not None else "N/A")

# ------------------------------------
# ALTMAN Z-SCORE FUNCTION
# ------------------------------------

import yfinance as yf
import pprint

def get_bs_value(bs, col, keys):
    for key in keys:
        for bs_key in bs.index:
            if bs_key.strip().lower() == key.strip().lower():
                return bs.loc[bs_key][col]
    return None

def get_fs_value(fs, col, keys):
    for key in keys:
        for fs_key in fs.index:
            if fs_key.strip().lower() == key.strip().lower():
                return fs.loc[fs_key][col]
    return None

def compute_altman_z(ticker: str):
    t = yf.Ticker(ticker)
    bs = t.balance_sheet
    fs = t.financials
    info = t.info

    try:
        bs_col = bs.columns[0]
        fs_col = fs.columns[0]
    except Exception:
        print("Could not determine the latest financial data period.")
        return None

    total_assets = get_bs_value(bs, bs_col, ["Total Assets"])
    total_liabilities = get_bs_value(bs, bs_col, ["Total Liab", "Total Liabilities", "Total Liabilities Net Minority Interest"])
    current_assets = get_bs_value(bs, bs_col, ["Total Current Assets", "Current Assets"])
    current_liabilities = get_bs_value(bs, bs_col, ["Total Current Liabilities", "Current Liabilities"])
    working_capital = current_assets - current_liabilities if current_assets is not None and current_liabilities is not None else None
    retained_earnings = get_bs_value(bs, bs_col, ["Retained Earnings"])
    ebit = get_fs_value(fs, fs_col, ["Operating Income", "EBIT"])
    sales = get_fs_value(fs, fs_col, ["Total Revenue", "Revenue", "Sales"])

    share_price = info.get("regularMarketPrice")
    shares_outstanding = info.get("sharesOutstanding")
    market_value_of_equity = share_price * shares_outstanding if share_price and shares_outstanding else None

    if total_assets is None or total_liabilities is None or market_value_of_equity is None:
        print(f"Essential data not available for ticker {ticker}")
        return None

    ratio1 = (working_capital / total_assets) if working_capital is not None else 0.0
    ratio2 = (retained_earnings / total_assets) if retained_earnings is not None else 0.0
    ratio3 = (ebit / total_assets) if ebit is not None else 0.0
    ratio4 = (market_value_of_equity / total_liabilities) if total_liabilities != 0 else 0.0
    ratio5 = (sales / total_assets) if sales is not None else 0.0

    z_score = 1.2 * ratio1 + 1.4 * ratio2 + 3.3 * ratio3 + 0.6 * ratio4 + ratio5

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
        "Ratio2 (RE/TA)": ratio2,
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
