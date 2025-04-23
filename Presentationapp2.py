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

def compute_altman_z(ticker: str):
    t_obj = yf.Ticker(ticker)
    bs = t_obj.balance_sheet
    fs = t_obj.financials
    info = t_obj.info

    if bs is None or bs.empty:
        return None, f"Balance sheet data not available for {ticker}."
    if fs is None or fs.empty:
        return None, f"Financial statements not available for {ticker}."

    bs_col = bs.columns[0]
    fs_col = fs.columns[0]

    def get_bs_value(df, col, keys):
        for key in keys:
            for k in df.index:
                if k.strip().lower() == key.strip().lower():
                    return df.loc[k][col]
        return None

    def get_fs_value(df, col, keys):
        for key in keys:
            for k in df.index:
                if k.strip().lower() == key.strip().lower():
                    return df.loc[k][col]
        return None

    TA = get_bs_value(bs, bs_col, ["Total Assets"])
    TL = get_bs_value(bs, bs_col, ["Total Liab", "Total Liabilities"])
    CA = get_bs_value(bs, bs_col, ["Total Current Assets"])
    CL = get_bs_value(bs, bs_col, ["Total Current Liabilities"])
    WC = CA - CL if CA and CL else None
    RE = get_bs_value(bs, bs_col, ["Retained Earnings"])
    EBIT = get_fs_value(fs, fs_col, ["Operating Income", "EBIT"])
    SALES = get_fs_value(fs, fs_col, ["Total Revenue", "Sales"])
    price = info.get('regularMarketPrice')
    shares = info.get('sharesOutstanding')
    MVE = price * shares if price and shares else None

    if not all([TA, TL, MVE]):
        return None, "Missing essential data."

    r1 = (WC / TA) if WC else 0
    r2 = (RE / TA) if RE else 0
    r3 = (EBIT / TA) if EBIT else 0
    r4 = (MVE / TL) if TL else 0
    r5 = (SALES / TA) if SALES else 0

    Z = 1.2*r1 + 1.4*r2 + 3.3*r3 + 0.6*r4 + r5
    cls = "Safe Zone" if Z>2.99 else "Grey Zone" if Z>=1.81 else "Distressed Zone"

    return Z, cls

# ------------------------------------
# STREAMLIT APP
# ------------------------------------

def dividend_page():
    st.header("Dividend Dashboard")
    ticker = st.text_input("Ticker", "AAPL")
    if st.button("Show Dividend Data"):
        display_dividend_dashboard(ticker)

def altman_page():
    st.header("Altman Z-Score Calculator")
    ticker = st.text_input("Ticker", "AAPL", key="alt")
    if st.button("Calculate Z-Score"):
        Z, cls = compute_altman_z(ticker)
        if Z is not None:
            st.success(f"Altman Z-Score: {Z:.2f}")
            st.info(f"Classification: {cls}")
        else:
            st.error(cls)

def main():
    st.title("Financial Dashboard")
    page = st.sidebar.radio("Select Page", ["Dividend Dashboard", "Altman Z-Score"])
    if page == "Dividend Dashboard":
        dividend_page()
    else:
        altman_page()

if __name__ == "__main__":
    main()
