import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

############################################
# Helper Functions for Altman Z‑Score
############################################

def get_bs_value(bs, col, keys):
    """
    Searches the balance sheet DataFrame for the first matching key.
    Comparison is done in a case‑insensitive way after stripping whitespace.
    Returns the value from the specified column if found; otherwise, None.
    """
    for key in keys:
        for bs_key in bs.index:
            if bs_key.strip().lower() == key.strip().lower():
                return bs.loc[bs_key][col]
    return None

def get_fs_value(fs, col, keys):
    """
    Searches the financials (income statement) DataFrame for the first matching key.
    Comparison is case‑insensitive.
    Returns the value from the specified column if found; otherwise, None.
    """
    for key in keys:
        for fs_key in fs.index:
            if fs_key.strip().lower() == key.strip().lower():
                return fs.loc[fs_key][col]
    return None

def compute_altman_z(ticker: str):
    """
    Retrieves data for the given ticker using yfinance and calculates the Altman Z‑Score.
    
    The formula is:
      Z = 1.2 * (Working Capital / Total Assets) +
          1.4 * (Retained Earnings / Total Assets) +
          3.3 * (EBIT / Total Assets) +
          0.6 * (Market Value of Equity / Total Liabilities) +
          (Sales / Total Assets)
    
    Essential data:
      - Total Assets (lookup using "Total Assets")
      - Total Liabilities (lookup using "Total Liab", "Total Liabilities", or "Total Liabilities Net Minority Interest")
      - Market Value of Equity (computed from share price * shares outstanding)
    
    Returns a tuple (z_score, classification) if successful; otherwise, (None, error_message).
    """
    t_obj = yf.Ticker(ticker)
    bs = t_obj.balance_sheet
    fs = t_obj.financials
    info = t_obj.info

    if bs is None or bs.empty:
        return None, f"Balance sheet data not available for ticker {ticker}."
    if fs is None or fs.empty:
        return None, f"Financial statement data not available for ticker {ticker}."

    try:
        bs_col = bs.columns[0]  # Most recent reporting period.
    except Exception:
        return None, "Could not determine the latest balance sheet period."
    try:
        fs_col = fs.columns[0]
    except Exception:
        return None, "Could not determine the latest financial statement period."

    # Retrieve balance sheet metrics.
    total_assets = get_bs_value(bs, bs_col, ["Total Assets"])
    total_liabilities = get_bs_value(bs, bs_col, ["Total Liab", "Total Liabilities", "Total Liabilities Net Minority Interest"])
    current_assets = get_bs_value(bs, bs_col, ["Total Current Assets", "Current Assets"])
    current_liabilities = get_bs_value(bs, bs_col, ["Total Current Liabilities", "Current Liabilities"])
    working_capital = current_assets - current_liabilities if (current_assets is not None and current_liabilities is not None) else None
    retained_earnings = get_bs_value(bs, bs_col, ["Retained Earnings"])

    # Retrieve income statement metrics.
    ebit = get_fs_value(fs, fs_col, ["Operating Income", "EBIT"])
    sales = get_fs_value(fs, fs_col, ["Total Revenue", "Revenue", "Sales"])

    # Compute Market Value of Equity.
    share_price = info.get('regularMarketPrice', None)
    shares_outstanding = info.get('sharesOutstanding', None)
    market_value_of_equity = share_price * shares_outstanding if (share_price is not None and shares_outstanding is not None) else None

    if total_assets is None or total_liabilities is None or market_value_of_equity is None:
        msg = (f"Essential data missing for ticker {ticker}. "
               f"Total Assets: {total_assets}, Total Liabilities: {total_liabilities}, "
               f"Market Value of Equity: {market_value_of_equity}")
        return None, msg

    # Compute ratios (default missing non-essential values to 0.0).
    ratio1 = (working_capital / total_assets) if working_capital is not None else 0.0
    ratio2 = (retained_earnings / total_assets) if retained_earnings is not None else 0.0
    ratio3 = (ebit / total_assets) if ebit is not None else 0.0
    ratio4 = (market_value_of_equity / total_liabilities) if total_liabilities != 0 else 0.0
    ratio5 = (sales / total_assets) if sales is not None else 0.0

    z_score = 1.2 * ratio1 + 1.4 * ratio2 + 3.3 * ratio3 + 0.6 * ratio4 + ratio5

    if z_score > 2.99:
        classification = "Safe Zone"
    elif z_score >= 1.81:
        classification = "Grey Zone"
    else:
        classification = "Distressed Zone"

    return z_score, classification

############################################
# Dividend Dashboard Function
############################################

def display_dividend_dashboard(ticker: str):
    """
    Retrieves and displays the dividend dashboard for the given ticker.
    Shows:
      - Company overview
      - Dividend history (last 10 entries) as a bar chart
      - Price history (last 1 year) as a line chart
      - Key financial metrics (Trailing EPS, Dividend Rate, Dividend Yield, Dividend Payout Ratio)
    """
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
        data_to_plot = dividends if len(dividends) <= 10 else dividends.tail(10)
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
    trailing_eps = info.get('trailingEps', None)
    dividend_rate = info.get('dividendRate', None)
    dividend_yield = info.get('dividendYield', None)
    if trailing_eps and trailing_eps != 0 and dividend_rate:
        dividend_payout_ratio = dividend_rate / trailing_eps
    else:
        dividend_payout_ratio = None

    st.write("Trailing EPS:", trailing_eps if trailing_eps is not None else "N/A")
    st.write("Dividend Rate:", dividend_rate if dividend_rate is not None else "N/A")
    st.write("Dividend Yield:", dividend_yield if dividend_yield is not None else "N/A")
    if dividend_payout_ratio is not None:
        st.write("Dividend Payout Ratio:", round(dividend_payout_ratio, 2))
    else:
        st.write("Dividend payout ratio could not be calculated due to missing data.")

############################################
# Main App
############################################

def main():
    st.title("Financial Dashboard")
    st.markdown("Welcome to your integrated financial dashboard. Use the sidebar to select your analysis.")

    analysis_choice = st.sidebar.radio("Select Analysis", options=["Dividend Dashboard", "Altman Z‑Score"])

    if analysis_choice == "Dividend Dashboard":
        st.header("Dividend Dashboard")
        ticker_div = st.text_input("Enter ticker symbol for Dividend Dashboard (e.g., AAPL)", value="AAPL", key="ticker_div")
        if st.button("Show Dividend Data", key="div_btn"):
            if ticker_div:
                with st.spinner("Fetching dividend and price data..."):
                    display_dividend_dashboard(ticker_div)
            else:
                st.error("Please enter a ticker symbol for the Dividend Dashboard.")

    elif analysis_choice == "Altman Z‑Score":
        st.header("Altman Z‑Score Calculator")
        st.markdown("""
        **Altman Z‑Score Explanation:**

        The Altman Z‑Score is a financial model used to predict the likelihood of bankruptcy. It combines five ratios derived from a company's financial statements:

        \\[
        Z = 1.2 \\times \\left(\\frac{\\text{Working Capital}}{\\text{Total Assets}}\\right) +
            1.4 \\times \\left(\\frac{\\text{Retained Earnings}}{\\text{Total Assets}}\\right) +
            3.3 \\times \\left(\\frac{\\text{EBIT}}{\\text{Total Assets}}\\right) +
            0.6 \\times \\left(\\frac{\\text{Market Value of Equity}}{\\text{Total Liabilities}}\\right) +
            \\left(\\frac{\\text{Sales}}{\\text{Total Assets}}\\right)
        \\]

        **Interpretation of the Z‑Score:**
        - **Safe Zone (Z > 2.99):** The company is financially healthy.
        - **Grey Zone (1.81 ≤ Z ≤ 2.99):** The company is in a cautionary zone.
        - **Distressed Zone (Z < 1.81):** The company has a high risk of financial distress.
        """)
        ticker_altman = st.text_input("Enter ticker symbol for Altman Z‑Score (e.g., AAPL)", value="AAPL", key="ticker_altman")
        if st.button("Calculate Altman Z‑Score", key="alt_btn"):
            if ticker_altman:
                with st.spinner("Calculating Altman Z‑Score..."):
                    z_score, classification = compute_altman_z(ticker_altman)
                    if z_score is not None:
                        st.success(f"Altman Z‑Score: {z_score:.2f}")
                        st.info(f"Classification: {classification}")
                    else:
                        st.error(f"Calculation failed: {classification}")
            else:
                st.error("Please enter a ticker symbol for Altman Z‑Score.")

if __name__ == "__main__":
    main()
