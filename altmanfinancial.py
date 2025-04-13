import streamlit as st
import yfinance as yf

############################################
# Helper Functions for Altman Z‑Score
############################################

def get_bs_value(bs, col, keys):
    """
    Searches the balance sheet DataFrame for the first matching key.
    Comparison is done in a case‑insensitive manner.
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
    Comparison is done in a case‑insensitive manner.
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
      - Total Assets (looked up using "Total Assets")
      - Total Liabilities (tries "Total Liab", "Total Liabilities", and "Total Liabilities Net Minority Interest")
      - Market Value of Equity (computed from share price and shares outstanding)
    
    Returns a tuple: (z_score, classification) if successful; otherwise, (None, error_message).
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
        bs_col = bs.columns[0]  # most recent annual reporting period
    except Exception:
        return None, "Could not determine the latest balance sheet period."
    try:
        fs_col = fs.columns[0]
    except Exception:
        return None, "Could not determine the latest financial statement period."

    total_assets = get_bs_value(bs, bs_col, ["Total Assets"])
    total_liabilities = get_bs_value(bs, bs_col, ["Total Liab", "Total Liabilities", "Total Liabilities Net Minority Interest"])
    current_assets = get_bs_value(bs, bs_col, ["Total Current Assets", "Current Assets"])
    current_liabilities = get_bs_value(bs, bs_col, ["Total Current Liabilities", "Current Liabilities"])
    working_capital = current_assets - current_liabilities if (current_assets is not None and current_liabilities is not None) else None
    retained_earnings = get_bs_value(bs, bs_col, ["Retained Earnings"])

    ebit = get_fs_value(fs, fs_col, ["Operating Income", "EBIT"])
    sales = get_fs_value(fs, fs_col, ["Total Revenue", "Revenue", "Sales"])

    share_price = info.get('regularMarketPrice', None)
    shares_outstanding = info.get('sharesOutstanding', None)
    market_value_of_equity = share_price * shares_outstanding if (share_price is not None and shares_outstanding is not None) else None

    if total_assets is None or total_liabilities is None or market_value_of_equity is None:
        msg = (f"Essential data missing for ticker {ticker}. "
               f"Total Assets: {total_assets}, Total Liabilities: {total_liabilities}, "
               f"Market Value of Equity: {market_value_of_equity}")
        return None, msg

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
# Dividend Dashboard Functions
############################################

def display_dividend_dashboard(ticker: str):
    """
    Retrieves dividend history and price history for the given ticker
    and displays them.
    """
    t_obj = yf.Ticker(ticker)
    dividends = t_obj.dividends
    price_history = t_obj.history(period="1y")
    
    st.subheader(f"Dividend History for {ticker}")
    if dividends is not None and not dividends.empty:
        # Limit to last 10 entries for clarity.
        data_to_plot = dividends.tail(10) if len(dividends) > 10 else dividends
        st.bar_chart(data_to_plot)
    else:
        st.write("No dividend data available.")

    st.subheader(f"Price History for {ticker} (Last 1 Year)")
    if price_history is not None and not price_history.empty:
        st.line_chart(price_history['Close'])
    else:
        st.write("No price data available.")


############################################
# Main App
############################################

st.title("Financial Dashboard")
st.markdown("Welcome to your integrated financial dashboard. Select your analysis from the sidebar.")

analysis_choice = st.sidebar.radio("Select Analysis", options=["Dividend Dashboard", "Altman Z‑Score"])

if analysis_choice == "Dividend Dashboard":
    st.header("Dividend Dashboard")
    ticker_div = st.text_input("Enter ticker symbol (e.g., AAPL) for Dividend Dashboard", value="AAPL", key="ticker_div")
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

    The Altman Z‑Score is a financial model used to predict the likelihood of bankruptcy. It combines five key ratios derived from a company’s financial statements. The formula is:

    \[
    Z = 1.2 \times \left(\frac{\text{Working Capital}}{\text{Total Assets}}\right) +
        1.4 \times \left(\frac{\text{Retained Earnings}}{\text{Total Assets}}\right) +
        3.3 \times \left(\frac{\text{EBIT}}{\text{Total Assets}}\right) +
        0.6 \times \left(\frac{\text{Market Value of Equity}}{\text{Total Liabilities}}\right) +
        \left(\frac{\text{Sales}}{\text{Total Assets}}\right)
    \]

    **Interpretation of the Z‑Score:**
    - **Safe Zone (Z > 2.99):** The company is financially healthy with a low risk of bankruptcy.
    - **Grey Zone (1.81 ≤ Z ≤ 2.99):** The company is in a cautionary zone and may need closer monitoring.
    - **Distressed Zone (Z < 1.81):** The company faces a high risk of financial distress.

    """)
    ticker_altman = st.text_input("Enter ticker symbol (e.g., AAPL) for Altman Z‑Score", value="AAPL", key="ticker_altman")
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
