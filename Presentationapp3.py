import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

############################################
# DIVIDEND DASHBOARD FUNCTIONS
############################################

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
        data_to_plot = dividends.tail(10) if len(dividends) > 10 else dividends
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
# ALTMAN Z-SCORE FUNCTIONS
############################################

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
    t_obj = yf.Ticker(ticker)
    bs = t_obj.balance_sheet
    fs = t_obj.financials
    info = t_obj.info

    if bs is None or bs.empty:
        return None, f"Balance sheet data not available for ticker {ticker}."
    if fs is None or fs.empty:
        return None, f"Financial statement data not available for ticker {ticker}."

    bs_col = bs.columns[0]
    fs_col = fs.columns[0]

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
        return None, f"Essential data missing for ticker {ticker}."

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
# EXPLANATION PAGES
############################################

def explain_dividend_code():
    st.header("Dividend Dashboard Code Explanation")
    st.subheader("1. Function Signature")
    st.code("def display_dividend_dashboard(ticker: str):")
    st.write("Defines a function that takes a ticker symbol and drives the entire dividend dashboard.")
    st.subheader("2. Fetching Data")
    snippet = "t_obj = yf.Ticker(ticker)\ninfo = t_obj.info"
    st.code(snippet)
    st.write("Creates a yfinance Ticker object and retrieves company info.")
    st.subheader("3. Plot Dividends")
    snippet = "ax_div.bar(data_to_plot.index, data_to_plot)"
    st.code(snippet)
    st.write("Builds a bar chart of the last 10 dividend payments.")
    st.subheader("4. Calculate Payout Ratio")
    snippet = "dividend_payout_ratio = dividend_rate / trailing_eps"
    st.code(snippet)
    st.write("Computes the dividend payout ratio as dividends per share over earnings per share.")

def explain_altman_code():
    st.header("Altman Z-Score Code Explanation")
    st.subheader("1. Retrieving Financial Data")
    st.code("bs = t_obj.balance_sheet\nfs = t_obj.financials")
    st.write("Loads balance sheet and income statement into DataFrames.")
    st.subheader("2. Ratio 1: Working Capital / Total Assets")
    st.code("ratio1 = (working_capital / total_assets) if working_capital is not None else 0.0")
    st.write("Measures liquidity by comparing available capital to total assets.")
    st.subheader("3. Ratio 2: Retained Earnings / Total Assets")
    st.code("ratio2 = (retained_earnings / total_assets) if retained_earnings is not None else 0.0")
    st.write("Shows cumulative profitability relative to asset base.")
    st.subheader("4. Ratio 3: EBIT / Total Assets")
    st.code("ratio3 = (ebit / total_assets) if ebit is not None else 0.0")
    st.write("Assesses operating efficiency by earnings before interest and taxes.")
    st.subheader("5. Ratio 4: Market Value of Equity / Total Liabilities")
    st.code("ratio4 = (market_value_of_equity / total_liabilities) if total_liabilities != 0 else 0.0")
    st.write("Compares market capitalization to liabilities to gauge solvency.")
    st.subheader("6. Ratio 5: Sales / Total Assets")
    st.code("ratio5 = (sales / total_assets) if sales is not None else 0.0")
    st.write("Evaluates asset turnover by linking revenue to assets.")
    st.subheader("7. Classification Logic")
    classification_snip = (
        "if z_score > 2.99:\n    classification = 'Safe Zone'\n"
        "elif z_score >= 1.81:\n    classification = 'Grey Zone'\n"
        "else:\n    classification = 'Distressed Zone'"
    )
    st.code(classification_snip)
    st.write("Determines financial health category based on Z-Score thresholds.")
    st.subheader("8. Handling Total Liabilities Field")
    liabilities_snip = (
        "total_liabilities = get_bs_value(bs, bs_col, ['Total Liab', 'Total Liabilities',"
        " 'Total Liabilities Net Minority Interest'])"
    )
    st.code(liabilities_snip)
    st.write(
        "Includes 'Total Liabilities Net Minority Interest' because some filings label liabilities differently,"
        " ensuring we capture the correct figure."
    )

############################################
# MAIN APP
############################################

def main():
    st.title("Financial Dashboard")
    page = st.sidebar.radio(
        "Select Analysis",
        [
            "Dividend Dashboard",
            "Altman Z-Score",
            "Explain Dividend Code",
            "Explain Altman Code"
        ]
    )

    if page == "Dividend Dashboard":
        ticker_div = st.text_input(
            "Enter ticker symbol for Dividend Dashboard (e.g., AAPL)",
            value="AAPL",
            key="ticker_div"
        )
        if st.button("Show Dividend Data", key="div_btn"):
            if ticker_div:
                with st.spinner("Fetching dividend and price data..."):
                    display_dividend_dashboard(ticker_div)
            else:
                st.error("Please enter a ticker symbol for the Dividend Dashboard.")

    elif page == "Altman Z-Score":
        ticker_alt = st.text_input(
            "Enter ticker symbol for Altman Z-Score (e.g., AAPL)",
            value="AAPL",
            key="ticker_alt"
        )
        if st.button("Calculate Altman Z-Score", key="alt_btn"):
            if ticker_alt:
                with st.spinner("Calculating Altman Z-Score..."):
                    z_score, classification = compute_altman_z(ticker_alt)
                    if z_score is not None:
                        st.success(f"Altman Z-Score: {z_score:.2f}")
                        st.info(f"Classification: {classification}")
                    else:
                        st.error(f"Calculation failed: {classification}")
            else:
                st.error("Please enter a ticker symbol for Altman Z-Score.")

    elif page == "Explain Dividend Code":
        explain_dividend_code()

    else:
        explain_altman_code()

if __name__ == "__main__":
    main()
