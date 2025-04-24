import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import requests
from bs4 import BeautifulSoup
from sklearn.cluster import KMeans

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
# INVESTING ANALYSIS FUNCTIONS
############################################

def get_sp500_tickers():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    table = soup.find('table', {'id': 'constituents'})
    df = pd.read_html(str(table))[0]
    return df['Symbol'].tolist()

def extract_features(tickers):
    data = []
    for ticker in tickers:
        stock = yf.Ticker(ticker)
        info = stock.info
        dividend_yield = info.get('dividendYield', np.nan)
        expected_return = info.get('regularMarketPrice', np.nan)
        stability = info.get('beta', np.nan)
        data.append([ticker, dividend_yield, expected_return, stability])
    return pd.DataFrame(data, columns=['Ticker', 'Dividend Yield', 'Expected Return', 'Stability'])

def elbow_method(df):
    df_clean = df.dropna()
    X = df_clean[['Dividend Yield', 'Expected Return', 'Stability']]
    inertia = []
    for k in range(1, 11):
        km = KMeans(n_clusters=k, random_state=42).fit(X)
        inertia.append(km.inertia_)
    fig, ax = plt.subplots()
    ax.plot(range(1, 11), inertia, marker='o')
    ax.set_xlabel("k")
    ax.set_ylabel("Inertia")
    st.pyplot(fig)

def perform_kmeans_clustering(df, num_clusters=3):
    df_clean = df.dropna()
    X = df_clean[['Dividend Yield', 'Expected Return', 'Stability']]
    km = KMeans(n_clusters=num_clusters, random_state=42).fit(X)
    df_clean['Cluster'] = km.labels_
    return df_clean, km

def recommend_stocks(df, user_budget):
    rec = df.sort_values('Dividend Yield', ascending=False).head(5)
    per = user_budget / len(rec)
    rec['Allocation'] = per
    return rec

############################################
# EXPLANATION PAGE
############################################

def explain_backend():
    st.header("Backend Code Explanation")

    with st.expander("Dividend Dashboard"):
        st.subheader("Function Signature")
        st.code("def display_dividend_dashboard(ticker: str):")
        st.subheader("Fetching Data")
        st.code("t_obj = yf.Ticker(ticker)\ninfo = t_obj.info")
        st.write("Retrieves company info and dividends via yfinance.")
        st.subheader("Plotting Dividends")
        st.code("ax_div.bar(data_to_plot.index, data_to_plot)")
        st.write("Creates bar chart of last 10 dividends.")

    with st.expander("Altman Z-Score"):
        st.subheader("Ratio Components")
        st.code("ratio1 = working_capital / total_assets")
        st.code("ratio2 = retained_earnings / total_assets")
        st.code("ratio3 = ebit / total_assets")
        st.code("ratio4 = market_value_of_equity / total_liabilities")
        st.code("ratio5 = sales / total_assets")
        st.subheader("Classification Logic")
        st.code("if z_score > 2.99:\n    classification = 'Safe Zone'\nelif z_score >= 1.81:\n    classification = 'Grey Zone'\nelse:\n    classification = 'Distressed Zone'")
        st.write("Assigns zone based on Z-Score thresholds.")

    with st.expander("Investing Analysis"):
        st.subheader("Fetching S&P 500 Tickers")
        st.code("def get_sp500_tickers(): ... # scrapes Wikipedia for tickers")
        st.subheader("Clustering (Elbow Method)")
        st.code("inertia.append(KMeans(n_clusters=k).inertia_)")
        st.write("Plots inertia to identify optimal clusters.")
        st.subheader("Recommendations")
        st.code("rec = df.sort_values('Dividend Yield', ascending=False).head(5)")
        st.write("Picks top dividend payers within budget.")

############################################
# STREAMLIT APP
############################################

def main():
    st.title("Financial Dashboard")

    page = st.sidebar.radio("Select Page", [
        "Dividend Dashboard",
        "Altman Z-Score",
        "Investing Analysis",
        "Explain Backend"
    ])

    if page == "Dividend Dashboard":
        ticker = st.text_input("Ticker", "AAPL")
        if st.button("Show Dividend Data"):
            display_dividend_dashboard(ticker)

    elif page == "Altman Z-Score":
        ticker = st.text_input("Ticker", "AAPL", key="alt")
        if st.button("Calculate Z-Score"):
            z, cls = compute_altman_z(ticker)
            if z is not None:
                st.success(f"Altman Z-Score: {z:.2f}")
                st.info(f"Classification: {cls}")
            else:
                st.error(cls)

    elif page == "Investing Analysis":
        if st.button("Fetch Tickers"):
            tickers = get_sp500_tickers()
            df = extract_features(tickers)
            st.write(df.head())
            k = st.slider("Clusters (k)", 1, 10, 3)
            elbow_method(df)
            dfc, _ = perform_kmeans_clustering(df, k)
            budget = st.number_input("Budget ($)", 1000.0)
            rec = recommend_stocks(dfc, budget)
            st.write(rec)

    else:
        explain_backend()

if __name__ == "__main__":
    main()
