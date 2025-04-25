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
    eps = info.get('trailingEps')
    rate = info.get('dividendRate')
    yld = info.get('dividendYield')
    payout = (rate / eps) if (eps and eps != 0 and rate) else None

    st.write("Trailing EPS:", eps or "N/A")
    st.write("Dividend Rate:", rate or "N/A")
    st.write("Dividend Yield:", yld or "N/A")
    st.write("Payout Ratio:", round(payout,2) if payout else "N/A")

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
        return None, f"Balance sheet data not available for {ticker}."
    if fs is None or fs.empty:
        return None, f"Financial statement data not available for {ticker}."

    bs_col, fs_col = bs.columns[0], fs.columns[0]
    TA = get_bs_value(bs, bs_col, ["Total Assets"])
    TL = get_bs_value(bs, bs_col, ["Total Liab","Total Liabilities","Total Liabilities Net Minority Interest"])
    CA = get_bs_value(bs, bs_col, ["Total Current Assets","Current Assets"])
    CL = get_bs_value(bs, bs_col, ["Total Current Liabilities","Current Liabilities"])
    WC = (CA - CL) if (CA is not None and CL is not None) else None
    RE = get_bs_value(bs, bs_col, ["Retained Earnings"])
    EBIT = get_fs_value(fs, fs_col, ["Operating Income","EBIT"])
    SALES = get_fs_value(fs, fs_col, ["Total Revenue","Revenue","Sales"])
    price = info.get('regularMarketPrice')
    shares = info.get('sharesOutstanding')
    MVE = (price * shares) if (price and shares) else None

    if not all([TA, TL, MVE]):
        return None, "Missing essential data."

    r1 = (WC/TA) if WC else 0.0
    r2 = (RE/TA) if RE else 0.0
    r3 = (EBIT/TA) if EBIT else 0.0
    r4 = (MVE/TL) if TL else 0.0
    r5 = (SALES/TA) if SALES else 0.0

    Z = 1.2*r1 + 1.4*r2 + 3.3*r3 + 0.6*r4 + r5
    cls = "Safe Zone" if Z>2.99 else "Grey Zone" if Z>=1.81 else "Distressed Zone"
    return Z, cls

############################################
# INVESTING ANALYSIS FUNCTIONS
############################################

def get_sp500_tickers():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    resp = requests.get(url)
    soup = BeautifulSoup(resp.text,'html.parser')
    table = soup.find('table',{'id':'constituents'})
    df = pd.read_html(str(table))[0]
    return df['Symbol'].tolist()

def extract_features(tickers):
    rows = []
    for t in tickers:
        info = yf.Ticker(t).info
        rows.append([
            t,
            info.get('dividendYield', np.nan),
            info.get('regularMarketPrice', np.nan),
            info.get('beta', np.nan)
        ])
    return pd.DataFrame(rows,columns=['Ticker','Dividend Yield','Expected Return','Stability'])

def perform_kmeans_clustering(df, k):
    dfc = df.dropna()
    X = dfc[['Dividend Yield','Expected Return','Stability']]
    km = KMeans(n_clusters=k, random_state=42).fit(X)
    dfc['Cluster'] = km.labels_
    return dfc, km

def recommend_stocks(df, budget):
    top = df.sort_values('Dividend Yield',ascending=False).head(5)
    alloc = budget/len(top)
    top['Allocation'] = alloc
    return top

############################################
# EXPLANATION PAGE
############################################

def explain_backend():
    st.header("Backend Code Explanation")
    with st.expander("Dividend Dashboard"):
        st.code("display_dividend_dashboard(ticker)")
        st.write("Fetch summary, plot dividends & price, compute payout ratio.")
    with st.expander("Altman Z-Score"):
        st.code("compute_altman_z(ticker)")
        st.write("Extract 5 ratios, combine into Z, classify into zones.")
    with st.expander("Investing Analysis"):
        st.code("tickers = get_sp500_tickers()  # 500 points")
        st.code("df = extract_features(tickers)    # yields Dividend/Price/Beta")
        st.code("dfc, km = perform_kmeans_clustering(df, k)")
        st.write("Clusters S&P 500 by those features, then recommends top dividend payers.")

############################################
# STREAMLIT APP
############################################

def main():
    st.title("Financial Dashboard")

    page = st.sidebar.radio("Select Page",[
        "Dividend Dashboard",
        "Altman Z-Score",
        "Investing Analysis",
        "Explain Backend"
    ])

    if page=="Dividend Dashboard":
        t = st.text_input("Ticker","AAPL")
        if st.button("Show Dividend Data"):
            display_dividend_dashboard(t)

    elif page=="Altman Z-Score":
        t = st.text_input("Ticker","AAPL",key="alt")
        if st.button("Calculate Z-Score"):
            z,cls = compute_altman_z(t)
            if z is not None:
                st.success(f"Altman Z-Score: {z:.2f}")
                st.info(f"Classification: {cls}")
            else:
                st.error(cls)

    elif page=="Investing Analysis":
        st.header("Investing Analysis")
        st.write("Clustering the full S&P 500 dataset:")
        with st.spinner("Fetching & featurizing..."):
            tickers = get_sp500_tickers()
            df = extract_features(tickers)
        if df.empty:
            st.error("Failed to load features.")
            return

        st.write(f"Loaded {len(df)} tickers; {df.dropna().shape[0]} with complete features.")
        max_k = min(df.dropna().shape[0], 10)
        k = st.slider("Number of clusters (k)",1,max_k,3)
        budget = st.number_input("Investment budget ($)",1000.0)

        if st.button("Run Analysis"):
            dfc, _ = perform_kmeans_clustering(df,k)
            # Elbow
            inertias=[]
            X = dfc[['Dividend Yield','Expected Return','Stability']]
            for i in range(1,max_k+1):
                inertias.append(KMeans(n_clusters=i,random_state=42).fit(X).inertia_)
            fig,ax=plt.subplots()
            ax.plot(range(1,max_k+1),inertias,marker='o')
            ax.set_xlabel("k"); ax.set_ylabel("Inertia"); ax.set_title("Elbow Method")
            st.pyplot(fig)

            rec = recommend_stocks(dfc,budget)
            st.write("### Top Recommendations",rec)

    else:
        explain_backend()

if __name__=="__main__":
    main()
