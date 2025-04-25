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
    total_liabilities = get_bs_value(bs, bs_col, [
        "Total Liab",
        "Total Liabilities",
        "Total Liabilities Net Minority Interest"
    ])
    current_assets = get_bs_value(bs, bs_col, ["Total Current Assets", "Current Assets"])
    current_liabilities = get_bs_value(bs, bs_col, ["Total Current Liabilities", "Current Liabilities"])
    working_capital = (
        current_assets - current_liabilities
        if (current_assets is not None and current_liabilities is not None)
        else None
    )
    retained_earnings = get_bs_value(bs, bs_col, ["Retained Earnings"])
    ebit = get_fs_value(fs, fs_col, ["Operating Income", "EBIT"])
    sales = get_fs_value(fs, fs_col, ["Total Revenue", "Revenue", "Sales"])
    share_price = info.get('regularMarketPrice', None)
    shares_outstanding = info.get('sharesOutstanding', None)
    market_value_of_equity = (
        share_price * shares_outstanding
        if (share_price is not None and shares_outstanding is not None)
        else None
    )

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
    resp = requests.get(url)
    soup = BeautifulSoup(resp.text, 'html.parser')
    table = soup.find('table', {'id': 'constituents'})
    df = pd.read_html(str(table))[0]
    return df['Symbol'].tolist()

def extract_features(tickers):
    data = []
    for t in tickers:
        try:
            info = yf.Ticker(t).info
            dy  = info.get('dividendYield',    np.nan)
            er  = info.get('regularMarketPrice', np.nan)
            stl = info.get('beta',              np.nan)
        except Exception:
            dy, er, stl = np.nan, np.nan, np.nan
        data.append([t, dy, er, stl])
    return pd.DataFrame(data, columns=['Ticker','Dividend Yield','Expected Return','Stability'])

def perform_kmeans_clustering(df, k):
    dfc = df.dropna()
    X = dfc[['Dividend Yield', 'Expected Return', 'Stability']]
    km = KMeans(n_clusters=k, random_state=42).fit(X)
    dfc['Cluster'] = km.labels_
    return dfc, km

def recommend_stocks(df, budget):
    top = df.sort_values('Dividend Yield', ascending=False).head(5)
    alloc = budget / len(top)
    top['Allocation'] = alloc
    return top

############################################
# EXPLANATION PAGE
############################################

def explain_backend():
    st.title("Backend Code Explanation")

    st.markdown("## Imports & Setup")
    st.code("""
import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import requests
from bs4 import BeautifulSoup
from sklearn.cluster import KMeans
""")
    st.write("These libraries handle the UI, data fetching, parsing, clustering, and plotting.")

    with st.expander("Dividend Dashboard Code"):
        st.subheader("Function Definition")
        st.code("def display_dividend_dashboard(ticker: str):")
        st.write("Main function driving the Dividend Dashboard page.")
        
        st.subheader("Data Fetching")
        st.code("""
t_obj = yf.Ticker(ticker)
info = t_obj.info
""")
        st.write("Creates a yfinance Ticker object and retrieves company metadata.")

        st.subheader("Plot Dividends")
        st.code("""
dividends = t_obj.dividends
data_to_plot = dividends.tail(10)
fig_div, ax_div = plt.subplots(figsize=(10,4))
ax_div.bar(data_to_plot.index, data_to_plot)
st.pyplot(fig_div)
""")
        st.write("Generates a bar chart of the last 10 dividend payments.")

        st.subheader("Payout Ratio Calculation")
        st.code("""
trailing_eps = info.get('trailingEps')
dividend_rate = info.get('dividendRate')
dividend_payout_ratio = dividend_rate / trailing_eps
""")
        st.write("Dividends per share divided by earnings per share.")

    with st.expander("Altman Z-Score Code"):
        st.subheader("Loading Statements")
        st.code("""
bs = t_obj.balance_sheet
fs = t_obj.financials
""")
        st.write("Loads the balance sheet and income statement as DataFrames.")

        st.subheader("Extracting Metrics")
        st.code("""
total_assets = get_bs_value(bs, bs_col, ['Total Assets'])
total_liabilities = get_bs_value(bs, bs_col, [
    'Total Liab',
    'Total Liabilities',
    'Total Liabilities Net Minority Interest'
])
""")
        st.write("Captures assets and liabilities, including alternate labels for liabilities.")

        st.subheader("Ratio Computations")
        st.code("""
ratio1 = working_capital / total_assets
ratio2 = retained_earnings / total_assets
ratio3 = ebit / total_assets
ratio4 = market_value_of_equity / total_liabilities
ratio5 = sales / total_assets
""")
        st.write("Five core components of the Altman Z-Score model.")

        st.subheader("Final Z-Score Formula")
        st.code("""
z_score = 1.2*ratio1 + 1.4*ratio2 + 3.3*ratio3 + 0.6*ratio4 + ratio5
""")
        st.write("Weighted sum of the five ratios per Altman's original research.")

        st.subheader("Classification Logic")
        st.code("""
if z_score > 2.99:
    classification = 'Safe Zone'
elif z_score >= 1.81:
    classification = 'Grey Zone'
else:
    classification = 'Distressed Zone'
""")
        st.write("Categorizes financial health based on Z-Score thresholds.")
        st.markdown("**Ranges:**  \n- Safe Zone: Z > 2.99  \n- Grey Zone: 1.81 ≤ Z ≤ 2.99  \n- Distressed Zone: Z < 1.81")

    with st.expander("Investing Analysis Code"):
        st.subheader("Ticker Fetching")
        st.code("""
tickers = get_sp500_tickers()
""")
        st.write("Scrapes Wikipedia to retrieve the current S&P 500 constituent list.")

        st.subheader("Feature Extraction")
        st.code("""
df = extract_features(tickers)
""")
        st.write("Pulls dividend yield, price, and beta for each ticker with error handling.")

        st.subheader("Elbow Method & Clustering")
        st.code("""
inertias = []
for i in range(1, max_k+1):
    model = KMeans(n_clusters=i).fit(df.dropna()[features])
    inertias.append(model.inertia_)
fig, ax = plt.subplots()
ax.plot(range(1, max_k+1), inertias, marker='o')
""")
        st.write("Plots inertia vs. k to identify the optimal cluster count.")

        st.subheader("Final Recommendation")
        st.code("""
dfc, km = perform_kmeans_clustering(df, k)
rec = recommend_stocks(dfc, budget)
st.write(rec)
""")
        st.write("Assigns your budget evenly to the top dividend-paying clusters.")

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
            result = compute_altman_z(ticker)
            if result[0] is not None:
                z_score, classification = result
                st.success(f"Altman Z-Score: {z_score:.2f}")
                st.info(f"Classification: {classification}")
                st.markdown("**Classification Ranges:**")
                st.markdown("- **Safe Zone:** Z > 2.99")
                st.markdown("- **Grey Zone:** 1.81 ≤ Z ≤ 2.99")
                st.markdown("- **Distressed Zone:** Z < 1.81")
            else:
                st.error(result[1])

    elif page == "Investing Analysis":
        st.header("Investing Analysis")
        st.write("Clustering the full S&P 500 dataset…")
        with st.spinner("Fetching & featurizing…"):
            tickers = get_sp500_tickers()
            df = extract_features(tickers)
        st.write(f"Loaded {len(df)} tickers; {df.dropna().shape[0]} with complete features.")

        features = ['Dividend Yield','Expected Return','Stability']
        max_k = min(df.dropna().shape[0], 10)
        k = st.slider("Number of clusters (k)", 1, max_k, 3)
        budget = st.number_input("Investment budget ($)", 1000.0)

        if st.button("Run Analysis"):
            # Elbow plot
            inertias = []
            X = df.dropna()[features]
            for i in range(1, max_k+1):
                inertias.append(KMeans(n_clusters=i, random_state=42).fit(X).inertia_)
            fig, ax = plt.subplots()
            ax.plot(range(1, max_k+1), inertias, marker='o')
            ax.set_xlabel("k")
            ax.set_ylabel("Inertia")
            ax.set_title("Elbow Method")
            st.pyplot(fig)

            # Clustering & recommendation
            dfc, _ = perform_kmeans_clustering(df, k)
            rec = recommend_stocks(dfc, budget)
            st.write("### Top Recommendations", rec)

    else:
        explain_backend()

if __name__ == "__main__":
    main()
