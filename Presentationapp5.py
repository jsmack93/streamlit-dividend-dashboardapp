import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import requests
from bs4 import BeautifulSoup
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy import stats

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

def perform_clustering(df):
    df_clean = df.dropna(subset=['Dividend Yield','Expected Return','Stability'])
    df_clean = df_clean[(np.abs(stats.zscore(df_clean[['Dividend Yield','Expected Return','Stability']]))<3).all(axis=1)]
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df_clean[['Dividend Yield','Expected Return','Stability']])
    model = KMeans(n_clusters=3, random_state=42).fit(scaled)
    df_clean['Cluster'] = model.predict(scaled)
    return model, df_clean

def recommend_stocks(df, budget, model=None, preferences=None, min_price_per_stock=20, max_price_per_stock=500):
    df_clean = df.dropna(subset=['Dividend Yield','Expected Return','Stability'])
    df_clean = df_clean[(df_clean['Price']>=min_price_per_stock) & (df_clean['Price']<=max_price_per_stock)]
    if model and preferences:
        df_clean['Cluster'] = model.predict(
            StandardScaler().fit_transform(df_clean[['Dividend Yield','Expected Return','Stability']])
        )
        best = df_clean['Cluster'].mode()[0]
        df_clean = df_clean[df_clean['Cluster']==best]
    if preferences:
        df_clean = df_clean.sort_values(preferences['priority'], ascending=False)
    selected = df_clean.head(5)
    selected['Allocation'] = budget / len(selected) if len(selected)>0 else 0
    return selected

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
from sklearn.preprocessing import StandardScaler
from scipy import stats
""")
    st.write("Libraries for UI, data retrieval, parsing, clustering, scaling, and statistics.")

    with st.expander("Dividend Dashboard Code"):
        st.subheader("Function Definition")
        st.code("def display_dividend_dashboard(ticker: str):")
        st.write("Main function driving the Dividend Dashboard page.")
        
        st.subheader("Data Fetching & Summary")
        st.code("""
t_obj = yf.Ticker(ticker)
info = t_obj.info
if 'longBusinessSummary' in info:
    st.write(info['longBusinessSummary'])
else:
    st.write("No overview available.")
""")
        st.write("Retrieves company metadata and displays the business summary.")

        st.subheader("Plotting Dividends")
        st.code("""
dividends = t_obj.dividends
data_to_plot = dividends.tail(10)
fig_div, ax_div = plt.subplots(figsize=(10,4))
ax_div.bar(data_to_plot.index, data_to_plot)
st.pyplot(fig_div)
""")
        st.write("Visualizes the last 10 dividend payments as a bar chart.")
        
        st.subheader("Key Metrics Calculation")
        st.code("""
trailing_eps = info.get('trailingEps')
dividend_rate = info.get('dividendRate')
dividend_payout_ratio = dividend_rate / trailing_eps
""")
        st.write("Calculates payout ratio from dividends and earnings.")

    with st.expander("Altman Z-Score Code"):
        st.subheader("Loading Financials")
        st.code("""
bs = t_obj.balance_sheet
fs = t_obj.financials
""")
        st.write("Loads balance sheet and income statement into DataFrames.")

        st.subheader("Extracting Key Values")
        st.code("""
total_assets = get_bs_value(bs, bs_col, ['Total Assets'])
total_liabilities = get_bs_value(bs, bs_col, [
    'Total Liab',
    'Total Liabilities',
    'Total Liabilities Net Minority Interest'
])
""")
        st.write("Handles multiple liability labels to ensure correct data capture.")

        st.subheader("Computing Ratios")
        st.code("""
ratio1 = working_capital / total_assets
ratio2 = retained_earnings / total_assets
ratio3 = ebit / total_assets
ratio4 = market_value_of_equity / total_liabilities
ratio5 = sales / total_assets
""")
        st.write("Five financial ratios per Altman's methodology.")

        st.subheader("Z-Score & Classification")
        st.code("""
z_score = 1.2*ratio1 + 1.4*ratio2 + 3.3*ratio3 + 0.6*ratio4 + ratio5

if z_score > 2.99:
    classification = 'Safe Zone'
elif z_score >= 1.81:
    classification = 'Grey Zone'
else:
    classification = 'Distressed Zone'
""")
        st.write("Combines ratios into Z and categorizes risk levels.")
        st.markdown("**Ranges:**  \n- Safe Zone: Z > 2.99  \n- Grey Zone: 1.81 ≤ Z ≤ 2.99  \n- Distressed Zone: Z < 1.81")

    with st.expander("Investing Analysis Code"):
        st.subheader("Fetching Tickers & Features")
        st.code("""
tickers = get_sp500_tickers()
df = extract_features(tickers)
""")
        st.write("Scrapes S&P 500 list and pulls Dividend Yield, Price, Stability for each.")

        st.subheader("Clustering Overview")
        st.write("""
We use KMeans with **n_clusters=3** to segment stocks into three groups:
- **Cluster 0 (Income Focus):** Stocks with higher dividend yields and moderate stability.
- **Cluster 1 (Growth Focus):** Stocks with higher expected returns and higher volatility.
- **Cluster 2 (Stability Focus):** Stocks with lower volatility and moderate returns.

This allows us to recommend stocks based on different investment styles.
""")

        st.subheader("Recommendation Logic")
        st.code("""
selected = recommend_stocks(clustered, budget, model, preferences, min_price, max_price)
""")
        st.write("Applies price & preference filters, then allocates budget across top picks.")

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
        st.subheader("Input preferences below for personalized investment analysis:")
        budget = st.number_input("Investment Budget ($)", min_value=0)
        investment_priority = st.selectbox(
            "Select Investment Priority",
            ['Dividend Yield', 'Expected Return', 'Stability']
        )
        min_price = st.number_input("Minimum Stock Price ($)", min_value=0, value=20)
        max_price = st.number_input("Maximum Stock Price ($)", min_value=0, value=500)

        if st.button("Get Stock Recommendations"):
            tickers = get_sp500_tickers()
            df_features = extract_features(tickers)
            model, clustered = perform_clustering(df_features)

            st.subheader("How Clustering Works")
            st.write("""
            Stocks are grouped into clusters based on similarities in their dividend yield, expected return, and stability (beta).
            We recommend stocks from the 'best' cluster that matches your selected priority.
            """)

            st.subheader("Cluster Visualization (3D)")
            fig = plt.figure(figsize=(15, 15))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(clustered['Dividend Yield'], clustered['Expected Return'], clustered['Stability'], 
                       c=clustered['Cluster'], cmap='viridis')

            ax.set_xlabel('Dividend Yield')
            ax.set_ylabel('Expected Return')
            ax.set_zlabel('Stability')
            ax.set_title('Stock Clusters in 3D')

            for cluster_num in clustered['Cluster'].unique():
                cluster_data = clustered[clustered['Cluster'] == cluster_num]
                center_x = cluster_data['Dividend Yield'].mean()
                center_y = cluster_data['Expected Return'].mean()
                center_z = cluster_data['Stability'].mean()
                ax.text(center_x, center_y, center_z, f'Cluster {cluster_num}', fontsize=12, weight='bold', 
                        ha='center', va='center', bbox=dict(facecolor='white', alpha=0.6, edgecolor='black'))

            st.pyplot(fig)

            preferences = {'priority': investment_priority}
            recommended_stocks = recommend_stocks(clustered, budget, model, preferences, min_price, max_price)

            st.subheader("Top Stock Picks for Your Budget and Preferences")
            st.write(recommended_stocks)

    else:
        explain_backend()

if __name__ == "__main__":
    main()
