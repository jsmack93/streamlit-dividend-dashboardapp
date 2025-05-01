# ============================================
# Imports and Page Config
# ============================================
import streamlit as st
st.set_page_config(page_title="Financial Dashboard — Christine, Omar, Emre (BA870)", layout="wide")

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import requests
from bs4 import BeautifulSoup
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy import stats
from sklearn.datasets import make_blobs
import plotly.graph_objects as go
import plotly.express as px
from sklearn.neighbors import KernelDensity

# ============================================
# Dividend Dashboard Functions
# ============================================
def display_dividend_dashboard(ticker: str):
    ticker_obj = yf.Ticker(ticker)
    info = ticker_obj.info

    st.subheader("Company Overview")
    st.write(info.get('longBusinessSummary', "No overview available."))

    st.subheader("Dividend History (Last 10 Entries)")
    dividends = ticker_obj.dividends

    if dividends.empty:
        st.write("No dividend data available for this ticker.")
    else:
        recent_dividends = dividends.tail(10)
        st.write(recent_dividends)

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(recent_dividends.index, recent_dividends.values)
        ax.set_title("Dividend History (Last 10 Entries)")
        ax.set_xlabel("Date")
        ax.set_ylabel("Dividend ($)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)

    st.subheader("Price History (Last 1 Year)")
    history = ticker_obj.history(period="1y")

    if history.empty:
        st.write("No price data available for this ticker.")
    else:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(history.index, history['Close'], label="Close Price")
        ax.set_title("Price History (Last 1 Year)")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price ($)")
        ax.legend()
        st.pyplot(fig)

    st.subheader("Key Financial Metrics")
    eps = info.get('trailingEps')
    dividend_rate = info.get('dividendRate')
    dividend_yield = info.get('dividendYield')

    payout_ratio = (dividend_rate / eps) if eps and dividend_rate else None

    st.write("Trailing EPS:", eps if eps is not None else "N/A")
    st.write("Dividend Rate:", dividend_rate if dividend_rate is not None else "N/A")
    st.write("Dividend Yield:", dividend_yield if dividend_yield is not None else "N/A")
    st.write("Dividend Payout Ratio:", round(payout_ratio, 2) if payout_ratio else "N/A")

# ============================================
# Altman Z-Score Functions
# ============================================
def compute_altman_z(ticker: str):
    ticker_obj = yf.Ticker(ticker)
    bs = ticker_obj.balance_sheet
    fs = ticker_obj.financials
    info = ticker_obj.info

    def fetch_balance(keys):
        for key in keys:
            for row in bs.index:
                if row.strip().lower() == key.strip().lower():
                    return bs.loc[row][0]
        return None

    def fetch_financials(keys):
        for key in keys:
            for row in fs.index:
                if row.strip().lower() == key.strip().lower():
                    return fs.loc[row][0]
        return None

    total_assets = fetch_balance(["Total Assets"])
    total_liabilities = fetch_balance(["Total Liabilities", "Total Liabilities Net Minority Interest"])
    current_assets = fetch_balance(["Current Assets", "Total Current Assets"])
    current_liabilities = fetch_balance(["Current Liabilities", "Total Current Liabilities"])
    retained_earnings = fetch_balance(["Retained Earnings"])
    ebit = fetch_financials(["EBIT", "Operating Income"])
    sales = fetch_financials(["Total Revenue", "Sales"])

    share_price = info.get('regularMarketPrice')
    shares_outstanding = info.get('sharesOutstanding')
    market_cap = share_price * shares_outstanding if share_price and shares_outstanding else None

    if not all([total_assets, total_liabilities, market_cap]):
        return None, "Essential data missing for computation."

    working_capital = (current_assets - current_liabilities) if current_assets and current_liabilities else 0
    ratio1 = working_capital / total_assets
    ratio2 = retained_earnings / total_assets
    ratio3 = ebit / total_assets
    ratio4 = market_cap / total_liabilities
    ratio5 = sales / total_assets

    z_score = 1.2 * ratio1 + 1.4 * ratio2 + 3.3 * ratio3 + 0.6 * ratio4 + ratio5

    if z_score > 2.99:
        classification = "Safe Zone"
    elif z_score >= 1.81:
        classification = "Grey Zone"
    else:
        classification = "Distressed Zone"

    return z_score, classification

# ============================================
# Investing Analysis Functions 
# ============================================

import numpy as np
import pandas as pd
import yfinance as yf
import requests
from bs4 import BeautifulSoup
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def extract_features(tickers):
    """
    Extracts Dividend Yield, Price, Beta (Stability), and computes Expected Return
    as Dividend Yield + Earnings Growth.
    """
    records = []
    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).info
            dy = info.get('dividendYield', np.nan)  # Dividend Yield (decimal form, e.g., 0.02)
            growth = info.get('earningsGrowth', np.nan)  # Earnings growth (e.g., 0.08)
            price = info.get('regularMarketPrice', np.nan)  # Current Price
            beta = info.get('beta', np.nan)  # Beta (stability measure)

            # Correct Expected Return definition
            expected_return = (dy or 0) + (growth or 0)
        except Exception:
            dy, growth, price, beta, expected_return = np.nan, np.nan, np.nan, np.nan, np.nan
        records.append([ticker, dy, price, beta, expected_return])

    return pd.DataFrame(records, columns=['Ticker', 'Dividend Yield', 'Price', 'Stability', 'Expected Return'])

def remove_outliers(df, columns):
    """
    Removes outliers from the specified columns using the Z-Score method.
    """
    z_scores = np.abs(stats.zscore(df[columns].dropna()))
    df_clean = df[(z_scores < 3).all(axis=1)]  # Keep rows where all z-scores < 3
    return df_clean

def perform_clustering(df):
    """
    Clusters stocks based on Dividend Yield, Expected Return, and Stability.
    """
    df_clean = df.dropna(subset=['Dividend Yield', 'Expected Return', 'Stability'])

    # Remove outliers
    df_clean = remove_outliers(df_clean, ['Dividend Yield', 'Expected Return', 'Stability'])

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(df_clean[['Dividend Yield', 'Expected Return', 'Stability']])

    model = KMeans(n_clusters=3, random_state=42)
    df_clean['Cluster'] = model.fit_predict(features_scaled)

    return model, df_clean

def recommend_stocks(df, budget, model=None, preferences=None, min_price_per_stock=20, max_price_per_stock=500):
    """
    Recommends a selection of stocks within budget and preference constraints.
    """
    df_clean = df.dropna(subset=['Dividend Yield', 'Expected Return', 'Stability'])

    # Remove outliers
    df_clean = remove_outliers(df_clean, ['Dividend Yield', 'Expected Return', 'Stability'])

    # Sort by user preference
    if preferences:
        priority = preferences.get('priority')
        if priority == 'Dividend Yield':
            df_clean = df_clean.sort_values('Dividend Yield', ascending=False)
        elif priority == 'Expected Return':
            df_clean = df_clean.sort_values('Expected Return', ascending=False)
        elif priority == 'Stability':
            df_clean = df_clean.sort_values('Stability', ascending=False)

    # Filter based on clustering if model is provided
    if model:
        features = df_clean[['Dividend Yield', 'Expected Return', 'Stability']]
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        df_clean['Cluster'] = model.predict(features_scaled)
        best_cluster = df_clean['Cluster'].mode()[0]
        df_clean = df_clean[df_clean['Cluster'] == best_cluster]

    # Filter by price constraints
    df_clean = df_clean[(df_clean['Price'] >= min_price_per_stock) & 
                        (df_clean['Price'] <= max_price_per_stock)]

    # Select top 5
    selected = df_clean.head(5)
    allocation = budget / len(selected) if len(selected) > 0 else 0
    selected['Allocation'] = allocation

    return selected

def get_sp500_tickers():
    """
    Scrapes the list of S&P 500 companies from Wikipedia.
    """
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    table = soup.find('table', {'id': 'constituents'})
    df = pd.read_html(str(table))[0]
    return df['Symbol'].tolist() 
  
#Sector Competitor Explorer

def sector_competitor_explorer():
    st.title("📈 Sector Competitor Explorer (Custom Dataset)")

    try:
        trimmed_df = pd.read_csv("your_cleaned_trimmed_df.csv")
    except Exception as e:
        st.error("❌ Data file could not be loaded. Please make sure 'your_cleaned_trimmed_df.csv' exists.")
        return

    ticker_input = st.text_input("Enter a Ticker to Find Sector Competitors", "AAPL").upper()

    if st.button("Find Competitors"):
        if ticker_input in trimmed_df['ticker'].values:
            sector = trimmed_df.loc[trimmed_df['ticker'] == ticker_input, 'sector'].values[0]
            competitors = trimmed_df[trimmed_df['sector'] == sector]
            st.success(f"Sector: {sector}")
            st.write(f"Found {len(competitors)} companies in this sector:")
            st.dataframe(competitors[['ticker', 'sector', 'profitability_ratio']].reset_index(drop=True))
        else:
            st.error("❌ Ticker not found in the dataset.")

# --- New Function: Hidden Competitor Neural Map ---
def hidden_competitor_neural_map():
    st.title("🧠 Hidden Competitor Neural Map")

    try:
        trimmed_df = pd.read_csv("your_cleaned_trimmed_df.csv")
        umap_embeddings_3d = np.load("your_umap_embeddings.npy")
    except Exception as e:
        st.error("❌ Data files not found. Please check your CSV and NPY files.")
        st.stop()

    plot_df_3d = pd.DataFrame({
        'x': umap_embeddings_3d[:, 0],
        'y': umap_embeddings_3d[:, 1],
        'z': umap_embeddings_3d[:, 2],
        'ticker': trimmed_df['ticker'],
        'sector': trimmed_df['sector'],
        'cluster': trimmed_df['hidden_competitor_cluster']
    })

    view_mode = st.radio("Choose View Mode", ["🔥 Sector Density Heatmap", "🌐 All Industry Cluster Map"])

    if view_mode == "🔥 Sector Density Heatmap":
        sectors = sorted(plot_df_3d['sector'].unique())
        selected_sector = st.sidebar.selectbox("Select Sector", sectors)
        sector_data = plot_df_3d[plot_df_3d['sector'] == selected_sector]
        if len(sector_data) < 10:
            st.warning("Not enough data points for density.")
            return
        xyz = np.vstack([sector_data['x'], sector_data['y'], sector_data['z']]).T
        kde = KernelDensity(bandwidth=0.5, kernel='gaussian').fit(xyz)
        density = np.exp(kde.score_samples(xyz))
        sector_data['density'] = density

        fig = px.scatter_3d(
            sector_data, x='x', y='y', z='z', color='density',
            color_continuous_scale='Hot', text='ticker',
            hover_data=['ticker', 'cluster', 'density'],
            title=f"🔥 Density Heatmap - {selected_sector} Sector"
        )
        st.plotly_chart(fig, use_container_width=True)

    else:
        cluster_ids = plot_df_3d['cluster'].unique()
        colors = px.colors.qualitative.Plotly
        color_map = {cid: colors[i % len(colors)] for i, cid in enumerate(cluster_ids)}

        fig = go.Figure()
        for cluster_id in cluster_ids:
            cluster_data = plot_df_3d[plot_df_3d['cluster'] == cluster_id]
            fig.add_trace(go.Scatter3d(
                x=cluster_data['x'], y=cluster_data['y'], z=cluster_data['z'],
                mode='markers',
                marker=dict(size=5, color=color_map[cluster_id], opacity=0.8),
                text=cluster_data['ticker'] + " (" + cluster_data['sector'] + ")",
                hoverinfo='text'
            ))

        fig.update_layout(
            title="🌐 Hidden Competitor Map — All Industries",
            scene=dict(xaxis_title='UMAP-1', yaxis_title='UMAP-2', zaxis_title='UMAP-3'),
            width=1100, height=900, showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)

# ============================================
# Explain Backend Functions
# ============================================
def explain_backend():
    st.title("🛠️ Backend Code Explanation")

    st.markdown("""
This section explains the backend logic for each component of the Financial Dashboard app.
The app uses data from Yahoo Finance and custom datasets to deliver analytical insights and recommendations.
    """)

    st.markdown("## 📦 Imports & Setup")
    st.code("""
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from bs4 import BeautifulSoup
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy import stats
    """)
    st.write("These libraries are used for building the user interface, retrieving financial data, performing clustering, and rendering visualizations.")

    # === Dividend Dashboard ===
    with st.expander("💰 Dividend Dashboard Code"):
        st.write("Fetches and displays dividend history, company overview, and financial ratios for a given ticker using the yfinance API.")
        st.code("def display_dividend_dashboard(ticker: str):")
        st.markdown("""
- Uses `yf.Ticker().dividends` to get dividend history  
- Uses `matplotlib` to plot recent dividend bars  
- Calculates dividend payout ratio using EPS and dividend rate  
        """)

    # === Altman Z-Score ===
    with st.expander("📉 Altman Z-Score Code"):
        st.write("Retrieves key financial statement values and applies the Altman Z-Score formula to assess bankruptcy risk.")
        st.code("def compute_altman_z(ticker: str):")
        st.markdown("""
- Parses balance sheet and income statement  
- Calculates five key ratios  
- Combines ratios using Altman Z-Score formula  
- Classifies result into Safe, Grey, or Distressed zones  
        """)

    # === Investing Analysis ===
    with st.expander("📈 Investing Analysis Code"):
        st.write("Fetches S&P 500 tickers, retrieves financial metrics, clusters them, and recommends stocks based on user preferences.")
        st.markdown("""
- Uses `extract_features()` to collect Dividend Yield, Growth, Beta  
- `perform_clustering()` groups stocks into 3 clusters  
- `recommend_stocks()` filters and allocates portfolio based on user budget and strategy  
- Calculates expected annual dividend income  
        """)

    # === Sector Competitor Explorer ===
    with st.expander("🏷️ Sector Competitor Explorer Code"):
        st.write("Finds companies in the same sector based on the user's selected ticker symbol from a custom CSV dataset.")
        st.code("""
def sector_competitor_explorer():
    trimmed_df = pd.read_csv("your_cleaned_trimmed_df.csv")
    ticker_input = st.text_input("Enter a Ticker", "AAPL").upper()

    if st.button("Find Competitors"):
        if ticker_input in trimmed_df['ticker'].values:
            sector = trimmed_df.loc[trimmed_df['ticker'] == ticker_input, 'sector'].values[0]
            competitors = trimmed_df[trimmed_df['sector'] == sector]
            st.dataframe(competitors[['ticker', 'sector', 'profitability_ratio']])
        else:
            st.error("Ticker not found in the dataset.")
        """)

    # === Hidden Competitor Neural Map ===
    with st.expander("🌌 Hidden Competitor Neural Map Code"):
        st.write("Visualizes companies in a 3D space using UMAP embeddings based on business descriptions and clusters.")
        st.markdown("""
- Loads embeddings from `.npy` file and joins with company info  
- Displays a 3D scatter plot using Plotly  
- Offers two views: full industry map or sector-specific density heatmap  
        """)


# ============================================
# Streamlit Main App
# ============================================
def main():
    st.title("🏦 Financial Dashboard — Christine, Omar, Emre (BA870)")

    page = st.sidebar.radio(
        "Navigation", 
        ["Dividend Dashboard", "Altman Z-Score", "Investing Analysis", "Sector Competitor Explorer", "Hidden Competitor Neural Map", "Explain Backend"]
    )

    if page == "Dividend Dashboard":
        ticker = st.text_input("Enter Ticker", "AAPL")
        if st.button("Show Dividend Info"):
            display_dividend_dashboard(ticker)

    elif page == "Altman Z-Score":
        ticker = st.text_input("Enter Ticker for Z-Score", "AAPL")
        if st.button("Compute Altman Z-Score"):
            z_score, classification = compute_altman_z(ticker)
            if z_score:
                st.success(f"Altman Z-Score: {z_score:.2f}")
                st.info(f"Classification: {classification}")
            else:
                st.error(f"Error: {classification}")

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
            Stocks are grouped into clusters based on similarities in their dividend yield, expected return (based on financial metrics), and stability (volatility measured by beta).
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

    elif page == "Sector Competitor Explorer":
        sector_competitor_explorer()

    elif page == "Hidden Competitor Neural Map":
        hidden_competitor_neural_map()

    elif page == "Explain Backend":
        explain_backend()


if __name__ == "__main__":
    main()
