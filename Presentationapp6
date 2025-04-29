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
        st.write("Scrapes the S&P 500 list and pulls Dividend Yield, Price, and Stability for each ticker.")

        st.subheader("Clustering Overview")
        st.write("""
We apply **KMeans** with **n_clusters=3** to segment our universe into three distinct groups:

- **Cluster 0 (Income Focus):**  
  High dividend yields with moderate stability.

- **Cluster 1 (Growth Focus):**  
  Elevated expected returns (price + earnings growth), but higher volatility.

- **Cluster 2 (Stability Focus):**  
  Lower volatility and stable returns, with moderate yields.

This replaces any elbow-method step: we fix three clusters to correspond to these three investment styles.
""")

        st.subheader("Recommendation Logic")
        st.code("""
selected = recommend_stocks(clustered, budget, model, preferences, min_price, max_price)
""")
        st.write("Filters by price & user preferences, then allocates your budget evenly across the top picks in the chosen cluster.")

# --- New Function: Sector Competitor Explorer ---
def sector_competitor_explorer():
    st.title("📈 Sector Competitor Explorer (Custom Dataset)")

    # Load your dataset
    try:
        trimmed_df = pd.read_csv("your_cleaned_trimmed_df.csv")
    except Exception as e:
        st.error("❌ Veri dosyası yüklenemedi. Lütfen 'your_cleaned_trimmed_df.csv' dosyasının projede olduğuna emin olun.")
        return

    # Ticker seçimi
    ticker_input = st.text_input("Enter a Ticker to Find Sector Competitors", "AAPL").upper()

    if st.button("Find Competitors"):
        if ticker_input in trimmed_df['ticker'].values:
            sector = trimmed_df.loc[trimmed_df['ticker'] == ticker_input, 'sector'].values[0]
            competitors = trimmed_df[trimmed_df['sector'] == sector]
            st.success(f"Sector: {sector}")
            st.write(f"Found {len(competitors)} companies in this sector:")
            st.dataframe(competitors[['ticker', 'sector', 'profitability_ratio']].reset_index(drop=True))
        else:
            st.error("Ticker not found in your dataset!")
# --- New Function: Hidden Competitor Neural Map ---

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

def hidden_competitor_neural_map():
    st.title("🧠 Hidden Competitor Neural Map")

    # --- Step 0: Load Data ---
    try:
        trimmed_df = pd.read_csv("your_cleaned_trimmed_df.csv")
        umap_embeddings_3d = np.load("your_umap_embeddings.npy")
    except Exception as e:
        st.error("❌ Veri dosyaları yüklenemedi. Lütfen CSV ve NPY dosyalarının bulunduğundan emin olun.")
        st.stop()

    # --- Step 1: Create Plot DataFrame ---
    plot_df_3d = pd.DataFrame({
        'x': umap_embeddings_3d[:, 0],
        'y': umap_embeddings_3d[:, 1],
        'z': umap_embeddings_3d[:, 2],
        'ticker': trimmed_df['ticker'],
        'sector': trimmed_df['sector'],
        'cluster': trimmed_df['hidden_competitor_cluster']
    })

    # --- Step 2: Display Mode Selection ---
    view_mode = st.radio("Choose view type:", ["🔥 Sector Density Heatmap", "🌐 All Industry Cluster Map"])

    if view_mode == "🔥 Sector Density Heatmap":
        sectors = sorted(plot_df_3d['sector'].unique())
        selected_sector = st.sidebar.selectbox("Select a sector to view density heatmap:", sectors)

        sector_data = plot_df_3d[plot_df_3d['sector'] == selected_sector]

        if len(sector_data) < 10:
            st.warning("Not enough data points in this sector to compute density.")
            return

        xyz = np.vstack([sector_data['x'], sector_data['y'], sector_data['z']]).T
        kde = KernelDensity(bandwidth=0.5, kernel='gaussian')
        kde.fit(xyz)
        density = np.exp(kde.score_samples(xyz))
        sector_data = sector_data.copy()
        sector_data['density'] = density

        fig = px.scatter_3d(
            sector_data,
            x='x', y='y', z='z',
            color='density',
            color_continuous_scale='Hot',
            text='ticker',
            hover_data=['ticker', 'cluster', 'density'],
            title=f"🔥 3D Density Heatmap for {selected_sector} Sector",
            width=1000,
            height=800
        )

        fig.update_layout(
            scene=dict(
                xaxis=dict(backgroundcolor='white', gridcolor='lightgrey'),
                yaxis=dict(backgroundcolor='white', gridcolor='lightgrey'),
                zaxis=dict(backgroundcolor='white', gridcolor='lightgrey'),
                bgcolor='white'
            ),
            paper_bgcolor='white',
            plot_bgcolor='white',
            font=dict(color='black', size=12),
            coloraxis_colorbar=dict(
                title="Density",
                tickvals=[density.min(), density.mean(), density.max()],
                ticktext=["Low", "Medium", "High"]
            )
        )

        st.plotly_chart(fig, use_container_width=True)

    elif view_mode == "🌐 All Industry Cluster Map":
        # Choose a ticker to highlight (optional)
        selected_ticker = st.sidebar.selectbox("Highlight specific ticker (optional):", ["None"] + plot_df_3d['ticker'].tolist())
        plot_df_3d['highlight'] = np.where(plot_df_3d['ticker'] == selected_ticker, 'Selected', 'Normal')

        cluster_ids = plot_df_3d['cluster'].unique()
        colors = px.colors.qualitative.Plotly
        color_map = {cid: colors[i % len(colors)] for i, cid in enumerate(cluster_ids)}

        fig = go.Figure()

        for cluster_id in cluster_ids:
            cluster_data = plot_df_3d[plot_df_3d['cluster'] == cluster_id]
            fig.add_trace(go.Scatter3d(
                x=cluster_data['x'],
                y=cluster_data['y'],
                z=cluster_data['z'],
                mode='markers',
                marker=dict(
                    size=5,
                    color=color_map[cluster_id],
                    opacity=0.8,
                    line=dict(width=0.5, color='white')
                ),
                name=f'Cluster {cluster_id}',
                text=cluster_data['ticker'] + " (" + cluster_data['sector'] + ")",
                hoverinfo='text'
            ))

        sector_centers = plot_df_3d.groupby('sector')[['x', 'y', 'z']].mean()
        for sector, row in sector_centers.iterrows():
            fig.add_trace(go.Scatter3d(
                x=[row['x']],
                y=[row['y']],
                z=[row['z']],
                mode='text',
                text=[sector],
                textposition='top center',
                textfont=dict(size=14, color='black'),
                showlegend=False
            ))

        fig.update_layout(
            title='🌐 3D Hidden Competitor Map – All Sectors & Clusters',
            scene=dict(xaxis_title='UMAP-1', yaxis_title='UMAP-2', zaxis_title='UMAP-3'),
            width=1100,
            height=900,
            showlegend=False
        )

        st.plotly_chart(fig, use_container_width=True)

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
