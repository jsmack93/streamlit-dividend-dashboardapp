# presentation_app.py

import streamlit as st
import yfinance as yf
import pandas as pd

# --- Page functions ---

def dividend_dashboard():
    # User inputs ticker symbol for dividend history
    ticker = st.text_input("Dividend Dashboard - Enter Ticker", "AAPL")
    if ticker:
        div = yf.Ticker(ticker).dividends
        st.line_chart(div)

def altman_zscore_dashboard():
    # Inputs for Altman Z-Score components
    st.write("### Altman Z-Score Calculator")
    wc = st.number_input("Working Capital / Total Assets", value=0.0)
    re = st.number_input("Retained Earnings / Total Assets", value=0.0)
    ebit = st.number_input("EBIT / Total Assets", value=0.0)
    mve = st.number_input("Market Value of Equity / Book Liabilities", value=0.0)
    sales = st.number_input("Sales / Total Assets", value=0.0)
    if st.button("Compute Z-Score"):
        z = 1.2*wc + 1.4*re + 3.3*ebit + 0.6*mve + 1.0*sales
        st.write(f"**Altman Z-Score:** {z:.2f}")

def investing_dashboard():
    # Historical price chart for investing
    st.write("### Simple Investing Dashboard")
    ticker = st.text_input("Investing Dashboard - Enter Ticker", "MSFT")
    start = st.date_input("Start Date")
    end = st.date_input("End Date")
    if ticker and start < end:
        data = yf.download(ticker, start=start, end=end)
        st.line_chart(data['Close'])

# --- Main multipage logic ---

st.sidebar.title("Navigate")
page = st.sidebar.radio("Go to", [
    "Dividend Code", "Dividend Explanation",
    "Altman Code", "Altman Explanation",
    "Investing Code", "Investing Explanation"
])

if page == "Dividend Code":
    st.header("Dividend Dashboard - Code View")
    with st.echo():
        dividend_dashboard()

elif page == "Dividend Explanation":
    st.header("Dividend Dashboard - Explanation")
    code = '''
def dividend_dashboard():
    ticker = st.text_input("Dividend Dashboard - Enter Ticker", "AAPL")
    if ticker:
        div = yf.Ticker(ticker).dividends
        st.line_chart(div)
'''
    st.code(code, language='python')
    st.write("This page lets the user input a stock ticker to fetch dividend history using yfinance, then displays it as a line chart.")

elif page == "Altman Code":
    st.header("Altman Z-Score Calculator - Code View")
    with st.echo():
        altman_zscore_dashboard()

elif page == "Altman Explanation":
    st.header("Altman Z-Score Calculator - Explanation")
    code = '''
def altman_zscore_dashboard():
    wc = st.number_input("Working Capital / Total Assets", value=0.0)
    re = st.number_input("Retained Earnings / Total Assets", value=0.0)
    ebit = st.number_input("EBIT / Total Assets", value=0.0)
    mve = st.number_input("Market Value of Equity / Book Liabilities", value=0.0)
    sales = st.number_input("Sales / Total Assets", value=0.0)
    if st.button("Compute Z-Score"):
        z = 1.2*wc + 1.4*re + 3.3*ebit + 0.6*mve + 1.0*sales
        st.write(f"**Altman Z-Score:** {z:.2f}")
'''
    st.code(code, language='python')
    st.write("This calculator takes five financial ratios, applies the Altman Z-Score formula, and outputs the company's Z-Score for bankruptcy risk assessment.")

elif page == "Investing Code":
    st.header("Investing Dashboard - Code View")
    with st.echo():
        investing_dashboard()

elif page == "Investing Explanation":
    st.header("Investing Dashboard - Explanation")
    code = '''
def investing_dashboard():
    ticker = st.text_input("Investing Dashboard - Enter Ticker", "MSFT")
    start = st.date_input("Start Date")
    end = st.date_input("End Date")
    if ticker and start < end:
        data = yf.download(ticker, start=start, end=end)
        st.line_chart(data['Close'])
'''
    st.code(code, language='python')
    st.write("This page fetches historical closing prices for a user-entered ticker over a specified date range and plots a time series chart for investment analysis.")
