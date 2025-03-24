import streamlit as st
import requests
import pandas as pd

backend = "http://stockapi:8000/get_stock"
st.set_page_config(layout="wide")


def get_listings():
    url = "https://www.alphavantage.co/query?function=LISTING_STATUS&apikey=demo"
    data = pd.read_csv(url)
    return data


companies = get_listings()


query = st.text_input("Choose your company from the list below.")

if query:
    mask = companies.map(lambda x: query in str(x).lower()).any(axis=1)
    companies = companies[mask]

st.data_editor(companies, hide_index=True, column_order=list(companies.columns))


companies_ticker = companies["symbol"].values
ticker = st.selectbox(
    "Choose the company ticker from the list below.", companies_ticker
)

st.title("Stock Price App")


if st.button("Get Price"):
    response = requests.post(backend, json={"ticker": ticker})
    if response.status_code == 200:
        stock_data = response.json()
        meta_data = stock_data["Meta Data"]
        latest_price = stock_data["Meta Data"]["3. Last Refreshed"]
        price = stock_data["Time Series (5min)"][latest_price]["2. high"]
        st.write(f"The price for {meta_data['2. Symbol']} is $ {price}")
    else:
        st.write("Error fetching the stock price")
