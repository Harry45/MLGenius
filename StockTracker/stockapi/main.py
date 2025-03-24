from fastapi import FastAPI
from pydantic import BaseModel
import os
from dotenv import load_dotenv
import requests

load_dotenv()

API_KEY = os.getenv("API_KEY")

app = FastAPI()


def get_price(ticker: str):
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={ticker}&interval=5min&apikey={API_KEY}"
    request = requests.get(url)
    data = request.json()
    return data


class StockRequest(BaseModel):
    ticker: str


@app.post("/get_stock")
async def get_stock(stock_request: StockRequest):
    data = get_price(stock_request.ticker)
    return data
